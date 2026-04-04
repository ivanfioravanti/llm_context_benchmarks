#!/usr/bin/env python3
"""
Benchmark script for Paroquant inference framework (MLX backend).

This script runs benchmarks using Paroquant's MLX backend, including
context-scaling benchmarks, perplexity computation, and batch inference
performance — matching the feature set of mlx_benchmark.py.

The model is loaded once via Paroquant's loader (which applies pairwise
Givens rotations and quantised layers) and then reused across all runs.

Usage:
    python paroquant_benchmark.py mlx-community/Qwen3-0.6B-4bit
    python paroquant_benchmark.py my-org/My-Model-PQ --max-tokens 500
    python paroquant_benchmark.py my-org/My-Model-PQ --no-batch --no-perplexity
"""

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import benchmark_common as common


# ---------------------------------------------------------------------------
# Install check
# ---------------------------------------------------------------------------

def check_paroquant_installed() -> bool:
    """Check if Paroquant (with MLX extras) is installed."""
    try:
        import paroquant.inference.backends.mlx  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str) -> Tuple:
    """Load a Paroquant MLX model and processor.

    Forces text-only loading (``force_text=True``) so that the returned model
    exposes ``make_cache`` and is fully compatible with ``mlx_lm`` generation,
    batch inference, and perplexity utilities.  Many Paroquant checkpoints
    ship with a ``vision_config`` key in their config (inherited from the
    base Qwen architecture) which would otherwise cause the loader to
    instantiate an ``mlx_vlm`` VLM wrapper that lacks the cache helpers
    ``mlx_lm.stream_generate`` relies on.

    Returns:
        (model, tokenizer, processor, is_vlm)
    """
    from paroquant.inference.backends.mlx.load import load

    model, processor, is_vlm = load(model_path, force_text=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    return model, tokenizer, processor, is_vlm


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def prepare_prompt(
    prompt_text: str,
    tokenizer,
    ignore_chat_template: bool = False,
) -> List[int]:
    """Wrap raw text through the chat template and return token IDs."""
    has_chat_template = bool(
        getattr(tokenizer, "has_chat_template", False)
        or getattr(tokenizer, "chat_template", None) is not None
    )
    if ignore_chat_template or not has_chat_template:
        return tokenizer.encode(prompt_text, add_special_tokens=False)

    messages = [{"role": "user", "content": prompt_text}]
    templated = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        continue_final_message=False,
        add_generation_prompt=True,
    )
    return tokenizer.encode(templated, add_special_tokens=False)


def safe_duration(tokens: int, tokens_per_sec: float) -> float:
    """Safely calculate duration from token count and throughput."""
    if tokens <= 0 or tokens_per_sec <= 0:
        return 0.0
    return float(tokens) / float(tokens_per_sec)


def _count_prompt_tokens(prepared_prompt, tokenizer) -> int:
    """Count the number of tokens in a prepared prompt."""
    if isinstance(prepared_prompt, list):
        return len(prepared_prompt)
    return len(tokenizer.encode(prepared_prompt, add_special_tokens=False))


def _show_prefill_progress(num_tokens: int, stop_event: threading.Event) -> None:
    """Display a live elapsed-time indicator while prefill is running."""
    start = time.time()
    while not stop_event.wait(0.2):
        elapsed = time.time() - start
        sys.stdout.write(f"\r  Prefilling {num_tokens} tokens... {elapsed:.1f}s")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Context-scaling benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    model,
    tokenizer,
    context_file: Path,
    max_tokens: int = 128,
    ignore_chat_template: bool = False,
) -> Optional[Dict]:
    """Run benchmark for a single context file using mlx_lm.stream_generate."""
    import mlx.core as mx
    import mlx_lm

    print(f"Running benchmark for {context_file}...")

    try:
        with open(context_file, "r") as f:
            prompt = f.read()

        prepared_prompt = prepare_prompt(prompt, tokenizer, ignore_chat_template)

        # Reset peak memory before each run
        mx.reset_peak_memory()

        # Count prompt tokens for the progress indicator
        num_prompt_tokens = _count_prompt_tokens(prepared_prompt, tokenizer)

        # Start live prefill progress indicator
        prefill_done = threading.Event()
        progress_thread = threading.Thread(
            target=_show_prefill_progress,
            args=(num_prompt_tokens, prefill_done),
            daemon=True,
        )
        progress_thread.start()

        start_time = time.time()
        first_token_received = False
        token_count = 0

        last_response = None
        generated_text = ""
        for response in mlx_lm.stream_generate(
            model, tokenizer, prompt=prepared_prompt, max_tokens=max_tokens,
        ):
            if not first_token_received:
                first_token_received = True
                prefill_done.set()
                progress_thread.join()
                prefill_time = time.time() - start_time
                pp_tps = num_prompt_tokens / prefill_time if prefill_time > 0 else 0
                sys.stdout.write(
                    f"\r  Prefill: {num_prompt_tokens} tokens in "
                    f"{prefill_time:.2f}s ({pp_tps:.0f} t/s)\n"
                )
                sys.stdout.flush()

            last_response = response
            generated_text += response.text
            token_count += 1
            pct = min(token_count * 100 // max_tokens, 100)
            sys.stdout.write(
                f"\r  Generating: {token_count}/{max_tokens} ({pct}%)"
            )
            sys.stdout.flush()

        # End generation progress line
        if first_token_received:
            sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            # No tokens were generated — stop the prefill indicator
            prefill_done.set()
            progress_thread.join()
            sys.stdout.write("\n")
            sys.stdout.flush()

        total_wall_time = time.time() - start_time

        if last_response is None:
            print(f"No response generated for {context_file}")
            return None

        prompt_tokens = last_response.prompt_tokens
        prompt_tps = last_response.prompt_tps
        generation_tokens = last_response.generation_tokens
        generation_tps = last_response.generation_tps
        peak_memory_gb = last_response.peak_memory
        prompt_eval_duration = safe_duration(prompt_tokens, prompt_tps)

        print(f"  Prompt: {prompt_tokens} tokens, {prompt_tps:.3f} tokens-per-sec")
        print(f"  Generation: {generation_tokens} tokens, {generation_tps:.3f} tokens-per-sec")
        print(f"  Peak memory: {peak_memory_gb:.3f} GB")
        if prompt_eval_duration > 0:
            print(f"  Time to first token: {prompt_eval_duration:.2f}s")
        print(f"  Total wall time: {total_wall_time:.2f}s")

        return {
            "context_size": Path(context_file).stem,
            "prompt_tokens": prompt_tokens,
            "prompt_tps": prompt_tps,
            "generation_tokens": generation_tokens,
            "generation_tps": generation_tps,
            "peak_memory_gb": peak_memory_gb,
            "total_time": total_wall_time,
            "generated_text": generated_text,
            "prompt_eval_duration": prompt_eval_duration,
            "time_to_first_token": prompt_eval_duration,
        }

    except Exception as e:
        print(f"Error running benchmark: {e}")
        return None


# ---------------------------------------------------------------------------
# Batch inference benchmark
# ---------------------------------------------------------------------------

def run_batch_benchmark(
    model,
    tokenizer,
    batch_sizes: List[int],
    prompt_tokens: int = 2048,
    gen_tokens: int = 256,
    num_trials: int = 3,
    vocab_size: Optional[int] = None,
) -> List[Dict]:
    """Measure throughput at different batch sizes.

    Uses mlx_lm.stream_generate (bs=1) and mlx_lm.batch_generate (bs>1).
    Returns list of result dicts with batch_size, prompt_tps, generation_tps,
    peak_memory_gb.
    """
    import mlx.core as mx
    from mlx_lm import batch_generate, stream_generate

    if vocab_size is None:
        vocab_size = tokenizer.vocab_size
    batch_results: List[Dict] = []

    # Disable EOS to avoid early stopping on random prompts
    original_eos = set(getattr(tokenizer, "eos_token_ids", set()))
    restored_via_private_attr = hasattr(tokenizer, "_eos_token_ids")
    if restored_via_private_attr:
        tokenizer._eos_token_ids = set()
    else:
        tokenizer.eos_token_ids = set()

    try:
        for bs in batch_sizes:
            print(
                f"\n  Batch size {bs} ({num_trials} trials, "
                f"{prompt_tokens} prompt tokens, {gen_tokens} gen tokens)..."
            )
            mx.reset_peak_memory()

            # Same random prompts for all trials
            prompts = mx.random.randint(0, vocab_size, (bs, prompt_tokens)).tolist()

            # Warmup
            print("    Warmup...")
            if bs == 1:
                for _ in stream_generate(
                    model, tokenizer, prompts[0], max_tokens=gen_tokens
                ):
                    pass
            else:
                batch_generate(model, tokenizer, prompts=prompts, max_tokens=gen_tokens)

            # Trials
            trial_prompt_tps: List[float] = []
            trial_gen_tps: List[float] = []

            for trial in range(num_trials):
                if bs == 1:
                    last_response = None
                    for response in stream_generate(
                        model, tokenizer, prompts[0], max_tokens=gen_tokens
                    ):
                        last_response = response
                    if last_response is not None:
                        trial_prompt_tps.append(last_response.prompt_tps)
                        trial_gen_tps.append(last_response.generation_tps)
                        print(
                            f"    Trial {trial + 1}: pp {last_response.prompt_tps:.1f} "
                            f"tg {last_response.generation_tps:.1f} t/s"
                        )
                else:
                    resp = batch_generate(
                        model, tokenizer, prompts=prompts, max_tokens=gen_tokens
                    )
                    trial_prompt_tps.append(resp.stats.prompt_tps)
                    trial_gen_tps.append(resp.stats.generation_tps)
                    print(
                        f"    Trial {trial + 1}: pp {resp.stats.prompt_tps:.1f} "
                        f"tg {resp.stats.generation_tps:.1f} t/s"
                    )

            if trial_prompt_tps:
                avg_prompt_tps = statistics.mean(trial_prompt_tps)
                avg_gen_tps = statistics.mean(trial_gen_tps)
                peak_mem = mx.get_peak_memory() / 1e9

                print(
                    f"  Avg: pp {avg_prompt_tps:.1f} tg {avg_gen_tps:.1f} t/s, "
                    f"peak mem {peak_mem:.2f} GB"
                )

                batch_results.append({
                    "batch_size": bs,
                    "prompt_tps": round(avg_prompt_tps, 2),
                    "generation_tps": round(avg_gen_tps, 2),
                    "peak_memory_gb": round(peak_mem, 3),
                })
    finally:
        if restored_via_private_attr:
            tokenizer._eos_token_ids = original_eos
        else:
            tokenizer.eos_token_ids = original_eos

    return batch_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Main function to run Paroquant MLX benchmarks."""
    parser = argparse.ArgumentParser(
        description="Run Paroquant (MLX) benchmarks on context files"
    )
    parser.add_argument(
        "model",
        help="Model path or HuggingFace repo (e.g., my-org/My-Model-PQ)",
    )
    parser.add_argument(
        "--ignore-chat-template",
        action="store_true",
        help="Use raw prompt text instead of tokenizer chat template",
    )

    # Batch benchmark options
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8,16,32",
        help="Comma-separated batch sizes for batch benchmark (default: 1,2,4,8,16,32)",
    )
    parser.add_argument(
        "--batch-prompt-tokens",
        type=int,
        default=2048,
        help="Number of prompt tokens per sequence in batch benchmark (default: 2048)",
    )
    parser.add_argument(
        "--batch-gen-tokens",
        type=int,
        default=256,
        help="Number of tokens to generate per sequence in batch benchmark (default: 256)",
    )
    parser.add_argument(
        "--batch-trials",
        type=int,
        default=3,
        help="Number of trials per batch size, takes mean (default: 3)",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        dest="no_batch",
        help="Skip batch benchmark",
    )
    parser.add_argument(
        "--no-perplexity",
        action="store_true",
        help="Skip perplexity computation",
    )

    # Common arguments
    common.setup_common_args(parser)

    args = parser.parse_args()

    # Check installation
    if not check_paroquant_installed():
        print(
            "Paroquant (MLX) is not installed. "
            "Install with: pip install 'paroquant[mlx]'"
        )
        return 1

    # Extract model name
    model_name = args.model.split("/")[-1]

    # Create output directory
    output_dir = common.create_output_directory("paroquant", model_name)

    # Find context files
    context_files = common.find_context_files(args.contexts)
    if not context_files:
        return 1

    # Load model once via Paroquant's loader
    print(f"\nLoading model: {args.model} (Paroquant MLX)...")
    load_start = time.time()
    try:
        model, tokenizer, processor, is_vlm = load_model(args.model)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return 1
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.1f}s")

    # Hardware info
    print("\nCollecting hardware information...")
    hardware_info = common.get_hardware_info()
    hardware_str = common.format_hardware_string(hardware_info)
    print(f"Hardware: {hardware_str}")
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    if args.ignore_chat_template:
        print("Chat template: disabled (raw prompt)")
    else:
        print("Chat template: enabled when tokenizer provides one")

    # Warmup run using the smallest context file
    import mlx_lm

    warmup_file = context_files[0]
    print(f"\nWarmup run ({warmup_file.name}, max_tokens=1)...")
    try:
        with open(warmup_file, "r") as f:
            warmup_prompt = f.read()
        warmup_prepared = prepare_prompt(
            warmup_prompt, tokenizer, args.ignore_chat_template,
        )
        for _ in mlx_lm.stream_generate(
            model, tokenizer, prompt=warmup_prepared, max_tokens=1,
        ):
            pass
        print("Warmup complete (result discarded)")
    except Exception as e:
        print(f"Warmup failed (continuing anyway): {e}")

    # ------------------------------------------------------------------
    # Perplexity
    # ------------------------------------------------------------------
    perplexity = None
    perplexity_data = None
    if args.no_perplexity:
        print("\nSkipping perplexity (--no-perplexity)")
    else:
        print("\nComputing perplexity...")
        try:
            from mlx_lm.perplexity import eval_ppl, load_data

            import mlx.core as mx
            import numpy as np

            np.random.seed(123)
            mx.random.seed(123)

            ppl_num_samples = 256
            ppl_seq_length = 512
            ppl_dataset = "allenai/tulu-3-sft-mixture"
            data = load_data(
                tokenizer, ppl_dataset,
                num_samples=ppl_num_samples, sequence_length=ppl_seq_length,
            )
            ppl, ppl_se = eval_ppl(model, data, batch_size=8)
            perplexity = float(ppl)
            perplexity_data = {
                "perplexity": perplexity,
                "std_error": float(ppl_se),
                "dataset": ppl_dataset,
                "num_samples": ppl_num_samples,
                "sequence_length": ppl_seq_length,
            }
            print(f"Perplexity: {perplexity:.2f} (±{float(ppl_se):.2f})")
        except Exception as e:
            print(f"Perplexity computation failed (continuing): {e}")

    # ------------------------------------------------------------------
    # Batch benchmark
    # ------------------------------------------------------------------
    batch_results = None
    if not args.no_batch:
        batch_sizes = [int(s.strip()) for s in args.batch_sizes.split(",")]
        print(f"\nRunning batch benchmark (sizes: {batch_sizes})...")

        import mlx.core as mx

        # Determine vocab size from model config or tokenizer
        model_vocab_size = tokenizer.vocab_size
        mx.random.seed(0)

        try:
            batch_results = run_batch_benchmark(
                model,
                tokenizer,
                batch_sizes,
                prompt_tokens=args.batch_prompt_tokens,
                gen_tokens=args.batch_gen_tokens,
                num_trials=args.batch_trials,
                vocab_size=model_vocab_size,
            )
            if batch_results:
                print(f"\nBatch benchmark complete: {len(batch_results)} sizes tested")
            else:
                print("\nBatch benchmark produced no results")
        except Exception as e:
            print(f"\nBatch benchmark failed (continuing): {e}")

    # ------------------------------------------------------------------
    # Context-scaling benchmarks
    # ------------------------------------------------------------------
    start_time = time.time()
    results: List[Dict] = []
    for file in context_files:
        print(f"\n{'=' * 50}")
        print(f"Benchmarking {file.name}...")
        print(f"{'=' * 50}")

        result = run_benchmark(
            model, tokenizer, file, args.max_tokens, args.ignore_chat_template,
        )
        if result:
            results.append(result)

            if args.save_responses:
                output_filename = output_dir / f"response_{result['context_size']}.txt"
                common.save_generated_text(
                    result, args.model, output_filename, "Paroquant"
                )

    total_benchmark_time = time.time() - start_time

    if not results:
        print("\nNo successful benchmark results")
        return 1

    # Save all outputs
    common.save_all_outputs(
        results, output_dir, model_name, "Paroquant", hardware_info, args,
        include_memory=True, perplexity=perplexity, perplexity_data=perplexity_data,
        batch_results=batch_results,
    )

    # Print summary
    common.print_benchmark_summary(
        results, model_name, "Paroquant", hardware_info, output_dir,
        total_benchmark_time, perplexity=perplexity, batch_results=batch_results,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
