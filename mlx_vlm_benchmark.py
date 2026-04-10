#!/usr/bin/env python3
"""
Benchmark script for MLX-VLM framework on Apple Silicon.

This script runs benchmarks using MLX-VLM for efficient inference of
vision-language models on Apple Silicon Macs. Supports KV cache quantization
via the kv-bits parameter, including TurboQuant (fractional bits like 3.5).

The model is loaded once and reused across all benchmark runs.

Usage:
    python mlx_vlm_benchmark.py mlx-community/Qwen2.5-VL-7B-Instruct-4bit
    python mlx_vlm_benchmark.py google/gemma-4-26b-a4b-it --kv-bits 4
    python mlx_vlm_benchmark.py google/gemma-4-26b-a4b-it --kv-bits 3.5
    python mlx_vlm_benchmark.py google/gemma-4-26b-a4b-it --kv-bits 4 --kv-quant-scheme turboquant
"""

import argparse
import statistics
import sys
import threading
import warnings

# Suppress harmless transformers audio mel filter warning (irrelevant for text benchmarking)
warnings.filterwarnings("ignore", message="At least one mel filter has all zero values")

import time
from pathlib import Path
from typing import Dict, List, Optional

import benchmark_common as common


def safe_duration(tokens: int, tokens_per_sec: float) -> float:
    """Safely calculate duration from token count and throughput."""
    if tokens <= 0 or tokens_per_sec <= 0:
        return 0.0
    return float(tokens) / float(tokens_per_sec)


def _show_prefill_progress(num_tokens: int, stop_event: threading.Event) -> None:
    """Display a live elapsed-time indicator while prefill is running."""
    start = time.time()
    while not stop_event.wait(0.2):
        elapsed = time.time() - start
        sys.stdout.write(f"\r  Prefilling {num_tokens} tokens... {elapsed:.1f}s")
        sys.stdout.flush()


def load_model(model_url: str, trust_remote_code: bool = False):
    """Load an MLX-VLM model and processor.

    Args:
        model_url: Model URL (e.g., mlx-community/Qwen2.5-VL-7B-Instruct-4bit)
        trust_remote_code: Allow running custom model/processor code

    Returns:
        Tuple of (model, processor)
    """
    import mlx_vlm

    kwargs = {}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True

    model, processor = mlx_vlm.load(model_url, **kwargs)
    return model, processor


def prepare_prompt(prompt_text: str, processor, model_config, ignore_chat_template: bool = False) -> str:
    """Prepare prompt text, optionally applying the chat template.

    Args:
        prompt_text: Raw prompt text
        processor: The loaded processor
        model_config: Model configuration dict/object
        ignore_chat_template: If true, return raw text without template

    Returns:
        Formatted prompt string
    """
    if ignore_chat_template:
        return prompt_text

    import mlx_vlm

    try:
        return mlx_vlm.apply_chat_template(processor, model_config, prompt_text)
    except Exception:
        return prompt_text


def run_benchmark(
    model,
    processor,
    context_file: Path,
    kv_bits: Optional[float] = None,
    kv_quant_scheme: Optional[str] = None,
    max_tokens: int = 128,
    max_kv_size: Optional[int] = None,
    ignore_chat_template: bool = False,
) -> Optional[Dict]:
    """Run MLX-VLM benchmark for a given context file.

    Args:
        model: Loaded MLX-VLM model
        processor: Loaded processor
        context_file: Path to the context file
        kv_bits: KV cache bit size (optional, supports float like 3.5 for TurboQuant)
        kv_quant_scheme: KV quantization scheme ("uniform" or "turboquant")
        max_tokens: Maximum tokens to generate
        max_kv_size: Maximum KV cache size in tokens (optional)
        ignore_chat_template: If true, skip chat template wrapping

    Returns:
        Dictionary with benchmark results or None if failed
    """
    import mlx.core as mx
    import mlx_vlm

    print(f"Running benchmark for {context_file}...")

    try:
        with open(context_file, "r") as f:
            prompt = f.read()

        formatted_prompt = prepare_prompt(prompt, processor, model.config, ignore_chat_template)

        kwargs = {"max_tokens": max_tokens}
        if kv_bits is not None:
            kwargs["kv_bits"] = kv_bits
        if kv_quant_scheme is not None:
            kwargs["kv_quant_scheme"] = kv_quant_scheme
        if max_kv_size is not None:
            kwargs["max_kv_size"] = max_kv_size

        # Reset peak memory before each run
        mx.reset_peak_memory()

        # We don't know exact token count before generation starts, show generic progress
        prefill_done = threading.Event()
        progress_thread = threading.Thread(
            target=_show_prefill_progress,
            args=(0, prefill_done),
            daemon=True,
        )
        progress_thread.start()

        start_time = time.time()
        first_token_received = False
        token_count = 0

        last_response = None
        generated_text = ""
        for response in mlx_vlm.stream_generate(model, processor, prompt=formatted_prompt, **kwargs):
            if not first_token_received:
                first_token_received = True
                prefill_done.set()
                progress_thread.join()
                prefill_time = time.time() - start_time
                num_prompt_tokens = response.prompt_tokens
                pp_tps = num_prompt_tokens / prefill_time if prefill_time > 0 else 0
                sys.stdout.write(f"\r  Prefill: {num_prompt_tokens} tokens in {prefill_time:.2f}s ({pp_tps:.0f} t/s)\n")
                sys.stdout.flush()

            last_response = response
            generated_text += response.text
            token_count += 1
            pct = min(token_count * 100 // max_tokens, 100)
            sys.stdout.write(f"\r  Generating: {token_count}/{max_tokens} ({pct}%)")
            sys.stdout.flush()

        # End generation progress line
        if first_token_received:
            sys.stdout.write("\n")
            sys.stdout.flush()
        else:
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


def run_batch_benchmark(
    model,
    processor,
    batch_sizes: List[int],
    prompt_tokens: int = 2048,
    gen_tokens: int = 128,
    num_trials: int = 3,
    vocab_size: Optional[int] = None,
) -> List[Dict]:
    """Run batch benchmark measuring throughput at different batch sizes.

    Mirrors `mlx_benchmark.run_batch_benchmark` (mlx_benchmark.py:489-590) but
    against mlx_vlm. The model is driven text-only (no images/audios), so we
    pass `prompts=` as a list of decoded random-token strings, one per batch
    slot, to mlx_vlm.batch_generate. For batch size 1 we use stream_generate
    and collect the final GenerationResult's library-reported TPS.

    Args:
        model: Loaded mlx_vlm model
        processor: mlx_vlm processor (contains tokenizer + vision/audio pre)
        batch_sizes: Concurrency levels to test
        prompt_tokens: Approximate prompt tokens per slot
        gen_tokens: Tokens to generate per slot
        num_trials: Trials per batch size (mean taken)
        vocab_size: Vocab to sample from; auto-detected from config if None

    Returns:
        List of result dicts with batch_size, prompt_tps, generation_tps, peak_memory_gb
    """
    import mlx.core as mx
    import mlx_vlm

    # Resolve the underlying tokenizer (processor wraps one for VLMs)
    tokenizer = getattr(processor, "tokenizer", processor)

    # Resolve vocab size. VLMs often keep vocab under `text_config` in the
    # model config, mirroring mlx_benchmark's fallback chain.
    if vocab_size is None:
        cfg = getattr(model, "config", None)
        cfg_get = (lambda k: getattr(cfg, k, None)) if cfg is not None else (lambda k: None)
        text_cfg = cfg_get("text_config")
        vocab_size = (
            cfg_get("vocab_size")
            or (getattr(text_cfg, "vocab_size", None) if text_cfg is not None else None)
            or getattr(tokenizer, "vocab_size", None)
        )
    if not vocab_size:
        print("  ERROR: could not determine vocab_size — skipping batch benchmark")
        return []

    batch_results = []

    # Disable EOS so random-token prompts don't trigger early stopping mid-
    # generation. Same pattern as mlx_benchmark.py:519-525. Restore in finally.
    _raw_eos = getattr(tokenizer, "eos_token_ids", set())
    original_eos = {_raw_eos} if isinstance(_raw_eos, int) else set(_raw_eos)
    restored_via_private_attr = hasattr(tokenizer, "_eos_token_ids")
    if restored_via_private_attr:
        tokenizer._eos_token_ids = set()
    else:
        try:
            tokenizer.eos_token_ids = set()
        except Exception:
            pass

    try:
        for bs in batch_sizes:
            print(
                f"\n  Batch size {bs} ({num_trials} trials, {prompt_tokens} prompt tokens, "
                f"{gen_tokens} gen tokens)..."
            )
            mx.reset_peak_memory()

            # Generate bs DIFFERENT random token sequences, one per slot, so
            # there's no shared-prefix KV reuse across slots. mlx_vlm expects
            # strings, so decode each row to text — the round-trip may shift
            # the exact token count by a few tokens but stays close enough.
            random_ids = mx.random.randint(0, vocab_size, (bs, prompt_tokens)).tolist()
            try:
                prompts = [tokenizer.decode(ids) for ids in random_ids]
            except Exception as e:
                print(f"    Failed to decode random prompts: {e} — skipping batch size {bs}")
                continue

            # Warmup at this batch size
            print("    Warmup...")
            try:
                if bs == 1:
                    for _ in mlx_vlm.stream_generate(
                        model, processor, prompt=prompts[0], max_tokens=gen_tokens
                    ):
                        pass
                else:
                    mlx_vlm.batch_generate(
                        model, processor, prompts=prompts, max_tokens=gen_tokens, verbose=False
                    )
            except Exception as e:
                print(f"    Warmup failed for batch size {bs}: {e} — skipping")
                continue

            trial_prompt_tps = []
            trial_gen_tps = []

            for trial in range(num_trials):
                try:
                    if bs == 1:
                        last_response = None
                        for response in mlx_vlm.stream_generate(
                            model, processor, prompt=prompts[0], max_tokens=gen_tokens
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
                        resp = mlx_vlm.batch_generate(
                            model,
                            processor,
                            prompts=prompts,
                            max_tokens=gen_tokens,
                            verbose=False,
                        )
                        trial_prompt_tps.append(resp.stats.prompt_tps)
                        trial_gen_tps.append(resp.stats.generation_tps)
                        print(
                            f"    Trial {trial + 1}: pp {resp.stats.prompt_tps:.1f} "
                            f"tg {resp.stats.generation_tps:.1f} t/s"
                        )
                except Exception as e:
                    print(f"    Trial {trial + 1} failed: {e}")
                    continue

            if trial_prompt_tps:
                avg_prompt_tps = statistics.mean(trial_prompt_tps)
                avg_gen_tps = statistics.mean(trial_gen_tps)
                peak_mem = mx.get_peak_memory() / 1e9

                print(
                    f"  Avg: pp {avg_prompt_tps:.1f} tg {avg_gen_tps:.1f} t/s, "
                    f"peak mem {peak_mem:.2f} GB"
                )

                batch_results.append(
                    {
                        "batch_size": bs,
                        "prompt_tps": round(avg_prompt_tps, 2),
                        "generation_tps": round(avg_gen_tps, 2),
                        "peak_memory_gb": round(peak_mem, 3),
                    }
                )
    finally:
        if restored_via_private_attr:
            tokenizer._eos_token_ids = original_eos
        else:
            try:
                tokenizer.eos_token_ids = original_eos
            except Exception:
                pass

    return batch_results


def check_mlx_vlm_installed() -> bool:
    """Check if MLX-VLM is installed."""
    try:
        import mlx_vlm

        return True
    except ImportError:
        return False


def main() -> int:
    """Main function to run MLX-VLM benchmarks."""
    parser = argparse.ArgumentParser(description="Run MLX-VLM benchmarks on context files")
    parser.add_argument("model", help="MLX-VLM model URL (e.g., mlx-community/Qwen2.5-VL-7B-Instruct-4bit)")
    parser.add_argument(
        "--kv-bits",
        type=float,
        default=None,
        help="KV cache bit size (e.g., 4, 8, or 3.5 for TurboQuant)",
    )
    parser.add_argument(
        "--kv-quant-scheme",
        default=None,
        choices=["uniform", "turboquant"],
        help="KV quantization scheme (default: uniform; turboquant for mixed-precision)",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="KV cache size in tokens (optional, e.g., 4096)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow running custom model/processor code when loading (HF)",
    )
    parser.add_argument(
        "--ignore-chat-template",
        action="store_true",
        help="Use raw prompt text instead of chat template",
    )
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
        default=128,
        help="Number of tokens to generate per sequence in batch benchmark (default: 128)",
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
        help="Skip batch benchmark",
    )

    # Add common arguments
    common.setup_common_args(parser)

    args = parser.parse_args()

    # Check if MLX-VLM is installed
    if not check_mlx_vlm_installed():
        print("MLX-VLM is not installed. Please install it with: uv add mlx-vlm")
        return 1

    # Extract model name from URL
    model_name = args.model.split("/")[-1]

    # Create output directory
    output_dir = common.create_output_directory("mlx_vlm", model_name)

    # Find context files
    context_files = common.find_context_files(args.contexts)
    if not context_files:
        return 1

    # Load model once
    print(f"\nLoading model: {args.model}...")
    load_start = time.time()
    try:
        model, processor = load_model(args.model, args.trust_remote_code)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return 1
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.1f}s")

    # Get hardware information
    print("\nCollecting hardware information...")
    hardware_info = common.get_hardware_info()
    hardware_str = common.format_hardware_string(hardware_info)
    print(f"Hardware: {hardware_str}")
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    if args.kv_bits:
        print(f"KV cache bits: {args.kv_bits}")
    if args.kv_quant_scheme:
        print(f"KV quant scheme: {args.kv_quant_scheme}")
    if args.max_kv_size:
        print(f"Max KV size: {args.max_kv_size}")
    if args.ignore_chat_template:
        print("Chat template: disabled (raw prompt)")
    else:
        print("Chat template: enabled when processor provides one")

    # Warmup run using the smallest context file
    warmup_file = context_files[0]
    print(f"\nWarmup run ({warmup_file.name}, max_tokens=1)...")
    import mlx_vlm

    try:
        with open(warmup_file, "r") as f:
            warmup_prompt = f.read()
        warmup_formatted = prepare_prompt(warmup_prompt, processor, model.config, args.ignore_chat_template)
        warmup_kwargs = {"max_tokens": 1}
        if args.kv_bits is not None:
            warmup_kwargs["kv_bits"] = args.kv_bits
        if args.kv_quant_scheme is not None:
            warmup_kwargs["kv_quant_scheme"] = args.kv_quant_scheme
        if args.max_kv_size is not None:
            warmup_kwargs["max_kv_size"] = args.max_kv_size
        for _ in mlx_vlm.stream_generate(model, processor, prompt=warmup_formatted, **warmup_kwargs):
            pass
        print("Warmup complete (result discarded)")
    except Exception as e:
        print(f"Warmup failed (continuing anyway): {e}")

    # Run benchmarks
    start_time = time.time()
    results = []
    for file in context_files:
        print(f"\n{'=' * 50}")
        print(f"Benchmarking {file.name}...")
        print(f"{'=' * 50}")

        result = run_benchmark(
            model,
            processor,
            file,
            kv_bits=args.kv_bits,
            kv_quant_scheme=args.kv_quant_scheme,
            max_tokens=args.max_tokens,
            max_kv_size=args.max_kv_size,
            ignore_chat_template=args.ignore_chat_template,
        )
        if result:
            results.append(result)

            if args.save_responses:
                output_filename = output_dir / f"response_{result['context_size']}.txt"
                common.save_generated_text(result, args.model, output_filename, "MLX-VLM")

    total_benchmark_time = time.time() - start_time

    if not results:
        print("\nNo successful benchmark results")
        return 1

    # Run batch benchmark (throughput at different batch sizes via mlx_vlm.batch_generate)
    batch_results = None
    if not args.no_batch:
        batch_sizes = [int(s.strip()) for s in args.batch_sizes.split(",")]
        print(f"\nRunning batch benchmark (sizes: {batch_sizes})...")

        # Seed MLX RNG for reproducible batch runs (same pattern as mlx_benchmark)
        import mlx.core as mx

        mx.random.seed(0)
        try:
            batch_results = run_batch_benchmark(
                model,
                processor,
                batch_sizes,
                prompt_tokens=args.batch_prompt_tokens,
                gen_tokens=args.batch_gen_tokens,
                num_trials=args.batch_trials,
            )
            if batch_results:
                print(f"\nBatch benchmark complete: {len(batch_results)} sizes tested")
            else:
                print("\nBatch benchmark produced no results")
        except Exception as e:
            print(f"\nBatch benchmark failed (continuing): {e}")

    # Save all outputs
    common.save_all_outputs(
        results,
        output_dir,
        model_name,
        "MLX-VLM",
        hardware_info,
        args,
        include_memory=True,
        batch_results=batch_results,
    )

    # Print summary
    common.print_benchmark_summary(
        results,
        model_name,
        "MLX-VLM",
        hardware_info,
        output_dir,
        total_benchmark_time,
        batch_results=batch_results,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
