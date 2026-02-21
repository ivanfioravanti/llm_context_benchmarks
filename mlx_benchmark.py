#!/usr/bin/env python3
"""
Benchmark script for MLX framework on Apple Silicon.

This script runs benchmarks using MLX-LM for efficient inference on Apple Silicon Macs.
The model is loaded once and reused across all benchmark runs.

Usage:
    python mlx_benchmark.py mlx-community/Qwen3-0.6B-4bit
    python mlx_benchmark.py mlx-community/Qwen3-0.6B-4bit --kv-bit 4 --max-tokens 500
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import benchmark_common as common


def safe_duration(tokens: int, tokens_per_sec: float) -> float:
    """Safely calculate duration from token count and throughput."""
    if tokens <= 0 or tokens_per_sec <= 0:
        return 0.0
    return float(tokens) / float(tokens_per_sec)


def load_model(model_url: str, trust_remote_code: bool = False) -> Tuple:
    """Load an MLX model and tokenizer.

    Args:
        model_url: MLX model URL (e.g., mlx-community/Qwen3-0.6B-4bit)
        trust_remote_code: Allow running custom model/tokenizer code

    Returns:
        Tuple of (model, tokenizer, config)
    """
    import mlx_lm

    model_config = {}
    tokenizer_config = {}
    if trust_remote_code:
        model_config["trust_remote_code"] = True
        tokenizer_config["trust_remote_code"] = True

    model, tokenizer, config = mlx_lm.load(
        model_url,
        model_config=model_config,
        tokenizer_config=tokenizer_config,
        return_config=True,
    )
    return model, tokenizer, config


def prepare_prompt(
    prompt_text: str,
    tokenizer,
    ignore_chat_template: bool = False,
    chat_template_config: Optional[str] = None,
) -> Union[str, List[int]]:
    """Prepare prompt input for mlx_lm.stream_generate.

    Mirrors mlx_lm.generate CLI behavior: when a tokenizer has a chat template,
    wrap the raw text as a user message and add the generation prompt.
    """
    has_chat_template = bool(
        getattr(tokenizer, "has_chat_template", False)
        or getattr(tokenizer, "chat_template", None) is not None
    )
    if ignore_chat_template or not has_chat_template:
        return prompt_text

    template_kwargs = {}
    if chat_template_config:
        template_kwargs = json.loads(chat_template_config)

    messages = [{"role": "user", "content": prompt_text}]
    templated_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        continue_final_message=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    return tokenizer.encode(templated_prompt, add_special_tokens=False)


def run_benchmark(
    model,
    tokenizer,
    context_file: Path,
    kv_bit: Optional[int] = None,
    max_tokens: int = 200,
    max_kv_size: Optional[int] = None,
    ignore_chat_template: bool = False,
    chat_template_config: Optional[str] = None,
) -> Optional[Dict]:
    """Run MLX benchmark for a given context file.

    Args:
        model: Loaded MLX model
        tokenizer: Loaded tokenizer
        context_file: Path to the context file
        kv_bit: KV cache bit size (optional)
        max_tokens: Maximum tokens to generate
        max_kv_size: Maximum KV cache size in tokens (optional)
        ignore_chat_template: If true, skip tokenizer chat template wrapping
        chat_template_config: JSON config passed to apply_chat_template

    Returns:
        Dictionary with benchmark results or None if failed
    """
    import mlx_lm

    print(f"Running benchmark for {context_file}...")

    try:
        with open(context_file, "r") as f:
            prompt = f.read()

        prepared_prompt = prepare_prompt(
            prompt,
            tokenizer,
            ignore_chat_template=ignore_chat_template,
            chat_template_config=chat_template_config,
        )

        kwargs = {}
        if kv_bit is not None:
            kwargs["kv_bits"] = kv_bit
        if max_kv_size is not None:
            kwargs["max_kv_size"] = max_kv_size

        # Reset peak memory before each run to get per-context-size measurement
        import mlx.core as mx
        mx.reset_peak_memory()

        start_time = time.time()

        last_response = None
        generated_text = ""
        for response in mlx_lm.stream_generate(
            model, tokenizer, prompt=prepared_prompt, max_tokens=max_tokens, **kwargs
        ):
            last_response = response
            generated_text += response.text

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
        print(
            f"  Generation: {generation_tokens} tokens, {generation_tps:.3f} tokens-per-sec"
        )
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
    tokenizer,
    batch_sizes: List[int],
    prompt_tokens: int = 2048,
    gen_tokens: int = 256,
    num_trials: int = 3,
    vocab_size: Optional[int] = None,
) -> List[Dict]:
    """Run batch benchmark measuring throughput at different batch sizes.

    Args:
        model: Loaded MLX model
        tokenizer: Loaded tokenizer
        batch_sizes: List of batch sizes to test
        prompt_tokens: Number of prompt tokens per sequence
        gen_tokens: Number of tokens to generate
        num_trials: Number of trials per batch size (takes median)
        vocab_size: Vocabulary size to sample synthetic prompts from

    Returns:
        List of result dicts with batch_size, prompt_tps, generation_tps, peak_memory_gb
    """
    import mlx.core as mx
    from mlx_lm import batch_generate, stream_generate

    if vocab_size is None:
        vocab_size = tokenizer.vocab_size
    batch_results = []

    # Match mlx_lm.benchmark: disable EOS to avoid early stopping on random prompts.
    original_eos = set(getattr(tokenizer, "eos_token_ids", set()))
    restored_via_private_attr = hasattr(tokenizer, "_eos_token_ids")
    if restored_via_private_attr:
        tokenizer._eos_token_ids = set()
    else:
        tokenizer.eos_token_ids = set()
    try:
        for bs in batch_sizes:
            print(f"\n  Batch size {bs} ({num_trials} trials, {prompt_tokens} prompt tokens, {gen_tokens} gen tokens)...")
            mx.reset_peak_memory()

            # Generate prompts once (same for all trials) - like mlx_lm.benchmark
            prompts = mx.random.randint(0, vocab_size, (bs, prompt_tokens)).tolist()

            # Warmup run
            print("    Warmup...")
            if bs == 1:
                for response in stream_generate(model, tokenizer, prompts[0], max_tokens=gen_tokens):
                    pass
            else:
                batch_generate(model, tokenizer, prompts=prompts, max_tokens=gen_tokens)

            # Actual trials - reuse same prompts
            trial_prompt_tps = []
            trial_gen_tps = []

            for trial in range(num_trials):
                if bs == 1:
                    # Pass token IDs directly to stream_generate (like mlx_lm.benchmark)
                    last_response = None
                    for response in stream_generate(
                        model, tokenizer, prompts[0], max_tokens=gen_tokens
                    ):
                        last_response = response
                    if last_response is not None:
                        trial_prompt_tps.append(last_response.prompt_tps)
                        trial_gen_tps.append(last_response.generation_tps)
                        print(f"    Trial {trial + 1}: pp {last_response.prompt_tps:.1f} tg {last_response.generation_tps:.1f} t/s")
                else:
                    # batch_generate expects List[List[int]] of token IDs
                    resp = batch_generate(
                        model, tokenizer, prompts=prompts, max_tokens=gen_tokens
                    )
                    trial_prompt_tps.append(resp.stats.prompt_tps)
                    trial_gen_tps.append(resp.stats.generation_tps)
                    print(f"    Trial {trial + 1}: pp {resp.stats.prompt_tps:.1f} tg {resp.stats.generation_tps:.1f} t/s")

            if trial_prompt_tps:
                avg_prompt_tps = statistics.mean(trial_prompt_tps)
                avg_gen_tps = statistics.mean(trial_gen_tps)
                peak_mem = mx.get_peak_memory() / 1e9

                print(f"  Avg: pp {avg_prompt_tps:.1f} tg {avg_gen_tps:.1f} t/s, peak mem {peak_mem:.2f} GB")

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


def check_mlx_installed() -> bool:
    """Check if MLX-LM is installed."""
    try:
        import mlx_lm
        return True
    except ImportError:
        return False


def main() -> int:
    """Main function to run MLX benchmarks."""
    parser = argparse.ArgumentParser(
        description="Run MLX benchmarks on context files"
    )
    parser.add_argument(
        "model", help="MLX model URL (e.g., mlx-community/Qwen3-0.6B-4bit)"
    )
    parser.add_argument(
        "--kv-bit",
        type=int,
        default=None,
        help="KV cache bit size (optional, e.g., 4 or 8)",
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
        help="Allow running custom model/tokenizer code when loading (HF)",
    )
    parser.add_argument(
        "--ignore-chat-template",
        action="store_true",
        help="Use raw prompt text instead of tokenizer chat template",
    )
    parser.add_argument(
        "--chat-template-config",
        default=None,
        help="JSON config passed to tokenizer.apply_chat_template",
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
        default=256,
        help="Number of tokens to generate per sequence in batch benchmark (default: 256)",
    )
    parser.add_argument(
        "--batch-trials",
        type=int,
        default=3,
        help="Number of trials per batch size, takes median (default: 3)",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        dest="no_batch",
        help="Skip batch benchmark",
    )
    # Backward-compatible alias for older invocations.
    parser.add_argument(
        "--no-batch-benchmark",
        action="store_true",
        dest="no_batch",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-perplexity",
        action="store_true",
        help="Skip perplexity computation",
    )

    # Add common arguments
    common.setup_common_args(parser)

    args = parser.parse_args()

    # Check if MLX-LM is installed
    if not check_mlx_installed():
        print("MLX-LM is not installed. Please install it with: pip install mlx-lm")
        return 1

    # Extract model name from URL
    model_name = args.model.split("/")[-1]

    # Create output directory using common function
    output_dir = common.create_output_directory("mlx", model_name)

    # Find context files using common module
    context_files = common.find_context_files(args.contexts)
    if not context_files:
        return 1

    # Load model once
    print(f"\nLoading model: {args.model}...")
    load_start = time.time()
    try:
        model, tokenizer, model_config = load_model(args.model, args.trust_remote_code)
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
    if args.kv_bit:
        print(f"KV cache bits: {args.kv_bit}")
    if args.max_kv_size:
        print(f"Max KV size: {args.max_kv_size}")
    if args.ignore_chat_template:
        print("Chat template: disabled (raw prompt)")
    else:
        print("Chat template: enabled when tokenizer provides one")

    # Warmup run using the smallest context file
    warmup_file = context_files[0]
    print(f"\nWarmup run ({warmup_file.name}, max_tokens=1)...")
    import mlx_lm

    warmup_kwargs = {}
    if args.kv_bit is not None:
        warmup_kwargs["kv_bits"] = args.kv_bit
    if args.max_kv_size is not None:
        warmup_kwargs["max_kv_size"] = args.max_kv_size

    try:
        with open(warmup_file, "r") as f:
            warmup_prompt = f.read()
        warmup_prepared_prompt = prepare_prompt(
            warmup_prompt,
            tokenizer,
            ignore_chat_template=args.ignore_chat_template,
            chat_template_config=args.chat_template_config,
        )
        for _ in mlx_lm.stream_generate(
            model, tokenizer, prompt=warmup_prepared_prompt, max_tokens=1, **warmup_kwargs
        ):
            pass
        print("Warmup complete (result discarded)")
    except Exception as e:
        print(f"Warmup failed (continuing anyway): {e}")

    perplexity = None
    perplexity_data = None
    if args.no_perplexity:
        print("\nSkipping perplexity (--no-perplexity)")
    else:
        # Compute perplexity
        print("\nComputing perplexity...")
        try:
            from mlx_lm.perplexity import eval_ppl, load_data

            import mlx.core as mx

            np_seed = 123
            import numpy as np_rng
            np_rng.random.seed(np_seed)
            mx.random.seed(np_seed)

            ppl_num_samples = 256
            ppl_seq_length = 512
            ppl_dataset = "allenai/tulu-3-sft-mixture"
            data = load_data(tokenizer, ppl_dataset, num_samples=ppl_num_samples, sequence_length=ppl_seq_length)
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

    # Run batch benchmark
    batch_results = None
    if not args.no_batch:
        batch_sizes = [int(s.strip()) for s in args.batch_sizes.split(",")]
        print(f"\nRunning batch benchmark (sizes: {batch_sizes})...")
        model_vocab_size = (
            model_config.get("vocab_size")
            or model_config.get("text_config", {}).get("vocab_size")
            or tokenizer.vocab_size
        )

        # Set seed for reproducibility (like mlx_lm.benchmark)
        import mlx.core as mx
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

    # Run benchmarks
    start_time = time.time()
    results = []
    for file in context_files:
        print(f"\n{'=' * 50}")
        print(f"Benchmarking {file.name}...")
        print(f"{'=' * 50}")

        result = run_benchmark(
            model,
            tokenizer,
            file,
            args.kv_bit,
            args.max_tokens,
            args.max_kv_size,
            args.ignore_chat_template,
            args.chat_template_config,
        )
        if result:
            results.append(result)

            # Save the generated text if requested
            if args.save_responses:
                output_filename = output_dir / f"response_{result['context_size']}.txt"
                common.save_generated_text(result, args.model, output_filename, "MLX")

    total_benchmark_time = time.time() - start_time

    if not results:
        print("\nNo successful benchmark results")
        return 1

    # Save all outputs using common function
    common.save_all_outputs(
        results, output_dir, model_name, "MLX", hardware_info, args,
        include_memory=True, perplexity=perplexity, perplexity_data=perplexity_data,
        batch_results=batch_results,
    )

    # Print summary using common function
    common.print_benchmark_summary(
        results, model_name, "MLX", hardware_info, output_dir,
        total_benchmark_time, perplexity=perplexity, batch_results=batch_results,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
