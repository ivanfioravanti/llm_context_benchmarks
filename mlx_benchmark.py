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
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

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
        Tuple of (model, tokenizer)
    """
    import mlx_lm

    model_config = {}
    tokenizer_config = {}
    if trust_remote_code:
        model_config["trust_remote_code"] = True
        tokenizer_config["trust_remote_code"] = True

    model, tokenizer = mlx_lm.load(
        model_url,
        model_config=model_config,
        tokenizer_config=tokenizer_config,
    )
    return model, tokenizer


def run_benchmark(
    model,
    tokenizer,
    context_file: Path,
    kv_bit: Optional[int] = None,
    max_tokens: int = 200,
    max_kv_size: Optional[int] = None,
) -> Optional[Dict]:
    """Run MLX benchmark for a given context file.

    Args:
        model: Loaded MLX model
        tokenizer: Loaded tokenizer
        context_file: Path to the context file
        kv_bit: KV cache bit size (optional)
        max_tokens: Maximum tokens to generate
        max_kv_size: Maximum KV cache size in tokens (optional)

    Returns:
        Dictionary with benchmark results or None if failed
    """
    import mlx_lm

    print(f"Running benchmark for {context_file}...")

    try:
        with open(context_file, "r") as f:
            prompt = f.read()

        kwargs = {}
        if kv_bit is not None:
            kwargs["kv_bits"] = kv_bit
        if max_kv_size is not None:
            kwargs["max_kv_size"] = max_kv_size

        start_time = time.time()

        last_response = None
        generated_text = ""
        for response in mlx_lm.stream_generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens, **kwargs
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
        model, tokenizer = load_model(args.model, args.trust_remote_code)
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
        for _ in mlx_lm.stream_generate(
            model, tokenizer, prompt=warmup_prompt, max_tokens=1, **warmup_kwargs
        ):
            pass
        print("Warmup complete (result discarded)")
    except Exception as e:
        print(f"Warmup failed (continuing anyway): {e}")

    # Compute perplexity
    print("\nComputing perplexity...")
    perplexity = None
    perplexity_data = None
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
    )

    # Print summary using common function
    common.print_benchmark_summary(
        results, model_name, "MLX", hardware_info, output_dir,
        total_benchmark_time, perplexity=perplexity,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
