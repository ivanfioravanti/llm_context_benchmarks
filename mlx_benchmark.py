#!/usr/bin/env python3
"""
Benchmark script for MLX framework on Apple Silicon.

This script runs benchmarks using MLX-LM for efficient inference on Apple Silicon Macs.

Usage:
    python mlx_benchmark.py mlx-community/Qwen3-0.6B-4bit
    python mlx_benchmark.py mlx-community/Qwen3-0.6B-4bit --kv-bit 4 --max-tokens 500
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import benchmark_common as common


def run_benchmark(
    model_url: str,
    context_file: Path,
    kv_bit: Optional[int] = None,
    max_tokens: int = 200,
    timeout: int = 1800,
    max_kv_size: Optional[int] = None,
    trust_remote_code: bool = False,
) -> Optional[Dict]:
    """Run MLX benchmark for a given context file.
    
    Args:
        model_url: MLX model URL
        context_file: Path to the context file
        kv_bit: KV cache bit size (optional)
        max_tokens: Maximum tokens to generate
        timeout: Timeout in seconds
        max_kv_size: Maximum KV cache size in tokens (optional)
    
    Returns:
        Dictionary with benchmark results or None if failed
    """
    print(f"Running benchmark for {context_file}...")

    # Check if we're in a virtual environment and use the correct python
    python_path = sys.executable

    cmd = [
        python_path,
        "-m",
        "mlx_lm",
        "generate",
        "--model",
        model_url,
        "--max-tokens",
        str(max_tokens),
    ]

    if trust_remote_code:
        cmd.append("--trust-remote-code")

    if max_kv_size is not None:
        cmd.extend(["--max-kv-size", str(max_kv_size)])

    if kv_bit is not None:
        cmd.extend(["--kv-bit", str(kv_bit)])

    cmd.extend(["--prompt", "-"])

    # Start timing
    start_time = time.time()

    try:
        with open(context_file, "r") as f:
            result = subprocess.run(
                cmd, stdin=f, capture_output=True, text=True, timeout=timeout
            )

        # Calculate total wall time
        total_wall_time = time.time() - start_time

        if result.returncode != 0:
            print(f"Error running benchmark: {result.stderr}")
            return None

        # Parse the output - MLX outputs metrics to stdout
        output = result.stdout

        # Extract the generated text (everything before the metrics section)
        metrics_start = output.find("\n==========\nPrompt:")
        if metrics_start != -1:
            generated_text = output[:metrics_start].strip()
            metrics_section = output[metrics_start:]
        else:
            # Fallback if separator not found
            generated_text = (
                output.split("Prompt:")[0].strip() if "Prompt:" in output else ""
            )
            metrics_section = output

        # Extract metrics using regex from MLX output format
        prompt_match = re.search(
            r"Prompt:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec", metrics_section
        )
        gen_match = re.search(
            r"Generation:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec",
            metrics_section,
        )
        memory_match = re.search(r"Peak memory:\s*([\d.]+)\s*GB", metrics_section)

        if not all([prompt_match, gen_match, memory_match]):
            print(f"Failed to parse output for {context_file}")
            print(f"Output was: {metrics_section[:500]}")
            return None

        # Display the metrics as MLX outputs them
        print(
            f"  Prompt: {prompt_match.group(1)} tokens, {prompt_match.group(2)} tokens-per-sec"
        )
        print(
            f"  Generation: {gen_match.group(1)} tokens, {gen_match.group(2)} tokens-per-sec"
        )
        print(f"  Peak memory: {memory_match.group(1)} GB")
        print(f"  Total wall time: {total_wall_time:.2f}s")

        return {
            "context_size": Path(context_file).stem,
            "prompt_tokens": int(prompt_match.group(1)),
            "prompt_tps": float(prompt_match.group(2)),
            "generation_tokens": int(gen_match.group(1)),
            "generation_tps": float(gen_match.group(2)),
            "peak_memory_gb": float(memory_match.group(1)),
            "total_time": total_wall_time,
            "generated_text": generated_text,
        }

    except subprocess.TimeoutExpired:
        print(f"Timeout running benchmark for {context_file}")
        return None
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

    # Run benchmarks
    import time
    start_time = time.time()
    results = []
    for file in context_files:
        print(f"\n{'=' * 50}")
        print(f"Benchmarking {file.name}...")
        print(f"{'=' * 50}")
        
        result = run_benchmark(
            args.model,
            file,
            args.kv_bit,
            args.max_tokens,
            args.timeout,
            args.max_kv_size,
            args.trust_remote_code,
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
        results, output_dir, model_name, "MLX", hardware_info, args, include_memory=True
    )

    # Print summary using common function
    common.print_benchmark_summary(results, model_name, "MLX", hardware_info, output_dir, total_benchmark_time)

    return 0


if __name__ == "__main__":
    sys.exit(main())
