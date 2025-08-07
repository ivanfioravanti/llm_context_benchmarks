#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import benchmark_common as common


def run_benchmark(model_url, context_file, kv_bit=None, max_tokens=200):
    """Run MLX benchmark for a given context file."""
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

    # Only add kv-bit if explicitly specified
    if kv_bit is not None:
        cmd.extend(["--kv-bit", str(kv_bit)])

    cmd.extend(["--prompt", "-"])

    # Start timing
    start_time = time.time()

    try:
        with open(context_file, "r") as f:
            result = subprocess.run(cmd, stdin=f, capture_output=True, text=True, timeout=600)  # 10 minute timeout

        # Calculate total wall time
        total_wall_time = time.time() - start_time

        if result.returncode != 0:
            print(f"Error running benchmark: {result.stderr}")
            return None

        # Parse the output - MLX outputs to both stdout and stderr
        output = result.stdout + result.stderr

        # The generated text is in stdout, metrics are in stderr
        generated_text = result.stdout.strip()

        # Extract metrics using regex (from stderr + stdout combined)
        prompt_match = re.search(r"Prompt:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec", output)
        gen_match = re.search(r"Generation:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec", output)
        memory_match = re.search(r"Peak memory:\s*([\d.]+)\s*GB", output)

        if not all([prompt_match, gen_match, memory_match]):
            print(f"Failed to parse output for {context_file}")
            return None

        # Debug logging
        print(
            f"  Prompt: {prompt_match.group(1)} tokens in {float(prompt_match.group(1))/float(prompt_match.group(2)):.2f}s = {prompt_match.group(2)} t/s"
        )
        print(
            f"  Generation: {gen_match.group(1)} tokens in {float(gen_match.group(1))/float(gen_match.group(2)):.2f}s = {gen_match.group(2)} t/s"
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
        print(f"Timeout running benchmark")
        return None
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return None


def check_mlx_installed():
    """Check if MLX-LM is installed."""
    try:
        import mlx_lm

        return True
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Run MLX benchmarks on context files")
    parser.add_argument("model", help="MLX model URL (e.g., mlx-community/Qwen3-0.6B-4bit)")
    parser.add_argument(
        "--contexts",
        type=str,
        default="2,4,8,16",
        help="Comma-separated list of context sizes to benchmark (default: 2,4,8,16)",
    )
    parser.add_argument("--kv-bit", type=int, default=None, help="KV cache bit size (optional, e.g., 4 or 8)")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate (default: 200)")
    parser.add_argument("--output-csv", default="benchmark_results.csv", help="Output CSV file")
    parser.add_argument("--output-chart", default="benchmark_chart.png", help="Output chart file")
    parser.add_argument("--save-responses", action="store_true", help="Save raw model responses to files")

    args = parser.parse_args()

    # Check if MLX-LM is installed
    if not check_mlx_installed():
        print("MLX-LM is not installed. Please install it with: pip install mlx-lm")
        return

    # Extract model name from URL and create output directory
    model_name = args.model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"benchmark_mlx_{model_name}_{timestamp}")
    output_dir.mkdir(exist_ok=True)

    # Update output paths to use the new directory
    csv_path = output_dir / args.output_csv
    chart_path = output_dir / args.output_chart
    tweet_path = output_dir / "tweet.txt"
    table_path = output_dir / "table.txt"
    hardware_path = output_dir / "hardware_info.json"

    # Find context files using common module
    context_files = common.find_context_files(args.contexts)
    if not context_files:
        return

    # Get hardware information
    print("\nCollecting hardware information...")
    hardware_info = common.get_hardware_info()
    hardware_str = common.format_hardware_string(hardware_info)
    print(f"Hardware: {hardware_str}")

    # Save hardware info
    common.save_hardware_info(hardware_info, hardware_path)

    # Run benchmarks
    results = []
    for file in context_files:
        result = run_benchmark(args.model, file, args.kv_bit, args.max_tokens)
        if result:
            results.append(result)

            # Save the generated text if requested
            if args.save_responses:
                output_filename = output_dir / f"generated_{result['context_size']}.txt"
                common.save_generated_text(result, args.model, output_filename, "MLX")

    if not results:
        print("No successful benchmark results")
        return

    # Save to CSV
    common.save_results_csv(results, csv_path)

    # Create chart
    chart_result = common.create_chart_mlx(results, model_name, hardware_info, str(chart_path))
    print(f"Chart saved to {chart_path}")

    # Generate tweet text
    tweet = common.generate_tweet_text(results, model_name, "MLX", hardware_info)
    print("\n--- Tweet Text ---")
    print(tweet)

    # Save tweet to file
    with open(tweet_path, "w") as f:
        f.write(tweet)
    print(f"\nTweet text saved to {tweet_path}")

    # Generate and display table (with memory info for MLX)
    table = common.generate_table(results, model_name, "MLX", hardware_info, include_memory=True)
    print("\n--- Table for X/Twitter Thread ---")
    print(table)

    # Save table to file
    with open(table_path, "w") as f:
        f.write(table)
    print(f"\nTable saved to {table_path}")

    print(f"\n✅ All outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
