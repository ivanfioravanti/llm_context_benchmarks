#!/usr/bin/env python3
import argparse
import csv
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil


def get_hardware_info():
    """Get hardware information for the system."""
    info = {}

    # Get platform info
    info["platform"] = platform.platform()
    info["processor"] = platform.processor()
    info["machine"] = platform.machine()

    # Get CPU info
    info["cpu_count"] = psutil.cpu_count(logical=False)
    info["cpu_count_logical"] = psutil.cpu_count(logical=True)

    # Get memory info
    mem = psutil.virtual_memory()
    info["memory_gb"] = round(mem.total / (1024**3), 1)

    # Try to get Mac-specific info if on macOS
    if platform.system() == "Darwin":
        try:
            # Get Mac model info
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                output = result.stdout

                # Extract chip info (M1, M2, M3, etc.)
                chip_match = re.search(r"Chip:\s+(.+)", output)
                if chip_match:
                    info["chip"] = chip_match.group(1).strip()

                # Extract model name
                model_match = re.search(r"Model Name:\s+(.+)", output)
                if model_match:
                    info["model"] = model_match.group(1).strip()

                # Extract total number of cores and breakdown
                cores_match = re.search(
                    r"Total Number of Cores:\s+(\d+)(?:\s*\((\d+)\s+performance\s+and\s+(\d+)\s+efficiency\))?",
                    output,
                )
                if cores_match:
                    info["total_cores"] = int(cores_match.group(1))
                    # Check if we have performance and efficiency breakdown
                    if cores_match.group(2) and cores_match.group(3):
                        info["performance_cores"] = int(cores_match.group(2))
                        info["efficiency_cores"] = int(cores_match.group(3))

            # Get GPU cores from SPDisplaysDataType
            gpu_result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, timeout=5
            )
            if gpu_result.returncode == 0:
                gpu_output = gpu_result.stdout
                # Look for GPU cores in display data
                gpu_cores_match = re.search(r"Total Number of Cores:\s+(\d+)", gpu_output)
                if gpu_cores_match:
                    info["gpu_cores"] = int(gpu_cores_match.group(1))
        except:
            pass

    return info


def format_hardware_string(hw_info):
    """Format hardware info into a readable string."""
    parts = []

    # For Mac with Apple Silicon
    if "chip" in hw_info:
        parts.append(hw_info["chip"])
    elif "processor" in hw_info and hw_info["processor"]:
        parts.append(hw_info["processor"][:50])  # Truncate long processor names

    # Add memory
    if "memory_gb" in hw_info:
        parts.append(f"{hw_info['memory_gb']}GB RAM")

    # Add CPU cores with performance/efficiency breakdown if available
    if "performance_cores" in hw_info and "efficiency_cores" in hw_info:
        parts.append(
            f"{hw_info['total_cores']} CPU cores ({hw_info['performance_cores']}P+{hw_info['efficiency_cores']}E)"
        )
    elif "total_cores" in hw_info:
        parts.append(f"{hw_info['total_cores']} CPU cores")
    elif "cpu_count" in hw_info:
        parts.append(f"{hw_info['cpu_count']} CPU cores")

    # Add GPU cores for Apple Silicon
    if "gpu_cores" in hw_info:
        parts.append(f"{hw_info['gpu_cores']} GPU cores")

    return ", ".join(parts) if parts else "Unknown hardware"


def run_benchmark(model_url, context_file, kv_bit=None, max_tokens=16000):
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


def create_chart(results, model_name, hardware_info, output_path="benchmark_chart.png"):
    """Create a chart with separate subplots for prompt TPS, generation TPS, and memory with tokens."""
    # Sort results by context size
    context_sizes = []
    prompt_tps = []
    gen_tps = []
    peak_memory = []
    generation_tokens = []

    for r in sorted(results, key=lambda x: int(x["context_size"][:-1])):
        context_sizes.append(r["context_size"])
        prompt_tps.append(r["prompt_tps"])
        gen_tps.append(r["generation_tps"])
        peak_memory.append(r["peak_memory_gb"])
        generation_tokens.append(r["generation_tokens"])

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), gridspec_kw={"height_ratios": [1, 1, 1]})

    # Model name and hardware in title
    hardware_str = format_hardware_string(hardware_info)
    fig.suptitle(f"{model_name} MLX Testing\n{hardware_str}", fontsize=16, fontweight="bold")

    x = np.arange(len(context_sizes))

    # First subplot - Prompt TPS
    ax1.set_title("Prompt Tokens per Second", fontsize=14, pad=10)
    color1 = "#ff9800"
    ax1.plot(x, prompt_tps, "o-", color=color1, linewidth=2, markersize=8)
    ax1.set_ylabel("Tokens/sec", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    # Add value labels for prompt
    if prompt_tps:
        max_prompt = max(prompt_tps) if prompt_tps else 1
        for i, p in enumerate(prompt_tps):
            ax1.text(i, p + max_prompt * 0.03, f"{p:.1f}", ha="center", va="bottom", fontsize=9, color=color1)

    ax1.set_xticks(x)
    ax1.set_xticklabels(context_sizes)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(prompt_tps) * 1.15 if prompt_tps and max(prompt_tps) > 0 else 1)

    # Second subplot - Generation TPS
    ax2.set_title("Generation Tokens per Second", fontsize=14, pad=10)
    color2 = "#ff5722"
    ax2.plot(x, gen_tps, "o-", color=color2, linewidth=2, markersize=8)
    ax2.set_ylabel("Tokens/sec", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Add value labels for generation
    if gen_tps:
        max_gen = max(gen_tps) if gen_tps else 1
        for i, g in enumerate(gen_tps):
            ax2.text(i, g + max_gen * 0.03, f"{g:.1f}", ha="center", va="bottom", fontsize=9, color=color2)

    ax2.set_xticks(x)
    ax2.set_xticklabels(context_sizes)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(gen_tps) * 1.15 if gen_tps and max(gen_tps) > 0 else 1)

    # Third subplot - Peak Memory Usage with Tokens Generated
    ax3.set_title("Peak Memory Usage & Tokens Generated", fontsize=14, pad=10)

    # Bar chart for memory (left y-axis)
    color_memory = "#ff9800"
    bars = ax3.bar(x, peak_memory, color=color_memory, width=0.6, alpha=0.7, label="Peak Memory")

    # Add value labels on bars
    if peak_memory:
        max_mem = max(peak_memory) if peak_memory else 1
        for i, (bar, mem) in enumerate(zip(bars, peak_memory)):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max_mem * 0.02,
                f"{mem:.1f} GB",
                ha="center",
                va="bottom",
                fontsize=9,
                color=color_memory,
            )

    ax3.set_xticks(x)
    ax3.set_xticklabels(context_sizes)
    ax3.set_ylabel("Memory (GB)", color=color_memory)
    ax3.tick_params(axis="y", labelcolor=color_memory)
    ax3.set_ylim(0, max(peak_memory) * 1.2 if peak_memory and max(peak_memory) > 0 else 1)

    # Create second y-axis for tokens generated
    ax3_right = ax3.twinx()
    color_tokens = "#4CAF50"
    ax3_right.plot(x, generation_tokens, "o-", color=color_tokens, linewidth=2, markersize=8, label="Tokens Generated")
    ax3_right.set_ylabel("Tokens Generated", color=color_tokens)
    ax3_right.tick_params(axis="y", labelcolor=color_tokens)

    # Add value labels for tokens
    if generation_tokens:
        for i, tokens in enumerate(generation_tokens):
            ax3_right.text(
                i,
                tokens + max(generation_tokens) * 0.03 if generation_tokens else 0,
                f"{tokens}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=color_tokens,
            )

    ax3_right.set_ylim(0, max(generation_tokens) * 1.15 if generation_tokens and max(generation_tokens) > 0 else 1)

    # Add grid
    ax3.grid(True, axis="y", alpha=0.3)

    # Add legends
    ax3.legend(loc="upper left")
    ax3_right.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


def generate_tweet_text(results, model_name, hardware_info=None):
    """Generate tweet text with results."""
    tweet = f"{model_name} MLX Benchmark Results\n"

    # Add hardware info if available
    if hardware_info:
        hardware_str = format_hardware_string(hardware_info)
        tweet += f"Hardware: {hardware_str}\n"

    tweet += "\n"

    for r in sorted(results, key=lambda x: int(x["context_size"][:-1])):
        tweet += f"{r['context_size']} Prompt: {r['prompt_tps']:.0f} - Gen: {r['generation_tps']:.0f} t/s\n"

    return tweet.strip()


def generate_table(results, model_name, hardware_info=None):
    """Generate a formatted table for posting to X/Twitter."""
    # Create header
    table = f"{model_name} MLX Benchmark Results\n"

    # Add hardware info if available
    if hardware_info:
        hardware_str = format_hardware_string(hardware_info)
        table += f"Hardware: {hardware_str}\n"

    table += "\nContext | Prompt TPS | Gen TPS | Memory\n"
    table += "--------|------------|---------|--------\n"

    # Add data rows
    for r in sorted(results, key=lambda x: int(x["context_size"][:-1])):
        table += f"{r['context_size']:>7} | {r['prompt_tps']:>10.1f} | {r['generation_tps']:>7.1f} | {r['peak_memory_gb']:>6.1f} GB\n"

    return table


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
        help="Comma-separated list of context sizes to benchmark (e.g., 2,4,8,16). If not specified, all .txt files will be used.",
    )
    parser.add_argument("--kv-bit", type=int, default=None, help="KV cache bit size (optional, e.g., 4 or 8)")
    parser.add_argument("--max-tokens", type=int, default=16000, help="Max tokens to generate (default: 16000)")
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

    # Find context files based on user input
    if args.contexts:
        # User specified which contexts to run
        context_sizes = [size.strip() for size in args.contexts.split(",")]
        context_files = []
        for size in context_sizes:
            file_path = Path(f"{size}k.txt")
            if file_path.exists():
                context_files.append(file_path)
            else:
                print(f"Warning: {file_path} not found, skipping...")

        if not context_files:
            print("No valid context files found from specified sizes")
            return

        # Sort by size
        context_files = sorted(context_files, key=lambda x: int(x.stem[:-1]))
    else:
        # Find all .txt files in current directory
        try:
            context_files = sorted(
                [f for f in Path(".").glob("*.txt") if len(f.stem) > 1 and f.stem[:-1].isdigit()],
                key=lambda x: int(x.stem[:-1]),
            )
        except:
            context_files = []

        if not context_files:
            print("No valid context files (e.g., 2k.txt, 4k.txt) found in current directory")
            return

    print(f"Will benchmark context files: {[f.name for f in context_files]}")

    # Get hardware information
    print("\nCollecting hardware information...")
    hardware_info = get_hardware_info()
    hardware_str = format_hardware_string(hardware_info)
    print(f"Hardware: {hardware_str}")

    # Save hardware info to JSON file
    with open(hardware_path, "w") as f:
        json.dump(hardware_info, f, indent=2)
    print(f"Hardware info saved to {hardware_path}")

    # Run benchmarks
    results = []
    for file in context_files:
        result = run_benchmark(args.model, file, args.kv_bit, args.max_tokens)
        if result:
            results.append(result)

            # Save the generated text to a separate file if requested
            if args.save_responses:
                output_filename = output_dir / f"generated_{result['context_size']}.txt"
                with open(output_filename, "w") as f:
                    f.write(f"Model: {args.model}\n")
                    f.write(f"Context size: {result['context_size']}\n")
                    f.write(f"Tokens generated: {result['generation_tokens']}\n")
                    f.write(f"Generation TPS: {result['generation_tps']:.1f} t/s\n")
                    f.write(f"Peak memory: {result['peak_memory_gb']:.1f} GB\n")
                    f.write(f"Total time: {result['total_time']:.2f}s\n")
                    f.write("-" * 80 + "\n\n")
                    f.write(result["generated_text"])
                print(f"Generated text saved to {output_filename}")

    if not results:
        print("No successful benchmark results")
        return

    # Save to CSV (excluding generated_text field for cleaner CSV)
    with open(csv_path, "w", newline="") as f:
        # Create a list of fieldnames excluding 'generated_text'
        fieldnames = [k for k in results[0].keys() if k != "generated_text"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Write rows without the generated_text field
        for row in results:
            csv_row = {k: v for k, v in row.items() if k != "generated_text"}
            writer.writerow(csv_row)
    print(f"Results saved to {csv_path}")

    # Create chart
    chart_result = create_chart(results, model_name, hardware_info, str(chart_path))
    print(f"Chart saved to {chart_path}")

    # Generate tweet text
    tweet = generate_tweet_text(results, model_name, hardware_info)
    print("\n--- Tweet Text ---")
    print(tweet)

    # Save tweet to file
    with open(tweet_path, "w") as f:
        f.write(tweet)
    print(f"\nTweet text saved to {tweet_path}")

    # Generate and display table
    table = generate_table(results, model_name, hardware_info)
    print("\n--- Table for X/Twitter Thread ---")
    print(table)

    # Save table to file
    with open(table_path, "w") as f:
        f.write(table)
    print(f"\nTable saved to {table_path}")

    print(f"\n✅ All outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
