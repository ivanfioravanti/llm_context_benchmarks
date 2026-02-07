#!/usr/bin/env python3
"""Common utilities for LLM benchmarking scripts."""

import argparse
import json
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def find_context_files(contexts_arg=None):
    """Find context files based on user input or auto-discover.

    Args:
        contexts_arg: Comma-separated string of context sizes or None for auto-discovery

    Returns:
        List of Path objects for context files, sorted by size
    """
    if contexts_arg:
        # User specified which contexts to run
        context_sizes = [size.strip() for size in contexts_arg.split(",")]
        context_files = []
        for size in context_sizes:
            file_path = Path(f"{size}k.txt")
            if file_path.exists():
                context_files.append(file_path)
            else:
                print(f"Warning: {file_path} not found, skipping...")

        if not context_files:
            print("No valid context files found from specified sizes")
            return []

        # Sort by size
        context_files = sorted(context_files, key=lambda x: float(x.stem[:-1]))
    else:
        # Find all .txt files in current directory
        try:
            # Filter files that have a numeric prefix followed by 'k' (e.g., 0.5k.txt, 2k.txt)
            def is_valid_context_file(f):
                stem = f.stem
                if len(stem) > 1 and stem.endswith('k'):
                    try:
                        float(stem[:-1])
                        return True
                    except ValueError:
                        return False
                return False

            context_files = sorted(
                [f for f in Path(".").glob("*.txt") if is_valid_context_file(f)],
                key=lambda x: float(x.stem[:-1]),
            )
        except:
            context_files = []

        if not context_files:
            print("No valid context files (e.g., 2k.txt, 4k.txt) found in current directory")
            return []

    print(f"Will benchmark context files: {[f.name for f in context_files]}")
    return context_files


def save_hardware_info(hardware_info, output_path):
    """Save hardware info to JSON file."""
    with open(output_path, "w") as f:
        json.dump(hardware_info, f, indent=2)
    print(f"Hardware info saved to {output_path}")


def save_generated_text(result, model_name, output_path, framework=""):
    """Save generated text to a file.

    Args:
        result: Benchmark result dictionary
        model_name: Name of the model
        output_path: Path to save the file
        framework: Framework name for display (e.g., "MLX", "Ollama CLI")
    """
    with open(output_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        if framework:
            f.write(f"Framework: {framework}\n")
        f.write(f"Context size: {result['context_size']}\n")
        f.write(f"Tokens generated: {result.get('generation_tokens', 0)}\n")

        # Handle different metric names
        if "eval_duration" in result:
            f.write(f"Generation time: {result['eval_duration']:.2f}s\n")
        elif "total_time" in result:
            f.write(f"Total time: {result['total_time']:.2f}s\n")

        f.write(f"Generation TPS: {result.get('generation_tps', 0):.1f} t/s\n")

        # Add MLX-specific metrics if present
        if "peak_memory_gb" in result:
            f.write(f"Peak memory: {result['peak_memory_gb']:.1f} GB\n")

        f.write("-" * 80 + "\n\n")
        f.write(result.get("generated_text", ""))
    print(f"Generated text saved to {output_path}")


def save_results_csv(results, csv_path, exclude_fields=None):
    """Save benchmark results to CSV, excluding specified fields.

    Args:
        results: List of result dictionaries
        csv_path: Path to save CSV file
        exclude_fields: List of field names to exclude from CSV (default: ['generated_text'])
    """
    import csv

    if exclude_fields is None:
        exclude_fields = ["generated_text"]

    if not results:
        print("Warning: No results to save to CSV")
        return

    with open(csv_path, "w", newline="") as f:
        # Create a list of fieldnames excluding specified fields
        fieldnames = [k for k in results[0].keys() if k not in exclude_fields]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Write rows without the excluded fields
        for row in results:
            csv_row = {k: v for k, v in row.items() if k not in exclude_fields}
            writer.writerow(csv_row)
    print(f"Results saved to {csv_path}")


def generate_xpost_text(results, model_name, framework, hardware_info=None):
    """Generate X post text with results.

    Args:
        results: List of benchmark results
        model_name: Name of the model
        framework: Framework name (e.g., "Ollama API", "Ollama CLI", "MLX")
        hardware_info: Hardware information dictionary
    """
    xpost = f"{model_name} {framework} Benchmark Results\n"

    # Add hardware info if available
    if hardware_info:
        hardware_str = format_hardware_string(hardware_info)
        xpost += f"Hardware: {hardware_str}\n"

    xpost += "\n"

    total_tokens = 0
    for r in sorted(results, key=lambda x: float(x["context_size"][:-1])):
        # Handle N/A prompt TPS for LM Studio
        if r.get("prompt_tps", 0) == 0 and "EXPERIMENTAL" in framework:
            prompt_part = f"pp {r.get('prompt_tokens', 0)} tok"
        else:
            prompt_part = f"pp {r['prompt_tps']:.0f}"

        line = f"{r['context_size']} {prompt_part} tg {r['generation_tps']:.0f} t/s"

        # Add memory information if available
        if "peak_memory_gb" in r:
            line += f" {r['peak_memory_gb']:.1f}GB"
        
        xpost += line + "\n"
        total_tokens += r.get("generation_tokens", 0)

    xpost += f"\nTotal generated tokens: {total_tokens}"

    return xpost.strip()


# Add a backward compatibility alias
generate_tweet_text = generate_xpost_text


def generate_table(results, model_name, framework, hardware_info=None, include_memory=False):
    """Generate a formatted table for posting to X/Twitter.

    Args:
        results: List of benchmark results
        model_name: Name of the model
        framework: Framework name (e.g., "Ollama API", "Ollama CLI", "MLX")
        hardware_info: Hardware information dictionary
        include_memory: Whether to include memory column (for MLX)
    """
    # Create header
    table = f"{model_name} {framework} Benchmark Results\n"

    # Add hardware info if available
    if hardware_info:
        hardware_str = format_hardware_string(hardware_info)
        table += f"Hardware: {hardware_str}\n"

    total_tokens = 0
    if include_memory:
        table += "\nContext | Prompt TPS | Gen TPS | Gen Tokens | Memory\n"
        table += "--------|------------|---------|------------|--------\n"

        # Add data rows with memory
        for r in sorted(results, key=lambda x: float(x["context_size"][:-1])):
            gen_tokens = r.get("generation_tokens", 0)
            # Handle N/A prompt TPS for LM Studio
            if r.get("prompt_tps", 0) == 0 and "EXPERIMENTAL" in framework:
                prompt_str = f"{r.get('prompt_tokens', 0)} tok"
            else:
                prompt_str = f"{r['prompt_tps']:>10.1f}"
            table += f"{r['context_size']:>7} | {prompt_str:>10} | {r['generation_tps']:>7.1f} | {gen_tokens:>10} | {r.get('peak_memory_gb', 0):>6.1f} GB\n"
            total_tokens += gen_tokens
    else:
        # Check if we need special handling for LM Studio
        if "EXPERIMENTAL" in framework:
            table += "\nContext | Prompt Tokens | Gen TPS | Gen Tokens | Total Time\n"
            table += "--------|---------------|---------|------------|------------\n"
        else:
            table += "\nContext | Prompt TPS | Gen TPS | Gen Tokens | Total Time\n"
            table += "--------|------------|---------|------------|------------\n"

        # Add data rows with total time
        for r in sorted(results, key=lambda x: float(x["context_size"][:-1])):
            total_time = r.get("total_time", r.get("wall_time", 0))
            gen_tokens = r.get("generation_tokens", 0)
            # Handle N/A prompt TPS for LM Studio
            if r.get("prompt_tps", 0) == 0 and "EXPERIMENTAL" in framework:
                prompt_str = f"{r.get('prompt_tokens', 0)} tok"
                table += f"{r['context_size']:>7} | {prompt_str:>13} | {r['generation_tps']:>7.1f} | {gen_tokens:>10} | {total_time:>9.1f}s\n"
            else:
                table += f"{r['context_size']:>7} | {r['prompt_tps']:>10.1f} | {r['generation_tps']:>7.1f} | {gen_tokens:>10} | {total_time:>9.1f}s\n"
            total_tokens += gen_tokens

    table += f"\nTotal generated tokens: {total_tokens}"

    return table


def create_chart_ollama(results, model_name, hardware_info, output_path="benchmark_chart.png", framework="Ollama"):
    """Create a chart for Ollama benchmarks with timing information."""
    # Sort results by context size
    context_sizes = []
    prompt_tps = []
    gen_tps = []
    total_times = []
    generation_tokens = []

    for r in sorted(results, key=lambda x: float(x["context_size"][:-1])):
        context_sizes.append(r["context_size"])
        prompt_tps.append(r["prompt_tps"])
        gen_tps.append(r["generation_tps"])
        total_times.append(r.get("total_time", r.get("wall_time", 0)))
        generation_tokens.append(r["generation_tokens"])

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), gridspec_kw={"height_ratios": [1, 1, 1]})

    # Model name and hardware in title
    hardware_str = format_hardware_string(hardware_info)
    fig.suptitle(f"{model_name} {framework} Testing\n{hardware_str}", fontsize=16, fontweight="bold")

    x = np.arange(len(context_sizes))

    # First subplot - Prompt TPS
    ax1.set_title("Prompt Tokens per Second", fontsize=14, pad=10)
    color1 = "#2196F3"  # Blue for Ollama
    ax1.plot(x, prompt_tps, "o-", color=color1, linewidth=2, markersize=8)
    ax1.set_ylabel("Tokens/sec", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    # Add value labels for prompt
    if prompt_tps:
        max_prompt = max(prompt_tps) if prompt_tps else 1
        for i, p in enumerate(prompt_tps):
            ax1.text(
                i,
                p + max_prompt * 0.03,
                f"{p:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=color1,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(context_sizes)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(prompt_tps) * 1.15 if prompt_tps and max(prompt_tps) > 0 else 1)

    # Second subplot - Generation TPS
    ax2.set_title("Generation Tokens per Second", fontsize=14, pad=10)
    color2 = "#00BCD4"  # Cyan for Ollama
    ax2.plot(x, gen_tps, "o-", color=color2, linewidth=2, markersize=8)
    ax2.set_ylabel("Tokens/sec", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Add value labels for generation
    if gen_tps:
        max_gen = max(gen_tps) if gen_tps else 1
        for i, g in enumerate(gen_tps):
            ax2.text(
                i,
                g + max_gen * 0.03,
                f"{g:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=color2,
            )

    ax2.set_xticks(x)
    ax2.set_xticklabels(context_sizes)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(gen_tps) * 1.15 if gen_tps and max(gen_tps) > 0 else 1)

    # Third subplot - Total Time with Tokens Generated
    ax3.set_title("Total Processing Time & Tokens Generated", fontsize=14, pad=10)

    # Bar chart for total time (left y-axis)
    color_time = "#2196F3"
    bars = ax3.bar(x, total_times, color=color_time, width=0.6, alpha=0.7, label="Total Time")

    # Add value labels on bars
    if total_times:
        max_time = max(total_times) if total_times else 1
        for i, (bar, time_val) in enumerate(zip(bars, total_times)):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max_time * 0.02,
                f"{time_val:.1f}s",
                ha="center",
                va="bottom",
                fontsize=9,
                color=color_time,
            )

    ax3.set_xticks(x)
    ax3.set_xticklabels(context_sizes)
    ax3.set_ylabel("Time (seconds)", color=color_time)
    ax3.tick_params(axis="y", labelcolor=color_time)
    ax3.set_ylim(0, max(total_times) * 1.2 if total_times and max(total_times) > 0 else 1)

    # Create second y-axis for tokens generated
    ax3_right = ax3.twinx()
    color_tokens = "#FF6B35"
    ax3_right.plot(
        x,
        generation_tokens,
        "o-",
        color=color_tokens,
        linewidth=2,
        markersize=8,
        label="Tokens Generated",
    )
    ax3_right.set_ylabel("Tokens Generated", color=color_tokens)
    ax3_right.tick_params(axis="y", labelcolor=color_tokens)

    # Add value labels for tokens positioned lower than bar labels
    if generation_tokens:
        for i, tokens in enumerate(generation_tokens):
            ax3_right.text(
                i,
                tokens + max(generation_tokens) * 0.02 if generation_tokens else 0,
                f"{tokens}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=color_tokens,
            )

    # Set right y-axis scale so the highest token point is positioned lower than the highest bar
    max_tokens = max(generation_tokens) if generation_tokens else 1
    if max_tokens > 0:
        # Scale to 1.5 times the max value so the highest point appears lower
        ax3_right.set_ylim(0, max_tokens * 1.5)
    else:
        ax3_right.set_ylim(0, 1)

    # Add grid
    ax3.grid(True, axis="y", alpha=0.3)

    # Add legends
    ax3.legend(loc="upper left")
    ax3_right.legend(loc="upper right")

    # Adjust layout with custom padding to prevent overlap
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95, hspace=0.4, wspace=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


def create_chart_mlx(results, model_name, hardware_info, output_path="benchmark_chart.png"):
    """Create a chart for MLX benchmarks with memory information."""
    # Sort results by context size
    context_sizes = []
    prompt_tps = []
    gen_tps = []
    peak_memory = []
    generation_tokens = []
    total_times = []

    for r in sorted(results, key=lambda x: float(x["context_size"][:-1])):
        context_sizes.append(r["context_size"])
        prompt_tps.append(r["prompt_tps"])
        gen_tps.append(r["generation_tps"])
        peak_memory.append(r["peak_memory_gb"])
        generation_tokens.append(r["generation_tokens"])
        total_times.append(r.get("total_time", 0))

    # Create figure with four subplots with more spacing
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12), gridspec_kw={"hspace": 0.4, "wspace": 0.3})

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

    # Third subplot - Total Time vs Tokens Generated
    ax3.set_title("Total Processing Time & Tokens Generated", fontsize=14, pad=10)

    # Bar chart for total time (left y-axis)
    color_time = "#9C27B0"
    bars = ax3.bar(x, total_times, color=color_time, width=0.6, alpha=0.7, label="Total Time")

    # Add value labels on bars
    if total_times:
        max_time = max(total_times) if total_times else 1
        for i, (bar, time_val) in enumerate(zip(bars, total_times)):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max_time * 0.02,
                f"{time_val:.1f}s",
                ha="center",
                va="bottom",
                fontsize=9,
                color=color_time,
            )

    ax3.set_xticks(x)
    ax3.set_xticklabels(context_sizes)
    ax3.set_ylabel("Time (seconds)", color=color_time)
    ax3.tick_params(axis="y", labelcolor=color_time)
    ax3.set_ylim(0, max(total_times) * 1.2 if total_times and max(total_times) > 0 else 1)

    # Create second y-axis for tokens generated
    ax3_right = ax3.twinx()
    color_tokens = "#4CAF50"
    ax3_right.plot(x, generation_tokens, "o-", color=color_tokens, linewidth=2, markersize=8, label="Tokens Generated")
    ax3_right.set_ylabel("Tokens Generated", color=color_tokens)
    ax3_right.tick_params(axis="y", labelcolor=color_tokens)

    # Add value labels for tokens positioned lower than bar labels
    if generation_tokens:
        for i, tokens in enumerate(generation_tokens):
            ax3_right.text(
                i,
                tokens + max(generation_tokens) * 0.02 if generation_tokens else 0,
                f"{tokens}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=color_tokens,
            )

    # Set right y-axis scale so the highest token point is positioned lower than the highest bar
    max_tokens = max(generation_tokens) if generation_tokens else 1
    if max_tokens > 0:
        # Scale to 1.5 times the max value so the highest point appears lower
        ax3_right.set_ylim(0, max_tokens * 1.5)
    else:
        ax3_right.set_ylim(0, 1)

    # Add grid
    ax3.grid(True, axis="y", alpha=0.3)

    # Add legends
    ax3.legend(loc="upper left")
    ax3_right.legend(loc="upper right")

    # Fourth subplot - Peak Memory Usage Only
    ax4.set_title("Peak Memory Usage", fontsize=14, pad=10)

    # Bar chart for memory
    color_memory = "#ff9800"
    bars = ax4.bar(x, peak_memory, color=color_memory, width=0.6, alpha=0.7)

    # Add value labels on bars
    if peak_memory:
        max_mem = max(peak_memory) if peak_memory else 1
        for i, (bar, mem) in enumerate(zip(bars, peak_memory)):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max_mem * 0.02,
                f"{mem:.1f} GB",
                ha="center",
                va="bottom",
                fontsize=9,
                color=color_memory,
            )

    ax4.set_xticks(x)
    ax4.set_xticklabels(context_sizes)
    ax4.set_ylabel("Memory (GB)", color=color_memory)
    ax4.tick_params(axis="y", labelcolor=color_memory)
    ax4.set_ylim(0, max(peak_memory) * 1.2 if peak_memory and max(peak_memory) > 0 else 1)

    # Add grid
    ax4.grid(True, axis="y", alpha=0.3)

    # Adjust layout with custom padding to prevent overlap
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95, hspace=0.4, wspace=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path
def create_output_directory(framework_name: str, model_name: str, base_dir: str = "output") -> Path:
    """Create timestamped output directory for benchmark results.
    
    Args:
        framework_name: Name of the framework (e.g., "ollama", "mlx", "llamacpp")
        model_name: Name of the model being benchmarked
        base_dir: Base directory for output (default: "output")
    
    Returns:
        Path object for the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(base_dir)
    base_output_dir.mkdir(exist_ok=True)
    
    # Sanitize model name for filesystem
    model_safe = model_name.replace("/", "_").replace(":", "_")
    output_dir = base_output_dir / f"benchmark_{framework_name}_{model_safe}_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    
    return output_dir


def setup_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common command-line arguments to parser.
    
    Args:
        parser: ArgumentParser instance to add arguments to
    """
    parser.add_argument(
        "--contexts",
        default="0.5,1,2,4,8,16,32",
        help="Comma-separated list of context sizes to benchmark (default: 0.5,1,2,4,8,16,32)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate (default: 200)",
    )
    parser.add_argument(
        "--save-responses",
        action="store_true",
        help="Save model responses to files",
    )
    parser.add_argument(
        "--output-csv",
        default="benchmark_results.csv",
        help="Output CSV filename (default: benchmark_results.csv)",
    )
    parser.add_argument(
        "--output-chart",
        default="benchmark_chart.png",
        help="Output chart filename (default: benchmark_chart.png)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds for each benchmark (default: 3600 = 60 minutes)",
    )


def save_all_outputs(
    results: List[Dict],
    output_dir: Path,
    model_name: str,
    framework: str,
    hardware_info: Dict,
    args: argparse.Namespace,
    include_memory: bool = False,
) -> None:
    """Save all benchmark outputs to files.
    
    Args:
        results: List of benchmark result dictionaries
        output_dir: Directory to save outputs
        model_name: Name of the model
        framework: Framework name (e.g., "Ollama API", "MLX")
        hardware_info: Hardware information dictionary
        args: Command-line arguments namespace
        include_memory: Whether to include memory in table (for MLX)
    """
    # Save hardware info
    hardware_path = output_dir / "hardware_info.json"
    save_hardware_info(hardware_info, hardware_path)
    
    # Save CSV results
    csv_path = output_dir / args.output_csv
    save_results_csv(results, csv_path)
    
    # Generate and save chart
    chart_path = output_dir / args.output_chart
    if "MLX" in framework:
        create_chart_mlx(results, model_name, hardware_info, chart_path)
    else:
        create_chart_ollama(results, model_name, hardware_info, chart_path, framework)
    print(f"Chart saved to {chart_path}")
    
    # Generate and save table
    table = generate_table(results, model_name, framework, hardware_info, include_memory)
    table_path = output_dir / "table.txt"
    with open(table_path, "w") as f:
        f.write(table)
    print(f"Table saved to {table_path}")
    
    # Generate and save X post
    xpost = generate_xpost_text(results, model_name, framework, hardware_info)
    xpost_path = output_dir / "xpost.txt"
    with open(xpost_path, "w") as f:
        f.write(xpost)
    print(f"X Post text saved to {xpost_path}")


def print_benchmark_summary(
    results: List[Dict],
    model_name: str,
    framework: str,
    hardware_info: Dict,
    output_dir: Path,
    total_benchmark_time: float = None,
) -> None:
    """Print benchmark summary to console.
    
    Args:
        results: List of benchmark result dictionaries
        model_name: Name of the model
        framework: Framework name
        hardware_info: Hardware information dictionary
        output_dir: Directory where outputs were saved
        total_benchmark_time: Total time to run all benchmarks in seconds
    """
    # Calculate total generated tokens
    total_generated_tokens = sum(r.get("generation_tokens", 0) for r in results)
    print(f"\n📊 Total generated tokens across all tests: {total_generated_tokens}")
    
    # Print total benchmark time if provided
    if total_benchmark_time is not None:
        print(f"⏱️  Total time to run benchmarks: {total_benchmark_time:.1f} seconds")
    
    # Print summary table
    print("\n" + "=" * 50)
    print("SUMMARY TABLE")
    print("=" * 50)
    table = generate_table(results, model_name, framework, hardware_info)
    print(table)
    
    # Print X post text
    print("\n" + "=" * 50)
    print("X POST TEXT")
    print("=" * 50)
    xpost = generate_xpost_text(results, model_name, framework, hardware_info)
    print(xpost)
    
    print(f"\n✅ All outputs saved to: {output_dir}/")


def validate_model_connection(test_func, *args, **kwargs) -> bool:
    """Generic function to validate model/server connection.
    
    Args:
        test_func: Function to test connection
        *args: Arguments to pass to test function
        **kwargs: Keyword arguments to pass to test function
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        return test_func(*args, **kwargs)
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False
