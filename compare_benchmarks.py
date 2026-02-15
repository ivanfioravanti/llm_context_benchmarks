#!/usr/bin/env python3
"""
Benchmark comparison tool for LLM context benchmarks.

This script aggregates results from multiple benchmark folders and creates
comparison charts to analyze performance across different engines and models.

Usage:
    # Compare all results in output directory
    python compare_benchmarks.py
    
    # Compare specific folders
    python compare_benchmarks.py output/benchmark_ollama_* output/benchmark_mlx_*
    
    # Save to custom location
    python compare_benchmarks.py --output comparison_results
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_benchmark_folder(folder_path: Path) -> Tuple[Dict, str]:
    """Parse a benchmark folder and extract results and metadata.

    Args:
        folder_path: Path to the benchmark folder

    Returns:
        Tuple of (results_dict, display_name)
    """
    # Read hardware info
    hardware_file = folder_path / "hardware_info.json"
    hardware_info = {}
    if hardware_file.exists():
        with open(hardware_file) as f:
            hardware_info = json.load(f)

    # Read benchmark results
    results_file = folder_path / "benchmark_results.csv"
    if not results_file.exists():
        return None, None

    results = []
    with open(results_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert string values to float where applicable
            processed_row = {}
            for key, value in row.items():
                try:
                    if key == "context_size":
                        processed_row[key] = value
                    else:
                        processed_row[key] = float(value)
                except ValueError:
                    processed_row[key] = value
            results.append(processed_row)

    # Extract engine and model from folder name
    folder_name = folder_path.name
    parts = folder_name.split("_")
    if len(parts) >= 3:
        engine = parts[1]  # benchmark_ENGINE_model_timestamp
        model_parts = parts[2:-1]  # Everything between engine and timestamp
        model = "_".join(model_parts)
    else:
        engine = "unknown"
        model = "unknown"

    # Read perplexity data if available
    perplexity_file = folder_path / "perplexity.json"
    perplexity_data = None
    if perplexity_file.exists():
        with open(perplexity_file) as f:
            perplexity_data = json.load(f)

    # Create display name
    display_name = f"{engine}: {model}"

    return {
        "results": results,
        "hardware_info": hardware_info,
        "engine": engine,
        "model": model,
        "folder_name": folder_name,
        "display_name": display_name,
        "perplexity_data": perplexity_data,
    }, display_name


def create_comparison_charts(benchmark_data: List[Dict], output_dir: Path):
    """Create comparison charts from multiple benchmark results."""

    # Check if any benchmark has memory data
    has_memory_data = any(
        any(r.get("peak_memory_gb", 0) > 0 for r in data["results"])
        for data in benchmark_data
    )

    # Check if any benchmark has perplexity data
    has_perplexity_data = any(
        data.get("perplexity_data") is not None
        for data in benchmark_data
    )

    # Set up the plot style
    plt.style.use("default")
    # Determine grid layout based on available data
    num_extra_plots = int(has_memory_data) + int(has_perplexity_data)
    if num_extra_plots == 2:
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 18))
        ax_memory = ax5 if has_memory_data else None
        ax_ppl = ax6 if has_perplexity_data else None
    elif num_extra_plots == 1:
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax_unused)) = plt.subplots(3, 2, figsize=(16, 18))
        ax_unused.set_visible(False)
        if has_memory_data:
            ax_memory = ax5
            ax_ppl = None
        else:
            ax_memory = None
            ax_ppl = ax5
    else:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        ax_memory = None
        ax_ppl = None
    fig.suptitle("LLM Benchmark Comparison", fontsize=16, fontweight="bold")

    # Colors for different benchmarks - using more readable, distinct colors
    readable_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
    ]
    colors = readable_colors[:len(benchmark_data)]
    if len(benchmark_data) > len(readable_colors):
        # Fall back to colormap if we have more data than predefined colors
        colors = plt.cm.tab10(np.linspace(0, 1, len(benchmark_data)))

    # Prepare data for plotting
    for i, data in enumerate(benchmark_data):
        results = data["results"]
        display_name = data["display_name"]
        color = colors[i]

        if not results:
            continue

        # Extract context sizes and metrics
        context_sizes = [r["context_size"] for r in results]
        prompt_tps = [r.get("prompt_tps", 0) for r in results]
        generation_tps = [r.get("generation_tps", 0) for r in results]
        total_times = [r.get("total_time", 0) for r in results]
        ttft_times = [r.get("time_to_first_token", r.get("prompt_eval_duration", 0)) for r in results]
        peak_memory = [r.get("peak_memory_gb", 0) for r in results]

        # Convert context sizes to numbers for sorting
        context_nums = []
        for ctx in context_sizes:
            if ctx.endswith("k"):
                context_nums.append(float(ctx[:-1]) * 1000)
            else:
                context_nums.append(int(ctx))

        # Sort all data by context size
        sorted_data = sorted(zip(context_nums, context_sizes, prompt_tps, generation_tps, total_times, ttft_times, peak_memory))
        context_nums, context_sizes, prompt_tps, generation_tps, total_times, ttft_times, peak_memory = zip(*sorted_data)

        # Plot 1: Prompt Processing Speed
        line1 = ax1.plot(context_sizes, prompt_tps, marker="o", linewidth=2, label=display_name, color=color)[0]
        # Add value annotations
        for x, y in zip(context_sizes, prompt_tps):
            ax1.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color=color)

        # Plot 2: Generation Speed
        line2 = ax2.plot(context_sizes, generation_tps, marker="s", linewidth=2, label=display_name, color=color)[0]
        # Add value annotations
        for x, y in zip(context_sizes, generation_tps):
            ax2.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color=color)

        # Plot 3: Total Time
        line3 = ax3.plot(context_sizes, total_times, marker="^", linewidth=2, label=display_name, color=color)[0]
        # Add value annotations
        for x, y in zip(context_sizes, total_times):
            ax3.annotate(f'{y:.1f}s', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color=color)

        # Plot 4: Time to First Token (TTFT)
        line4 = ax4.plot(context_sizes, ttft_times, marker="d", linewidth=2, label=display_name, color=color)[0]
        # Add value annotations
        for x, y in zip(context_sizes, ttft_times):
            ax4.annotate(f'{y:.2f}s', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color=color)

        if has_memory_data and ax_memory is not None:
            # Plot: Peak Memory Usage
            ax_memory.plot(context_sizes, peak_memory, marker="p", linewidth=2, label=display_name, color=color)
            # Add value annotations
            for x, y in zip(context_sizes, peak_memory):
                if y > 0:  # Only annotate non-zero values
                    ax_memory.annotate(f'{y:.1f}GB', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color=color)

    # Configure subplot 1: Prompt Processing Speed
    ax1.set_title("Prompt Processing Speed", fontweight="bold")
    ax1.set_xlabel("Context Size")
    ax1.set_ylabel("Tokens/sec")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)

    # Configure subplot 2: Generation Speed
    ax2.set_title("Text Generation Speed", fontweight="bold")
    ax2.set_xlabel("Context Size")
    ax2.set_ylabel("Tokens/sec")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="x", rotation=45)

    # Configure subplot 3: Total Time
    ax3.set_title("Total Processing Time", fontweight="bold")
    ax3.set_xlabel("Context Size")
    ax3.set_ylabel("Seconds")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis="x", rotation=45)

    # Configure subplot 4: TTFT
    ax4.set_title("Time to First Token (TTFT)\nLower is better", fontweight="bold")
    ax4.set_xlabel("Context Size")
    ax4.set_ylabel("Time (seconds)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis="x", rotation=45)

    if has_memory_data and ax_memory is not None:
        ax_memory.set_title("Peak Memory Usage", fontweight="bold")
        ax_memory.set_xlabel("Context Size")
        ax_memory.set_ylabel("Memory (GB)")
        ax_memory.legend()
        ax_memory.grid(True, alpha=0.3)
        ax_memory.tick_params(axis="x", rotation=45)

    if has_perplexity_data and ax_ppl is not None:
        # Build perplexity bar chart
        ppl_names = []
        ppl_values = []
        ppl_errors = []
        ppl_colors = []
        for i, data in enumerate(benchmark_data):
            ppl_data = data.get("perplexity_data")
            if ppl_data is not None:
                ppl_names.append(data["display_name"])
                ppl_values.append(ppl_data["perplexity"])
                ppl_errors.append(ppl_data.get("std_error", 0))
                ppl_colors.append(colors[i])

        if ppl_values:
            bars = ax_ppl.bar(range(len(ppl_values)), ppl_values, yerr=ppl_errors,
                              color=ppl_colors, alpha=0.8, capsize=5, width=0.6)
            ax_ppl.set_title("Perplexity (lower is better)", fontweight="bold")
            ax_ppl.set_ylabel("Perplexity")
            ax_ppl.set_xticks(range(len(ppl_names)))
            ax_ppl.set_xticklabels(ppl_names, rotation=30, ha="right", fontsize=9)
            ax_ppl.grid(True, axis="y", alpha=0.3)
            # Add value labels on bars
            for bar, val in zip(bars, ppl_values):
                ax_ppl.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                            f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()

    # Save the chart
    chart_path = output_dir / "comparison_chart.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    print(f"Comparison chart saved to: {chart_path}")

    return chart_path


def create_comparison_table(benchmark_data: List[Dict], output_dir: Path):
    """Create a comparison table with key metrics in X post-friendly format."""

    table_data = []

    for data in benchmark_data:
        results = data["results"]
        hardware_info = data["hardware_info"]
        display_name = data["display_name"]

        if not results:
            continue

        # Calculate average metrics
        avg_prompt_tps = np.mean([r.get("prompt_tps", 0) for r in results])
        avg_generation_tps = np.mean([r.get("generation_tps", 0) for r in results])
        total_tokens_generated = sum([r.get("generation_tokens", 0) for r in results])
        total_time = sum([r.get("total_time", 0) for r in results])

        # Peak memory (max across all context sizes)
        peak_mem_values = [r.get("peak_memory_gb", 0) for r in results]
        peak_memory = max(peak_mem_values) if peak_mem_values else 0

        # Hardware info
        chip = hardware_info.get("chip", "Unknown")
        memory = hardware_info.get("memory_gb", 0)
        cores = hardware_info.get("total_cores", hardware_info.get("cpu_count", 0))

        # Perplexity
        ppl_data = data.get("perplexity_data")
        ppl_str = f"{ppl_data['perplexity']:.2f}" if ppl_data else "N/A"

        table_data.append(
            {
                "Engine/Model": display_name,
                "Hardware": f"{chip}, {memory}GB RAM, {cores} cores",
                "Avg Prompt TPS": f"{avg_prompt_tps:.1f}",
                "Avg Gen TPS": f"{avg_generation_tps:.1f}",
                "Peak Memory": f"{peak_memory:.1f}GB" if peak_memory > 0 else "N/A",
                "Perplexity": ppl_str,
                "Total Tokens": f"{total_tokens_generated:,}",
                "Total Time": f"{total_time:.1f}s",
            }
        )

    # Create DataFrame and save
    df = pd.DataFrame(table_data)

    # Save as CSV
    csv_path = output_dir / "comparison_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Comparison results saved to: {csv_path}")

    # Create X post-friendly comparison text
    xpost_text = "LLM Benchmark Comparison\n"

    # Add hardware info from first benchmark if available
    if benchmark_data and benchmark_data[0].get("hardware_info"):
        hw = benchmark_data[0]["hardware_info"]
        chip = hw.get("chip", "Unknown")
        memory = hw.get("memory_gb", 0)
        xpost_text += f"Hardware: {chip}, {memory}GB RAM\n"

    xpost_text += "\n"

    for entry in table_data:
        xpost_text += f"{entry['Engine/Model']}\n"
        xpost_text += f"  pp {entry['Avg Prompt TPS']} tg {entry['Avg Gen TPS']} t/s"
        if entry["Peak Memory"] != "N/A":
            xpost_text += f" {entry['Peak Memory']}"
        xpost_text += f"\n  {entry['Total Tokens']} tokens in {entry['Total Time']}"
        if entry.get("Perplexity", "N/A") != "N/A":
            xpost_text += f"\n  Perplexity: {entry['Perplexity']}"
        xpost_text += "\n\n"

    table_path = output_dir / "comparison_table.txt"
    with open(table_path, "w") as f:
        f.write(xpost_text.strip())
    print(f"Comparison table saved to: {table_path}")

    # Print to console
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(xpost_text.strip())

    return csv_path, table_path


def find_benchmark_folders(search_paths: List[str] = None) -> List[Path]:
    """Find all benchmark folders to compare."""

    if search_paths:
        # Use provided paths
        folders = []
        for path_pattern in search_paths:
            path = Path(path_pattern)
            if path.is_dir() and path.name.startswith("benchmark_"):
                folders.append(path)
            else:
                # Try glob pattern
                parent = path.parent if path.parent != Path(".") else Path(".")
                pattern = path.name
                folders.extend(parent.glob(pattern))
        return sorted(folders)
    else:
        # Auto-discover in output directory
        output_dir = Path("output")
        if output_dir.exists():
            folders = [f for f in output_dir.iterdir() if f.is_dir() and f.name.startswith("benchmark_")]
            return sorted(folders, key=lambda x: x.name)
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Compare benchmark results from multiple folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all benchmarks in output directory
  python compare_benchmarks.py
  
  # Compare specific folders
  python compare_benchmarks.py output/benchmark_ollama_* output/benchmark_mlx_*
  
  # Compare with custom output location
  python compare_benchmarks.py --output my_comparison
        """,
    )

    parser.add_argument("folders", nargs="*", help="Benchmark folders to compare (default: all in output/)")

    parser.add_argument(
        "--output",
        default="comparison_results",
        help="Output directory for comparison results (default: comparison_results)",
    )

    args = parser.parse_args()

    # Find benchmark folders
    benchmark_folders = find_benchmark_folders(args.folders)

    if not benchmark_folders:
        print("No benchmark folders found!")
        if not args.folders:
            print("Try running some benchmarks first, or specify folder paths explicitly.")
        return 1

    print(f"Found {len(benchmark_folders)} benchmark folders:")
    for folder in benchmark_folders:
        print(f"  - {folder}")

    # Parse benchmark data
    benchmark_data = []
    for folder in benchmark_folders:
        data, display_name = parse_benchmark_folder(folder)
        if data and display_name:
            benchmark_data.append(data)
            print(f"✓ Loaded: {display_name}")
        else:
            print(f"✗ Failed to load: {folder}")

    if not benchmark_data:
        print("No valid benchmark data found!")
        return 1

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Create comparison charts and tables
    create_comparison_charts(benchmark_data, output_dir)
    create_comparison_table(benchmark_data, output_dir)

    print(f"\n✅ Comparison complete! Results saved to: {output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
