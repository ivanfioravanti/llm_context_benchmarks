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
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _clean_display_name(name: str) -> str:
    """Strip trailing timestamp suffixes like _20260215 from display names."""
    return re.sub(r"_\d{8,}$", "", name)


def _extract_base_model(model: str, hardware_info: dict, quant: str) -> str:
    """Strip quantization, hardware, and date tokens from model name to get the base model name."""
    base = model
    # Strip date/time fragments like _20260324 or _20260324_092407
    base = re.sub(r"[-_]\d{8,}(?:[-_]\d+)*", "", base)
    # Strip hardware chip compact form (e.g. "M5 Max" → "M5Max")
    chip = hardware_info.get("chip", "").replace("Apple ", "")
    chip_compact = re.sub(r"\s+", "", chip)
    if chip_compact:
        base = re.sub(r"[-_]?" + re.escape(chip_compact) + r"[-_]?", "", base, flags=re.IGNORECASE)
    # Strip quantization token
    if quant and quant != "unknown":
        base = re.sub(r"[-_]?" + re.escape(quant) + r"[-_]?", "", base, flags=re.IGNORECASE)
    return base.strip("-_") or model


def _extract_quantization(model_name: str) -> str:
    """Extract quantization info from model name (e.g. 8bit, 4-bit, Q4_K_M, fp16, int8)."""
    for pattern in [
        r"(?<![a-zA-Z])(\d+[-]?bit)(?![a-zA-Z])",   # 2bit 4bit 6bit 8bit 4-bit …
        r"(?<![a-zA-Z])(Q\d+(?:[_-][\w]+)*)(?![\w])",  # Q4_0 Q4_K_M Q8_0 …
        r"(?<![a-zA-Z])(fp16|bf16|fp32|f16|f32|int4|int8)(?![\w])",  # float/int types
    ]:
        match = re.search(pattern, model_name, re.IGNORECASE)
        if match:
            return match.group(1)
    return "unknown"


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

    # Read batch benchmark data if available
    batch_file = folder_path / "batch_benchmark.json"
    batch_data = None
    if batch_file.exists():
        with open(batch_file) as f:
            batch_data = json.load(f)

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
        "batch_data": batch_data,
    }, display_name


def create_comparison_charts(benchmark_data: List[Dict], output_dir: Path, charts: List[str] = None):
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

    # Check if any benchmark has batch data
    has_batch_data = any(
        data.get("batch_data") is not None
        for data in benchmark_data
    )

    # Set up the plot style
    plt.style.use("default")

    # Build list of subplot specs: each is (key, title, ylabel, is_bar)
    subplot_specs = [
        ("prompt_tps", "Prompt Processing Speed", "Tokens/sec"),
        ("generation_tps", "Text Generation Speed", "Tokens/sec"),
        ("ttft", "Time to First Token (TTFT)", "Time (seconds)"),
    ]
    if has_memory_data:
        subplot_specs.append(("memory", "Peak Memory Usage", "Memory (GB)"))
    if has_perplexity_data:
        subplot_specs.append(("perplexity", "Perplexity (lower is better)", "Perplexity"))
    if has_batch_data:
        subplot_specs.append(("batch_prompt", "Batch Prompt TPS", "Tokens/sec"))
        subplot_specs.append(("batch_gen", "Batch Generation TPS", "Tokens/sec"))

    if charts:
        subplot_specs = [spec for spec in subplot_specs if spec[0] in charts]

    num_plots = len(subplot_specs)
    if num_plots == 0:
        print("No valid charts to generate based on requested elements.")
        return None

    num_rows = (num_plots + 1) // 2
    cols = 2 if num_plots > 1 else 1
    
    fig, axes_flat = plt.subplots(num_rows, cols, figsize=(8 * cols, 5.5 * num_rows))
    axes_all = np.array(axes_flat).flatten()

    # Hide unused axes
    for idx in range(num_plots, len(axes_all)):
        axes_all[idx].set_visible(False)

    # Colors for different benchmarks
    readable_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]
    colors = readable_colors[:len(benchmark_data)]
    if len(benchmark_data) > len(readable_colors):
        colors = plt.cm.tab10(np.linspace(0, 1, len(benchmark_data)))

    # Clean display names (strip timestamps)
    clean_names = [_clean_display_name(d["display_name"]) for d in benchmark_data]

    # Pre-extract and sort data per benchmark
    all_series = []
    for i, data in enumerate(benchmark_data):
        results = data["results"]
        if not results:
            all_series.append(None)
            continue

        context_sizes = [r["context_size"] for r in results]
        prompt_tps = [r.get("prompt_tps", 0) for r in results]
        generation_tps = [r.get("generation_tps", 0) for r in results]
        total_times = [r.get("total_time", 0) for r in results]
        ttft_times = [r.get("time_to_first_token", r.get("prompt_eval_duration", 0)) for r in results]
        peak_memory = [r.get("peak_memory_gb", 0) for r in results]

        context_nums = []
        for ctx in context_sizes:
            context_nums.append(float(ctx[:-1]) * 1000 if ctx.endswith("k") else int(ctx))

        sorted_data = sorted(zip(context_nums, context_sizes, prompt_tps, generation_tps, total_times, ttft_times, peak_memory))
        cn, cs, pp, gn, tt, tf, pm = zip(*sorted_data)
        all_series.append({
            "context_nums": cn, "context_labels": cs,
            "prompt_tps": pp, "generation_tps": gn,
            "total_time": tt, "ttft": tf, "memory": pm,
        })

    # Markers for each series
    markers = ["o", "s", "^", "d", "p", "h", "v", "<", ">", "D"]

    # For TTFT labels, annotate only the lowest value for each context size.
    min_ttft_by_context = {}
    for series in all_series:
        if series is None:
            continue
        for ctx, ttft in zip(series["context_labels"], series["ttft"]):
            current = min_ttft_by_context.get(ctx)
            if current is None or ttft < current:
                min_ttft_by_context[ctx] = ttft

    # Plot each subplot
    for plot_idx, (key, title, ylabel) in enumerate(subplot_specs):
        ax = axes_all[plot_idx]
        if key == "ttft":
            ax.set_title(
                "Time to First Token (TTFT)",
                fontweight="bold",
                fontsize=13,
                pad=22,
            )
            ax.text(
                0.5,
                1.005,
                "(only lower values labeled)",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
            )
        else:
            ax.set_title(title, fontweight="bold", fontsize=13)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        if key == "perplexity":
            # Bar chart for perplexity
            ppl_names, ppl_values, ppl_errors, ppl_colors = [], [], [], []
            for i, data in enumerate(benchmark_data):
                ppl_data = data.get("perplexity_data")
                if ppl_data is not None:
                    ppl_names.append(clean_names[i])
                    ppl_values.append(ppl_data["perplexity"])
                    ppl_errors.append(ppl_data.get("std_error", 0))
                    ppl_colors.append(colors[i])
            if ppl_values:
                bars = ax.bar(range(len(ppl_values)), ppl_values, yerr=ppl_errors,
                              color=ppl_colors, alpha=0.8, capsize=5, width=0.6)
                ax.set_xticks(range(len(ppl_names)))
                ax.set_xticklabels(ppl_names, rotation=30, ha="right", fontsize=9)
                for bar, val in zip(bars, ppl_values):
                    ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                            f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
            continue

        if key in ("batch_prompt", "batch_gen"):
            # Batch prompt/generation TPS vs batch size
            data_key = "prompt_tps" if key == "batch_prompt" else "generation_tps"
            title = "Batch Prompt TPS" if key == "batch_prompt" else "Batch Generation TPS"
            ax.set_title(title, fontweight="bold", fontsize=13)
            for i, data in enumerate(benchmark_data):
                batch = data.get("batch_data")
                if not batch:
                    continue
                batch_sizes = [r["batch_size"] for r in batch]
                batch_tps = [r[data_key] for r in batch]
                marker = markers[i % len(markers)]
                ax.plot(batch_sizes, batch_tps, marker=marker, linewidth=2,
                        label=clean_names[i], color=colors[i])
                for x, y in zip(batch_sizes, batch_tps):
                    ax.annotate(
                        f"{y:.1f}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha="center",
                        fontsize=7,
                        color=colors[i],
                    )
            ax.set_xlabel("Batch Size")
            ax.legend(fontsize=9)
            continue

        # Line charts: context-size based
        ax.set_xlabel("Context Size")
        ax.tick_params(axis="x", rotation=45)
        for i, series in enumerate(all_series):
            if series is None:
                continue
            x_vals = series["context_labels"]
            y_vals = series[key]
            marker = markers[i % len(markers)]
            ax.plot(x_vals, y_vals, marker=marker, linewidth=2,
                    label=clean_names[i], color=colors[i], markersize=6)
            # Add value annotations
            for x, y in zip(x_vals, y_vals):
                if key == "ttft":
                    if np.isclose(y, min_ttft_by_context.get(x, y)):
                        ax.annotate(
                            f"{y:.2f}s",
                            (x, y),
                            textcoords="offset points",
                            xytext=(0, 8),
                            ha="center",
                            fontsize=7,
                            color=colors[i],
                        )
                else:
                    ax.annotate(
                        f"{y:.1f}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha="center",
                        fontsize=7,
                        color=colors[i],
                    )

    # Single shared legend at the top of the figure
    handles, labels = axes_all[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(handles), 4),
                   fontsize=10, bbox_to_anchor=(0.5, 1.0), frameon=True,
                   fancybox=True, shadow=True)

    fig.suptitle("LLM Benchmark Comparison", fontsize=16, fontweight="bold", y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save the chart
    chart_path = output_dir / "comparison_chart.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Comparison chart saved to: {chart_path}")

    return chart_path


def create_comparison_table(benchmark_data: List[Dict], output_dir: Path):
    """Create a comparison table with key metrics in X post-friendly format."""

    table_data = []

    for data in benchmark_data:
        results = data["results"]
        hardware_info = data["hardware_info"]
        display_name = _clean_display_name(data["display_name"])

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

        # Peak batch generation TPS
        batch = data.get("batch_data")
        if batch:
            peak_batch_gen = max(r["generation_tps"] for r in batch)
            peak_batch_prompt = max(r["prompt_tps"] for r in batch)
            peak_batch_str = f"{peak_batch_gen:.1f}"
            peak_batch_prompt_str = f"{peak_batch_prompt:.1f}"
        else:
            peak_batch_str = "N/A"
            peak_batch_prompt_str = "N/A"

        table_data.append(
            {
                "Engine/Model": display_name,
                "Hardware": f"{chip}, {memory}GB RAM, {cores} cores",
                "Avg Prompt TPS": f"{avg_prompt_tps:.1f}",
                "Avg Gen TPS": f"{avg_generation_tps:.1f}",
                "Peak Memory": f"{peak_memory:.1f}GB" if peak_memory > 0 else "N/A",
                "Perplexity": ppl_str,
                "Peak Batch Prompt TPS": peak_batch_prompt_str,
                "Peak Batch Gen TPS": peak_batch_str,
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

    # Create detailed data table with all context sizes
    detailed_data = []
    for data in benchmark_data:
        results = data["results"]
        display_name = _clean_display_name(data["display_name"])
        if not results:
            continue
        for r in results:
            detailed_data.append({
                "Model": display_name,
                "Context": r["context_size"],
                "Prompt TPS": round(r.get("prompt_tps", 0), 2),
                "Generation TPS": round(r.get("generation_tps", 0), 2),
                "Total Time": round(r.get("total_time", 0), 2),
                "TTFT": round(r.get("time_to_first_token", r.get("prompt_eval_duration", 0)), 2),
                "Peak Memory GB": round(r.get("peak_memory_gb", 0), 2),
            })
        # Add batch data if present
        batch = data.get("batch_data")
        if batch:
            for b in batch:
                detailed_data.append({
                    "Model": display_name,
                    "Context": f"batch_{b['batch_size']}",
                    "Prompt TPS": round(b.get("prompt_tps", 0), 2),
                    "Generation TPS": round(b.get("generation_tps", 0), 2),
                    "Total Time": "",
                    "TTFT": "",
                    "Peak Memory GB": round(b.get("peak_memory_gb", 0), 2),
                })

    if detailed_data:
        detailed_df = pd.DataFrame(detailed_data)
        detailed_csv_path = output_dir / "comparison_detailed.csv"
        detailed_df.to_csv(detailed_csv_path, index=False)
        print(f"Detailed results saved to: {detailed_csv_path}")

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
        xpost_text += f"\n  {entry['Total Tokens']} tokens in {entry['Total Time']}"
        if entry["Peak Memory"] != "N/A":
            xpost_text += f"\n  Peak Memory: {entry['Peak Memory']}"
        if entry.get("Perplexity", "N/A") != "N/A":
            xpost_text += f"\n  Perplexity: {entry['Perplexity']}"
        if entry.get("Peak Batch Prompt TPS", "N/A") != "N/A":
            xpost_text += f"\n  Peak Batch pp: {entry['Peak Batch Prompt TPS']}"
        if entry.get("Peak Batch Gen TPS", "N/A") != "N/A":
            xpost_text += f"\n  Peak Batch tg: {entry['Peak Batch Gen TPS']}"
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


def create_heatmap(benchmark_data: List[Dict], output_dir: Path):
    """Create a performance heatmap with a separate section per quantization level.

    Each section normalises independently: 100% = best within that quantization group.
    Columns = Avg Prompt TPS, Avg Gen TPS, Peak Batch Prompt TPS, Peak Batch Gen TPS.
    Title uses the model name when all runs share the same base model; otherwise the model
    name is included in each row label.
    """
    metric_keys = ["avg_prompt_tps", "avg_gen_tps", "peak_batch_prompt_tps", "peak_batch_gen_tps"]
    col_labels = ["Avg Prompt TPS", "Avg Gen TPS", "Peak Batch\nPrompt TPS", "Peak Batch\nGen TPS"]
    n_cols = len(metric_keys)

    rows = []
    for data in benchmark_data:
        results = data["results"]
        if not results:
            continue

        hardware_info = data["hardware_info"]
        chip_short = hardware_info.get("chip", "Unknown").replace("Apple ", "")
        quant = _extract_quantization(data["model"])
        base_model = _extract_base_model(data["model"], hardware_info, quant)

        avg_prompt_tps = float(np.mean([r.get("prompt_tps", 0) for r in results]))
        avg_gen_tps = float(np.mean([r.get("generation_tps", 0) for r in results]))

        batch = data.get("batch_data")
        peak_batch_prompt = float(max(r["prompt_tps"] for r in batch)) if batch else float("nan")
        peak_batch_gen = float(max(r["generation_tps"] for r in batch)) if batch else float("nan")

        rows.append({
            "quant": quant,
            "chip_short": chip_short,
            "base_model": base_model,
            "avg_prompt_tps": avg_prompt_tps,
            "avg_gen_tps": avg_gen_tps,
            "peak_batch_prompt_tps": peak_batch_prompt,
            "peak_batch_gen_tps": peak_batch_gen,
        })

    if not rows:
        print("No data available for heatmap.")
        return None

    # Group by quantization level
    quant_groups: dict = defaultdict(list)
    for row in rows:
        quant_groups[row["quant"]].append(row)
    ordered_quants = sorted(quant_groups.keys())
    n_groups = len(ordered_quants)

    # Determine title: single model name or generic
    base_models = {r["base_model"] for r in rows}
    single_model = len(base_models) == 1
    model_title = next(iter(base_models)) if single_model else None

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#cccccc")

    total_data_rows = sum(len(quant_groups[q]) for q in ordered_quants)
    height_ratios = [max(1, len(quant_groups[q])) for q in ordered_quants]
    fig_w = max(10, n_cols * 3)
    fig_h = max(5, total_data_rows * 1.4 + n_groups * 1.2)

    fig, axes = plt.subplots(
        n_groups, 1, figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.55},
    )
    if n_groups == 1:
        axes = [axes]

    for ax_idx, (ax, quant) in enumerate(zip(axes, ordered_quants)):
        group_rows = quant_groups[quant]
        n_group_rows = len(group_rows)

        raw_matrix = np.array([[r[k] for k in metric_keys] for r in group_rows], dtype=float)

        # Normalise per column within this group only
        normalised = np.full_like(raw_matrix, np.nan)
        for col_idx in range(n_cols):
            col = raw_matrix[:, col_idx]
            col_max = np.nanmax(col) if not np.all(np.isnan(col)) else 0
            if col_max > 0:
                normalised[:, col_idx] = col / col_max * 100

        masked = np.ma.array(normalised, mask=np.isnan(normalised))
        im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")

        # Column labels only on the first subplot (at the top)
        ax.set_xticks(range(n_cols))
        if ax_idx == 0:
            ax.set_xticklabels(col_labels, fontsize=11, fontweight="bold")
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
        else:
            ax.set_xticklabels([""] * n_cols)
            ax.tick_params(axis="x", length=0)

        # Row labels: include model name only when multiple base models exist
        if single_model:
            row_labels = [r["chip_short"] for r in group_rows]
        else:
            row_labels = [f"{r['base_model']} / {r['chip_short']}" for r in group_rows]
        ax.set_yticks(range(n_group_rows))
        ax.set_yticklabels(row_labels, fontsize=10)

        # Quantization level as the subplot title (left-aligned)
        ax.set_title(f"◆ {quant}", loc="left", fontsize=12, fontweight="bold", pad=6, color="#333333")

        # Cell annotations
        for i in range(n_group_rows):
            for j in range(n_cols):
                raw_val = raw_matrix[i, j]
                pct = normalised[i, j]
                if np.isnan(pct):
                    cell_text = "N/A"
                    text_color = "#555555"
                else:
                    cell_text = f"{pct:.0f}%\n({raw_val:.1f})"
                    text_color = "white" if pct < 25 else "black"
                ax.text(j, i, cell_text, ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax, label="% of Best", shrink=0.8, pad=0.02)
        cbar.ax.tick_params(labelsize=9)

    if model_title:
        fig.suptitle(
            f"Performance Heatmap — {model_title}\n(% of best per quantization group)",
            fontweight="bold", fontsize=14, y=1.01,
        )
    else:
        fig.suptitle(
            "Performance Heatmap (% of best per quantization group)",
            fontweight="bold", fontsize=14, y=1.01,
        )

    plt.tight_layout()
    heatmap_path = output_dir / "comparison_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to: {heatmap_path}")
    return heatmap_path


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

    parser.add_argument(
        "--charts",
        nargs="+",
        choices=["prompt_tps", "generation_tps", "ttft", "memory", "perplexity", "batch_prompt", "batch_gen"],
        help="Specific charts to generate (default: all available)",
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
    create_comparison_charts(benchmark_data, output_dir, charts=args.charts)
    create_comparison_table(benchmark_data, output_dir)
    create_heatmap(benchmark_data, output_dir)

    print(f"\n✅ Comparison complete! Results saved to: {output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
