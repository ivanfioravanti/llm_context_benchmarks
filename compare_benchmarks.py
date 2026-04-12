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


# Known engine names used in output folder prefixes. Multi-word names are listed
# first so longest-prefix matching picks e.g. "mlx_vlm" before "mlx".
KNOWN_ENGINES = (
    "ollama_api",
    "ollama_cli",
    "mlx_vlm",
    "mlx-distributed",
    "llamacpp_embed",
    "openai_compat",
    "deepseek",
    "exo",
    "grok",
    "llamacpp",
    "lmstudio",
    "mlx",
    "paroquant",
    "vllm",
)


def _parse_folder_metadata(folder_name: str, hardware_info: dict) -> Tuple[str, str, str]:
    """Extract (engine, model, cache_mode) from a benchmark folder name.

    Folder format: ``benchmark_{framework}_{model}[_{cache_mode}]_{machine}_{YYYYMMDD}_{HHMMSS}``
    where ``{framework}`` may itself contain underscores (e.g. ``ollama_api``,
    ``mlx_vlm``) and ``{machine}`` is omitted in older folders.
    ``{cache_mode}`` is ``cache`` or ``nocache`` when present.
    """
    body = folder_name
    if body.startswith("benchmark_"):
        body = body[len("benchmark_") :]

    # Strip trailing _YYYYMMDD_HHMMSS timestamp if present
    body = re.sub(r"_\d{8}_\d{6}$", "", body)

    # Identify the engine using a longest-prefix match against known names
    engine = None
    for known in KNOWN_ENGINES:
        if body == known or body.startswith(known + "_"):
            engine = known
            body = body[len(known) :].lstrip("_")
            break
    if engine is None:
        first_underscore = body.find("_")
        if first_underscore == -1:
            return body or "unknown", "unknown", ""
        engine = body[:first_underscore]
        body = body[first_underscore + 1 :]

    # body is now "{model}" or "{model}_{machine}". Strip a trailing machine
    # token if it matches the chip recorded in hardware_info or the Apple
    # Silicon naming convention (e.g. M1, M2Max, M5Ultra).
    chip_compact = re.sub(r"\s+", "", (hardware_info or {}).get("chip", "").replace("Apple ", ""))
    if "_" in body:
        head, tail = body.rsplit("_", 1)
        if chip_compact and tail == chip_compact:
            body = head
        elif re.match(r"^M\d[A-Za-z0-9]*$", tail):
            body = head

    # Detect cache mode suffix (cache / nocache)
    cache_mode = ""
    cache_match = re.search(r"_(nocache|cache)$", body)
    if cache_match:
        cache_mode = cache_match.group(1)
        body = body[: -len(cache_match.group(0))]

    return engine, body or "unknown", cache_mode


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
        r"(?<![a-zA-Z])(\d+[-]?bit)(?![a-zA-Z])",  # 2bit 4bit 6bit 8bit 4-bit …
        r"(?<![a-zA-Z])(Q\d+(?:[_-][\w]+)*)(?![\w])",  # Q4_0 Q4_K_M Q8_0 …
        r"(?<![a-zA-Z])(fp16|bf16|fp32|f16|f32|int4|int8)(?![\w])",  # float/int types
        r"(?<![a-zA-Z])(TBQ)(?![\w])",  # TurboQuant
    ]:
        match = re.search(pattern, model_name, re.IGNORECASE)
        if match:
            return match.group(1)
    return "bf16"


def _quant_group_key(quant: str) -> str:
    """Bucket TurboQuant runs together with bf16 in the heatmap.

    TBQ is conceptually bf16 weights with a KV-cache quantization trick on top,
    so for grouping purposes the two belong in the same heatmap section. The
    original ``quant`` value is still kept on the row so the label rendering
    can tag TBQ rows distinctly within the merged section.
    """
    return "bf16" if quant.upper() == "TBQ" else quant


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
    engine, model, cache_mode = _parse_folder_metadata(folder_name, hardware_info)

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

    # Read cached benchmark results if available
    cached_results = []
    cached_json_file = folder_path / "all_results_cached.json"
    if cached_json_file.exists():
        with open(cached_json_file) as f:
            cached_data = json.load(f)
        cached_results = [r for r in cached_data.get("results", []) if r.get("cached_tokens", 0) > 0]

    # Create display name
    cache_label = " (cached)" if cache_mode == "cache" else " (no cache)" if cache_mode == "nocache" else ""
    display_name = f"{engine}: {model}{cache_label}"

    return {
        "results": results,
        "hardware_info": hardware_info,
        "engine": engine,
        "model": model,
        "cache_mode": cache_mode,
        "folder_name": folder_name,
        "display_name": display_name,
        "perplexity_data": perplexity_data,
        "batch_data": batch_data,
        "cached_results": cached_results,
    }, display_name


def create_comparison_charts(benchmark_data: List[Dict], output_dir: Path, charts: List[str] = None):
    """Create comparison charts from multiple benchmark results."""

    # Check if any benchmark has memory data
    has_memory_data = any(any(r.get("peak_memory_gb", 0) > 0 for r in data["results"]) for data in benchmark_data)

    # Only show KV cache subplot if every benchmark in the comparison reports
    # it — otherwise providers without the metric would draw lines at 0.
    has_kv_cache_data = all(any(r.get("kv_cache_gb", 0) > 0 for r in data["results"]) for data in benchmark_data)

    # Check if any benchmark has perplexity data
    has_perplexity_data = any(data.get("perplexity_data") is not None for data in benchmark_data)

    # Check if any benchmark has batch data
    has_batch_data = any(data.get("batch_data") is not None for data in benchmark_data)

    # Check if any benchmark has KV cache reported on its batch rows
    has_batch_kv_data = any(any("kv_cache_gb" in r for r in (data.get("batch_data") or [])) for data in benchmark_data)

    # Check if any benchmark has cached KV results (from all_results_cached.json)
    has_cached_data = any(bool(data.get("cached_results")) for data in benchmark_data)

    # Detect cache modes from folder names
    cache_modes = {d.get("cache_mode", "") for d in benchmark_data}
    split_by_cache = "cache" in cache_modes and "nocache" in cache_modes

    # Set up the plot style
    plt.style.use("default")

    # Build list of subplot specs: each is (key, title, ylabel, is_bar)
    if split_by_cache:
        subplot_specs = [
            ("prompt_tps_nocache", "Prefill TPS (No Cache)", "Tokens/sec"),
            ("prompt_tps_cache", "Prefill TPS (Cached)", "Tokens/sec"),
        ]
    else:
        subplot_specs = [
            ("prompt_tps", "Prompt Processing Speed", "Tokens/sec"),
        ]
    if has_cached_data:
        subplot_specs.append(("inc_prompt_tps", "Incremental Prompt TPS (Cached KV)", "Tokens/sec"))
    subplot_specs += [
        ("generation_tps", "Text Generation Speed", "Tokens/sec"),
        ("ttft", "Time to First Token (TTFT)", "Time (seconds)"),
    ]
    if has_memory_data:
        subplot_specs.append(("memory", "Peak Memory Usage", "Memory (GB)"))
    if has_kv_cache_data:
        subplot_specs.append(("kv_cache", "KV Cache Size", "KV Cache (GB)"))
    if has_perplexity_data:
        subplot_specs.append(("perplexity", "Perplexity (lower is better)", "Perplexity"))
    if has_batch_data:
        subplot_specs.append(("batch_prompt", "Batch Prompt TPS", "Tokens/sec"))
        subplot_specs.append(("batch_gen", "Batch Generation TPS", "Tokens/sec"))
    if has_batch_kv_data:
        subplot_specs.append(("batch_kv", "Batch KV Cache Size", "KV Cache (GB)"))

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
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    colors = readable_colors[: len(benchmark_data)]
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
        kv_cache_gb = [r.get("kv_cache_gb", 0) for r in results]

        context_nums = []
        for ctx in context_sizes:
            context_nums.append(float(ctx[:-1]) * 1000 if ctx.endswith("k") else int(ctx))

        sorted_data = sorted(
            zip(
                context_nums,
                context_sizes,
                prompt_tps,
                generation_tps,
                total_times,
                ttft_times,
                peak_memory,
                kv_cache_gb,
            )
        )
        cn, cs, pp, gn, tt, tf, pm, kv = zip(*sorted_data)
        all_series.append(
            {
                "context_nums": cn,
                "context_labels": cs,
                "prompt_tps": pp,
                "generation_tps": gn,
                "total_time": tt,
                "ttft": tf,
                "memory": pm,
                "kv_cache": kv,
            }
        )

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
                bars = ax.bar(
                    range(len(ppl_values)),
                    ppl_values,
                    yerr=ppl_errors,
                    color=ppl_colors,
                    alpha=0.8,
                    capsize=5,
                    width=0.6,
                )
                ax.set_xticks(range(len(ppl_names)))
                ax.set_xticklabels(ppl_names, rotation=30, ha="right", fontsize=9)
                for bar, val in zip(bars, ppl_values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height(),
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        fontweight="bold",
                    )
            continue

        if key in ("batch_prompt", "batch_gen", "batch_kv"):
            # Batch prompt/generation TPS or KV cache vs batch size
            if key == "batch_prompt":
                data_key, title = "prompt_tps", "Batch Prompt TPS"
                fmt = "{:.1f}"
            elif key == "batch_gen":
                data_key, title = "generation_tps", "Batch Generation TPS"
                fmt = "{:.1f}"
            else:
                data_key, title = "kv_cache_gb", "Batch KV Cache Size"
                fmt = "{:.2f}"
            ax.set_title(title, fontweight="bold", fontsize=13)
            for i, data in enumerate(benchmark_data):
                batch = data.get("batch_data")
                if not batch:
                    continue
                # For batch_kv, only plot rows that actually have the field —
                # avoids drawing a line at zero for engines that don't track it.
                rows = [r for r in batch if (key != "batch_kv" or "kv_cache_gb" in r)]
                if not rows:
                    continue
                batch_sizes = [r["batch_size"] for r in rows]
                batch_y = [r.get(data_key, 0) for r in rows]
                marker = markers[i % len(markers)]
                ax.plot(batch_sizes, batch_y, marker=marker, linewidth=2, label=clean_names[i], color=colors[i])
                for x, y in zip(batch_sizes, batch_y):
                    ax.annotate(
                        fmt.format(y),
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
            # Filter by cache mode for cache-separated subplots
            if key == "prompt_tps_nocache" and benchmark_data[i].get("cache_mode") == "cache":
                continue
            if key == "prompt_tps_cache" and benchmark_data[i].get("cache_mode") != "cache":
                continue
            # Map cache-separated keys to the actual data key
            data_key = "prompt_tps" if key in ("prompt_tps_nocache", "prompt_tps_cache") else key
            x_vals = series["context_labels"]
            y_vals = series[data_key]
            marker = markers[i % len(markers)]
            ax.plot(x_vals, y_vals, marker=marker, linewidth=2, label=clean_names[i], color=colors[i], markersize=6)
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

        if key == "inc_prompt_tps":
            ax.set_xlabel("Context Size")
            ax.tick_params(axis="x", rotation=45)
            for i, data in enumerate(benchmark_data):
                cached = data.get("cached_results", [])
                if not cached:
                    continue
                sorted_cached = sorted(cached, key=lambda r: float(r["context_size"].replace("k", "")))
                x_vals = [r["context_size"] for r in sorted_cached]
                y_vals = [r.get("incremental_prompt_tps", 0) for r in sorted_cached]
                marker = markers[i % len(markers)]
                ax.plot(
                    x_vals,
                    y_vals,
                    marker=marker,
                    linewidth=2,
                    label=clean_names[i],
                    color=colors[i],
                    markersize=6,
                )
                for x, y in zip(x_vals, y_vals):
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
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(handles), 4),
            fontsize=10,
            bbox_to_anchor=(0.5, 1.0),
            frameon=True,
            fancybox=True,
            shadow=True,
        )

    fig.suptitle("LLM Benchmark Comparison", fontsize=16, fontweight="bold", y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save the chart
    chart_path = output_dir / "comparison_chart.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Comparison chart saved to: {chart_path}")

    return chart_path


def create_speed_chart(benchmark_data: List[Dict], output_dir: Path):
    """Create a focused 1x2 chart with only Prompt Processing and Generation speed."""

    plt.style.use("default")

    readable_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    colors = readable_colors[: len(benchmark_data)]
    if len(benchmark_data) > len(readable_colors):
        colors = plt.cm.tab10(np.linspace(0, 1, len(benchmark_data)))

    markers = ["o", "s", "^", "d", "p", "h", "v", "<", ">", "D"]
    clean_names = [_clean_display_name(d["display_name"]) for d in benchmark_data]
    has_cached_data = any(bool(data.get("cached_results")) for data in benchmark_data)

    # Detect cache modes from folder names
    cache_modes = {d.get("cache_mode", "") for d in benchmark_data}
    split_by_cache = "cache" in cache_modes and "nocache" in cache_modes

    # Pre-extract and sort series per benchmark
    all_series = []
    for data in benchmark_data:
        results = data["results"]
        if not results:
            all_series.append(None)
            continue

        context_sizes = [r["context_size"] for r in results]
        prompt_tps = [r.get("prompt_tps", 0) for r in results]
        generation_tps = [r.get("generation_tps", 0) for r in results]
        context_nums = [float(c[:-1]) * 1000 if c.endswith("k") else int(c) for c in context_sizes]

        cn, cs, pp, gn = zip(*sorted(zip(context_nums, context_sizes, prompt_tps, generation_tps)))
        all_series.append({"context_labels": cs, "prompt_tps": pp, "generation_tps": gn})

    # Build subplot specs: separate nocache/cache panels when folder names indicate it
    specs = []
    if split_by_cache:
        specs.append(("prompt_tps_nocache", "Prefill TPS (No Cache)"))
        specs.append(("prompt_tps_cache", "Prefill TPS (Cached)"))
    else:
        specs.append(("prompt_tps", "Prompt Processing Speed"))
    if has_cached_data:
        specs.append(("inc_prompt_tps", "Incremental Prompt TPS (Cached KV)"))
    specs.append(("generation_tps", "Text Generation Speed"))

    n_cols_speed = len(specs)
    fig, axes = plt.subplots(1, n_cols_speed, figsize=(8 * n_cols_speed, 5.5))

    for ax, (key, title) in zip(axes, specs):
        ax.set_title(title, fontweight="bold", fontsize=13)
        ax.set_ylabel("Tokens/sec")
        ax.set_xlabel("Context Size")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

        if key == "inc_prompt_tps":
            for i, data in enumerate(benchmark_data):
                cached = data.get("cached_results", [])
                if not cached:
                    continue
                sorted_cached = sorted(cached, key=lambda r: float(r["context_size"].replace("k", "")))
                x_vals = [r["context_size"] for r in sorted_cached]
                y_vals = [r.get("incremental_prompt_tps", 0) for r in sorted_cached]
                marker = markers[i % len(markers)]
                ax.plot(
                    x_vals,
                    y_vals,
                    marker=marker,
                    linewidth=2,
                    label=clean_names[i],
                    color=colors[i],
                    markersize=6,
                )
                for x, y in zip(x_vals, y_vals):
                    ax.annotate(
                        f"{y:.1f}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha="center",
                        fontsize=7,
                        color=colors[i],
                    )
        else:
            for i, series in enumerate(all_series):
                if series is None:
                    continue
                # Filter by cache mode for cache-separated panels
                if key == "prompt_tps_nocache" and benchmark_data[i].get("cache_mode") == "cache":
                    continue
                if key == "prompt_tps_cache" and benchmark_data[i].get("cache_mode") != "cache":
                    continue
                data_key = "prompt_tps" if key in ("prompt_tps_nocache", "prompt_tps_cache") else key
                x_vals = series["context_labels"]
                y_vals = series[data_key]
                ax.plot(
                    x_vals,
                    y_vals,
                    marker=markers[i % len(markers)],
                    linewidth=2,
                    label=clean_names[i],
                    color=colors[i],
                    markersize=6,
                )
                for x, y in zip(x_vals, y_vals):
                    ax.annotate(
                        f"{y:.1f}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha="center",
                        fontsize=7,
                        color=colors[i],
                    )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(handles), 4),
            fontsize=10,
            bbox_to_anchor=(0.5, 1.0),
            frameon=True,
            fancybox=True,
            shadow=True,
        )

    fig.suptitle("LLM Speed Comparison", fontsize=16, fontweight="bold", y=1.06)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    chart_path = output_dir / "comparison_speed_chart.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Speed comparison chart saved to: {chart_path}")
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

        # Peak KV cache (max across all context sizes) — mlx / mlx-vlm only
        kv_cache_values = [r.get("kv_cache_gb", 0) for r in results]
        peak_kv_cache = max(kv_cache_values) if kv_cache_values else 0

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

        # Avg incremental prompt TPS from cached KV benchmark
        cached = data.get("cached_results", [])
        if cached:
            avg_inc_prompt_tps = np.mean([r.get("incremental_prompt_tps", 0) for r in cached])
            avg_inc_prompt_tps_str = f"{avg_inc_prompt_tps:.1f}"
        else:
            avg_inc_prompt_tps_str = "N/A"

        table_data.append(
            {
                "Engine/Model": display_name,
                "Hardware": f"{chip}, {memory}GB RAM, {cores} cores",
                "Avg Prompt TPS": f"{avg_prompt_tps:.1f}",
                "Avg Gen TPS": f"{avg_generation_tps:.1f}",
                "Avg Inc Prompt TPS": avg_inc_prompt_tps_str,
                "Peak Memory": f"{peak_memory:.1f}GB" if peak_memory > 0 else "N/A",
                "Peak KV Cache": f"{peak_kv_cache:.2f}GB" if peak_kv_cache > 0 else "N/A",
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
            detailed_data.append(
                {
                    "Model": display_name,
                    "Context": r["context_size"],
                    "Prompt TPS": round(r.get("prompt_tps", 0), 2),
                    "Generation TPS": round(r.get("generation_tps", 0), 2),
                    "Total Time": round(r.get("total_time", 0), 2),
                    "TTFT": round(r.get("time_to_first_token", r.get("prompt_eval_duration", 0)), 2),
                    "Peak Memory GB": round(r.get("peak_memory_gb", 0), 2),
                    "KV Cache GB": round(r.get("kv_cache_gb", 0), 3),
                }
            )
        # Add batch data if present
        batch = data.get("batch_data")
        if batch:
            for b in batch:
                detailed_data.append(
                    {
                        "Model": display_name,
                        "Context": f"batch_{b['batch_size']}",
                        "Prompt TPS": round(b.get("prompt_tps", 0), 2),
                        "Generation TPS": round(b.get("generation_tps", 0), 2),
                        "Total Time": "",
                        "TTFT": "",
                        "Peak Memory GB": round(b.get("peak_memory_gb", 0), 2),
                        "KV Cache GB": round(b.get("kv_cache_gb", 0), 3),
                    }
                )
        # Add cached KV data if present
        for r in data.get("cached_results", []):
            detailed_data.append(
                {
                    "Model": display_name,
                    "Context": f"{r['context_size']}_cached",
                    "Prompt TPS": round(r.get("incremental_prompt_tps", 0), 2),
                    "Generation TPS": round(r.get("generation_tps", 0), 2),
                    "Total Time": round(r.get("total_time", 0), 2),
                    "TTFT": round(r.get("time_to_first_token", 0), 2),
                    "Peak Memory GB": round(r.get("peak_memory_gb", 0), 2),
                    "KV Cache GB": round(r.get("kv_cache_gb", 0), 3),
                }
            )

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
        if entry.get("Peak KV Cache", "N/A") != "N/A":
            xpost_text += f"\n  Peak KV Cache: {entry['Peak KV Cache']}"
        if entry.get("Perplexity", "N/A") != "N/A":
            xpost_text += f"\n  Perplexity: {entry['Perplexity']}"
        if entry.get("Avg Inc Prompt TPS", "N/A") != "N/A":
            xpost_text += f"\n  Cached inc pp: {entry['Avg Inc Prompt TPS']}"
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


def create_comparison_table_image(benchmark_data: List[Dict], output_dir: Path):
    """Create a styled comparison table image comparing benchmarks side by side.

    Each value cell shows the raw number plus a percentage vs the best value
    in its group for that row.  The best value gets a green highlight; others
    show how far behind they are.  Works for 2+ benchmarks.
    """
    if len(benchmark_data) < 2:
        print("Need at least 2 benchmarks for comparison table image.")
        return None

    # Collect all context sizes across benchmarks
    all_contexts = set()
    for data in benchmark_data:
        for r in data["results"]:
            all_contexts.add(r["context_size"])

    def ctx_sort_key(ctx):
        return float(ctx.replace("k", "")) if ctx.endswith("k") else float(ctx)

    sorted_contexts = sorted(all_contexts, key=ctx_sort_key)

    # Build lookup: benchmark_index -> context_size -> result
    lookups = []
    for data in benchmark_data:
        lookup = {}
        for r in data["results"]:
            lookup[r["context_size"]] = r
        lookups.append(lookup)

    cached_lookups = []
    for data in benchmark_data:
        lookup = {}
        for r in data.get("cached_results", []):
            lookup[r["context_size"]] = r
        cached_lookups.append(lookup)

    clean_names = [_clean_display_name(d["display_name"]) for d in benchmark_data]
    # Only show the memory / KV-cache columns if every benchmark in the
    # comparison reports them — otherwise we end up with a half-empty column
    # dominated by N/A cells from providers that don't measure them.
    has_memory = all(any(r.get("peak_memory_gb", 0) > 0 for r in data["results"]) for data in benchmark_data)
    has_kv_cache = all(any(r.get("kv_cache_gb", 0) > 0 for r in data["results"]) for data in benchmark_data)
    n_benchmarks = len(benchmark_data)
    aliases = [chr(ord("A") + i) for i in range(n_benchmarks)]

    # Detect cache modes from folder names
    cache_modes = {d.get("cache_mode", "") for d in benchmark_data}
    split_by_cache = "cache" in cache_modes and "nocache" in cache_modes
    cache_indices = {i for i, d in enumerate(benchmark_data) if d.get("cache_mode") == "cache"}
    nocache_indices = {i for i, d in enumerate(benchmark_data) if d.get("cache_mode") != "cache"}

    # Build column structure and track which columns belong to which metric group.
    # A "group" is a set of columns (one per benchmark) that should be compared
    # against each other to find the best value per row.
    # higher_is_better: True for TPS metrics, False for memory.
    col_labels = ["Context"]
    # groups: list of (start_col, end_col_exclusive, higher_is_better)
    groups = []

    sep_indices = set()

    if split_by_cache:
        # ---- No-cache side: Prefill + Decode ----
        start = len(col_labels)
        for i in nocache_indices:
            col_labels.append(f"No-Cache\nPrefill\n{aliases[i]}")
        groups.append((start, start + len(nocache_indices), True))

        start = len(col_labels)
        for i in nocache_indices:
            col_labels.append(f"No-Cache\nDecode\n{aliases[i]}")
        groups.append((start, start + len(nocache_indices), True))

        # Separator between no-cache and cached sides
        sep_indices.add(len(col_labels))
        col_labels.append("")

        # ---- Cached side: Prefill + Decode ----
        start = len(col_labels)
        for i in cache_indices:
            col_labels.append(f"Cached\nPrefill\n{aliases[i]}")
        groups.append((start, start + len(cache_indices), True))

        start = len(col_labels)
        for i in cache_indices:
            col_labels.append(f"Cached\nDecode\n{aliases[i]}")
        groups.append((start, start + len(cache_indices), True))
    else:
        # All benchmarks in a single group
        start = len(col_labels)
        for alias in aliases:
            col_labels.append(f"Prefill\n{alias}")
        groups.append((start, start + n_benchmarks, True))

        start = len(col_labels)
        for alias in aliases:
            col_labels.append(f"Decode\n{alias}")
        groups.append((start, start + n_benchmarks, True))

    if has_memory:
        start = len(col_labels)
        for alias in aliases:
            col_labels.append(f"Mem (GB)\n{alias}")
        groups.append((start, start + n_benchmarks, False))

    if has_kv_cache:
        start = len(col_labels)
        for alias in aliases:
            col_labels.append(f"KV (GB)\n{alias}")
        groups.append((start, start + n_benchmarks, False))

    n_cols = len(col_labels)

    # Build row data (raw numeric values parallel to cell text)
    rows = []  # display strings
    raw_values = []  # float values for comparison (0 means missing)

    for ctx in sorted_contexts:
        row = [ctx]
        raw_row = [0.0]  # context column placeholder
        results = [lookups[i].get(ctx) for i in range(n_benchmarks)]

        # Prefill + Decode TPS
        if split_by_cache:
            # No-cache prefill
            for i in nocache_indices:
                r = results[i]
                v = r.get("prompt_tps", 0) if r else 0
                row.append(f"{v:.1f}" if v > 0 else "\u2014")
                raw_row.append(v)
            # No-cache decode
            for i in nocache_indices:
                r = results[i]
                v = r.get("generation_tps", 0) if r else 0
                row.append(f"{v:.1f}" if v > 0 else "\u2014")
                raw_row.append(v)
            # Separator
            row.append("")
            raw_row.append(0)
            # Cached prefill
            for i in cache_indices:
                r = results[i]
                v = r.get("prompt_tps", 0) if r else 0
                row.append(f"{v:.1f}" if v > 0 else "\u2014")
                raw_row.append(v)
            # Cached decode
            for i in cache_indices:
                r = results[i]
                v = r.get("generation_tps", 0) if r else 0
                row.append(f"{v:.1f}" if v > 0 else "\u2014")
                raw_row.append(v)
        else:
            prefill_vals = [r.get("prompt_tps", 0) if r else 0 for r in results]
            for v in prefill_vals:
                row.append(f"{v:.1f}" if v > 0 else "\u2014")
                raw_row.append(v)
            decode_vals = [r.get("generation_tps", 0) if r else 0 for r in results]
            for v in decode_vals:
                row.append(f"{v:.1f}" if v > 0 else "\u2014")
                raw_row.append(v)

        # Memory
        if has_memory:
            mem_vals = [r.get("peak_memory_gb", 0) if r else 0 for r in results]
            for v in mem_vals:
                row.append(f"{v:.2f}" if v > 0 else "\u2014")
                raw_row.append(v)

        # KV cache
        if has_kv_cache:
            kv_vals = [r.get("kv_cache_gb", 0) if r else 0 for r in results]
            for v in kv_vals:
                row.append(f"{v:.2f}" if v > 0 else "\u2014")
                raw_row.append(v)

        rows.append(row)
        raw_values.append(raw_row)

    # --- Render with matplotlib ---
    fig_w = max(14, n_cols * 1.8)
    fig_h = max(3, (len(rows) + 2) * 0.65)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    bg_color = "#1a1a2e"
    header_bg = "#16213e"
    row_even = "#1a1a2e"
    row_odd = "#0f3460"
    text_color = "#e0e0e0"
    header_text = "#ffffff"
    accent_green = "#00b894"
    accent_green_dim = "#1e5c4a"
    accent_red = "#d63031"

    fig.patch.set_facecolor(bg_color)

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Style header
    for col_idx in range(n_cols):
        cell = table[0, col_idx]
        if col_idx in sep_indices:
            cell.set_facecolor(bg_color)
            cell.set_text_props(color=bg_color, fontsize=1)
            cell.set_edgecolor("#444444")
            cell.set_linewidth(0.5)
            cell.set_width(0.08)
        else:
            cell.set_facecolor(header_bg)
            cell.set_text_props(color=header_text, fontweight="bold", fontsize=9)
            cell.set_edgecolor("#2d3436")
            cell.set_linewidth(0.5)

    # Style data rows with best-value highlighting
    for row_idx in range(len(rows)):
        bg = row_even if row_idx % 2 == 0 else row_odd

        # Pre-compute best value per group for this row
        group_best = {}  # group_start -> (best_val, best_col)
        for g_start, g_end, higher in groups:
            vals = [(raw_values[row_idx][c], c) for c in range(g_start, g_end) if raw_values[row_idx][c] > 0]
            if vals:
                if higher:
                    best_val, best_col = max(vals, key=lambda x: x[0])
                else:
                    best_val, best_col = min(vals, key=lambda x: x[0])
                group_best[g_start] = (best_val, best_col)

        for col_idx in range(n_cols):
            cell = table[row_idx + 1, col_idx]
            if col_idx in sep_indices:
                cell.set_facecolor(bg_color)
                cell.set_text_props(color=bg_color, fontsize=1)
                cell.set_edgecolor("#444444")
                cell.set_linewidth(0.5)
                cell.set_width(0.08)
                continue
            cell.set_facecolor(bg)
            cell.set_text_props(color=text_color, fontsize=10)
            cell.set_edgecolor("#2d3436")
            cell.set_linewidth(0.5)

            # Check if this column belongs to a group
            for g_start, g_end, higher in groups:
                if g_start <= col_idx < g_end:
                    val = raw_values[row_idx][col_idx]
                    if val <= 0 or g_start not in group_best:
                        break
                    best_val, best_col = group_best[g_start]
                    base_text = rows[row_idx][col_idx]

                    if col_idx == best_col:
                        # Best value — green highlight
                        cell.set_facecolor(accent_green_dim)
                        cell.set_text_props(color=accent_green, fontweight="bold", fontsize=10)
                    else:
                        # Show percentage difference from best
                        if higher:
                            pct = (val - best_val) / best_val * 100  # negative = worse
                        else:
                            pct = (best_val - val) / best_val * 100  # negative = worse (higher mem)
                        pct_text = f"{pct:+.0f}%"
                        cell.get_text().set_text(f"{base_text}\n{pct_text}")
                        if pct < -5:
                            cell.set_text_props(color="#ff7675", fontsize=9)
                        else:
                            cell.set_text_props(color="#b0b0b0", fontsize=9)
                    break

    # Title with hardware info
    hw = benchmark_data[0].get("hardware_info", {})
    chip = hw.get("chip", "Unknown").replace("Apple ", "")
    mem_gb = hw.get("memory_gb", "")
    title_parts = [chip]
    if mem_gb:
        title_parts.append(f"{mem_gb} GB")
    title = " \u00b7 ".join(title_parts)

    fig.text(
        0.5,
        0.98,
        title,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=header_text,
        fontfamily="monospace",
        transform=fig.transFigure,
    )

    # Legend mapping aliases to benchmark names
    legend_lines = [f"{aliases[i]} = {clean_names[i]}" for i in range(n_benchmarks)]
    legend_text = "    ".join(legend_lines)
    # Wrap long legends
    if len(legend_text) > 120:
        mid = len(legend_lines) // 2
        line1 = "    ".join(legend_lines[:mid])
        line2 = "    ".join(legend_lines[mid:])
        legend_text = f"{line1}\n{line2}"
    fig.text(
        0.5,
        0.94,
        legend_text,
        ha="center",
        va="top",
        fontsize=8,
        color="#aaaaaa",
        fontfamily="monospace",
        transform=fig.transFigure,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    table_img_path = output_dir / "comparison_table.png"
    plt.savefig(table_img_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Comparison table image saved to: {table_img_path}")
    return table_img_path


def create_heatmap(benchmark_data: List[Dict], output_dir: Path):
    """Create a performance heatmap with a separate section per quantization level
    and cache mode.

    Each section normalises independently: 100% = best within that group.
    Columns = Avg Prompt TPS, Avg Gen TPS, Peak Batch Prompt TPS, Peak Batch Gen TPS.
    Title uses the model name when all runs share the same base model; otherwise the model
    name is included in each row label.
    """
    has_cached_data = any(bool(data.get("cached_results")) for data in benchmark_data)

    metric_keys = ["avg_prompt_tps", "avg_gen_tps"]
    col_labels = ["Avg Prompt TPS", "Avg Gen TPS"]
    if has_cached_data:
        metric_keys.append("avg_inc_prompt_tps")
        col_labels.append("Avg Inc\nPrompt TPS")
    metric_keys += ["peak_batch_prompt_tps", "peak_batch_gen_tps"]
    col_labels += ["Peak Batch\nPrompt TPS", "Peak Batch\nGen TPS"]
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
        cache_mode = data.get("cache_mode", "")

        avg_prompt_tps = float(np.mean([r.get("prompt_tps", 0) for r in results]))
        avg_gen_tps = float(np.mean([r.get("generation_tps", 0) for r in results]))

        cached = data.get("cached_results", [])
        avg_inc_prompt_tps = (
            float(np.mean([r.get("incremental_prompt_tps", 0) for r in cached])) if cached else float("nan")
        )

        batch = data.get("batch_data")
        peak_batch_prompt = float(max(r["prompt_tps"] for r in batch)) if batch else float("nan")
        peak_batch_gen = float(max(r["generation_tps"] for r in batch)) if batch else float("nan")

        rows.append(
            {
                "engine": data.get("engine", ""),
                "quant": quant,
                "quant_group": _quant_group_key(quant),
                "cache_mode": cache_mode,
                "chip_short": chip_short,
                "base_model": base_model,
                "avg_prompt_tps": avg_prompt_tps,
                "avg_gen_tps": avg_gen_tps,
                "avg_inc_prompt_tps": avg_inc_prompt_tps,
                "peak_batch_prompt_tps": peak_batch_prompt,
                "peak_batch_gen_tps": peak_batch_gen,
            }
        )

    if not rows:
        print("No data available for heatmap.")
        return None

    # Detect cache modes to decide grouping strategy
    cache_modes = {r["cache_mode"] for r in rows}
    split_by_cache = "cache" in cache_modes and "nocache" in cache_modes

    # Group by (cache_mode, quantization) when both modes present, otherwise just quantization
    if split_by_cache:
        sections: dict = defaultdict(list)
        for row in rows:
            cache_label = "Cached" if row["cache_mode"] == "cache" else "No Cache"
            section_key = f"{cache_label} · {_quant_group_key(row['quant'])}"
            sections[section_key].append(row)
        # Order: no-cache sections first, then cached
        ordered_sections = sorted(sections.keys(), key=lambda k: (0 if "No Cache" in k else 1, k))
    else:
        sections: dict = defaultdict(list)
        for row in rows:
            sections[_quant_group_key(row["quant"])].append(row)
        ordered_sections = sorted(sections.keys())

    n_groups = len(ordered_sections)

    # Filter out columns where ALL values are NaN across ALL groups
    active_col_indices = []
    for col_idx, key in enumerate(metric_keys):
        has_data = any(not np.isnan(row[key]) for group_rows in sections.values() for row in group_rows)
        if has_data:
            active_col_indices.append(col_idx)

    metric_keys = [metric_keys[i] for i in active_col_indices]
    col_labels = [col_labels[i] for i in active_col_indices]
    n_cols = len(metric_keys)

    if n_cols == 0:
        print("No data available for heatmap.")
        return None

    # Determine title: single model name or generic
    base_models = {r["base_model"] for r in rows}
    single_model = len(base_models) == 1
    model_title = next(iter(base_models)) if single_model else None

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#cccccc")

    total_data_rows = sum(len(sections[s]) for s in ordered_sections)
    height_ratios = [max(1, len(sections[s])) for s in ordered_sections]
    fig_w = max(10, n_cols * 3.5)
    fig_h = max(5, total_data_rows * 2.0 + n_groups * 1.5)

    fig, axes = plt.subplots(
        n_groups,
        1,
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.65},
    )
    if n_groups == 1:
        axes = [axes]

    for ax_idx, (ax, section_key) in enumerate(zip(axes, ordered_sections)):
        group_rows = sections[section_key]
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

        # Row labels: include the engine and cache mode when there is
        # more than one engine or cache mode in the group.
        engines_in_group = {r.get("engine", "") for r in group_rows}
        multi_engine = len(engines_in_group) > 1

        def _row_label(r):
            parts = []
            if multi_engine:
                parts.append(r["engine"])
            if not single_model:
                parts.append(r["base_model"])
            parts.append(r["chip_short"])
            return " / ".join(parts)

        row_labels = [_row_label(r) for r in group_rows]
        ax.set_yticks(range(n_group_rows))
        ax.set_yticklabels(row_labels, fontsize=10)

        # Section title (left-aligned)
        ax.set_title(f"◆ {section_key}", loc="left", fontsize=12, fontweight="bold", pad=6, color="#333333")

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
                ax.text(j, i, cell_text, ha="center", va="center", fontsize=9, color=text_color, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax, label="% of Best", shrink=0.8, pad=0.02)
        cbar.ax.tick_params(labelsize=9)

    if model_title:
        fig.suptitle(
            f"Performance Heatmap — {model_title}\n(% of best per quantization group)",
            fontweight="bold",
            fontsize=14,
            y=1.01,
        )
    else:
        fig.suptitle(
            "Performance Heatmap (% of best per quantization group)",
            fontweight="bold",
            fontsize=14,
            y=1.01,
        )

    plt.tight_layout()
    heatmap_path = output_dir / "comparison_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to: {heatmap_path}")
    return heatmap_path


def create_speed_heatmap(benchmark_data: List[Dict], output_dir: Path):
    """Create side-by-side heatmaps: Prompt TPS and Gen TPS vs context sizes.

    Rows = benchmark runs (model/quantization), Columns = context sizes.
    Cells show raw tokens/sec, colored green=fast, red=slow.
    """
    if len(benchmark_data) < 2:
        print("Need at least 2 benchmarks for speed heatmap.")
        return None

    # Collect all context sizes and build per-run lookup
    all_context_sizes = set()
    run_data = []
    for data in benchmark_data:
        results = data["results"]
        if not results:
            continue

        # Build context_size -> result mapping
        ctx_map = {}
        for r in results:
            ctx = r.get("context_size", "")
            all_context_sizes.add(ctx)
            ctx_map[ctx] = r

        hardware_info = data["hardware_info"]
        chip_short = hardware_info.get("chip", "Unknown").replace("Apple ", "")
        cache_mode = data.get("cache_mode", "")
        cache_label = " (cached)" if cache_mode == "cache" else ""

        # Use full model name (includes quant variant) for the speed heatmap
        label = f"{data['model']} / {chip_short}{cache_label}"

        run_data.append({"label": label, "ctx_map": ctx_map, "engine": data.get("engine", "")})

    if len(run_data) < 2:
        print("Need at least 2 benchmarks with data for speed heatmap.")
        return None

    # Sort context sizes numerically (strip the 'k' suffix)
    context_sizes = sorted(all_context_sizes, key=lambda s: float(s.rstrip("k")))

    # Check if multiple engines
    engines = {rd["engine"] for rd in run_data}
    multi_engine = len(engines) > 1
    if multi_engine:
        for rd in run_data:
            rd["label"] = f"{rd['engine']}: {rd['label']}"

    row_labels = [rd["label"] for rd in run_data]
    n_rows = len(run_data)
    n_cols = len(context_sizes)

    # Build prompt_tps and gen_tps matrices
    prompt_matrix = np.full((n_rows, n_cols), np.nan)
    gen_matrix = np.full((n_rows, n_cols), np.nan)

    for i, rd in enumerate(run_data):
        for j, ctx in enumerate(context_sizes):
            r = rd["ctx_map"].get(ctx)
            if r:
                prompt_matrix[i, j] = r.get("prompt_tps", np.nan)
                gen_matrix[i, j] = r.get("generation_tps", np.nan)

    # Create figure with two side-by-side heatmaps
    fig_height = max(5, n_rows * 1.2 + 2)
    fig_width = max(12, n_cols * 1.8 + 4)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height), gridspec_kw={"wspace": 0.35})

    def _draw_heatmap(ax, matrix, title):
        # Normalize per-column for coloring
        col_max = np.nanmax(matrix, axis=0)
        col_max = np.where(col_max == 0, 1, col_max)
        normalized = matrix / col_max * 100

        masked = np.ma.array(normalized, mask=np.isnan(matrix))
        cmap = plt.cm.RdYlGn.copy()
        cmap.set_bad(color="#cccccc")

        im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")

        # Annotate cells
        for i in range(n_rows):
            for j in range(n_cols):
                val = matrix[i, j]
                if np.isnan(val):
                    ax.text(j, i, "N/A", ha="center", va="center", fontsize=9, color="#555555")
                else:
                    pct = normalized[i, j]
                    text_color = "white" if pct < 25 else "black"
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9, fontweight="bold", color=text_color)

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(context_sizes, fontsize=10)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=10)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

        # Grid lines
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)

        return im

    _draw_heatmap(ax1, prompt_matrix, "Prompt Processing Speed (tokens/sec)")
    _draw_heatmap(ax2, gen_matrix, "Generation Speed (tokens/sec)")

    plt.tight_layout()
    heatmap_path = output_dir / "comparison_speed_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Speed heatmap saved to: {heatmap_path}")
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
        choices=[
            "prompt_tps",
            "generation_tps",
            "ttft",
            "memory",
            "perplexity",
            "batch_prompt",
            "batch_gen",
        ],
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
    create_speed_chart(benchmark_data, output_dir)
    create_comparison_table(benchmark_data, output_dir)
    create_comparison_table_image(benchmark_data, output_dir)
    create_heatmap(benchmark_data, output_dir)
    create_speed_heatmap(benchmark_data, output_dir)

    print(f"\n✅ Comparison complete! Results saved to: {output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
