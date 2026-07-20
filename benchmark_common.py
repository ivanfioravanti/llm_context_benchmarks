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

    # Get memory info. Some macOS/Python/psutil combinations can fail inside
    # host_statistics64(); keep benchmarking usable and fall back to sysctl.
    try:
        mem = psutil.virtual_memory()
        info["memory_gb"] = round(mem.total / (1024**3), 1)
    except Exception:
        if platform.system() == "Darwin":
            try:
                result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    info["memory_gb"] = round(int(result.stdout.strip()) / (1024**3), 1)
            except Exception:
                pass

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


def find_warmup_file() -> Optional[Path]:
    """Return the 0.5k.txt warmup file if it exists in the current directory."""
    warmup = Path("0.5k.txt")
    return warmup if warmup.exists() else None


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
                if len(stem) > 1 and stem.endswith("k"):
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


def kv_cache_bytes(cache_list) -> int:
    """Sum the byte size of an mlx-lm / mlx-vlm prompt cache list.

    Each cache layer in mlx-lm exposes an ``nbytes`` property. mlx-vlm reuses
    those classes but adds a few of its own (``SimpleKVCache``,
    ``SlidingWindowCache``, ``StaticKVCache``) that don't define ``nbytes`` —
    for those we fall back to summing ``keys.nbytes + values.nbytes``.
    """
    total = 0
    for c in cache_list or []:
        try:
            total += int(c.nbytes)
            continue
        except (AttributeError, NotImplementedError):
            pass
        k = getattr(c, "keys", None)
        v = getattr(c, "values", None)
        if k is not None and hasattr(k, "nbytes"):
            total += int(k.nbytes)
        if v is not None and hasattr(v, "nbytes"):
            total += int(v.nbytes)
    return total


def add_throughput_metrics(result: Dict, prompt_text: str = "") -> Dict:
    """Add tokenizer-independent throughput metrics in place.

    Adds five keys to ``result``:
      - ``generation_utf8_bytes_per_sec``  — UTF-8 bytes of generated_text / eval_duration
      - ``generation_chars_per_sec``       — Unicode code points / eval_duration
      - ``prompt_utf8_bytes_per_sec``      — UTF-8 bytes of prompt_text / prompt_eval_duration
      - ``prompt_chars_per_sec``           — Unicode code points / prompt_eval_duration
      - ``time_per_output_token``          — eval_duration / (generation_tokens - 1), i.e. TPOT

    These metrics complement the tokenizer-dependent ``*_tps`` fields and let
    different tokenizers be compared on the same textual workload.
    """
    # Reasoning/thinking text is generated in the same decode window but some
    # engines store it separately from generated_text — count both.
    gen_text = (result.get("generated_text", "") or "") + (result.get("reasoning_text", "") or "")
    prompt_text = prompt_text or ""
    eval_dur = result.get("eval_duration", 0) or 0
    p_eval_dur = result.get("prompt_eval_duration", 0) or 0

    # Fall back to deriving duration from tokens / tps when the explicit
    # duration field is missing (some engines report rate but not duration).
    if eval_dur <= 0:
        gen_tokens = result.get("generation_tokens", 0) or 0
        gen_tps = result.get("generation_tps", 0) or 0
        if gen_tokens > 0 and gen_tps > 0:
            eval_dur = gen_tokens / gen_tps
    if p_eval_dur <= 0:
        p_tokens = result.get("prompt_tokens", 0) or 0
        p_tps = result.get("prompt_tps", 0) or 0
        if p_tokens > 0 and p_tps > 0:
            p_eval_dur = p_tokens / p_tps

    gen_bytes = len(gen_text.encode("utf-8"))
    gen_chars = len(gen_text)
    p_bytes = len(prompt_text.encode("utf-8"))
    p_chars = len(prompt_text)

    result["generation_utf8_bytes_per_sec"] = gen_bytes / eval_dur if eval_dur > 0 else 0.0
    result["generation_chars_per_sec"] = gen_chars / eval_dur if eval_dur > 0 else 0.0
    result["prompt_utf8_bytes_per_sec"] = p_bytes / p_eval_dur if p_eval_dur > 0 else 0.0
    result["prompt_chars_per_sec"] = p_chars / p_eval_dur if p_eval_dur > 0 else 0.0

    # TPOT: average time per output token during decode (excluding first token).
    gen_tokens = result.get("generation_tokens", 0) or 0
    if eval_dur > 0 and gen_tokens > 1:
        result["time_per_output_token"] = eval_dur / (gen_tokens - 1)
    else:
        result["time_per_output_token"] = 0.0

    return result


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


def generate_xpost_text(
    results, model_name, framework, hardware_info=None, perplexity=None, batch_results=None, cached_results=None
):
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

        # Add memory information if available. When both peak memory and KV
        # cache are present, label both explicitly so the line stays readable.
        has_kv = "kv_cache_gb" in r
        if "peak_memory_gb" in r:
            if has_kv:
                line += f" mem {r['peak_memory_gb']:.1f}GB"
            else:
                line += f" {r['peak_memory_gb']:.1f}GB"
        if has_kv:
            line += f" kv {r['kv_cache_gb']:.2f}GB"
        if "kv_cache_usage_perc" in r:
            line += f" kv{r['kv_cache_usage_perc'] * 100:.0f}%"

        xpost += line + "\n"
        total_tokens += r.get("generation_tokens", 0)

    xpost += f"\nTotal generated tokens: {total_tokens}"

    if perplexity is not None:
        xpost += f"\nPerplexity: {perplexity:.2f}"

    if batch_results:
        parts = [f"b{r['batch_size']} {r['generation_tps']:.0f}" for r in batch_results]
        xpost += f"\nBatch TPS: {' '.join(parts)}"
        if any("kv_cache_gb" in r for r in batch_results):
            kv_parts = [f"b{r['batch_size']} {r.get('kv_cache_gb', 0):.2f}GB" for r in batch_results]
            xpost += f"\nBatch KV : {' '.join(kv_parts)}"
        if any("kv_cache_usage_perc" in r for r in batch_results):
            kv_parts = [f"b{r['batch_size']} {r.get('kv_cache_usage_perc', 0) * 100:.0f}%" for r in batch_results]
            xpost += f"\nBatch KV : {' '.join(kv_parts)}"

    if cached_results:
        xpost += "\n\nCached KV Cache (incremental prefill)\n"
        for r in sorted(cached_results, key=lambda x: float(x["context_size"][:-1])):
            if r.get("cached_tokens", 0) == 0:
                continue
            line = (
                f"{r['context_size']} cached {r.get('cached_tokens', 0)} "
                f"inc_pp {r.get('incremental_prompt_tps', 0):.0f} "
                f"tg {r['generation_tps']:.0f} t/s"
            )
            has_kv = "kv_cache_gb" in r
            if "peak_memory_gb" in r:
                if has_kv:
                    line += f" mem {r['peak_memory_gb']:.1f}GB"
                else:
                    line += f" {r['peak_memory_gb']:.1f}GB"
            if has_kv:
                line += f" kv {r['kv_cache_gb']:.2f}GB"
            xpost += line + "\n"

    return xpost.strip()


# Add a backward compatibility alias
generate_tweet_text = generate_xpost_text


def generate_table(
    results,
    model_name,
    framework,
    hardware_info=None,
    include_memory=False,
    perplexity=None,
    batch_results=None,
    cached_results=None,
):
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
        # Trailing memory columns render only when at least one result reports a
        # non-zero value, so engines that don't expose a metric (e.g. vLLM has no
        # peak VRAM) don't show an all-zero column.
        trail = []
        if any(r.get("peak_memory_gb", 0) > 0 for r in results):
            trail.append(("Memory", 8, lambda r: f"{r.get('peak_memory_gb', 0):>6.1f} GB"))
        if any("kv_cache_gb" in r for r in results):
            trail.append(("KV Cache", 8, lambda r: f"{r.get('kv_cache_gb', 0):>6.2f} GB"))
        if any("kv_cache_usage_perc" in r for r in results):
            trail.append(("KV Cache %", 9, lambda r: f"{r.get('kv_cache_usage_perc', 0) * 100:>7.1f}%"))

        base_cols = ["Context", "Prompt TPS", "Gen TPS", "Gen B/s", "Gen Ch/s", "Gen Tokens", "TPOT (ms)"]
        base_widths = [7, 10, 7, 7, 8, 10, 9]
        cols = base_cols + [t[0] for t in trail]
        widths = base_widths + [t[1] for t in trail]
        table += "\n" + " | ".join(c.rjust(w) for c, w in zip(cols, widths)) + "\n"
        table += "-|-".join("-" * w for w in widths) + "\n"

        for r in sorted(results, key=lambda x: float(x["context_size"][:-1])):
            gen_tokens = r.get("generation_tokens", 0)
            tpot_ms = r.get("time_per_output_token", 0) * 1000  # Convert to ms
            # Handle N/A prompt TPS for LM Studio
            if r.get("prompt_tps", 0) == 0 and "EXPERIMENTAL" in framework:
                prompt_str = f"{r.get('prompt_tokens', 0)} tok"
            else:
                prompt_str = f"{r['prompt_tps']:>10.1f}"
            parts = [
                f"{r['context_size']:>7}",
                prompt_str,
                f"{r['generation_tps']:>7.1f}",
                f"{r.get('generation_utf8_bytes_per_sec', 0):>7.1f}",
                f"{r.get('generation_chars_per_sec', 0):>8.1f}",
                f"{gen_tokens:>10}",
                f"{tpot_ms:>8.2f}",
            ] + [fmt(r) for _, _, fmt in trail]
            table += " | ".join(parts) + "\n"
            total_tokens += gen_tokens
    else:
        # Check if we need special handling for LM Studio
        if "EXPERIMENTAL" in framework:
            table += "\nContext | Prompt Tokens | Gen TPS | Gen B/s | Gen Ch/s | Gen Tokens | TPOT (ms) | Total Time\n"
            table += "--------|---------------|---------|---------|----------|------------|-----------|------------\n"
        else:
            table += "\nContext | Prompt TPS | Gen TPS | Gen B/s | Gen Ch/s | Gen Tokens | TPOT (ms) | Total Time\n"
            table += "--------|------------|---------|---------|----------|------------|-----------|------------\n"

        # Add data rows with total time
        for r in sorted(results, key=lambda x: float(x["context_size"][:-1])):
            total_time = r.get("total_time", r.get("wall_time", 0))
            gen_tokens = r.get("generation_tokens", 0)
            gen_bps = r.get("generation_utf8_bytes_per_sec", 0)
            gen_chps = r.get("generation_chars_per_sec", 0)
            tpot_ms = r.get("time_per_output_token", 0) * 1000  # Convert to ms
            # Handle N/A prompt TPS for LM Studio
            if r.get("prompt_tps", 0) == 0 and "EXPERIMENTAL" in framework:
                prompt_str = f"{r.get('prompt_tokens', 0)} tok"
                table += (
                    f"{r['context_size']:>7} | {prompt_str:>13} | {r['generation_tps']:>7.1f} | "
                    f"{gen_bps:>7.1f} | {gen_chps:>8.1f} | {gen_tokens:>10} | {tpot_ms:>8.2f} | {total_time:>9.1f}s\n"
                )
            else:
                table += (
                    f"{r['context_size']:>7} | {r['prompt_tps']:>10.1f} | {r['generation_tps']:>7.1f} | "
                    f"{gen_bps:>7.1f} | {gen_chps:>8.1f} | {gen_tokens:>10} | {tpot_ms:>8.2f} | {total_time:>9.1f}s\n"
                )
            total_tokens += gen_tokens

    table += f"\nTotal generated tokens: {total_tokens}"

    if perplexity is not None:
        table += f"\nPerplexity: {perplexity:.2f}"

    if batch_results:
        batch_has_bps = any(r.get("generation_utf8_bytes_per_sec", 0) > 0 for r in batch_results)
        batch_has_ttft = any(r.get("time_to_first_token", 0) > 0 for r in batch_results)
        batch_has_tpot = any(r.get("time_per_output_token", 0) > 0 for r in batch_results)
        # Same all-zero suppression as the main table for batch trailing columns.
        batch_trail = []
        if any(r.get("peak_memory_gb", 0) > 0 for r in batch_results):
            batch_trail.append(("Memory", 9, lambda r: f"{r.get('peak_memory_gb', 0):>6.1f} GB"))
        if any("kv_cache_gb" in r for r in batch_results):
            batch_trail.append(("KV Cache", 9, lambda r: f"{r.get('kv_cache_gb', 0):>6.2f} GB"))
        if any("kv_cache_usage_perc" in r for r in batch_results):
            batch_trail.append(("KV Cache %", 10, lambda r: f"{r.get('kv_cache_usage_perc', 0) * 100:>8.1f}%"))

        table += "\n\nBatch Benchmark\n"
        # Build header with optional Gen B/s column (only when populated by the engine)
        cols = ["Batch", "Prompt TPS", "Gen TPS"]
        widths = [5, 10, 7]
        if batch_has_bps:
            cols.append("Gen B/s")
            widths.append(7)
        if batch_has_ttft:
            cols.append("TTFT")
            widths.append(8)
        if batch_has_tpot:
            cols.append("TPOT")
            widths.append(8)
        cols += [t[0] for t in batch_trail]
        widths += [t[1] for t in batch_trail]
        header = " | ".join(c.rjust(w) for c, w in zip(cols, widths))
        sep = "-|-".join("-" * w for w in widths)
        table += header + "\n" + sep + "\n"
        for r in batch_results:
            parts = [
                f"{r['batch_size']:>5}",
                f"{r['prompt_tps']:>10.1f}",
                f"{r['generation_tps']:>7.1f}",
            ]
            if batch_has_bps:
                parts.append(f"{r.get('generation_utf8_bytes_per_sec', 0):>7.1f}")
            if batch_has_ttft:
                ttft_ms = r.get("time_to_first_token", 0) * 1000
                parts.append(f"{ttft_ms:>6.0f}ms" if ttft_ms > 0 else f"{'':>8}")
            if batch_has_tpot:
                tpot_ms = r.get("time_per_output_token", 0) * 1000
                parts.append(f"{tpot_ms:>6.1f}ms" if tpot_ms > 0 else f"{'':>8}")
            parts += [fmt(r) for _, _, fmt in batch_trail]
            table += " | ".join(parts) + "\n"

    if cached_results:
        cached_has_kv = any("kv_cache_gb" in r for r in cached_results)
        table += "\n\nCached KV Cache (incremental prefill)\n"
        if cached_has_kv:
            table += "Context | Total Tok | Delta Tok | Cached Tok | Inc Prefill TPS | Gen TPS | Gen B/s | KV Cache\n"
            table += "--------|-----------|-----------|------------|-----------------|---------|---------|----------\n"
        else:
            table += "Context | Total Tok | Delta Tok | Cached Tok | Inc Prefill TPS | Gen TPS | Gen B/s\n"
            table += "--------|-----------|-----------|------------|-----------------|---------|--------\n"
        for r in sorted(cached_results, key=lambda x: float(x["context_size"][:-1])):
            if r.get("cached_tokens", 0) == 0:
                continue
            row = (
                f"{r['context_size']:>7} | {r['prompt_tokens']:>9} | "
                f"{r.get('delta_tokens', 0):>9} | {r.get('cached_tokens', 0):>10} | "
                f"{r.get('incremental_prompt_tps', 0):>15.1f} | {r['generation_tps']:>7.1f} | "
                f"{r.get('generation_utf8_bytes_per_sec', 0):>7.1f}"
            )
            if cached_has_kv:
                row += f" | {r.get('kv_cache_gb', 0):>6.2f} GB"
            table += row + "\n"

    return table


def _plot_throughput_panel(ax, x, context_sizes, bytes_per_sec, chars_per_sec, title):
    """Draw a throughput panel: UTF-8 bytes/sec (bars) + chars/sec (twin axis line).

    Used by both create_chart_ollama and create_chart_mlx to render the
    tokenizer-independent throughput metrics consistently.
    """
    ax.set_title(title, fontsize=14, pad=10)
    color_bytes = "#3949AB"
    color_chars = "#43A047"

    bars = ax.bar(x, bytes_per_sec, color=color_bytes, width=0.6, alpha=0.7, label="UTF-8 bytes/s")
    if bytes_per_sec:
        max_b = max(bytes_per_sec) if bytes_per_sec else 1
        for bar, val in zip(bars, bytes_per_sec):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max_b * 0.02,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=color_bytes,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(context_sizes)
    ax.set_ylabel("Bytes/sec", color=color_bytes)
    ax.tick_params(axis="y", labelcolor=color_bytes)
    ax.set_ylim(0, max(bytes_per_sec) * 1.2 if bytes_per_sec and max(bytes_per_sec) > 0 else 1)
    ax.grid(True, axis="y", alpha=0.3)

    ax_right = ax.twinx()
    ax_right.plot(x, chars_per_sec, "o-", color=color_chars, linewidth=2, markersize=8, label="Chars/s")
    ax_right.set_ylabel("Chars/sec", color=color_chars)
    ax_right.tick_params(axis="y", labelcolor=color_chars)
    if chars_per_sec:
        max_c = max(chars_per_sec) if chars_per_sec else 1
        for i, c in enumerate(chars_per_sec):
            ax_right.text(i, c + max_c * 0.02, f"{c:.0f}", ha="center", va="bottom", fontsize=8, color=color_chars)
        ax_right.set_ylim(0, max(chars_per_sec) * 1.5 if max(chars_per_sec) > 0 else 1)

    ax.legend(loc="upper left", fontsize=9)
    ax_right.legend(loc="upper right", fontsize=9)


def create_chart_ollama(results, model_name, hardware_info, output_path="benchmark_chart.png", framework="Ollama"):
    """Create a chart for Ollama benchmarks with timing information."""
    # Sort results by context size
    context_sizes = []
    prompt_tps = []
    gen_tps = []
    total_times = []
    generation_tokens = []
    ttft_times = []
    gen_bytes_ps = []
    gen_chars_ps = []
    prompt_bytes_ps = []
    prompt_chars_ps = []

    for r in sorted(results, key=lambda x: float(x["context_size"][:-1])):
        context_sizes.append(r["context_size"])
        prompt_tps.append(r["prompt_tps"])
        gen_tps.append(r["generation_tps"])
        total_times.append(r.get("total_time", r.get("wall_time", 0)))
        generation_tokens.append(r["generation_tokens"])
        ttft_times.append(r.get("time_to_first_token", r.get("prompt_eval_duration", 0)))
        gen_bytes_ps.append(r.get("generation_utf8_bytes_per_sec", 0))
        gen_chars_ps.append(r.get("generation_chars_per_sec", 0))
        prompt_bytes_ps.append(r.get("prompt_utf8_bytes_per_sec", 0))
        prompt_chars_ps.append(r.get("prompt_chars_per_sec", 0))

    # Create figure with six subplots (3x2 grid): existing 4 + 2 throughput panels
    fig, axes = plt.subplots(3, 2, figsize=(15, 18), gridspec_kw={"hspace": 0.4, "wspace": 0.3})
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]
    # Throughput row: prompt on the left, generation (decode) on the right
    ax_thru_prompt, ax_thru_gen = axes[2]

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

    # Fourth subplot - Time to First Token (TTFT) and Time Per Output Token (TPOT)
    ax4.set_title("Latency: TTFT (left) and TPOT (right)", fontsize=14, pad=10)
    color_ttft = "#E91E63"  # Pink
    ax4.plot(x, ttft_times, "o-", color=color_ttft, linewidth=2, markersize=8, label="TTFT")
    ax4.set_ylabel("TTFT (seconds)", color=color_ttft)
    ax4.tick_params(axis="y", labelcolor=color_ttft)

    # Add value labels for TTFT
    if ttft_times:
        max_ttft = max(ttft_times) if ttft_times else 1
        for i, t in enumerate(ttft_times):
            ax4.text(i, t + max_ttft * 0.03, f"{t:.2f}s", ha="center", va="bottom", fontsize=9, color=color_ttft)

    ax4.set_xticks(x)
    ax4.set_xticklabels(context_sizes)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, max(ttft_times) * 1.15 if ttft_times and max(ttft_times) > 0 else 1)

    # Twin axis for TPOT (in milliseconds)
    ax4_tpot = ax4.twinx()
    tpot_times_ms = [r.get("time_per_output_token", 0) * 1000 for r in results]
    color_tpot = "#9C27B0"  # Purple
    ax4_tpot.plot(x, tpot_times_ms, "s--", color=color_tpot, linewidth=2, markersize=6, label="TPOT")
    ax4_tpot.set_ylabel("TPOT (ms)", color=color_tpot)
    ax4_tpot.tick_params(axis="y", labelcolor=color_tpot)

    # Add value labels for TPOT
    if tpot_times_ms:
        max_tpot = max(tpot_times_ms) if tpot_times_ms else 1
        for i, t in enumerate(tpot_times_ms):
            ax4_tpot.text(i, t + max_tpot * 0.03, f"{t:.1f}ms", ha="center", va="bottom", fontsize=8, color=color_tpot)
    ax4_tpot.set_ylim(0, max(tpot_times_ms) * 1.15 if tpot_times_ms and max(tpot_times_ms) > 0 else 1)

    # Add legends
    ax4.legend(loc="upper left", fontsize=9)
    ax4_tpot.legend(loc="upper right", fontsize=9)

    # Throughput panels (tokenizer-independent): prompt (left) + generation (right).
    # Hide the row entirely when an engine reports no usable throughput data.
    if any(v > 0 for v in prompt_bytes_ps + prompt_chars_ps + gen_bytes_ps + gen_chars_ps):
        _plot_throughput_panel(
            ax_thru_prompt,
            x,
            context_sizes,
            prompt_bytes_ps,
            prompt_chars_ps,
            "Prompt Throughput (tokenizer-free)",
        )
        _plot_throughput_panel(
            ax_thru_gen, x, context_sizes, gen_bytes_ps, gen_chars_ps, "Generation Throughput (tokenizer-free)"
        )
    else:
        ax_thru_prompt.set_visible(False)
        ax_thru_gen.set_visible(False)

    # Adjust layout with custom padding to prevent overlap
    plt.subplots_adjust(top=0.93, bottom=0.05, left=0.1, right=0.95, hspace=0.4, wspace=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


def create_chart_mlx(
    results,
    model_name,
    hardware_info,
    output_path="benchmark_chart.png",
    perplexity=None,
    batch_results=None,
    framework="MLX",
    cached_results=None,
):
    """Create a chart for MLX benchmarks with memory information."""
    # Sort results by context size
    context_sizes = []
    prompt_tps = []
    gen_tps = []
    peak_memory = []
    kv_cache = []
    generation_tokens = []
    total_times = []
    ttft_times = []
    gen_bytes_ps = []
    gen_chars_ps = []
    prompt_bytes_ps = []
    prompt_chars_ps = []

    for r in sorted(results, key=lambda x: float(x["context_size"][:-1])):
        context_sizes.append(r["context_size"])
        prompt_tps.append(r["prompt_tps"])
        gen_tps.append(r["generation_tps"])
        peak_memory.append(r.get("peak_memory_gb", 0))
        kv_cache.append(r.get("kv_cache_gb", 0))
        generation_tokens.append(r["generation_tokens"])
        total_times.append(r.get("total_time", 0))
        ttft_times.append(r.get("time_to_first_token", r.get("prompt_eval_duration", 0)))
        gen_bytes_ps.append(r.get("generation_utf8_bytes_per_sec", 0))
        gen_chars_ps.append(r.get("generation_chars_per_sec", 0))
        prompt_bytes_ps.append(r.get("prompt_utf8_bytes_per_sec", 0))
        prompt_chars_ps.append(r.get("prompt_chars_per_sec", 0))

    has_kv_cache = any(v > 0 for v in kv_cache)
    # vLLM reports KV cache pool utilization (0-1) instead of peak VRAM GB.
    has_kv_perc = any("kv_cache_usage_perc" in r for r in results)

    # Detect whether the batch sweep also tracked KV cache (mlx-lm always does;
    # mlx-vlm does once the patched fork is in place). When present, we add a
    # 5th row showing batch peak memory + batch kv cache vs batch size so the
    # KV growth across batch sizes is visible alongside the TPS panels.
    batch_has_kv = bool(batch_results) and any("kv_cache_gb" in r for r in batch_results)

    # Layout: base 3 rows + always-on throughput row + optional batch rows.
    # The throughput row is appended at the end so existing axes indices stay stable.
    if batch_has_kv:
        num_rows = 6
        fig_height = 28
    elif batch_results:
        num_rows = 5
        fig_height = 24
    else:
        num_rows = 4
        fig_height = 20
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, fig_height), gridspec_kw={"hspace": 0.4, "wspace": 0.3})
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]
    ax5, ax6 = axes[2]
    if batch_results:
        ax7, ax8 = axes[3]
    if batch_has_kv:
        ax9, ax10 = axes[4]
    # Throughput row is always the last row: prompt on the left, generation (decode) on the right
    ax_thru_prompt, ax_thru_gen = axes[num_rows - 1]

    # Model name and hardware in title
    hardware_str = format_hardware_string(hardware_info)
    fig.suptitle(f"{model_name} {framework} Testing\n{hardware_str}", fontsize=16, fontweight="bold")

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

    # Overlay cached incremental prompt TPS if available
    if cached_results:
        cached_data = sorted(cached_results, key=lambda r: float(r["context_size"].replace("k", "")))
        # Skip baseline point (cached_tokens=0 means cold prefill, identical to cold results)
        cached_data = [r for r in cached_data if r.get("cached_tokens", 0) > 0]
        # Only plot points whose context size exists in cold results
        cached_data = [r for r in cached_data if r["context_size"] in context_sizes]
        cached_sizes = [r["context_size"] for r in cached_data]
        if cached_sizes:
            cached_x = [context_sizes.index(s) for s in cached_sizes]
            cached_inc_tps = [r.get("incremental_prompt_tps", 0) for r in cached_data]
            ax1.plot(
                cached_x,
                cached_inc_tps,
                "s--",
                color="#2196F3",
                linewidth=2,
                markersize=8,
                label="Cached (incremental)",
            )
            for i, p in zip(cached_x, cached_inc_tps):
                ax1.text(i + 0.15, p, f"{p:.0f}", ha="left", va="bottom", fontsize=8, color="#2196F3")
            ax1.legend()

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

    # Overlay cached generation TPS if available
    if cached_results:
        cached_data = sorted(cached_results, key=lambda r: float(r["context_size"].replace("k", "")))
        cached_data = [r for r in cached_data if r.get("cached_tokens", 0) > 0]
        cached_data = [r for r in cached_data if r["context_size"] in context_sizes]
        cached_sizes = [r["context_size"] for r in cached_data]
        if cached_sizes:
            cached_x = [context_sizes.index(s) for s in cached_sizes]
            cached_gen_tps = [r.get("generation_tps", 0) for r in cached_data]
            ax2.plot(
                cached_x,
                cached_gen_tps,
                "s--",
                color="#2196F3",
                linewidth=2,
                markersize=8,
                label="Cached (incremental)",
            )
            ax2.legend()

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

    # Fourth subplot - Time to First Token (TTFT) and Time Per Output Token (TPOT)
    ax4.set_title("Latency: TTFT (left) and TPOT (right)", fontsize=14, pad=10)
    color_ttft = "#E91E63"  # Pink
    ax4.plot(x, ttft_times, "o-", color=color_ttft, linewidth=2, markersize=8, label="TTFT")
    ax4.set_ylabel("TTFT (seconds)", color=color_ttft)
    ax4.tick_params(axis="y", labelcolor=color_ttft)

    # Add value labels for TTFT
    if ttft_times:
        max_ttft = max(ttft_times) if ttft_times else 1
        for i, t in enumerate(ttft_times):
            ax4.text(i, t + max_ttft * 0.03, f"{t:.2f}s", ha="center", va="bottom", fontsize=9, color=color_ttft)

    ax4.set_xticks(x)
    ax4.set_xticklabels(context_sizes)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, max(ttft_times) * 1.15 if ttft_times and max(ttft_times) > 0 else 1)

    # Overlay cached TTFT if available
    if cached_results:
        cached_data = sorted(cached_results, key=lambda r: float(r["context_size"].replace("k", "")))
        cached_data = [r for r in cached_data if r.get("cached_tokens", 0) > 0]
        cached_data = [r for r in cached_data if r["context_size"] in context_sizes]
        cached_sizes = [r["context_size"] for r in cached_data]
        if cached_sizes:
            cached_x = [context_sizes.index(s) for s in cached_sizes]
            cached_ttft = [r.get("time_to_first_token", 0) for r in cached_data]
            ax4.plot(
                cached_x, cached_ttft, "s--", color="#2196F3", linewidth=2, markersize=8, label="Cached (incremental)"
            )
            for i, t in zip(cached_x, cached_ttft):
                ax4.text(i + 0.15, t, f"{t:.2f}s", ha="left", va="bottom", fontsize=8, color="#2196F3")
            ax4.legend()

    # Twin axis for TPOT (in milliseconds)
    ax4_tpot = ax4.twinx()
    tpot_times_ms = [r.get("time_per_output_token", 0) * 1000 for r in results]
    color_tpot = "#9C27B0"  # Purple
    ax4_tpot.plot(x, tpot_times_ms, "s--", color=color_tpot, linewidth=2, markersize=6, label="TPOT")
    ax4_tpot.set_ylabel("TPOT (ms)", color=color_tpot)
    ax4_tpot.tick_params(axis="y", labelcolor=color_tpot)

    # Add value labels for TPOT
    if tpot_times_ms:
        max_tpot = max(tpot_times_ms) if tpot_times_ms else 1
        for i, t in enumerate(tpot_times_ms):
            ax4_tpot.text(i, t + max_tpot * 0.03, f"{t:.1f}ms", ha="center", va="bottom", fontsize=8, color=color_tpot)
    ax4_tpot.set_ylim(0, max(tpot_times_ms) * 1.15 if tpot_times_ms and max(tpot_times_ms) > 0 else 1)

    # Add legends (avoid duplicate if cached TTFT already added a legend)
    if not cached_results:
        ax4.legend(loc="upper left", fontsize=9)
        ax4_tpot.legend(loc="upper right", fontsize=9)

    # Fifth subplot - memory / KV cache. Draw Peak Memory bars when an engine
    # reports VRAM; otherwise fall back to KV-cache pool utilization (vLLM);
    # hide the panel entirely when there is no memory data, so we never render
    # an empty all-zero chart.
    if has_kv_cache or any(r.get("peak_memory_gb", 0) > 0 for r in results):
        title5 = "Peak Memory Usage" + (" & KV Cache" if has_kv_cache else "")
        ax5.set_title(title5, fontsize=14, pad=10)

        # Bar chart for memory
        color_memory = "#ff9800"
        bars = ax5.bar(x, peak_memory, color=color_memory, width=0.6, alpha=0.7, label="Peak Memory")

        # Add value labels on bars
        if peak_memory:
            max_mem = max(peak_memory) if peak_memory else 1
            for i, (bar, mem) in enumerate(zip(bars, peak_memory)):
                height = bar.get_height()
                ax5.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max_mem * 0.02,
                    f"{mem:.1f} GB",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=color_memory,
                )

        ax5.set_xticks(x)
        ax5.set_xticklabels(context_sizes)
        ax5.set_ylabel("Memory (GB)", color=color_memory)
        ax5.tick_params(axis="y", labelcolor=color_memory)
        ax5.set_ylim(0, max(peak_memory) * 1.2 if peak_memory and max(peak_memory) > 0 else 1)

        # Overlay KV cache as a line on the same y-axis (both are GB)
        if has_kv_cache:
            color_kv = "#2196F3"
            ax5.plot(x, kv_cache, "s-", color=color_kv, linewidth=2, markersize=8, label="KV Cache")
            for i, kv in enumerate(kv_cache):
                if kv > 0:
                    ax5.text(
                        i,
                        kv,
                        f"{kv:.2f} GB",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color=color_kv,
                    )
            ax5.legend(loc="upper left", fontsize=9)

        # Add grid
        ax5.grid(True, axis="y", alpha=0.3)
    elif has_kv_perc:
        # vLLM: KV cache pool utilization (%) in place of peak VRAM GB.
        kv_perc = [
            r.get("kv_cache_usage_perc", 0) * 100 for r in sorted(results, key=lambda x: float(x["context_size"][:-1]))
        ]
        ax5.set_title("KV Cache Usage", fontsize=14, pad=10)
        color_kv = "#2196F3"
        bars = ax5.bar(x, kv_perc, color=color_kv, width=0.6, alpha=0.7, label="KV cache used")
        if kv_perc:
            max_kv = max(kv_perc) if kv_perc else 1
            for bar, kv in zip(bars, kv_perc):
                ax5.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + max_kv * 0.02,
                    f"{kv:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=color_kv,
                )
        ax5.set_xticks(x)
        ax5.set_xticklabels(context_sizes)
        ax5.set_ylabel("KV cache pool used (%)", color=color_kv)
        ax5.tick_params(axis="y", labelcolor=color_kv)
        ax5.set_ylim(0, max(kv_perc) * 1.2 if kv_perc and max(kv_perc) > 0 else 100)
        ax5.grid(True, axis="y", alpha=0.3)
    else:
        ax5.set_visible(False)

    # Sixth subplot - Perplexity info or hidden
    if perplexity is not None:
        ax6.set_visible(True)
        ax6.axis("off")
        ax6.text(
            0.5,
            0.5,
            f"Perplexity: {perplexity:.2f}",
            transform=ax6.transAxes,
            fontsize=24,
            fontweight="bold",
            ha="center",
            va="center",
            color="#d62728",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3f3", edgecolor="#d62728", alpha=0.8),
        )
        ax6.text(
            0.5,
            0.2,
            "Dataset: allenai/tulu-3-sft-mixture\n256 samples, seq_len 512",
            transform=ax6.transAxes,
            fontsize=10,
            ha="center",
            va="center",
            color="#666666",
        )
    else:
        ax6.set_visible(False)

    # Row 4: Batch benchmark charts (if present)
    if batch_results:
        batch_sizes = [r["batch_size"] for r in batch_results]
        batch_prompt_tps = [r["prompt_tps"] for r in batch_results]
        batch_gen_tps = [r["generation_tps"] for r in batch_results]
        batch_ttft_ms = [r.get("time_to_first_token", 0) * 1000 for r in batch_results]
        batch_tpot_ms = [r.get("time_per_output_token", 0) * 1000 for r in batch_results]
        bx = np.arange(len(batch_sizes))
        batch_labels = [str(bs) for bs in batch_sizes]

        # Batch Prompt TPS
        ax7.set_title("Batch Prompt Tokens per Second", fontsize=14, pad=10)
        color_bp = "#3F51B5"
        ax7.plot(bx, batch_prompt_tps, "o-", color=color_bp, linewidth=2, markersize=8)
        ax7.set_ylabel("Tokens/sec", color=color_bp)
        ax7.set_xlabel("Batch Size")
        ax7.tick_params(axis="y", labelcolor=color_bp)
        if batch_prompt_tps:
            max_bp = max(batch_prompt_tps) if batch_prompt_tps else 1
            for i, p in enumerate(batch_prompt_tps):
                ax7.text(i, p + max_bp * 0.03, f"{p:.1f}", ha="center", va="bottom", fontsize=9, color=color_bp)
        ax7.set_xticks(bx)
        ax7.set_xticklabels(batch_labels)
        ax7.grid(True, alpha=0.3)
        ax7.set_ylim(0, max(batch_prompt_tps) * 1.15 if batch_prompt_tps and max(batch_prompt_tps) > 0 else 1)

        if any(v > 0 for v in batch_ttft_ms):
            ax7_ttft = ax7.twinx()
            color_bttft = "#E91E63"
            ax7_ttft.plot(bx, batch_ttft_ms, "s--", color=color_bttft, linewidth=2, markersize=6, label="TTFT")
            ax7_ttft.set_ylabel("TTFT (ms)", color=color_bttft)
            ax7_ttft.tick_params(axis="y", labelcolor=color_bttft)
            max_bttft = max(batch_ttft_ms) if batch_ttft_ms else 1
            for i, t in enumerate(batch_ttft_ms):
                if t > 0:
                    ax7_ttft.text(
                        i, t + max_bttft * 0.03, f"{t:.0f}ms", ha="center", va="bottom", fontsize=8, color=color_bttft
                    )
            ax7_ttft.set_ylim(0, max_bttft * 1.15 if max_bttft > 0 else 1)
            ax7.legend(loc="upper left", fontsize=9)
            ax7_ttft.legend(loc="upper right", fontsize=9)

        # Batch Generation TPS
        ax8.set_title("Batch Generation Tokens per Second", fontsize=14, pad=10)
        color_bg = "#009688"
        ax8.plot(bx, batch_gen_tps, "o-", color=color_bg, linewidth=2, markersize=8)
        ax8.set_ylabel("Tokens/sec", color=color_bg)
        ax8.set_xlabel("Batch Size")
        ax8.tick_params(axis="y", labelcolor=color_bg)
        if batch_gen_tps:
            max_bg = max(batch_gen_tps) if batch_gen_tps else 1
            for i, g in enumerate(batch_gen_tps):
                ax8.text(i, g + max_bg * 0.03, f"{g:.1f}", ha="center", va="bottom", fontsize=9, color=color_bg)
        ax8.set_xticks(bx)
        ax8.set_xticklabels(batch_labels)
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim(0, max(batch_gen_tps) * 1.15 if batch_gen_tps and max(batch_gen_tps) > 0 else 1)

        if any(v > 0 for v in batch_tpot_ms):
            ax8_tpot = ax8.twinx()
            color_btpot = "#9C27B0"
            ax8_tpot.plot(bx, batch_tpot_ms, "s--", color=color_btpot, linewidth=2, markersize=6, label="TPOT")
            ax8_tpot.set_ylabel("TPOT (ms)", color=color_btpot)
            ax8_tpot.tick_params(axis="y", labelcolor=color_btpot)
            max_btpot = max(batch_tpot_ms) if batch_tpot_ms else 1
            for i, t in enumerate(batch_tpot_ms):
                if t > 0:
                    ax8_tpot.text(
                        i, t + max_btpot * 0.03, f"{t:.1f}ms", ha="center", va="bottom", fontsize=8, color=color_btpot
                    )
            ax8_tpot.set_ylim(0, max_btpot * 1.15 if max_btpot > 0 else 1)
            ax8.legend(loc="upper left", fontsize=9)
            ax8_tpot.legend(loc="upper right", fontsize=9)

    # Row 5: Batch Peak Memory + Batch KV Cache (when KV is tracked per batch)
    if batch_has_kv:
        batch_peak_mem = [r.get("peak_memory_gb", 0) for r in batch_results]
        batch_kv = [r.get("kv_cache_gb", 0) for r in batch_results]

        # Batch Peak Memory bars
        ax9.set_title("Batch Peak Memory Usage", fontsize=14, pad=10)
        color_bm = "#ff9800"
        ax9_bars = ax9.bar(bx, batch_peak_mem, color=color_bm, width=0.6, alpha=0.7)
        if batch_peak_mem:
            max_bm = max(batch_peak_mem) if batch_peak_mem else 1
            for bar, mem in zip(ax9_bars, batch_peak_mem):
                ax9.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + max_bm * 0.02,
                    f"{mem:.1f} GB",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=color_bm,
                )
        ax9.set_xticks(bx)
        ax9.set_xticklabels(batch_labels)
        ax9.set_xlabel("Batch Size")
        ax9.set_ylabel("Memory (GB)", color=color_bm)
        ax9.tick_params(axis="y", labelcolor=color_bm)
        ax9.set_ylim(0, max(batch_peak_mem) * 1.2 if batch_peak_mem and max(batch_peak_mem) > 0 else 1)
        ax9.grid(True, axis="y", alpha=0.3)

        # Batch KV Cache bars
        ax10.set_title("Batch KV Cache Usage", fontsize=14, pad=10)
        color_bkv = "#2196F3"
        ax10_bars = ax10.bar(bx, batch_kv, color=color_bkv, width=0.6, alpha=0.7)
        if batch_kv:
            max_bkv = max(batch_kv) if batch_kv else 1
            for bar, kv in zip(ax10_bars, batch_kv):
                ax10.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + max_bkv * 0.02,
                    f"{kv:.2f} GB",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=color_bkv,
                )
        ax10.set_xticks(bx)
        ax10.set_xticklabels(batch_labels)
        ax10.set_xlabel("Batch Size")
        ax10.set_ylabel("KV Cache (GB)", color=color_bkv)
        ax10.tick_params(axis="y", labelcolor=color_bkv)
        ax10.set_ylim(0, max(batch_kv) * 1.2 if batch_kv and max(batch_kv) > 0 else 1)
        ax10.grid(True, axis="y", alpha=0.3)

    # Final row: tokenizer-independent throughput panels (prompt left, generation right).
    # Hide the row entirely when an engine reports no usable throughput data.
    if any(v > 0 for v in prompt_bytes_ps + prompt_chars_ps + gen_bytes_ps + gen_chars_ps):
        _plot_throughput_panel(
            ax_thru_prompt,
            x,
            context_sizes,
            prompt_bytes_ps,
            prompt_chars_ps,
            "Prompt Throughput (tokenizer-free)",
        )
        _plot_throughput_panel(
            ax_thru_gen, x, context_sizes, gen_bytes_ps, gen_chars_ps, "Generation Throughput (tokenizer-free)"
        )
    else:
        ax_thru_prompt.set_visible(False)
        ax_thru_gen.set_visible(False)

    # Adjust layout with custom padding to prevent overlap
    plt.subplots_adjust(top=0.93, bottom=0.05, left=0.1, right=0.95, hspace=0.4, wspace=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


def _get_machine_name() -> str:
    """Extract a short machine identifier for output directory naming."""
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                chip_match = re.search(r"Chip:\s+(.+)", result.stdout)
                model_match = re.search(r"Model Name:\s+(.+)", result.stdout)
                if chip_match and model_match:
                    # "Apple M3 Ultra" → "M3Ultra", "MacBook Pro" is dropped
                    chip = chip_match.group(1).strip()
                    # Remove "Apple " prefix and spaces
                    short = chip.replace("Apple ", "").replace(" ", "")
                    return short
                if chip_match:
                    return chip_match.group(1).strip().replace("Apple ", "").replace(" ", "")
        except Exception:
            pass
    return platform.node().split(".")[0]


def create_output_directory(
    framework_name: str,
    model_name: str,
    base_dir: str = "output",
    cold_prefill: bool = True,
    machine_name: Optional[str] = None,
) -> Path:
    """Create timestamped output directory for benchmark results.

    Args:
        framework_name: Name of the framework (e.g., "ollama", "mlx", "llamacpp")
        model_name: Name of the model being benchmarked
        base_dir: Base directory for output (default: "output")
        cold_prefill: If True, append _nocache suffix to directory name
        machine_name: Optional hardware label to use in the directory name

    Returns:
        Path object for the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(base_dir)
    base_output_dir.mkdir(exist_ok=True)

    # Sanitize model name for filesystem
    model_safe = model_name.replace("/", "_").replace(":", "_")

    # Extract short machine name for the directory, unless a server-backed
    # benchmark provided a remote hardware label.
    machine_name = machine_name or _get_machine_name()

    cache_tag = "_nocache" if cold_prefill else "_cache"
    output_dir = base_output_dir / f"benchmark_{framework_name}_{model_safe}-{machine_name}{cache_tag}_{timestamp}"
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
        default=128,
        help="Maximum tokens to generate (default: 128)",
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
        "--runs",
        type=int,
        default=2,
        help="Number of runs per context size; peak score is kept (default: 2)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds for each benchmark (default: 3600 = 60 minutes)",
    )


def make_cache_buster(run_idx: Optional[int] = None) -> str:
    """Generate a prefix to bust server-side KV prompt cache.

    Many inference servers (Ollama, llama.cpp, vLLM, LM Studio, etc.) reuse the
    KV cache when consecutive prompts share a prefix.  That causes
    prompt_eval_duration to report only the *uncached delta* while
    prompt_eval_count still reports the full length, inflating the derived
    prompt tokens/sec.

    When ``run_idx`` is provided, the buster is deterministic per run index:
    all calls within the same run share the same prefix (so KV cache carries
    over across context sizes), while different runs get different prefixes
    (so runs don't interfere). When ``run_idx`` is None, a random UUID is
    used for full cold-prefill busting. ~10 tokens of overhead per prompt.
    """
    if run_idx is not None:
        return f"[run-{run_idx}]\n"
    import uuid

    return f"[session-{uuid.uuid4().hex[:16]}]\n"


def _is_degenerate_generation(result: Dict) -> bool:
    """Return True if the generation phase was too short for a reliable TPS measurement."""
    if result.get("generation_tokens", 0) < 2:
        return True
    eval_duration = result.get("eval_duration", 0)
    if eval_duration > 0 and eval_duration < 0.01:
        return True
    return False


def run_benchmark_peak(run_fn, *args, n_runs=2, metric="generation_tps", **kwargs):
    """Run benchmark N times and return the result with peak generation_tps.

    Each run gets a unique ``_run_idx`` keyword argument so the engine can
    bust any server-side KV cache that may carry over from a previous run.
    Engines that accept ``_run_idx`` should use it to vary the prompt prefix
    (e.g. ``common.make_cache_buster()`` already includes a UUID; the run
    index is an additional differentiator).

    Degenerate runs (near-zero generation time or <2 tokens generated) are
    discarded so they don't inflate the peak score.

    Args:
        run_fn: Engine-specific run_benchmark function.
        *args: Positional arguments forwarded to run_fn.
        n_runs: Number of runs per context size.
        metric: Metric to maximize when selecting the peak run.
        **kwargs: Keyword arguments forwarded to run_fn.

    Returns:
        Result dict from the run with the highest metric value, or None if all runs fail.
    """
    best_result = None
    best_score = -1
    for run_idx in range(1, n_runs + 1):
        print(f"  Run {run_idx}/{n_runs}...")
        kwargs["_run_idx"] = run_idx
        result = run_fn(*args, **kwargs)
        if result:
            if _is_degenerate_generation(result):
                print(
                    f"    Skipping degenerate run: {result.get('generation_tokens', 0)} tokens "
                    f"in {result.get('eval_duration', 0):.6f}s "
                    f"(tps={result.get('generation_tps', 0):.0f})"
                )
                continue
            score = result.get(metric, 0)
            print(f"    {metric}: {score:.2f}")
            if score > best_score:
                best_score = score
                best_result = result
    if best_result:
        print(f"  Peak {metric}: {best_score:.2f}")
    # Clean up so the kwarg doesn't leak if the dict is reused
    kwargs.pop("_run_idx", None)
    return best_result


def run_benchmark_peak_per_run(run_fn, context_files, n_runs=2, metric="generation_tps", **kwargs):
    """Run benchmark with each run completing all context sizes sequentially.

    Unlike ``run_benchmark_peak`` (which runs all runs per context size), this
    runs all context sizes per run.  This ensures KV cache accumulates within a
    run — e.g. run 1 processes 1k→2k→4k→…, then run 2 starts fresh at
    1k→2k→4k→….  Each run gets a deterministic prefix (via ``_run_idx``) so
    different runs are cache-isolated while the same run reuses cache across
    context sizes.

    For each context size, the result with the peak metric value across runs is
    returned.

    Args:
        run_fn: Engine-specific run_benchmark function.
        context_files: List of context file Paths to benchmark.
        n_runs: Number of runs per context size.
        metric: Metric to maximize when selecting the peak run.
        **kwargs: Keyword arguments forwarded to run_fn.

    Returns:
        List of result dicts (one per context file), ordered by context file.
    """
    all_results = {cf.stem: [] for cf in context_files}

    for run_idx in range(1, n_runs + 1):
        for context_file in context_files:
            print(f"\n{'=' * 50}")
            print(f"Run {run_idx}/{n_runs} — Benchmarking {context_file.name}...")
            print(f"{'=' * 50}")

            kwargs["_run_idx"] = run_idx
            try:
                result = run_fn(context_file=context_file, **kwargs)
            except Exception as exc:
                print(f"    Error: {exc}")
                result = None
            if result:
                if _is_degenerate_generation(result):
                    print(
                        f"    Skipping degenerate run: {result.get('generation_tokens', 0)} tokens "
                        f"in {result.get('eval_duration', 0):.6f}s "
                        f"(tps={result.get('generation_tps', 0):.0f})"
                    )
                    continue
                score = result.get(metric, 0)
                print(f"    {metric}: {score:.2f}")
                all_results[context_file.stem].append(result)

    results = []
    for context_file in context_files:
        run_results = all_results[context_file.stem]
        if run_results:
            best = max(run_results, key=lambda r: r.get(metric, 0))
            print(f"  {context_file.name}: Peak {metric}: {best.get(metric, 0):.2f}")
            results.append(best)

    kwargs.pop("_run_idx", None)
    return results


def save_all_outputs(
    results: List[Dict],
    output_dir: Path,
    model_name: str,
    framework: str,
    hardware_info: Dict,
    args: argparse.Namespace,
    include_memory: bool = False,
    perplexity: Optional[float] = None,
    perplexity_data: Optional[Dict] = None,
    batch_results: Optional[List[Dict]] = None,
    cached_results: Optional[List[Dict]] = None,
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
        perplexity: Perplexity score (optional, MLX only)
        perplexity_data: Full perplexity data dict to save as JSON (optional)
        batch_results: Batch benchmark results (optional)
        cached_results: Cached KV cache benchmark results (optional)
    """
    # Save hardware info
    hardware_path = output_dir / "hardware_info.json"
    save_hardware_info(hardware_info, hardware_path)

    # Save CSV results
    csv_path = output_dir / args.output_csv
    save_results_csv(results, csv_path)

    # Save perplexity data if available
    if perplexity_data is not None:
        ppl_path = output_dir / "perplexity.json"
        with open(ppl_path, "w") as f:
            json.dump(perplexity_data, f, indent=2)
        print(f"Perplexity data saved to {ppl_path}")

    # Save batch benchmark results if available
    if batch_results:
        batch_path = output_dir / "batch_benchmark.json"
        with open(batch_path, "w") as f:
            json.dump(batch_results, f, indent=2)
        print(f"Batch benchmark saved to {batch_path}")

    # Save cached benchmark results if available
    if cached_results:
        cached_csv_path = output_dir / "benchmark_results_cached.csv"
        save_results_csv(cached_results, cached_csv_path)
        cached_json = {
            "model": model_name,
            "framework": framework,
            "cached": True,
            "results": cached_results,
        }
        cached_json_path = output_dir / "all_results_cached.json"
        with open(cached_json_path, "w") as f:
            json.dump(cached_json, f, indent=2)
        print(f"Cached results saved to {cached_json_path}")

    # Generate and save chart
    chart_path = output_dir / args.output_chart
    if include_memory:
        create_chart_mlx(
            results,
            model_name,
            hardware_info,
            chart_path,
            perplexity=perplexity,
            batch_results=batch_results,
            framework=framework,
            cached_results=cached_results,
        )
    else:
        create_chart_ollama(results, model_name, hardware_info, chart_path, framework)
    print(f"Chart saved to {chart_path}")

    # Generate and save table
    table = generate_table(
        results,
        model_name,
        framework,
        hardware_info,
        include_memory,
        perplexity=perplexity,
        batch_results=batch_results,
        cached_results=cached_results,
    )
    table_path = output_dir / "table.txt"
    with open(table_path, "w") as f:
        f.write(table)
    print(f"Table saved to {table_path}")

    # Generate and save X post
    xpost = generate_xpost_text(
        results,
        model_name,
        framework,
        hardware_info,
        perplexity=perplexity,
        batch_results=batch_results,
        cached_results=cached_results,
    )
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
    perplexity: Optional[float] = None,
    batch_results: Optional[List[Dict]] = None,
    cached_results: Optional[List[Dict]] = None,
) -> None:
    """Print benchmark summary to console.

    Args:
        results: List of benchmark result dictionaries
        model_name: Name of the model
        framework: Framework name
        hardware_info: Hardware information dictionary
        output_dir: Directory where outputs were saved
        total_benchmark_time: Total time to run all benchmarks in seconds
        perplexity: Perplexity score (optional)
        batch_results: Batch benchmark results (optional)
        cached_results: Cached KV cache benchmark results (optional)
    """
    # Calculate total generated tokens
    total_generated_tokens = sum(r.get("generation_tokens", 0) for r in results)
    print(f"\n📊 Total generated tokens across all tests: {total_generated_tokens}")

    # Print total benchmark time if provided
    if total_benchmark_time is not None:
        print(f"⏱️  Total time to run benchmarks: {total_benchmark_time:.1f} seconds")

    # Print perplexity if available
    if perplexity is not None:
        print(f"🎯 Perplexity: {perplexity:.2f}")

    # Print batch results if available
    if batch_results:
        print("\n" + "=" * 50)
        print("BATCH BENCHMARK RESULTS")
        print("=" * 50)
        for r in batch_results:
            line = f"  Batch {r['batch_size']:>2}: pp {r['prompt_tps']:.1f} tg {r['generation_tps']:.1f} t/s"
            if r.get("time_to_first_token", 0) > 0:
                line += f", TTFT {r['time_to_first_token'] * 1000:.0f}ms"
            if r.get("time_per_output_token", 0) > 0:
                line += f", TPOT {r['time_per_output_token'] * 1000:.0f}ms"
            if r.get("peak_memory_gb", 0) > 0:
                line += f", mem {r['peak_memory_gb']:.2f} GB"
            if "kv_cache_gb" in r:
                line += f", kv {r['kv_cache_gb']:.2f} GB"
            if "kv_cache_usage_perc" in r:
                line += f", kv {r['kv_cache_usage_perc'] * 100:.0f}%"
            print(line)

    # Print cached benchmark results if available
    if cached_results:
        print("\n" + "=" * 50)
        print("CACHED KV CACHE BENCHMARK RESULTS")
        print("=" * 50)
        print(
            f"{'Context':>8} | {'Total':>6} | {'Delta':>6} | {'Cached':>6} | "
            f"{'Inc Prefill TPS':>14} | {'Gen TPS':>10}"
        )
        print(f"{'':>8} | {'Tok':>6} | {'Tok':>6} | {'Tok':>6} | {'':>14} | {'':>10}")
        print("-" * 72)
        for r in sorted(cached_results, key=lambda x: float(x["context_size"].replace("k", ""))):
            print(
                f"{r['context_size']:>8} | {r['prompt_tokens']:>6} | "
                f"{r.get('delta_tokens', 0):>6} | "
                f"{r.get('cached_tokens', 0):>6} | "
                f"{r.get('incremental_prompt_tps', 0):>14.1f} | "
                f"{r['generation_tps']:>10.1f}"
            )

    # Print summary table
    print("\n" + "=" * 50)
    print("SUMMARY TABLE")
    print("=" * 50)
    table = generate_table(
        results,
        model_name,
        framework,
        hardware_info,
        perplexity=perplexity,
        batch_results=batch_results,
        cached_results=cached_results,
    )
    print(table)

    # Print X post text
    print("\n" + "=" * 50)
    print("X POST TEXT")
    print("=" * 50)
    xpost = generate_xpost_text(
        results,
        model_name,
        framework,
        hardware_info,
        perplexity=perplexity,
        batch_results=batch_results,
        cached_results=cached_results,
    )
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
