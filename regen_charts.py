#!/usr/bin/env python3
"""Regenerate charts and text outputs from existing benchmark result folders.

Useful when the folder was renamed but the chart PNG still has the old
(or missing) model name baked into the image.

Usage:
    uv run regen-charts output/benchmark_mlx_MiniMax-M2.7-6bit_nocache_M3Ultra_20260412_140357
    uv run regen-charts output/benchmark_mlx_*              # regenerate all MLX folders
    uv run regen-charts output/                             # regenerate everything
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import benchmark_common as common
from compare_benchmarks import read_folder_identity

# Display name per engine folder tag, for folders older than metadata.json
# (newer folders record the framework name they were run with).
FRAMEWORK_NAMES = {
    "afms": "Apple Foundation Models Serve",
    "deepseek": "DeepSeek API",
    "dflash": "dflash-mlx API",
    "exo": "Exo API",
    "grok": "Grok API",
    "llamacpp": "llama.cpp",
    "llamacpp_embed": "llama.cpp Embed [EXPERIMENTAL]",
    "lucebox": "Lucebox",
    "lmstudio": "LM Studio",
    "mlx": "MLX",
    "mlx-distributed": "MLX Distributed",
    "mlx_vlm": "MLX-VLM",
    "mlx_vlm_server": "MLX-VLM Server",
    "mlx_serve": "MLX-Serve",
    "mtplx": "MTPLX API",
    "ollama_api": "Ollama API",
    "ollama_cli": "Ollama CLI",
    "omlx": "oMLX API",
    "openai_compat": "OpenAI Compat",
    "paroquant": "Paroquant",
    "sglang": "SGLang",
    "unsloth": "Unsloth",
    "vllm": "vLLM",
    "vmlx": "vMLX API",
}


def load_csv_results(csv_path: Path):
    """Load benchmark results from CSV file."""
    results = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in row:
                try:
                    if key == "context_size":
                        continue
                    row[key] = float(row[key])
                except (ValueError, TypeError):
                    pass
            results.append(dict(row))
    return results


def regen_folder(folder: Path, model_override: str = None, dry_run: bool = False):
    """Regenerate outputs for a single benchmark folder."""
    csv_path = folder / "benchmark_results.csv"
    hw_path = folder / "hardware_info.json"
    batch_path = folder / "batch_benchmark.json"
    ppl_path = folder / "perplexity.json"
    cached_csv_path = folder / "benchmark_results_cached.csv"

    if not csv_path.exists():
        print(f"  Skipping {folder.name}: no benchmark_results.csv")
        return False

    hardware_info = json.loads(hw_path.read_text()) if hw_path.exists() else {}
    identity = read_folder_identity(folder, hardware_info)
    metadata = identity["metadata"] or {}
    model_name = model_override or metadata.get("model_label") or metadata.get("model") or identity["model"]
    if not model_name or model_name == "unknown":
        print(f"  Skipping {folder.name}: could not determine model name")
        return False
    framework = metadata.get("framework") or FRAMEWORK_NAMES.get(identity["engine"], identity["engine"])

    # Load data
    results = load_csv_results(csv_path)
    batch_results = json.loads(batch_path.read_text()) if batch_path.exists() else None
    perplexity_data = json.loads(ppl_path.read_text()) if ppl_path.exists() else None
    perplexity = perplexity_data.get("perplexity") if perplexity_data else None

    # Load cached results if available
    cached_results = None
    if cached_csv_path.exists():
        cached_results = load_csv_results(cached_csv_path)

    include_memory = any(r.get("peak_memory_gb", 0) > 0 for r in results)

    print(f"  {folder.name}")
    print(f"    Model: {model_name}, Framework: {framework}, Contexts: {len(results)}, Memory: {include_memory}")

    if dry_run:
        return True

    # Build a minimal args namespace
    args = argparse.Namespace(
        output_csv="benchmark_results.csv",
        output_chart="benchmark_chart.png",
    )

    # Regenerate all outputs
    common.save_all_outputs(
        results,
        folder,
        model_name,
        framework,
        hardware_info,
        args,
        include_memory=include_memory,
        perplexity=perplexity,
        perplexity_data=perplexity_data,
        batch_results=batch_results,
        cached_results=cached_results,
        record_run=False,
    )

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate charts and text outputs from existing benchmark folders",
    )
    parser.add_argument(
        "folders",
        nargs="+",
        help="Benchmark output folder(s) to regenerate. Accepts globs.",
    )
    parser.add_argument(
        "--model",
        help="Override model name (applied to all folders)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing files",
    )

    args = parser.parse_args()

    # Collect folders
    folders = []
    for pattern in args.folders:
        p = Path(pattern)
        if p.is_dir():
            # If it's a parent dir (like output/), find benchmark subdirs
            if (p / "benchmark_results.csv").exists():
                folders.append(p)
            else:
                for child in sorted(p.iterdir()):
                    if child.is_dir() and (child / "benchmark_results.csv").exists():
                        folders.append(child)
        else:
            # Try glob
            import glob

            for match in sorted(glob.glob(pattern)):
                mp = Path(match)
                if mp.is_dir() and (mp / "benchmark_results.csv").exists():
                    folders.append(mp)

    if not folders:
        print("No benchmark folders found.")
        return 1

    print(f"Regenerating outputs for {len(folders)} folder(s):\n")
    success = 0
    for folder in folders:
        if regen_folder(folder, model_override=args.model, dry_run=args.dry_run):
            success += 1

    print(f"\nDone: {success}/{len(folders)} folder(s) regenerated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
