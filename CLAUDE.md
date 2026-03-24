# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM context benchmarking toolkit that measures prompt processing speed and generation speed (tokens/sec) across increasing context sizes (0.5k–128k tokens) for multiple inference engines (Ollama, MLX, llama.cpp, Exo, DeepSeek, Grok, OpenAI-compatible, etc.).

## Commands

```bash
uv sync                              # Install dependencies
pre-commit install                   # Set up formatting hooks (one-time)
pre-commit run --all-files           # Run Black + isort manually

uv run benchmark -- --list-engines   # List available engines
uv run benchmark -- ollama-api <model>
uv run benchmark -- mlx <model>
uv run benchmark -- llamacpp <model> --host localhost --port 8080
uv run benchmark -- <engine> <model> --contexts 2,4,8,16 --max-tokens 500 --timeout 7200

uv run compare-benchmarks            # Compare benchmark results
uv run generate-context-files -- <source.txt> --sizes 2,4,8,16,32,64,128
```

Scripts can also be run directly: `python mlx_benchmark.py <model>`

There are no automated tests in this project.

## Architecture

**Dispatcher + engine plugins** pattern. All files are flat Python modules in the project root (no package hierarchy).

- **`benchmark.py`** — Unified CLI entry point. Maps engine names to engine-specific scripts and delegates via `subprocess.run()`.
- **`benchmark_common.py`** — Shared library used by all engines. Provides: hardware detection, context file discovery, CLI argument setup (`setup_common_args()`), result serialization (CSV, JSON, charts), chart generation (`create_chart_ollama()`, `create_chart_mlx()`), and summary formatting.
- **`{engine}_benchmark.py`** — Engine-specific scripts (ollama_api, ollama_cli, mlx, mlx_distributed, llamacpp, lmstudio, exo, deepseek, grok, openai, vllm). Each follows: parse args → verify engine → collect hardware → warmup → iterate contexts → save outputs → print summary.
- **`compare_benchmarks.py`** — Multi-benchmark comparison tool. Reads result directories, produces side-by-side charts/CSV/tables.
- **`generate_context_files.py`** — Generates token-precise context files (`{size}k.txt`) using tiktoken.

## Key Conventions

- **Use `uv`** for all Python operations (install, run, sync).
- **Code style**: Black (line-length 120) + isort (profile: black, line-length 120).
- **Python 3.13+** required.
- **Result dict contract**: All engines produce dicts with core keys: `context_size`, `prompt_tokens`, `prompt_tps`, `generation_tokens`, `generation_tps`, `total_time`, `eval_duration`, `prompt_eval_duration`, `time_to_first_token`, `generated_text`. MLX additionally includes `peak_memory_gb`.
- **Common args**: Every engine calls `benchmark_common.setup_common_args(parser)` for shared CLI arguments (`--contexts`, `--max-tokens`, `--save-responses`, `--output-csv`, `--output-chart`, `--timeout`).
- **Context files**: Named `{size}k.txt` (e.g., `2k.txt`, `0.5k.txt`). Discovered via glob.
- **Output directories**: `output/benchmark_{engine}_{model}_{YYYYMMDD_HHMMSS}/`
- **Conditional imports**: Framework-specific deps (mlx, paroquant) use `try/except ImportError`.
- **`lmstudio_benchmark.py`** is an older script that doesn't use `setup_common_args()` or `save_all_outputs()` — it has its own manual argparse and output logic.
- **`vllm_benchmark.py`** and **`llamacpp_embed_benchmark.py`** exist but are not registered as entry points in `pyproject.toml`.
- **Two chart types**: `create_chart_ollama()` (2x2: prompt TPS, gen TPS, total time, TTFT) and `create_chart_mlx()` (3x2 or 4x2: adds memory and optional batch/perplexity charts).
