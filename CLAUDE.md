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
- **Result dict contract**: All engines produce dicts with core keys: `context_size`, `prompt_tokens`, `prompt_tps`, `generation_tokens`, `generation_tps`, `total_time`, `eval_duration`, `prompt_eval_duration`, `time_to_first_token`, `generated_text`. MLX additionally includes `peak_memory_gb`. Tokenizer-independent throughput keys (added by `benchmark_common.add_throughput_metrics`): `generation_utf8_bytes_per_sec`, `generation_chars_per_sec`, `prompt_utf8_bytes_per_sec`, `prompt_chars_per_sec` — every engine should call the helper once before returning so cross-tokenizer comparisons are possible.
- **Common args**: Every engine calls `benchmark_common.setup_common_args(parser)` for shared CLI arguments (`--contexts`, `--max-tokens`, `--save-responses`, `--output-csv`, `--output-chart`, `--timeout`).
- **Context files**: Named `{size}k.txt` (e.g., `2k.txt`, `0.5k.txt`). Discovered via glob.
- **Output directories**: `output/benchmark_{engine}_{model}_{YYYYMMDD_HHMMSS}/`
- **Conditional imports**: Framework-specific deps (mlx, paroquant) use `try/except ImportError`.
- **`lmstudio_benchmark.py`** is an older script that doesn't use `setup_common_args()` or `save_all_outputs()` — it has its own manual argparse and output logic.
- **`llamacpp_embed_benchmark.py`** exists but is not registered as an entry point in `pyproject.toml`. (`vllm_benchmark.py` is registered as the `vllm` engine + `vllm-benchmark` script and supports continuous-batch sweeps via `--batch-sizes`.)
- **Two chart types**: `create_chart_ollama()` (3x2: prompt TPS, gen TPS, total time, TTFT, plus a row of tokenizer-free throughput panels — gen + prompt bytes/sec with chars/sec on a twin axis) and `create_chart_mlx()` (4x2 base, grows to 5x2/6x2 with batch/batch+KV: standard panels + memory + tokenizer-free throughput row + optional perplexity/batch rows).
