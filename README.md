# LLM Context Benchmarks

Benchmark prompt-processing and generation throughput across context sizes
(0.5k–128k tokens) for many inference engines: Ollama (API & CLI), MLX,
MLX Distributed, MLX-VLM, llama.cpp, LM Studio, Exo, vMLX, oMLX, Paroquant,
and any OpenAI-compatible endpoint.

Optimized for Apple Silicon but works anywhere Python runs.

## Installation

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

Engine-specific setup:

| Engine | Setup |
| --- | --- |
| Ollama | Install [Ollama](https://ollama.com), `ollama pull <model>` |
| MLX / MLX-VLM / vMLX / oMLX | Apple Silicon only; models download from Hugging Face on first run |
| MLX Distributed | Requires `mlx.launch` and a hostfile JSON |
| llama.cpp | Run `llama-server -m model.gguf --port 8080` |
| LM Studio | Install [LM Studio](https://lmstudio.ai), start the local server |
| Exo / OpenAI-compatible | Any server exposing `/v1/chat/completions` |

(Optional) pre-commit hooks for Black + isort:

```bash
pre-commit install
```

## Running Benchmarks

```bash
# List engines
uv run benchmark --list-engines

# Generate test files (only needed once)
uv run generate-context-files pride_and_prejudice.txt

# Run a benchmark (engine + model)
uv run benchmark mlx mlx-community/Qwen3-4B-Instruct-2507-4bit
uv run benchmark ollama-api gpt-oss:20b
uv run benchmark llamacpp gpt-oss:20b --host localhost --port 8080

# Generic OpenAI-compatible endpoint (separate entry point)
uv run openai-benchmark --model llama3.2 --base-url http://localhost:11434/v1
```

Common options:

- `--contexts 0.5,1,2,4,8,16,32` — context sizes to test (in thousands of tokens)
- `--max-tokens 200` — generation cap per run
- `--timeout 7200` — per-context timeout (default 3600s)
- `--save-responses` — save model outputs to `response_<size>.txt`
- `--runs 3` — repeat each context size and keep the peak

Engine-specific options worth knowing:

- `--kv-bit 4|8`, `--max-kv-size N` — MLX KV cache quantization / cap
- `--host`, `--port` — llama.cpp server target
- `--backend`, `--hostfile`, `--env`, `--pipeline` — MLX Distributed
- `--base-url`, `--api-key` — OpenAI-compatible endpoints

## Comparing Runs

After running multiple benchmarks, aggregate them:

```bash
# Auto-discover everything in output/
uv run compare-benchmarks

# Compare specific folders
uv run compare-benchmarks output/benchmark_ollama_* output/benchmark_mlx_*

# Custom output directory
uv run compare-benchmarks --output my_comparison
```

Generates `comparison_chart.png`, `comparison_results.csv`,
`comparison_table.txt`, plus per-engine heatmaps.

### KL Divergence (MLX and llama.cpp)

MLX and llama.cpp benchmarks automatically capture top-K logprobs over a fixed
reference (the first ~512 tokens of `2k.txt`) into `logprobs.json` in each run
directory. Cost: one extra forward pass, ~50 KB of disk.

To compare distributions:

```bash
uv run compare-benchmarks --kl-baseline output/benchmark_mlx_<bf16-run>
```

Outputs:

- `kl_divergence.csv` — mean KL per target run
- `kl_divergence.png` — bar chart + per-position trace
- A KL panel inside `comparison_chart.png`, paired with perplexity

**Use bf16 as the baseline when possible.** Lower-precision runs (8-bit, 6-bit,
4-bit, …) are quantizations of the bf16 weights, so KL(bf16 || quantized)
directly measures how much the quantization distorts the output distribution.
A quantized baseline conflates errors and is harder to interpret.

Caveats:

- Both runs must use the **same tokenizer** — KL is computed on display-string
  tokens, so different tokenizer families produce noise.
- llama.cpp capture needs a recent server build with OpenAI-compat
  `echo + logprobs` support.
- Pass `--no-kl-capture` to either benchmark to skip the capture step.
- Don't put `--` between the command and `--kl-baseline`; uv passes the `--`
  through and argparse then treats the flag as positional.

## Output Files

Each run writes a timestamped directory under `output/`:

| File | Contents |
| --- | --- |
| `hardware_info.json` | CPU/GPU/memory specs |
| `benchmark_results.csv` | Per-context metrics (TPS, TTFT, total time, …) |
| `benchmark_chart.png` | Visual chart with hardware in the title |
| `table.txt` | Formatted results table |
| `xpost.txt` | Summary text for social posts |
| `perplexity.json` | Perplexity score (MLX) |
| `batch_benchmark.json` | Batch-size sweep (MLX) |
| `logprobs.json` | Top-K logprobs for KL comparison (MLX, llama.cpp) |
| `response_<size>.txt` | Model outputs, when `--save-responses` is set |

## Project Layout

```
llm_context_benchmarks/
├── benchmark.py              # Unified CLI dispatcher
├── benchmark_common.py       # Shared utilities (hardware, charts, CSV, …)
├── compare_benchmarks.py     # Multi-run comparison
├── generate_context_files.py # Token-precise context file generation
├── kl_capture.py             # Logprob capture + KL divergence (MLX, llama.cpp)
├── <engine>_benchmark.py     # One file per engine (mlx, llamacpp, ollama_*, …)
├── pyproject.toml            # uv-managed dependencies
└── output/                   # Timestamped result directories
```

## Requirements

- Python 3.13+
- `uv` for dependency management
- Engine-specific runtime (see Installation table)

## Contributing

```bash
pre-commit install
# make changes
pre-commit run --all-files
```

Benchmark-result PRs are welcome — they help build a cross-hardware picture.
The `output/` folder is gitignored, so either rename your folder to include
your hardware (e.g. `benchmark_m3_ultra_512gb_mlx_qwen3_4bit`) or whitelist
it in `.gitignore` before committing.
