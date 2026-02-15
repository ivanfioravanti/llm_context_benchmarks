# LLM Context Benchmarks

A comprehensive benchmarking tool for testing Large Language Models (LLMs) with different context sizes across multiple inference engines including Ollama, MLX, MLX Distributed (beta), llama.cpp, LM Studio (beta), and Exo (OpenAI-compatible).

## Features

- 📊 **Multiple Benchmark Engines**: Test models using Ollama (API & CLI), MLX, MLX Distributed (beta), llama.cpp, LM Studio (beta), and Exo (OpenAI-compatible)
- 🔧 **Automatic Hardware Detection**: Captures system specs including:
  - CPU cores (with performance/efficiency breakdown on Apple Silicon)
  - GPU cores (Apple Silicon)
  - System memory
- 📈 **Visual Performance Charts**: Generate detailed performance graphs with hardware info
- 💾 **Context File Generation**: Create test files with precise token counts
- 🖥️ **Apple Silicon Optimized**: Full support for M1/M2/M3/M4 chips with MLX
- 📝 **Jupyter Notebook Support**: Interactive benchmarking and analysis
- 🔄 **Pre-commit Hooks**: Automated code formatting with Black and isort
- 📄 **Complete Output Capture**: Saves model responses for analysis

## Installation

1. Install Python dependencies using uv:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

2. Install framework-specific requirements:

### For Ollama:
- Install Ollama from https://ollama.com
- Pull the model you want to test:
  ```bash
  ollama pull gpt-oss:20b
  # or
  ollama pull llama3.2
  # or any other Ollama model
  ```

### For MLX (Apple Silicon only):
- Requires Apple Silicon and the `mlx-lm` dependency (installed via `uv sync`).
- Models will be downloaded automatically from Hugging Face when running benchmarks.
- The model is loaded once and reused across all context sizes, with an automatic warmup pass before benchmarking begins.

### For llama.cpp:
- Run a llama.cpp server instance:
  ```bash
  # Example: Start llama.cpp server on port 8080
  ./llama-server -m model.gguf --port 8080
  ```

### For LM Studio (Beta):
- Install LM Studio from https://lmstudio.ai
- Start the local server from LM Studio UI
- Load your desired model in LM Studio

3. (Optional) Set up pre-commit hooks for code quality:

```bash
# Install pre-commit hooks (only runs Black and isort for Python formatting)
pre-commit install

# Run hooks manually on all files
pre-commit run --all-files
```

## Quick Start

```bash
# 1. Generate test data
uv run generate-context-files -- pride_and_prejudice.txt

# 2. Run benchmark with unified interface
uv run benchmark -- ollama-api gpt-oss:20b

# 3. View available engines
uv run benchmark -- --list-engines
```

## Usage

All scripts can still be run directly with `python <script>.py ...`, but the
recommended approach is `uv run <command> -- ...` using the script entry points
defined in `pyproject.toml`.

Common commands:

```bash
uv run benchmark -- --list-engines
uv run compare-benchmarks -- --output my_comparison
uv run generate-context-files -- pride_and_prejudice.txt --sizes 2,4,8,16
```

### Generate Context Files

Use `uv run generate-context-files -- <source> [options]` to create test files from any source text.

```bash
# Generate context files from Pride and Prejudice
uv run generate-context-files -- pride_and_prejudice.txt

# Generate specific context sizes (in thousands of tokens)
uv run generate-context-files -- source.txt --sizes 2,4,8,16,32,64,128
```

Options:

- `--sizes`: Comma-separated list of sizes in thousands of tokens (default: 2,4,8,16,32,64,128)
- `--encoding`: Tiktoken encoding to use (default: cl100k_base for GPT-3.5/GPT-4)
- `--output-dir`: Directory to save context files (default: current directory)
- `--prompt-suffix`: Custom prompt to append to each file (default: "Please provide a summary of the above text.")

### Run Benchmarks

```bash
# List available engines
uv run benchmark -- --list-engines

# Run Ollama API benchmark
uv run benchmark -- ollama-api gpt-oss:20b

# Run Ollama CLI benchmark
uv run benchmark -- ollama-cli llama3.2

# Run MLX benchmark (Apple Silicon only)
uv run benchmark -- mlx mlx-community/Qwen3-4B-Instruct-2507-4bit

# Run MLX distributed benchmark via mlx.launch (Beta - for example JACCL)
uv run benchmark -- mlx-distributed /Users/ifioravanti/MiniMax-M2.5-6bit \
  --hostfile /Users/ifioravanti/github/mlx-lm/m3-ultra-jaccl.json \
  --backend jaccl \
  --env MLX_METAL_FAST_SYNCH=1 \
  --env HF_HOME=/Users/Shared/.cache/huggingface

# Run llama.cpp benchmark (defaults to localhost:8080)
uv run benchmark -- llamacpp gpt-oss:20b

# Run llama.cpp with custom host and port
uv run benchmark -- llamacpp gpt-oss:20b --host 192.168.1.100 --port 9000

# Run LM Studio benchmark (Beta - requires LM Studio server)
uv run benchmark -- lmstudio local-model

# Run Exo benchmark (OpenAI-compatible endpoint on http://0.0.0.0:52415)
uv run benchmark -- exo local-model

# Custom options
uv run benchmark -- ollama-api gpt-oss:20b --contexts 0.5,1,2,4,8,16,32 --max-tokens 500 --save-responses

# Increase timeout for large context benchmarks
uv run benchmark -- mlx mlx-community/Qwen3-4B-Instruct-2507-4bit --contexts 64,128 --timeout 7200
```

Common options:

- `--contexts`: Context sizes to test (default: 0.5,1,2,4,8,16,32)
- `--max-tokens`: Maximum tokens to generate (default: 200)
- `--timeout`: Timeout in seconds for each benchmark (default: 3600 = 60 minutes)
- `--save-responses`: Save model responses to files
- `--output-csv`: Output CSV filename
- `--output-chart`: Output chart filename

Engine-specific options:

- `--kv-bit`: KV cache bit size for MLX (e.g., 4 or 8)
- `--host`: Host for llama.cpp server (default: localhost)
- `--port`: Port for llama.cpp server (default: 8080)
- `--backend`: Distributed backend for `mlx-distributed` (default: `jaccl`)
- `--hostfile`: Required hostfile JSON for `mlx-distributed`
- `--env`: Repeatable `KEY=VALUE` for `mlx.launch` environment variables in `mlx-distributed`
- `--sharded-script`: Path to `mlx_lm/examples/sharded_generate.py` for `mlx-distributed`
- `--pipeline`: Enable pipeline parallelism for `mlx-distributed`
- `--max-kv-size`: KV cache size in tokens for `mlx`
- `--base-url`: Override OpenAI-compatible endpoint (`exo` only)

### Compare Results (Optional)

After running multiple benchmarks, use the comparison tool to analyze performance differences:

```bash
# Compare all benchmark results in output directory
uv run compare-benchmarks

# Compare specific benchmark folders
uv run compare-benchmarks -- output/benchmark_ollama_* output/benchmark_mlx_*

# Save comparison to custom location
uv run compare-benchmarks -- --output my_comparison
```

The comparison tool generates:
- **comparison_chart.png**: Side-by-side performance charts
- **comparison_results.csv**: Aggregated metrics in CSV format  
- **comparison_table.txt**: Formatted comparison table

#### Output Files

All benchmark scripts create a timestamped directory containing:

1. **hardware_info.json** - Detailed system specifications:
   ```json
   {
     "chip": "Apple M3 Ultra",
     "total_cores": 32,
     "performance_cores": 24,
     "efficiency_cores": 8,
     "gpu_cores": 80,
     "memory_gb": 512
   }
   ```

2. **benchmark_results.csv** - Detailed metrics for each context size:
   - Prompt tokens and tokens per second
   - Generation tokens and tokens per second
   - Total processing time
   - Additional engine-specific timing columns may appear (e.g., `prompt_eval_duration`, `time_to_first_token`)
   - Peak memory usage (MLX only)

3. **benchmark_chart.png** - Visual charts showing:
   - Hardware specs in the title
   - Prompt processing speed (tokens/sec)
   - Generation speed (tokens/sec)
   - Total processing time and tokens generated (Ollama)
   - Peak memory usage and tokens generated (MLX)

4. **generated_*.txt** - Complete model responses including (Ollama only):
   - Model metadata
   - Token counts and timing
   - Full generated text (including thinking process for models that show it)

5. **table.txt** - Formatted table with hardware info and results:
   ```
   gpt-oss:20b Ollama CLI Benchmark Results
   Hardware: Apple M3 Ultra, 512GB RAM, 32 CPU cores (24P+8E), 80 GPU cores
   
   Context | Prompt TPS | Gen TPS | Total Time
   --------|------------|---------|------------
        2k |      521.4 |    52.0 |      14.9s
   ```

6. **tweet.txt** - Summary formatted for social media sharing

## Jupyter Notebook

The project includes `ollama_benchmark_notebook.ipynb` for interactive benchmarking:

- Generate context files interactively
- Run both CLI and API benchmarks
- Compare results side-by-side
- Create comparison charts
- Display hardware information

## Hardware Detection

The tool automatically detects and reports:

### Apple Silicon (M1/M2/M3/M4)
- Chip model and variant
- Total CPU cores with performance/efficiency breakdown
- GPU core count
- System memory
- Optimized for MLX framework performance

### Other Systems
- Processor information
- CPU core count
- System memory

This information is:
- Displayed in chart titles
- Saved to `hardware_info.json`
- Included in output tables
- Shown during benchmark execution

## Development

### Code Quality

This project uses pre-commit hooks for code formatting:

- **Black**: Python code formatting (120 char line length)
- **isort**: Import sorting

The hooks run automatically on commit if installed. To manually run:

```bash
# Check all files
pre-commit run --all-files

# Update hook versions
pre-commit autoupdate
```

### Project Structure

```
llm_context_benchmarks/
├── benchmark.py                 # Unified benchmark interface (main entry point)
├── benchmark_common.py          # Shared utilities and functions
├── ollama_api_benchmark.py      # Ollama API-based benchmarking
├── ollama_cli_benchmark.py      # Ollama CLI-based benchmarking
├── mlx_benchmark.py             # MLX single-node benchmarking (loads model once + warmup)
├── mlx_distributed_benchmark.py # MLX distributed benchmarking via mlx.launch (Beta)
├── llamacpp_benchmark.py        # llama.cpp server benchmarking
├── lmstudio_benchmark.py        # LM Studio benchmarking (Beta)
├── compare_benchmarks.py        # Multi-benchmark comparison tool
├── generate_context_files.py    # Context file generation
├── ollama_benchmark_notebook.ipynb  # Interactive notebook
├── pyproject.toml               # Python dependencies (uv)
├── uv.lock                      # Resolved dependency lockfile (uv)
├── .pre-commit-config.yaml     # Pre-commit configuration
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Install pre-commit hooks: `pre-commit install`
4. Make your changes
5. Run tests and benchmarks
6. Submit a pull request

### Contributing Benchmark Results

We welcome benchmark contributions from different hardware configurations! To share your benchmark results:

1. Run benchmarks on your hardware
2. The output folders (`benchmark_ollama_*`) are normally gitignored
3. To commit your results, either:
   - Comment out the relevant lines in `.gitignore`, or
   - Rename your folder to include your hardware (e.g., `benchmark_m2_max_64gb_ollama_cli_llama3.2`)
4. Create a PR with your benchmark results
5. Include hardware details in your PR description

This helps the community understand performance across different systems!

## Requirements

- Python 3.13+
- Sufficient RAM for the model and context sizes you want to test
- psutil (for hardware detection)
- matplotlib, numpy (for charts)
- tiktoken (for token counting)

### Framework-specific:
- **Ollama**: Ollama installed and running
- **MLX**: Apple Silicon Mac (M1/M2/M3/M4), mlx-lm package
- **MLX Distributed** (Beta): `mlx.launch` available and a valid hostfile JSON (for example with `--backend jaccl`)
- **llama.cpp**: llama.cpp server running
- **LM Studio** (Beta): LM Studio installed with server running

## Notes

- Larger context sizes require more memory
- Performance varies significantly between models and hardware
- The tool automatically handles models that support different maximum context lengths
- Hardware information is automatically collected on macOS (Apple Silicon) and Linux systems
- Generated text files include both the model's thinking process (if shown) and final response (Ollama only)
- All outputs are organized in timestamped directories for easy comparison
- MLX benchmark loads the model once and runs a warmup pass before benchmarking, using the `mlx_lm` Python API directly for efficient inference
- MLX supports quantized models (4-bit, 8-bit) for efficient inference on Apple Silicon
- MLX Distributed (beta) launches `mlx_lm/examples/sharded_generate.py` through `mlx.launch` on each benchmark run
- llama.cpp integration requires a running server instance with your model loaded
- LM Studio support is currently in beta - ensure your server is running before benchmarking
