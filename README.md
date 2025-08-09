# LLM Context Benchmarks

A comprehensive benchmarking tool for testing Large Language Models (LLMs) with different context sizes across multiple inference engines including Ollama, MLX, llama.cpp, and LM Studio (beta).

## Features

- 📊 **Multiple Benchmark Engines**: Test models using Ollama (API & CLI), MLX, llama.cpp, and LM Studio
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
uv pip install -r requirements.txt
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
- Install MLX-LM:
  ```bash
  pip install mlx-lm
  ```
- Models will be downloaded automatically from Hugging Face when running benchmarks

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
python generate_context_files.py pride_and_prejudice.txt

# 2. Run benchmark with unified interface
python benchmark.py ollama-api gpt-oss:20b

# 3. View available engines
python benchmark.py --list-engines
```

## Usage

### Step 1: Generate Test Data

Use `generate_context_files.py` to create test files from any source text (e.g., Pride and Prejudice or any other text file).

#### Basic Usage

```bash
# Generate context files from Pride and Prejudice
python generate_context_files.py pride_and_prejudice.txt
```

This will create files in the root directory:

- `2k.txt` (2,000 tokens)
- `4k.txt` (4,000 tokens)
- `8k.txt` (8,000 tokens)
- `16k.txt` (16,000 tokens)
- `32k.txt` (32,000 tokens)
- `64k.txt` (64,000 tokens)
- `128k.txt` (128,000 tokens)

#### Custom Sizes

```bash
# Generate specific context sizes (in thousands of tokens)
python generate_context_files.py source.txt --sizes 2,4,8,16,32,64,128
```

#### Options

- `--sizes`: Comma-separated list of sizes in thousands of tokens (default: 2,4,8,16,32,64,128)
- `--encoding`: Tiktoken encoding to use (default: cl100k_base for GPT-3.5/GPT-4)
- `--output-dir`: Directory to save context files (default: current directory)
- `--prompt-suffix`: Custom prompt to append to each file (default: "Please provide a summary of the above text.")

### Step 2: Run Benchmarks

Run benchmarks with different engines and models to collect performance data.

### Step 3: Compare Results (Optional)

After running multiple benchmarks, use the comparison tool to analyze performance differences:

```bash
# Compare all benchmark results in output directory
python compare_benchmarks.py

# Compare specific benchmark folders
python compare_benchmarks.py output/benchmark_ollama_* output/benchmark_mlx_*

# Save comparison to custom location
python compare_benchmarks.py --output my_comparison
```

The comparison tool generates:
- **comparison_chart.png**: Side-by-side performance charts
- **comparison_results.csv**: Aggregated metrics in CSV format  
- **comparison_table.txt**: Formatted comparison table

### Step 2 (Alternative): Run Individual Benchmarks

#### Unified Benchmark Interface (Recommended)

The simplest way to run benchmarks is using the unified `benchmark.py` script:

```bash
# List available engines
python benchmark.py --list-engines

# Run Ollama API benchmark
python benchmark.py ollama-api gpt-oss:20b

# Run Ollama CLI benchmark
python benchmark.py ollama-cli llama3.2

# Run MLX benchmark (Apple Silicon only)
python benchmark.py mlx mlx-community/Qwen3-4B-Instruct-2507-4bit

# Run llama.cpp benchmark (defaults to localhost:8080)
python benchmark.py llamacpp gpt-oss:20b

# Run llama.cpp with custom port (e.g., 9000)
python benchmark.py llamacpp gpt-oss:20b --port 9000

# Run llama.cpp with custom host and port
python benchmark.py llamacpp gpt-oss:20b --host 192.168.1.100 --port 9000

# Run LM Studio benchmark (Beta - requires LM Studio server)
python benchmark.py lmstudio local-model

# Custom options
python benchmark.py ollama-api gpt-oss:20b --contexts 2,4,8,16,32 --max-tokens 500 --save-responses

# Increase timeout for large context benchmarks
python benchmark.py mlx mlx-community/Qwen3-4B-Instruct-2507-4bit --contexts 64,128 --timeout 7200
```

##### Common Options

- `--contexts`: Context sizes to test (default: 2,4,8,16)
- `--max-tokens`: Maximum tokens to generate (default: 200)
- `--timeout`: Timeout in seconds for each benchmark (default: 3600 = 60 minutes)
- `--save-responses`: Save model responses to files
- `--output-csv`: Output CSV filename
- `--output-chart`: Output chart filename

##### Engine-Specific Options

- `--kv-bit`: KV cache bit size for MLX (e.g., 4 or 8)
- `--host`: Host for llama.cpp server (default: localhost)
- `--port`: Port for llama.cpp server (default: 8080)

#### Direct Script Usage (Alternative)

You can also run the benchmark scripts directly:

##### Ollama API Benchmark

```bash
python ollama_api_benchmark.py gpt-oss:20b
python ollama_api_benchmark.py gpt-oss:20b --contexts 2,4,8,16 --max-tokens 200
```

##### Ollama CLI Benchmark

```bash
python ollama_cli_benchmark.py gpt-oss:20b
python ollama_cli_benchmark.py gpt-oss:20b --contexts 2,4,8,16,32
```

##### MLX Benchmark (Apple Silicon Only)

```bash
python mlx_benchmark.py mlx-community/Qwen3-4B-Instruct-2507-4bit
python mlx_benchmark.py mlx-community/Qwen3-4B-Instruct-2507-4bit --kv-bit 8
python mlx_benchmark.py mlx-community/Qwen3-4B-Instruct-2507-4bit --timeout 7200  # 2 hours for large contexts
```

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

## Example Workflows

### Multi-Engine Comparison Workflow

```bash
# 1. Install dependencies
uv pip install -r requirements.txt

# 2. Generate context files (only needed once)
python generate_context_files.py pride_and_prejudice.txt --sizes 2,4,8,16

# 3. Run benchmarks on different engines
# Ollama (requires model pulled)
python benchmark.py ollama-api gpt-oss:20b --contexts 2,4,8,16

# MLX (Apple Silicon, auto-downloads model)
python benchmark.py mlx mlx-community/Qwen2.5-3B-Instruct-4bit --contexts 2,4,8,16

# llama.cpp (requires server running on port 9000)
python benchmark.py llamacpp gpt-oss:20b --port 9000 --contexts 2,4,8,16

# 4. Compare all results
python compare_benchmarks.py

# View results
ls comparison_results/
```

This creates a comprehensive comparison showing:
- **Prompt processing speed** across engines
- **Text generation speed** by context size  
- **Total processing time** comparison
- **Efficiency metrics** (tokens/sec per 1K context)

### Single Engine Workflows

### Quick Benchmark with Unified Interface

```bash
# 1. Install dependencies
uv pip install -r requirements.txt

# 2. Download a source text
curl https://www.gutenberg.org/files/1342/1342-0.txt -o pride_and_prejudice.txt

# 3. Generate context files (only needed once)
python generate_context_files.py pride_and_prejudice.txt --sizes 2,4,8,16

# 4. List available engines
python benchmark.py --list-engines

# 5. Run benchmarks using unified interface
# Ollama API (requires: ollama pull gpt-oss:20b)
python benchmark.py ollama-api gpt-oss:20b

# Ollama CLI
python benchmark.py ollama-cli llama3.2

# MLX (Apple Silicon only, auto-downloads model)
python benchmark.py mlx mlx-community/Qwen2.5-3B-Instruct-4bit

# llama.cpp (defaults to localhost:8080)
python benchmark.py llamacpp gpt-oss:20b

# llama.cpp with custom port
python benchmark.py llamacpp gpt-oss:20b --port 9000

# LM Studio (Beta - requires LM Studio server)
python benchmark.py lmstudio local-model

# 6. View results
ls benchmark_*/
```

### Detailed Ollama Workflow

```bash
# 1. Install dependencies
uv pip install -r requirements.txt

# 2. Download source text
curl https://www.gutenberg.org/files/1342/1342-0.txt -o pride_and_prejudice.txt

# 3. Generate context files
python generate_context_files.py pride_and_prejudice.txt

# 4. Pull Ollama model
ollama pull gpt-oss:20b

# 5. Run benchmark with unified interface
python benchmark.py ollama-api gpt-oss:20b

# Or run directly
python ollama_api_benchmark.py gpt-oss:20b --contexts 2,4,8,16 --max-tokens 200

# 6. View results
ls benchmark_ollama_*/
cat benchmark_ollama_*/hardware_info.json
```

### MLX Workflow (Apple Silicon)

```bash
# 1. Install dependencies
uv pip install -r requirements.txt
pip install mlx-lm

# 2. Generate test data
python generate_context_files.py pride_and_prejudice.txt

# 3. Run MLX benchmark (model downloads automatically)
python benchmark.py mlx mlx-community/Qwen2.5-3B-Instruct-4bit

# With KV cache quantization
python benchmark.py mlx mlx-community/Qwen2.5-3B-Instruct-4bit --kv-bit 4

# 4. View results
ls benchmark_mlx_*/
cat benchmark_mlx_*/hardware_info.json
```

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
├── mlx_benchmark.py             # MLX framework benchmarking
├── llamacpp_benchmark.py        # llama.cpp server benchmarking
├── lmstudio_benchmark.py        # LM Studio benchmarking (Beta)
├── compare_benchmarks.py        # Multi-benchmark comparison tool
├── generate_context_files.py    # Context file generation
├── ollama_benchmark_notebook.ipynb  # Interactive notebook
├── requirements.txt             # Python dependencies
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

- Python 3.7+
- Sufficient RAM for the model and context sizes you want to test
- psutil (for hardware detection)
- matplotlib, numpy (for charts)
- tiktoken (for token counting)

### Framework-specific:
- **Ollama**: Ollama installed and running
- **MLX**: Apple Silicon Mac (M1/M2/M3/M4), mlx-lm package
- **llama.cpp**: llama.cpp server running
- **LM Studio** (Beta): LM Studio installed with server running

## Notes

- Larger context sizes require more memory
- Performance varies significantly between models and hardware
- The tool automatically handles models that support different maximum context lengths
- Hardware information is automatically collected on macOS (Apple Silicon) and Linux systems
- Generated text files include both the model's thinking process (if shown) and final response (Ollama only)
- All outputs are organized in timestamped directories for easy comparison
- MLX models are optimized for Apple Silicon and provide excellent performance with low memory usage
- MLX supports quantized models (4-bit, 8-bit) for efficient inference
- llama.cpp integration requires a running server instance with your model loaded
- LM Studio support is currently in beta - ensure your server is running before benchmarking