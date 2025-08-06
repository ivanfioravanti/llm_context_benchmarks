# LLM Context Benchmarks

A comprehensive benchmarking tool for testing Large Language Models (LLMs) with different context sizes using Ollama's API and CLI interfaces.

## Features

- 📊 **Dual Benchmark Modes**: Test models using both Ollama API and CLI
- 🔧 **Automatic Hardware Detection**: Captures system specs including:
  - CPU cores (with performance/efficiency breakdown on Apple Silicon)
  - GPU cores (Apple Silicon)
  - System memory
- 📈 **Visual Performance Charts**: Generate detailed performance graphs with hardware info
- 💾 **Context File Generation**: Create test files with precise token counts
- 🖥️ **Apple Silicon Optimized**: Full support for M1/M2/M3 chips
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

2. Install Ollama from https://ollama.com

3. (Optional) Set up pre-commit hooks for code quality:

```bash
# Install pre-commit hooks (only runs Black and isort for Python formatting)
pre-commit install

# Run hooks manually on all files
pre-commit run --all-files
```

4. Pull the model you want to test:

```bash
ollama pull gpt-oss:20b
# or
ollama pull llama3.2
# or any other Ollama model
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

You can benchmark Ollama models using either the API or CLI approach:

#### Option A: API Benchmark (ollama_api_benchmark.py)

Uses the Ollama Python API for benchmarking. This provides programmatic access to the model.

```bash
# Test with all .txt files in the current directory
python ollama_api_benchmark.py gpt-oss:20b

# Test only specific context sizes
python ollama_api_benchmark.py gpt-oss:20b --contexts 2,4,8,16
```

#### Option B: CLI Benchmark (ollama_cli_benchmark.py)

Uses the `ollama run` command with `--verbose` flag to get detailed metrics. This approach directly calls the Ollama CLI and parses the verbose output.

```bash
# Test with all .txt files in the current directory
python ollama_cli_benchmark.py gpt-oss:20b

# Test only specific context sizes
python ollama_cli_benchmark.py gpt-oss:20b --contexts 2,4,8,16,32
```

The CLI version provides the same metrics as shown in the verbose output:

- `prompt eval count/duration/rate`: Prompt processing metrics
- `eval count/duration/rate`: Generation metrics
- `total duration`: Total processing time

#### Common Options (both scripts)

- `--contexts`: Comma-separated list of context sizes to test (e.g., 2,4,8,16)
- `--max-tokens`: Maximum tokens to generate per test (default: 16000)
- `--output-csv`: Output CSV filename (default: benchmark_results.csv)
- `--output-chart`: Output chart filename (default: benchmark_chart.png)

#### Output Files

Both benchmark scripts create a timestamped directory containing:

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

3. **benchmark_chart.png** - Visual charts showing:
   - Hardware specs in the title
   - Prompt processing speed (tokens/sec)
   - Generation speed (tokens/sec)
   - Total processing time and tokens generated

4. **generated_*.txt** - Complete model responses including:
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

## Example Workflow

```bash
# 1. Install dependencies
uv pip install -r requirements.txt

# 2. Download a source text (e.g., Pride and Prejudice from Project Gutenberg)
curl https://www.gutenberg.org/files/1342/1342-0.txt -o pride_and_prejudice.txt

# 3. Generate context files
python generate_context_files.py pride_and_prejudice.txt

# 4. Pull an Ollama model
ollama pull gpt-oss:20b

# 5. Run the benchmark (choose one)
# Using API:
python ollama_api_benchmark.py gpt-oss:20b
# Or using CLI:
python ollama_cli_benchmark.py gpt-oss:20b

# 6. View results in the generated directory
ls benchmark_ollama_*/
cat benchmark_ollama_*/hardware_info.json
```

## Hardware Detection

The tool automatically detects and reports:

### Apple Silicon (M1/M2/M3)
- Chip model and variant
- Total CPU cores with performance/efficiency breakdown
- GPU core count
- System memory

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
├── ollama_api_benchmark.py      # API-based benchmarking
├── ollama_cli_benchmark.py      # CLI-based benchmarking
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
- Ollama installed and running
- Sufficient RAM for the model and context sizes you want to test
- psutil (for hardware detection)
- matplotlib, numpy (for charts)
- tiktoken (for token counting)

## Notes

- Larger context sizes require more memory
- Performance varies significantly between models and hardware
- The tool automatically handles models that support different maximum context lengths
- Hardware information is automatically collected on macOS (Apple Silicon) and Linux systems
- Generated text files include both the model's thinking process (if shown) and final response
- All outputs are organized in timestamped directories for easy comparison