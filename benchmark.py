#!/usr/bin/env python3
"""
Unified benchmark runner for multiple LLM engines.

Usage:
    python benchmark.py ollama-api gpt-oss:20b
    python benchmark.py ollama-cli llama3.2
    python benchmark.py mlx mlx-community/Qwen3-0.6B-4bit
    python benchmark.py llamacpp gpt-oss:20b  # defaults to localhost:8080
    python benchmark.py llamacpp gpt-oss:20b --port 9000  # custom port
    python benchmark.py lmstudio local-model
"""

import argparse
import subprocess
import sys
from pathlib import Path


def get_available_engines():
    """Return list of available benchmark engines."""
    engines = {
        "ollama-api": {
            "script": "ollama_api_benchmark.py",
            "description": "Ollama Python API",
            "example": "gpt-oss:20b",
        },
        "ollama-cli": {
            "script": "ollama_cli_benchmark.py",
            "description": "Ollama CLI with verbose output",
            "example": "llama3.2",
        },
        "mlx": {
            "script": "mlx_benchmark.py",
            "description": "MLX framework (Apple Silicon only)",
            "example": "mlx-community/Qwen3-0.6B-4bit",
        },
        "llamacpp": {
            "script": "llamacpp_benchmark.py",
            "description": "llama.cpp server via HTTP API",
            "example": "gpt-oss:20b --port 9000",
        },
        "lmstudio": {
            "script": "lmstudio_benchmark.py",
            "description": "LM Studio local server",
            "example": "local-model",
        },
    }
    return engines


def list_engines():
    """Print available engines and their descriptions."""
    engines = get_available_engines()
    print("\nAvailable benchmark engines:\n")
    print(f"{'Engine':<12} {'Status':<10} {'Description':<40} {'Example Model'}")
    print("-" * 90)

    for name, info in engines.items():
        status = info.get("status", "ready")
        status_symbol = "✓" if status == "ready" else "○"
        print(f"{name:<12} {status_symbol} {status:<8} {info['description']:<40} {info['example']}")

    print("\nUsage: python benchmark.py <engine> <model> [options]")
    print("\nExamples:")
    for name, info in engines.items():
        if info.get("status", "ready") == "ready":
            print(f"  python benchmark.py {name} {info['example']}")


def run_benchmark(engine, model, args):
    """Run the appropriate benchmark script based on engine selection."""
    engines = get_available_engines()

    if engine not in engines:
        print(f"Error: Unknown engine '{engine}'")
        list_engines()
        return 1

    engine_info = engines[engine]

    # Check if engine is implemented
    if engine_info.get("status") == "planned":
        print(f"Error: Engine '{engine}' is planned but not yet implemented.")
        print(
            f"Currently available engines: {', '.join([e for e, info in engines.items() if info.get('status', 'ready') == 'ready'])}"
        )
        return 1

    # Check if script exists
    script_path = Path(__file__).parent / engine_info["script"]
    if not script_path.exists():
        print(f"Error: Benchmark script '{engine_info['script']}' not found.")
        return 1

    # Build command
    cmd = [sys.executable, str(script_path), model] + args

    print(f"Running {engine} benchmark for model: {model}")
    print(f"Command: {' '.join(cmd)}\n")

    # Run the benchmark
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        return 130
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return 1


def main():
    engines = get_available_engines()
    available_engines = [name for name, info in engines.items() if info.get("status", "ready") == "ready"]

    parser = argparse.ArgumentParser(
        description="Unified LLM benchmark runner for multiple engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py ollama-api gpt-oss:20b
  python benchmark.py ollama-cli llama3.2 --contexts 2,4,8
  python benchmark.py mlx mlx-community/Qwen3-0.6B-4bit --kv-bit 4
  
  # List available engines
  python benchmark.py --list-engines
        """,
    )

    # Special argument to list engines
    parser.add_argument(
        "--list-engines",
        action="store_true",
        help="List available benchmark engines and exit",
    )

    # Positional arguments
    parser.add_argument(
        "engine",
        nargs="?",
        choices=available_engines,
        help=f"Benchmark engine to use: {', '.join(available_engines)}",
    )

    parser.add_argument(
        "model",
        nargs="?",
        help="Model identifier or path (format depends on engine)",
    )

    # Common options that will be passed through
    parser.add_argument(
        "--contexts",
        default="2,4,8,16",
        help="Comma-separated list of context sizes to benchmark (default: 2,4,8,16)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate (default: 200)",
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
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds for each benchmark (default: 3600 = 60 minutes)",
    )

    # Engine-specific options
    parser.add_argument(
        "--kv-bit",
        type=int,
        help="KV cache bit size for MLX (e.g., 4 or 8)",
    )

    parser.add_argument(
        "--host",
        default="localhost",
        help="Host for llama.cpp server (default: localhost)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for llama.cpp server (default: 8080)",
    )

    # Parse arguments
    args, unknown_args = parser.parse_known_args()

    # Handle --list-engines
    if args.list_engines:
        list_engines()
        return 0

    # Check required arguments
    if not args.engine:
        parser.print_help()
        print("\nError: Engine is required.")
        list_engines()
        return 1

    if not args.model:
        parser.print_help()
        print(f"\nError: Model is required for engine '{args.engine}'.")
        engine_info = engines.get(args.engine, {})
        if engine_info.get("example"):
            print(f"Example: python benchmark.py {args.engine} {engine_info['example']}")
        return 1

    # Build arguments to pass to the specific benchmark script
    pass_through_args = []

    # Add common arguments
    pass_through_args.extend(["--contexts", args.contexts])
    pass_through_args.extend(["--max-tokens", str(args.max_tokens)])
    pass_through_args.extend(["--timeout", str(args.timeout)])

    if args.save_responses:
        pass_through_args.append("--save-responses")

    pass_through_args.extend(["--output-csv", args.output_csv])
    pass_through_args.extend(["--output-chart", args.output_chart])

    # Add engine-specific arguments
    if args.engine == "mlx" and args.kv_bit is not None:
        pass_through_args.extend(["--kv-bit", str(args.kv_bit)])

    if args.engine == "llamacpp":
        pass_through_args.extend(["--host", args.host])
        pass_through_args.extend(["--port", str(args.port)])

    # Add any unknown arguments (for future compatibility)
    pass_through_args.extend(unknown_args)

    # Run the benchmark
    return run_benchmark(args.engine, args.model, pass_through_args)


if __name__ == "__main__":
    sys.exit(main())
