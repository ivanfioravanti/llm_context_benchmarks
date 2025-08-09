#!/usr/bin/env python3
"""
Benchmark script for llama.cpp server using HTTP API.

This script connects to a running llama.cpp server and benchmarks its performance
across different context sizes.

Usage:
    # Start llama.cpp server first:
    # ./llama-server -m model.gguf --host 0.0.0.0 --port 8080
    
    # Then run benchmark (default localhost:8080):
    python llamacpp_benchmark.py gpt-oss:20b
    
    # Or specify custom host and port:
    python llamacpp_benchmark.py gpt-oss:20b --host localhost --port 9000
"""

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from benchmark_common import (
    create_chart_ollama,
    find_context_files,
    format_hardware_string,
    generate_table,
    generate_xpost_text,
    get_hardware_info,
    save_generated_text,
    save_hardware_info,
    save_results_csv,
)


def test_server_connection(server_url):
    """Test if the llama.cpp server is running and accessible."""
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server: {e}")
    return False


def get_server_info(server_url):
    """Get model and server information from llama.cpp server."""
    try:
        # Try to get model metadata
        response = requests.get(f"{server_url}/props", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass

    # Return empty dict if no info available
    return {}


def benchmark_llamacpp(server_url, context_file, max_tokens=200):
    """Benchmark llama.cpp server with a given context file.

    Args:
        server_url: URL of the llama.cpp server
        context_file: Path to the context file
        max_tokens: Maximum number of tokens to generate

    Returns:
        Dictionary with benchmark results
    """
    # Read the context file
    with open(context_file, "r") as f:
        prompt = f.read()

    # Prepare the request payload
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.95,
        "stream": False,
    }

    # Record start time
    start_time = time.time()

    # Make the request to the server
    try:
        response = requests.post(
            f"{server_url}/completion", json=payload, timeout=300  # 5 minute timeout for large contexts
        )
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error during benchmark: {e}")
        return None

    # Calculate timings
    total_time = time.time() - start_time

    # Extract metrics from response
    generated_text = result.get("content", "")

    # llama.cpp server provides timing information
    timings = result.get("timings", {})

    # Get token counts
    prompt_tokens = timings.get("prompt_n", 0)
    generation_tokens = timings.get("predicted_n", 0)

    # Get processing times (in milliseconds from server)
    prompt_ms = timings.get("prompt_ms", 0)
    predict_ms = timings.get("predicted_ms", 0)

    # Convert to seconds
    prompt_time = prompt_ms / 1000.0
    predict_time = predict_ms / 1000.0

    # Calculate tokens per second
    prompt_tps = prompt_tokens / prompt_time if prompt_time > 0 else 0
    generation_tps = generation_tokens / predict_time if predict_time > 0 else 0

    return {
        "context_size": context_file.stem,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "prompt_time": prompt_time,
        "eval_duration": predict_time,
        "total_time": total_time,
        "prompt_tps": prompt_tps,
        "generation_tps": generation_tps,
        "generated_text": generated_text,
        "wall_time": total_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark llama.cpp server across different context sizes")
    parser.add_argument(
        "model",
        help="Model name or identifier (used for display purposes)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host of the llama.cpp server (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port of the llama.cpp server (default: 8080)",
    )
    parser.add_argument(
        "--contexts",
        help="Comma-separated list of context sizes to benchmark (e.g., 2,4,8,16)",
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

    args = parser.parse_args()

    # Construct server URL from host and port
    server_url = f"http://{args.host}:{args.port}"

    # Test server connection
    print(f"Testing connection to llama.cpp server at {server_url}...")
    if not test_server_connection(server_url):
        print(f"Error: Cannot connect to llama.cpp server at {server_url}")
        print("Please ensure the server is running:")
        print("  ./llama-server -m model.gguf --host 0.0.0.0 --port 8080")
        return 1

    print("Successfully connected to llama.cpp server")

    # Get server info
    server_info = get_server_info(server_url)
    # Use the model name from args, fallback to server info if available
    model_name = args.model or server_info.get("default_generation_settings", {}).get("model", "llama.cpp model")

    # Get hardware info
    hardware_info = get_hardware_info()
    hardware_str = format_hardware_string(hardware_info)

    print(f"\nHardware: {hardware_str}")
    print(f"Model: {model_name}")
    print(f"Max tokens: {args.max_tokens}")

    # Find context files
    context_files = find_context_files(args.contexts)
    if not context_files:
        return 1

    # Create output directory structure
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path("output")
    base_output_dir.mkdir(exist_ok=True)
    # Use model name in directory, sanitizing for filesystem
    model_safe = args.model.replace("/", "_").replace(":", "_")
    output_dir = base_output_dir / f"benchmark_llamacpp_{model_safe}_{timestamp}"
    output_dir.mkdir(exist_ok=True)

    # Save hardware info
    save_hardware_info(hardware_info, output_dir / "hardware_info.json")

    results = []

    # Run benchmarks
    for context_file in context_files:
        print(f"\n{'=' * 50}")
        print(f"Benchmarking {context_file.name}...")
        print(f"{'=' * 50}")

        result = benchmark_llamacpp(server_url, context_file, args.max_tokens)

        if result:
            results.append(result)

            # Print results
            print(f"\nResults for {context_file.name}:")
            print(f"  Prompt tokens: {result['prompt_tokens']}")
            print(f"  Generated tokens: {result['generation_tokens']}")
            print(f"  Prompt time: {result['prompt_time']:.2f}s")
            print(f"  Generation time: {result['eval_duration']:.2f}s")
            print(f"  Total time: {result['total_time']:.2f}s")
            print(f"  Prompt TPS: {result['prompt_tps']:.1f} tokens/sec")
            print(f"  Generation TPS: {result['generation_tps']:.1f} tokens/sec")

            # Save response if requested
            if args.save_responses:
                response_file = output_dir / f"response_{context_file.stem}.txt"
                save_generated_text(result, model_name, response_file, framework="llama.cpp")
        else:
            print(f"Failed to benchmark {context_file.name}")

    if not results:
        print("\nNo successful benchmark results")
        return 1

    # Calculate total generated tokens
    total_generated_tokens = sum(r.get("generation_tokens", 0) for r in results)
    print(f"\n📊 Total generated tokens across all tests: {total_generated_tokens}")

    # Save results to CSV
    csv_path = output_dir / args.output_csv
    save_results_csv(results, csv_path)

    # Generate chart
    chart_path = output_dir / args.output_chart
    create_chart_ollama(results, model_name, hardware_info, chart_path, framework="llama.cpp")
    print(f"\nChart saved to {chart_path}")

    # Generate summary table
    print("\n" + "=" * 50)
    print("SUMMARY TABLE")
    print("=" * 50)
    table = generate_table(results, model_name, "llama.cpp", hardware_info)
    print(table)

    # Generate X post text
    print("\n" + "=" * 50)
    print("X POST TEXT")
    print("=" * 50)
    xpost = generate_xpost_text(results, model_name, "llama.cpp", hardware_info)
    print(xpost)

    print(f"\n✅ All outputs saved to: {output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
