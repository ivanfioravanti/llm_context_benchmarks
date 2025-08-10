#!/usr/bin/env python3
"""
[EXPERIMENTAL] Benchmark script for LM Studio local server using native API.

This script connects to a running LM Studio server and benchmarks its performance
across different context sizes using LM Studio's native API.

NOTE: This is experimental. LM Studio's API does not return accurate prompt processing
timing, only the number of prompt tokens processed.

Requirements:
    - LM Studio with native API enabled (beta feature)
    - Server running on http://localhost:1234 or custom URL

Usage:
    # Start LM Studio server first with native API enabled
    # Then run benchmark:
    python lmstudio_benchmark.py local-model
    python lmstudio_benchmark.py http://localhost:1234
"""

import argparse
import json
import sys
import time
from datetime import datetime
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
    """Test if the LM Studio server is running and accessible."""
    try:
        # LM Studio uses OpenAI-compatible API
        response = requests.get(f"{server_url}/v1/models", timeout=5)
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server: {e}")
    return False


def get_server_info(server_url):
    """Get model information from LM Studio server."""
    try:
        # Get available models
        response = requests.get(f"{server_url}/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()

            # DEBUG: Print full models API response
            print("\n" + "=" * 60)
            print("DEBUG: Full Models API Response")
            print("=" * 60)
            print(json.dumps(data, indent=2))
            print("=" * 60 + "\n")

            models = data.get("data", [])
            if models:
                # Return the first model's info
                return {"model": models[0].get("id", "unknown")}
    except:
        pass

    return {"model": "lmstudio-model"}


def benchmark_lmstudio(server_url, context_file, max_tokens=200, model_name=None, timeout=300):
    """Benchmark LM Studio server with a given context file using native API.

    Args:
        server_url: URL of the LM Studio server
        context_file: Path to the context file
        max_tokens: Maximum number of tokens to generate
        model_name: Optional model name to use
        timeout: Request timeout in seconds

    Returns:
        Dictionary with benchmark results
    """
    # Read the context file
    with open(context_file, "r") as f:
        prompt = f.read()

    # Prepare the request payload for LM Studio native API
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": False,
    }

    # Add model if specified
    if model_name:
        payload["model"] = model_name

    # Record start time
    start_time = time.time()

    # Use LM Studio native API for accurate metrics
    try:
        response = requests.post(
            f"{server_url}/api/v0/chat/completions", json=payload, timeout=timeout
        )
        response.raise_for_status()
        result = response.json()

        # DEBUG: Print full API response
        print("\n" + "=" * 60)
        print("DEBUG: Full API Response from LM Studio")
        print("=" * 60)
        print(json.dumps(result, indent=2))
        print("=" * 60 + "\n")

    except requests.exceptions.RequestException as e:
        print(f"Error during benchmark: {e}")
        print(f"Note: This benchmark requires LM Studio's native API (beta)")
        print(f"Ensure LM Studio is running and the native API is enabled")
        return None

    # Calculate timings
    total_time = time.time() - start_time

    # Extract metrics from response
    choices = result.get("choices", [])
    generated_text = choices[0]["message"]["content"] if choices else ""

    # Get token usage information
    usage = result.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    generation_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + generation_tokens)

    # Get accurate timing from LM Studio native API
    if "stats" not in result:
        print(f"Warning: No stats found in response. Ensure you're using LM Studio's native API.")
        return None

    stats = result["stats"]
    # Use actual metrics from LM Studio
    time_to_first_token = stats.get("time_to_first_token", 0)
    generation_time = stats.get("generation_time", 0)
    tokens_per_second = stats.get("tokens_per_second", 0)

    # Calculate metrics
    # NOTE: LM Studio API doesn't provide accurate prompt processing speed
    # We only get the time to first token, not the actual prompt processing time
    prompt_time = time_to_first_token
    predict_time = generation_time - time_to_first_token if generation_time > time_to_first_token else generation_time

    # Calculate tokens per second
    # For prompt, we can't calculate accurate TPS since we don't have proper timing
    # We'll set it to 0 to indicate it's not available
    prompt_tps = 0  # Not available from LM Studio API
    generation_tps = (
        tokens_per_second if tokens_per_second > 0 else (generation_tokens / predict_time if predict_time > 0 else 0)
    )

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
    parser = argparse.ArgumentParser(description="Benchmark LM Studio server across different context sizes")
    parser.add_argument(
        "server_or_model",
        help="LM Studio server URL (e.g., http://localhost:1234) or model name (e.g., local-model)",
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
        default=300,
        help="Timeout in seconds for each benchmark (default: 300)",
    )

    args = parser.parse_args()

    # Determine if input is URL or model name
    if args.server_or_model.startswith("http://") or args.server_or_model.startswith("https://"):
        server_url = args.server_or_model.rstrip("/")
        model_name = None
    else:
        # Default LM Studio server URL
        server_url = "http://localhost:1234"
        model_name = args.server_or_model

    # Test server connection
    print(f"Testing connection to LM Studio server at {server_url}...")
    if not test_server_connection(server_url):
        print(f"Error: Cannot connect to LM Studio server at {server_url}")
        print("Please ensure LM Studio is running with a model loaded.")
        print("LM Studio typically runs on http://localhost:1234")
        return 1

    print("Successfully connected to LM Studio server")

    # Get server info
    server_info = get_server_info(server_url)
    if not model_name:
        model_name = server_info.get("model", "lmstudio-model")

    # Get hardware info
    hardware_info = get_hardware_info()
    hardware_str = format_hardware_string(hardware_info)

    print(f"\n[EXPERIMENTAL] LM Studio Benchmark")
    print(f"Note: Prompt processing speed is not available from LM Studio API")
    print(f"\nHardware: {hardware_str}")
    print(f"Model: {model_name}")
    print(f"Max tokens: {args.max_tokens}")

    # Find context files
    context_files = find_context_files(args.contexts)
    if not context_files:
        return 1

    # Create output directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path("output")
    base_output_dir.mkdir(exist_ok=True)
    # Sanitize model name for directory
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    output_dir = base_output_dir / f"benchmark_lmstudio_{safe_model_name}_{timestamp}"
    output_dir.mkdir(exist_ok=True)

    # Save hardware info
    save_hardware_info(hardware_info, output_dir / "hardware_info.json")

    results = []

    # Run benchmarks
    for context_file in context_files:
        print(f"\n{'=' * 50}")
        print(f"Benchmarking {context_file.name}...")
        print(f"{'=' * 50}")

        result = benchmark_lmstudio(server_url, context_file, args.max_tokens, model_name, args.timeout)

        if result:
            results.append(result)

            # Print results
            print(f"\nResults for {context_file.name}:")
            print(f"  Prompt tokens processed: {result['prompt_tokens']}")
            print(f"  Generated tokens: {result['generation_tokens']}")
            print(f"  Time to first token: {result['prompt_time']:.2f}s")
            print(f"  Generation time: {result['eval_duration']:.2f}s")
            print(f"  Total time: {result['total_time']:.2f}s")
            print(f"  Prompt TPS: N/A (not provided by LM Studio API)")
            print(f"  Generation TPS: {result['generation_tps']:.1f} tokens/sec")

            # Save response if requested
            if args.save_responses:
                response_file = output_dir / f"response_{context_file.stem}.txt"
                save_generated_text(result, model_name, response_file, framework="LM Studio [EXPERIMENTAL]")
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
    create_chart_ollama(results, model_name, hardware_info, chart_path, framework="LM Studio [EXPERIMENTAL]")
    print(f"\nChart saved to {chart_path}")

    # Generate summary table
    print("\n" + "=" * 50)
    print("SUMMARY TABLE")
    print("=" * 50)
    table = generate_table(results, model_name, "LM Studio [EXPERIMENTAL]", hardware_info)
    print(table)

    # Generate X post text
    print("\n" + "=" * 50)
    print("X POST TEXT")
    print("=" * 50)
    xpost = generate_xpost_text(results, model_name, "LM Studio [EXPERIMENTAL]", hardware_info)
    print(xpost)

    print(f"\n✅ All outputs saved to: {output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
