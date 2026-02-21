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
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import requests

from benchmark_common import (
    create_output_directory,
    find_context_files,
    format_hardware_string,
    get_hardware_info,
    print_benchmark_summary,
    save_all_outputs,
    save_generated_text,
    setup_common_args,
)


def test_server_connection(server_url: str) -> bool:
    """Test if the llama.cpp server is running and accessible."""
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server: {e}")
    return False


def get_server_info(server_url: str) -> Dict:
    """Get model and server information from llama.cpp server."""
    try:
        # Try to get model metadata
        response = requests.get(f"{server_url}/props", timeout=5)
        if response.status_code == 200:
            return response.json()
    except (requests.exceptions.RequestException, Exception):
        pass

    # Return empty dict if no info available
    return {}


def benchmark_llamacpp(
    server_url: str, context_file: Path, max_tokens: int = 200, timeout: int = 3600
) -> Optional[Dict]:
    """Benchmark llama.cpp server with a given context file.

    Args:
        server_url: URL of the llama.cpp server
        context_file: Path to the context file
        max_tokens: Maximum number of tokens to generate
        timeout: Request timeout in seconds

    Returns:
        Dictionary with benchmark results or None if failed
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
        "stream": True,
        "cache_prompt": False,
    }

    # Record start time
    start_time = time.time()
    first_token_time = None
    generated_text = ""
    result = {}

    # Make the request to the server
    try:
        import json
        response = requests.post(
            f"{server_url}/completion", json=payload, timeout=timeout, stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    try:
                        chunk = json.loads(data_str)
                        # Mark first token time when we receive content
                        if first_token_time is None and chunk.get("content"):
                            first_token_time = time.time()
                            
                        generated_text += chunk.get("content", "")
                        
                        # Stop chunk usually contains timings
                        if chunk.get("stop"):
                            result = chunk
                            break
                    except json.JSONDecodeError:
                        pass
    except requests.exceptions.RequestException as e:
        print(f"Error during benchmark: {e}")
        return None

    # Calculate timings
    total_time = time.time() - start_time
    time_to_first_token = (first_token_time - start_time) if first_token_time else 0.0

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
        "prompt_eval_duration": prompt_time,
        "time_to_first_token": time_to_first_token,
        "eval_duration": predict_time,
        "total_time": total_time,
        "prompt_tps": prompt_tps,
        "generation_tps": generation_tps,
        "generated_text": generated_text,
        "wall_time": total_time,
    }


def main() -> int:
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark llama.cpp server across different context sizes"
    )
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
    
    # Add common arguments
    setup_common_args(parser)
    
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
    model_name = args.model or server_info.get("default_generation_settings", {}).get(
        "model", "llama.cpp model"
    )

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

    # Create output directory using common function
    output_dir = create_output_directory("llamacpp", args.model)

    results = []

    # Run benchmarks
    import time
    start_time = time.time()
    for context_file in context_files:
        print(f"\n{'=' * 50}")
        print(f"Benchmarking {context_file.name}...")
        print(f"{'=' * 50}")

        result = benchmark_llamacpp(
            server_url, context_file, args.max_tokens, args.timeout
        )

        if result:
            results.append(result)

            # Print results
            print(f"\nResults for {context_file.name}:")
            print(f"  Prompt tokens: {result['prompt_tokens']}")
            print(f"  Generated tokens: {result['generation_tokens']}")
            print(f"  Time to first token: {result['time_to_first_token']:.2f}s")
            print(f"  Prompt time: {result['prompt_time']:.2f}s")
            print(f"  Generation time: {result['eval_duration']:.2f}s")
            print(f"  Total time: {result['total_time']:.2f}s")
            print(f"  Prompt TPS: {result['prompt_tps']:.1f} tokens/sec")
            print(f"  Generation TPS: {result['generation_tps']:.1f} tokens/sec")

            # Save response if requested
            if args.save_responses:
                response_file = output_dir / f"response_{context_file.stem}.txt"
                save_generated_text(
                    result, model_name, response_file, framework="llama.cpp"
                )
        else:
            print(f"Failed to benchmark {context_file.name}")
    
    total_benchmark_time = time.time() - start_time

    if not results:
        print("\nNo successful benchmark results")
        return 1

    # Save all outputs using common function
    save_all_outputs(
        results, output_dir, model_name, "llama.cpp", hardware_info, args
    )

    # Print summary using common function
    print_benchmark_summary(results, model_name, "llama.cpp", hardware_info, output_dir, total_benchmark_time)

    return 0


if __name__ == "__main__":
    sys.exit(main())