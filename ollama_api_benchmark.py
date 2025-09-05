#!/usr/bin/env python3
"""
Benchmark script for Ollama using the Python API.

This script benchmarks Ollama models using the official Python API for programmatic access.

Usage:
    python ollama_api_benchmark.py llama3.2
    python ollama_api_benchmark.py gpt-oss:20b --contexts 2,4,8,16 --max-tokens 500
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import ollama

import benchmark_common as common


def run_benchmark(
    model_name: str, context_file: Path, max_tokens: int = 200
) -> Optional[Dict]:
    """Run Ollama benchmark for a given context file.
    
    Args:
        model_name: Name of the Ollama model
        context_file: Path to the context file
        max_tokens: Maximum tokens to generate
        
    Returns:
        Dictionary with benchmark results or None if failed
    """
    print(f"Running benchmark for {context_file}...")

    # Read the prompt from file
    with open(context_file) as f:
        prompt = f.read()

    # Count prompt tokens (approximate)
    prompt_tokens = len(prompt.split())

    # Start timing
    start_time = time.time()

    try:
        # Run the model
        response = ollama.generate(
            model=model_name, prompt=prompt, options={"num_predict": max_tokens}
        )

        # Calculate total time
        total_time = time.time() - start_time

        # Parse the response
        generated_text = response.get("response", "")
        generation_tokens = response.get("eval_count", 0)
        eval_duration = response.get("eval_duration", 0) / 1e9  # Convert nanoseconds to seconds
        prompt_eval_duration = response.get("prompt_eval_duration", 0) / 1e9  # Convert to seconds
        prompt_eval_count = response.get("prompt_eval_count", prompt_tokens)

        # Calculate tokens per second
        prompt_tps = (
            prompt_eval_count / prompt_eval_duration if prompt_eval_duration > 0 else 0
        )
        generation_tps = generation_tokens / eval_duration if eval_duration > 0 else 0

        # Debug logging
        print(
            f"  Prompt: {prompt_eval_count} tokens in {prompt_eval_duration:.2f}s = {prompt_tps:.1f} t/s"
        )
        print(
            f"  Generation: {generation_tokens} tokens in {eval_duration:.2f}s = {generation_tps:.1f} t/s"
        )
        print(f"  Total time: {total_time:.2f}s")

        return {
            "context_size": Path(context_file).stem,
            "prompt_tokens": prompt_eval_count,
            "prompt_tps": prompt_tps,
            "generation_tokens": generation_tokens,
            "generation_tps": generation_tps,
            "total_time": total_time,
            "eval_duration": eval_duration,
            "prompt_eval_duration": prompt_eval_duration,
            "generated_text": generated_text,
        }

    except Exception as e:
        print(f"Error running benchmark: {e}")
        return None


def check_model_available(model_name: str) -> bool:
    """Check if the model is available in Ollama.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        True if model is available, False otherwise
    """
    try:
        # Try to list models and check if our model is there
        response = ollama.list()

        # The response is a dictionary with 'models' key
        models = response.get("models", [])

        # Extract model names - the model object has a 'model' attribute
        model_names = []
        for model_obj in models:
            # Access the model attribute directly if it's an object
            if hasattr(model_obj, "model"):
                name = model_obj.model
            elif isinstance(model_obj, dict):
                # If it's a dict, try to get the 'model' key
                name = model_obj.get("model", str(model_obj))
            else:
                # Fallback to string representation
                name = str(model_obj)
            model_names.append(name)

        # Check exact match (including tags)
        if model_name in model_names:
            print(f"Found model: {model_name}")
            return True

        # Also check without considering digest (for models like "gpt-oss:20b")
        for available_model in model_names:
            if (
                available_model == model_name
                or available_model.split(":")[0] == model_name.split(":")[0]
            ):
                print(f"Found model: {available_model}")
                return True

        print(f"Model '{model_name}' not found. Available models:")
        for model in model_names:
            print(f"  - {model}")
        return False

    except Exception as e:
        print(f"Error checking model availability: {e}")
        print(f"Attempting to use model anyway...")
        return True  # Try to proceed anyway


def main() -> int:
    """Main function to run Ollama API benchmarks."""
    parser = argparse.ArgumentParser(
        description="Run Ollama benchmarks using Python API"
    )
    parser.add_argument(
        "model", help="Ollama model name (e.g., llama3.2, mistral)"
    )
    
    # Add common arguments
    common.setup_common_args(parser)
    
    args = parser.parse_args()

    # Check if model is available
    if not check_model_available(args.model):
        print(f"\nPlease pull the model first with: ollama pull {args.model}")
        return 1

    # Create output directory using common function
    output_dir = common.create_output_directory("ollama_api", args.model)

    # Find context files using common module
    context_files = common.find_context_files(args.contexts)
    if not context_files:
        return 1

    # Get hardware information
    print("\nCollecting hardware information...")
    hardware_info = common.get_hardware_info()
    hardware_str = common.format_hardware_string(hardware_info)
    print(f"Hardware: {hardware_str}")
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")

    # Run benchmarks
    import time
    start_time = time.time()
    results = []
    for file in context_files:
        print(f"\n{'=' * 50}")
        print(f"Benchmarking {file.name}...")
        print(f"{'=' * 50}")
        
        result = run_benchmark(args.model, file, args.max_tokens)
        if result:
            results.append(result)

            # Save the generated text if requested
            if args.save_responses:
                output_filename = output_dir / f"response_{result['context_size']}.txt"
                common.save_generated_text(
                    result, args.model, output_filename, "Ollama API"
                )
    
    total_benchmark_time = time.time() - start_time

    if not results:
        print("\nNo successful benchmark results")
        return 1

    # Save all outputs using common function
    common.save_all_outputs(
        results, output_dir, args.model, "Ollama API", hardware_info, args
    )

    # Print summary using common function
    common.print_benchmark_summary(
        results, args.model, "Ollama API", hardware_info, output_dir, total_benchmark_time
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())