#!/usr/bin/env python3
"""
Benchmark script for Ollama using the CLI.

This script benchmarks Ollama models using the command-line interface for direct testing.

Usage:
    python ollama_cli_benchmark.py llama3.2
    python ollama_cli_benchmark.py gpt-oss:20b --contexts 2,4,8,16 --max-tokens 500
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import benchmark_common as common


def parse_ollama_output(output: str) -> Dict:
    """Parse the verbose output from ollama run command.
    
    Args:
        output: Raw output from ollama CLI
        
    Returns:
        Dictionary with parsed metrics
    """
    # Remove ANSI escape sequences and terminal control codes
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    output = ansi_escape.sub("", output)

    metrics = {}

    # Parse total duration
    total_match = re.search(r"total duration:\s+([\d.]+)([a-z]+)", output)
    if total_match:
        value = float(total_match.group(1))
        unit = total_match.group(2)
        if unit == "ms":
            metrics["total_duration"] = value / 1000
        elif unit == "s":
            metrics["total_duration"] = value
        else:
            metrics["total_duration"] = value

    # Parse load duration
    load_match = re.search(r"load duration:\s+([\d.]+)([a-z]+)", output)
    if load_match:
        value = float(load_match.group(1))
        unit = load_match.group(2)
        if unit == "ms":
            metrics["load_duration"] = value / 1000
        elif unit == "s":
            metrics["load_duration"] = value
        else:
            metrics["load_duration"] = value

    # Parse prompt eval count
    prompt_count_match = re.search(r"prompt eval count:\s+(\d+)", output)
    if prompt_count_match:
        metrics["prompt_eval_count"] = int(prompt_count_match.group(1))

    # Parse prompt eval duration
    prompt_dur_match = re.search(r"prompt eval duration:\s+([\d.]+)([a-z]+)", output)
    if prompt_dur_match:
        value = float(prompt_dur_match.group(1))
        unit = prompt_dur_match.group(2)
        if unit == "ms":
            metrics["prompt_eval_duration"] = value / 1000
        elif unit == "s":
            metrics["prompt_eval_duration"] = value
        else:
            metrics["prompt_eval_duration"] = value

    # Parse prompt eval rate
    prompt_rate_match = re.search(r"prompt eval rate:\s+([\d.]+)\s+tokens/s", output)
    if prompt_rate_match:
        metrics["prompt_eval_rate"] = float(prompt_rate_match.group(1))

    # Parse eval count (generation tokens) - looking for line without "prompt" prefix
    eval_count_match = re.search(r"^eval count:\s+(\d+)", output, re.MULTILINE)
    if eval_count_match:
        metrics["eval_count"] = int(eval_count_match.group(1))

    # Parse eval duration (generation time) - looking for line without "prompt" prefix
    eval_dur_match = re.search(
        r"^eval duration:\s+([\d.]+)([a-z]+)", output, re.MULTILINE
    )
    if eval_dur_match:
        value = float(eval_dur_match.group(1))
        unit = eval_dur_match.group(2)
        if unit == "ms":
            metrics["eval_duration"] = value / 1000
        elif unit == "s":
            metrics["eval_duration"] = value
        else:
            metrics["eval_duration"] = value

    # Parse eval rate (generation tokens per second) - looking for line without "prompt" prefix
    eval_rate_match = re.search(
        r"^eval rate:\s+([\d.]+)\s+tokens/s", output, re.MULTILINE
    )
    if eval_rate_match:
        metrics["eval_rate"] = float(eval_rate_match.group(1))

    return metrics


def extract_generated_text(stdout: str, stderr: str, prompt: str) -> str:
    """Extract the generated text from the ollama output.

    With --verbose flag:
    - The generated text appears in stdout
    - The metrics appear in stderr
    
    Args:
        stdout: Standard output from ollama
        stderr: Standard error from ollama
        prompt: The original prompt
        
    Returns:
        The generated text
    """
    # The model's response should be the entire stdout
    # since metrics go to stderr with --verbose
    generated_text = stdout.strip()

    # If the stdout starts with the prompt (sometimes ollama echoes it),
    # remove it to get just the generated text
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt) :].strip()

    return generated_text


def run_cli_benchmark(
    model_name: str, context_file: Path, max_tokens: int = 200
) -> Optional[Dict]:
    """Run Ollama benchmark using CLI for a given context file.
    
    Args:
        model_name: Name of the Ollama model
        context_file: Path to the context file
        max_tokens: Maximum tokens to generate
        
    Returns:
        Dictionary with benchmark results or None if failed
    """
    print(f"Running CLI benchmark for {context_file}...")

    # Read the prompt from file
    with open(context_file) as f:
        prompt = f.read()

    # Count prompt tokens (approximate)
    prompt_tokens = len(prompt.split())

    # Prepare the command
    cmd = ["ollama", "run", model_name, "--verbose", prompt]

    # Start timing
    start_time = time.time()

    try:
        # Run the command and capture output
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600
        )  # 60 minute timeout

        # Calculate total wall time
        total_wall_time = time.time() - start_time

        # Parse the metrics from stderr (with --verbose, metrics go to stderr)
        metrics = parse_ollama_output(result.stderr)

        # Extract generated text from stdout
        generated_text = extract_generated_text(result.stdout, result.stderr, prompt)

        # If we didn't get prompt eval rate from parsing, calculate it
        if (
            "prompt_eval_rate" not in metrics
            and "prompt_eval_duration" in metrics
            and "prompt_eval_count" in metrics
        ):
            if metrics["prompt_eval_duration"] > 0:
                metrics["prompt_eval_rate"] = (
                    metrics["prompt_eval_count"] / metrics["prompt_eval_duration"]
                )

        # If we didn't get eval rate from parsing, calculate it
        if (
            "eval_rate" not in metrics
            and "eval_duration" in metrics
            and "eval_count" in metrics
        ):
            if metrics["eval_duration"] > 0:
                metrics["eval_rate"] = metrics["eval_count"] / metrics["eval_duration"]

        # Ensure we have the required metrics
        if "eval_count" not in metrics or "eval_rate" not in metrics:
            print(f"Failed to parse required metrics from output")
            print(f"Parsed metrics: {metrics}")
            return None

        # Debug logging
        print(
            f"  Prompt: {metrics.get('prompt_eval_count', 0)} tokens at {metrics.get('prompt_eval_rate', 0):.1f} t/s"
        )
        print(
            f"  Generation: {metrics.get('eval_count', 0)} tokens at {metrics.get('eval_rate', 0):.1f} t/s"
        )
        print(f"  Total wall time: {total_wall_time:.2f}s")

        return {
            "context_size": Path(context_file).stem,
            "prompt_tokens": metrics.get("prompt_eval_count", prompt_tokens),
            "prompt_tps": metrics.get("prompt_eval_rate", 0),
            "generation_tokens": metrics.get("eval_count", 0),
            "generation_tps": metrics.get("eval_rate", 0),
            "total_time": total_wall_time,
            "eval_duration": metrics.get("eval_duration", 0),
            "prompt_eval_duration": metrics.get("prompt_eval_duration", 0),
            "generated_text": generated_text,
            "wall_time": total_wall_time,
        }

    except subprocess.TimeoutExpired:
        print(f"Timeout running benchmark for {context_file}")
        return None
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
        # Use ollama list command
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )

        if result.returncode != 0:
            print(f"Error checking model availability: {result.stderr}")
            return False

        # Parse the output
        lines = result.stdout.strip().split("\n")

        # Skip header line
        if len(lines) > 1:
            for line in lines[1:]:
                # Split by whitespace and get the first column (model name)
                if line.strip():
                    parts = line.split()
                    if parts:
                        available_model = parts[0]
                        # Check for exact match or base model match
                        if available_model == model_name:
                            print(f"Found model: {model_name}")
                            return True
                        # Check without tag
                        if ":" in model_name and ":" in available_model:
                            if available_model.split(":")[0] == model_name.split(":")[0]:
                                print(f"Found model: {available_model}")
                                return True

        print(f"Model '{model_name}' not found in Ollama")
        print("Available models:")
        if len(lines) > 1:
            for line in lines[1:]:
                if line.strip():
                    parts = line.split()
                    if parts:
                        print(f"  - {parts[0]}")
        return False

    except Exception as e:
        print(f"Error checking model availability: {e}")
        print(f"Attempting to use model anyway...")
        return True  # Try to proceed anyway


def main() -> int:
    """Main function to run Ollama CLI benchmarks."""
    parser = argparse.ArgumentParser(
        description="Run Ollama benchmarks using command-line interface"
    )
    parser.add_argument("model", help="Ollama model name (e.g., llama3.2, mistral)")

    # Add common arguments
    common.setup_common_args(parser)

    args = parser.parse_args()

    # Check if model is available
    if not check_model_available(args.model):
        print(f"\nPlease pull the model first with: ollama pull {args.model}")
        return 1

    # Create output directory using common function
    output_dir = common.create_output_directory("ollama_cli", args.model)

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

        result = run_cli_benchmark(args.model, file, args.max_tokens)
        if result:
            results.append(result)

            # Save the generated text if requested
            if args.save_responses:
                output_filename = output_dir / f"response_{result['context_size']}.txt"
                common.save_generated_text(
                    result, args.model, output_filename, "Ollama CLI"
                )
    
    total_benchmark_time = time.time() - start_time

    if not results:
        print("\nNo successful benchmark results")
        return 1

    # Save all outputs using common function
    common.save_all_outputs(
        results, output_dir, args.model, "Ollama CLI", hardware_info, args
    )

    # Print summary using common function
    common.print_benchmark_summary(
        results, args.model, "Ollama CLI", hardware_info, output_dir, total_benchmark_time
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())