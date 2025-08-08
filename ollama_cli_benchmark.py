#!/usr/bin/env python3
import argparse
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import benchmark_common as common


def parse_ollama_output(output):
    """Parse the verbose output from ollama run command."""
    # Remove ANSI escape sequences and terminal control codes
    import re as regex

    ansi_escape = regex.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
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
    # Need to avoid matching "prompt eval count"
    eval_count_match = re.search(r"^eval count:\s+(\d+)", output, re.MULTILINE)
    if eval_count_match:
        metrics["eval_count"] = int(eval_count_match.group(1))

    # Parse eval duration (generation time) - looking for line without "prompt" prefix
    eval_dur_match = re.search(r"^eval duration:\s+([\d.]+)([a-z]+)", output, re.MULTILINE)
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
    eval_rate_match = re.search(r"^eval rate:\s+([\d.]+)\s+tokens/s", output, re.MULTILINE)
    if eval_rate_match:
        metrics["eval_rate"] = float(eval_rate_match.group(1))

    return metrics


def extract_generated_text(stdout, stderr, prompt):
    """Extract the generated text from the ollama output.

    With --verbose flag:
    - The generated text appears in stdout
    - The metrics appear in stderr
    """
    # The model's response should be the entire stdout
    # since metrics go to stderr with --verbose
    generated_text = stdout.strip()

    # If the stdout starts with the prompt (sometimes ollama echoes it),
    # remove it to get just the generated text
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt) :].strip()

    return generated_text


def run_cli_benchmark(model_name, context_file, max_tokens=200):
    """Run Ollama benchmark using CLI for a given context file."""
    print(f"Running CLI benchmark for {context_file}...")

    # Read the prompt from file
    with open(context_file) as f:
        prompt = f.read()

    # Count prompt tokens (approximate)
    prompt_tokens = len(prompt.split())

    # Prepare the command
    # Using echo to pipe the prompt to ollama
    cmd = ["ollama", "run", model_name, "--verbose", prompt]

    # Start timing
    start_time = time.time()

    try:
        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 minute timeout

        # Calculate total wall time
        total_wall_time = time.time() - start_time

        # Parse the metrics from stderr (with --verbose, metrics go to stderr)
        metrics = parse_ollama_output(result.stderr)

        # Extract generated text from stdout
        generated_text = extract_generated_text(result.stdout, result.stderr, prompt)

        # If we didn't get prompt eval rate from parsing, calculate it
        if "prompt_eval_rate" not in metrics and "prompt_eval_duration" in metrics and "prompt_eval_count" in metrics:
            if metrics["prompt_eval_duration"] > 0:
                metrics["prompt_eval_rate"] = metrics["prompt_eval_count"] / metrics["prompt_eval_duration"]

        # If we didn't get eval rate from parsing, calculate it
        if "eval_rate" not in metrics and "eval_duration" in metrics and "eval_count" in metrics:
            if metrics["eval_duration"] > 0:
                metrics["eval_rate"] = metrics["eval_count"] / metrics["eval_duration"]

        # Debug logging
        print(
            f"  Prompt: {metrics.get('prompt_eval_count', prompt_tokens)} tokens in {metrics.get('prompt_eval_duration', 0):.2f}s = {metrics.get('prompt_eval_rate', 0):.1f} t/s"
        )
        print(
            f"  Generation: {metrics.get('eval_count', 0)} tokens in {metrics.get('eval_duration', 0):.2f}s = {metrics.get('eval_rate', 0):.1f} t/s"
        )
        print(f"  Total wall time: {total_wall_time:.2f}s")
        print(f"  Total duration (from ollama): {metrics.get('total_duration', 0):.2f}s")

        return {
            "context_size": Path(context_file).stem,
            "prompt_tokens": metrics.get("prompt_eval_count", prompt_tokens),
            "prompt_tps": metrics.get("prompt_eval_rate", 0),
            "generation_tokens": metrics.get("eval_count", 0),
            "generation_tps": metrics.get("eval_rate", 0),
            "total_time": metrics.get("total_duration", total_wall_time),
            "wall_time": total_wall_time,
            "prompt_eval_duration": metrics.get("prompt_eval_duration", 0),
            "eval_duration": metrics.get("eval_duration", 0),
            "load_duration": metrics.get("load_duration", 0),
            "generated_text": generated_text,
        }

    except subprocess.TimeoutExpired:
        print(f"Timeout running benchmark")
        return None
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return None


def check_model_available(model_name):
    """Check if the model is available in Ollama."""
    try:
        # Run ollama list command
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error checking models: {result.stderr}")
            return True  # Try to proceed anyway

        # Parse the output
        lines = result.stdout.strip().split("\n")

        # Skip header line if present
        if lines and "NAME" in lines[0]:
            lines = lines[1:]

        # Extract model names
        model_names = []
        for line in lines:
            if line.strip():
                # Model name is usually the first column
                parts = line.split()
                if parts:
                    model_names.append(parts[0])

        # Check if our model is available
        if model_name in model_names:
            print(f"Found model: {model_name}")
            return True

        # Also check without considering tags
        model_base = model_name.split(":")[0]
        for available_model in model_names:
            if available_model == model_name or available_model.startswith(model_base):
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


def main():
    parser = argparse.ArgumentParser(description="Run Ollama CLI benchmarks on context files")
    parser.add_argument("model", help="Ollama model name (e.g., gpt-oss:20b, llama3.2, mistral)")
    parser.add_argument(
        "--contexts",
        type=str,
        default="2,4,8,16",
        help="Comma-separated list of context sizes to benchmark (default: 2,4,8,16)",
    )
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate (default: 200)")
    parser.add_argument("--output-csv", default="benchmark_results.csv", help="Output CSV file")
    parser.add_argument("--output-chart", default="benchmark_chart.png", help="Output chart file")
    parser.add_argument("--save-responses", action="store_true", help="Save raw model responses to files")

    args = parser.parse_args()

    # Check if model is available
    if not check_model_available(args.model):
        print(f"\nPlease pull the model first with: ollama pull {args.model}")
        return

    # Extract model name and create output directory
    model_name = args.model.replace(":", "_")  # Replace : with _ for filesystem compatibility
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path("output")
    base_output_dir.mkdir(exist_ok=True)
    output_dir = base_output_dir / f"benchmark_ollama_cli_{model_name}_{timestamp}"
    output_dir.mkdir(exist_ok=True)

    # Update output paths to use the new directory
    csv_path = output_dir / args.output_csv
    chart_path = output_dir / args.output_chart
    xpost_path = output_dir / "xpost.txt"
    table_path = output_dir / "table.txt"
    hardware_path = output_dir / "hardware_info.json"

    # Find context files using common module
    context_files = common.find_context_files(args.contexts)
    if not context_files:
        return

    # Get hardware information
    print("\nCollecting hardware information...")
    hardware_info = common.get_hardware_info()
    hardware_str = common.format_hardware_string(hardware_info)
    print(f"Hardware: {hardware_str}")

    # Save hardware info
    common.save_hardware_info(hardware_info, hardware_path)

    # Run benchmarks
    results = []
    for file in context_files:
        result = run_cli_benchmark(args.model, file, args.max_tokens)
        if result:
            results.append(result)

            # Save the generated text if requested
            if args.save_responses:
                output_filename = output_dir / f"generated_{result['context_size']}.txt"
                common.save_generated_text(result, args.model, output_filename, "Ollama CLI")

    if not results:
        print("No successful benchmark results")
        return

    # Calculate total generated tokens
    total_generated_tokens = sum(r.get("generation_tokens", 0) for r in results)
    print(f"\n📊 Total generated tokens across all tests: {total_generated_tokens}")

    # Save to CSV
    common.save_results_csv(results, csv_path)

    # Create chart
    chart_result = common.create_chart_ollama(results, model_name, hardware_info, str(chart_path), "Ollama CLI")
    print(f"Chart saved to {chart_path}")

    # Generate X post text
    xpost = common.generate_xpost_text(results, model_name, "Ollama CLI", hardware_info)
    print("\n--- X Post Text ---")
    print(xpost)

    # Save X post to file
    with open(xpost_path, "w") as f:
        f.write(xpost)
    print(f"\nX Post text saved to {xpost_path}")

    # Generate and display table
    table = common.generate_table(results, model_name, "Ollama CLI", hardware_info)
    print("\n--- Table for X/Twitter Thread ---")
    print(table)

    # Save table to file
    with open(table_path, "w") as f:
        f.write(table)
    print(f"\nTable saved to {table_path}")

    print(f"\n✅ All outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
