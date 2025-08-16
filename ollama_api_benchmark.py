#!/usr/bin/env python3
import argparse
import time
from datetime import datetime
from pathlib import Path

import ollama

import benchmark_common as common


def run_benchmark(model_name, context_file, max_tokens=200):
    """Run Ollama benchmark for a given context file."""
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
        response = ollama.generate(model=model_name, prompt=prompt, options={"num_predict": max_tokens})

        # Calculate total time
        total_time = time.time() - start_time

        # Parse the response
        generated_text = response.get("response", "")
        generation_tokens = response.get("eval_count", 0)
        eval_duration = response.get("eval_duration", 0) / 1e9  # Convert nanoseconds to seconds
        prompt_eval_duration = response.get("prompt_eval_duration", 0) / 1e9  # Convert to seconds
        prompt_eval_count = response.get("prompt_eval_count", prompt_tokens)

        # Calculate tokens per second
        prompt_tps = prompt_eval_count / prompt_eval_duration if prompt_eval_duration > 0 else 0
        generation_tps = generation_tokens / eval_duration if eval_duration > 0 else 0

        # Debug logging
        print(f"  Prompt: {prompt_eval_count} tokens in {prompt_eval_duration:.2f}s = {prompt_tps:.1f} t/s")
        print(f"  Generation: {generation_tokens} tokens in {eval_duration:.2f}s = {generation_tps:.1f} t/s")
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


def check_model_available(model_name):
    """Check if the model is available in Ollama."""
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
            if available_model == model_name or available_model.split(":")[0] == model_name.split(":")[0]:
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
    parser = argparse.ArgumentParser(description="Run Ollama benchmarks on context files")
    parser.add_argument("model", help="Ollama model name (e.g., llama3.2, mistral)")
    parser.add_argument(
        "--contexts",
        type=str,
        default="2,4,8,16",
        help="Comma-separated list of context sizes to benchmark (default: 2,4,8,16)",
    )
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds (default: 3600)")
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
    output_dir = base_output_dir / f"benchmark_ollama_api_{model_name}_{timestamp}"
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
        result = run_benchmark(args.model, file, args.max_tokens)
        if result:
            results.append(result)

            # Save the generated text if requested
            if args.save_responses:
                output_filename = output_dir / f"generated_{result['context_size']}.txt"
                common.save_generated_text(result, args.model, output_filename, "Ollama API")

    if not results:
        print("No successful benchmark results")
        return

    # Calculate total generated tokens
    total_generated_tokens = sum(r.get("generation_tokens", 0) for r in results)
    print(f"\n📊 Total generated tokens across all tests: {total_generated_tokens}")

    # Save to CSV
    common.save_results_csv(results, csv_path)

    # Create chart
    chart_result = common.create_chart_ollama(results, model_name, hardware_info, str(chart_path), "Ollama API")
    print(f"Chart saved to {chart_path}")

    # Generate X post text
    xpost = common.generate_xpost_text(results, model_name, "Ollama API", hardware_info)
    print("\n--- X Post Text ---")
    print(xpost)

    # Save X post to file
    with open(xpost_path, "w") as f:
        f.write(xpost)
    print(f"\nX Post text saved to {xpost_path}")

    # Generate and display table
    table = common.generate_table(results, model_name, "Ollama API", hardware_info)
    print("\n--- Table for X/Twitter Thread ---")
    print(table)

    # Save table to file
    with open(table_path, "w") as f:
        f.write(table)
    print(f"\nTable saved to {table_path}")

    print(f"\n✅ All outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
