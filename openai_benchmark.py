#!/usr/bin/env python3
"""
Benchmark script for any OpenAI-compatible API endpoint.

Connects to any server that implements the OpenAI Chat Completions API
and benchmarks its performance across different context sizes.

Supports vLLM, llama.cpp server, Ollama, LM Studio, text-generation-webui,
and any other OpenAI-compatible server.

Usage:
    # Against a local vLLM / llama.cpp / Ollama server:
    python openai_benchmark.py --model meta-llama/Llama-3.1-8B-Instruct
    python openai_benchmark.py --model llama3.2 --base-url http://localhost:11434/v1

    # Against a remote server with an API key:
    python openai_benchmark.py --model gpt-4o --base-url https://api.openai.com/v1 --api-key sk-...

    # Custom contexts and token budget:
    python openai_benchmark.py --model mistral --contexts 2,4,8,16 --max-tokens 500
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from openai import OpenAI

import benchmark_common as common


def build_client(base_url: str, api_key: str) -> OpenAI:
    """Create an OpenAI client pointed at the given base URL."""
    return OpenAI(base_url=base_url, api_key=api_key)


def test_server_connection(client: OpenAI) -> bool:
    """Check that the server is reachable and returns a model list."""
    try:
        client.models.list()
        return True
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return False


def get_available_model(client: OpenAI) -> Optional[str]:
    """Return the first model ID reported by the server, or None."""
    try:
        models = list(client.models.list())
        if models:
            return models[0].id
    except Exception:
        pass
    return None


def run_benchmark(
    client: OpenAI,
    model: str,
    context_file: Path,
    max_tokens: int = 200,
    timeout: int = 3600,
) -> Optional[Dict]:
    """Benchmark a single context file against the OpenAI-compatible endpoint.

    Uses streaming so we can measure time-to-first-token (TTFT) accurately.

    Returns a result dict on success, None on failure.
    """
    with open(context_file) as f:
        prompt = f.read()

    start_time = time.time()
    first_token_time: Optional[float] = None
    generated_text = ""
    prompt_tokens = 0
    completion_tokens = 0

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
            stream=True,
            stream_options={"include_usage": True},
            timeout=timeout,
        )

        for chunk in stream:
            # Capture first token time
            if first_token_time is None and chunk.choices and chunk.choices[0].delta.content:
                first_token_time = time.time()

            # Accumulate generated text
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    generated_text += delta

            # Final chunk may carry usage stats
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0

    except Exception as e:
        print(f"Error during benchmark: {e}")
        return None

    end_time = time.time()
    total_time = end_time - start_time

    ttft = (first_token_time - start_time) if first_token_time else 0.0
    generation_time = (end_time - first_token_time) if first_token_time else total_time

    # Fallback: approximate prompt token count from word count
    if prompt_tokens == 0:
        prompt_tokens = len(prompt.split())

    # Fallback: approximate completion tokens from generated text
    if completion_tokens == 0:
        completion_tokens = len(generated_text.split())

    prompt_tps = prompt_tokens / ttft if ttft > 0 else 0.0
    generation_tps = completion_tokens / generation_time if generation_time > 0 else 0.0

    print(f"  Prompt tokens:      {prompt_tokens}")
    print(f"  Completion tokens:  {completion_tokens}")
    print(f"  TTFT:               {ttft:.3f}s")
    print(f"  Generation time:    {generation_time:.2f}s")
    print(f"  Total time:         {total_time:.2f}s")
    print(f"  Prompt TPS:         {prompt_tps:.1f} t/s")
    print(f"  Generation TPS:     {generation_tps:.1f} t/s")

    return {
        "context_size": context_file.stem,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": completion_tokens,
        "time_to_first_token": ttft,
        "eval_duration": generation_time,
        "total_time": total_time,
        "prompt_tps": prompt_tps,
        "generation_tps": generation_tps,
        "generated_text": generated_text,
    }


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark any OpenAI-compatible API endpoint across different context sizes"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name/ID to use (auto-detected from server if omitted)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080/v1",
        help="Base URL of the OpenAI-compatible server (default: http://localhost:8080/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "no-key"),
        help="API key (default: OPENAI_API_KEY env var or 'no-key' for local servers)",
    )

    common.setup_common_args(parser)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    client = build_client(base_url, args.api_key)

    print(f"Testing connection to {base_url} ...")
    if not test_server_connection(client):
        print(f"Error: Cannot reach server at {base_url}")
        print("Make sure the server is running and the base URL is correct.")
        return 1
    print("Connected successfully.")

    # Resolve model name
    model = args.model
    if not model:
        model = get_available_model(client)
        if not model:
            print("Error: No model specified and could not auto-detect one from the server.")
            return 1
        print(f"Auto-detected model: {model}")

    hardware_info = common.get_hardware_info()
    hardware_str = common.format_hardware_string(hardware_info)

    print(f"\nOpenAI-compatible Endpoint Benchmark")
    print(f"Server:     {base_url}")
    print(f"Model:      {model}")
    print(f"Hardware:   {hardware_str}")
    print(f"Max tokens: {args.max_tokens}")

    context_files = common.find_context_files(args.contexts)
    if not context_files:
        return 1

    output_dir = common.create_output_directory("openai_compat", model)

    results = []
    benchmark_start = time.time()

    for ctx_file in context_files:
        print(f"\n{'=' * 50}")
        print(f"Benchmarking {ctx_file.name} ...")
        print(f"{'=' * 50}")

        result = run_benchmark(client, model, ctx_file, args.max_tokens, args.timeout)
        if result:
            results.append(result)

            if args.save_responses:
                resp_path = output_dir / f"response_{result['context_size']}.txt"
                common.save_generated_text(result, model, resp_path, "OpenAI Compat")

    total_benchmark_time = time.time() - benchmark_start

    if not results:
        print("\nNo successful benchmark results.")
        return 1

    common.save_all_outputs(
        results, output_dir, model, "OpenAI Compat", hardware_info, args
    )

    common.print_benchmark_summary(
        results, model, "OpenAI Compat", hardware_info, output_dir, total_benchmark_time
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
