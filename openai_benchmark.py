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
from typing import Dict, List, Optional

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
    max_tokens: int = 128,
    timeout: int = 3600,
    cold_prefill: bool = True,
    _run_idx: Optional[int] = None,
) -> Optional[Dict]:
    """Benchmark a single context file against the OpenAI-compatible endpoint.

    Uses streaming so we can measure time-to-first-token (TTFT) accurately.

    Returns a result dict on success, None on failure.
    """
    with open(context_file) as f:
        prompt = f.read()

    if cold_prefill:
        prompt = common.make_cache_buster() + prompt
    elif _run_idx is not None:
        prompt = common.make_cache_buster(run_idx=_run_idx) + prompt

    start_time = time.time()
    first_token_time: Optional[float] = None
    generated_text = ""
    prompt_tokens = 0
    completion_tokens = 0
    server_prompt_tps = 0.0
    server_generation_tps = 0.0
    peak_memory_gb = 0.0

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

                # Extract server-reported stats (mlx-vlm, etc.)
                usage_extra = chunk.usage.model_dump() if hasattr(chunk.usage, "model_dump") else {}
                # Some servers use input_tokens/output_tokens instead
                if prompt_tokens == 0:
                    prompt_tokens = usage_extra.get("input_tokens", 0) or 0
                if completion_tokens == 0:
                    completion_tokens = usage_extra.get("output_tokens", 0) or 0
                server_prompt_tps = usage_extra.get("prompt_tps", 0.0) or 0.0
                server_generation_tps = usage_extra.get("generation_tps", 0.0) or 0.0
                peak_memory_gb = usage_extra.get("peak_memory", 0.0) or 0.0

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

    # Prefer server-reported TPS when available, fall back to client-side calculation
    prompt_tps = server_prompt_tps if server_prompt_tps > 0 else (prompt_tokens / ttft if ttft > 0 else 0.0)
    generation_tps = (
        server_generation_tps
        if server_generation_tps > 0
        else (completion_tokens / generation_time if generation_time > 0 else 0.0)
    )

    print(f"  Prompt tokens:      {prompt_tokens}")
    print(f"  Completion tokens:  {completion_tokens}")
    print(f"  TTFT:               {ttft:.3f}s")
    print(f"  Generation time:    {generation_time:.2f}s")
    print(f"  Total time:         {total_time:.2f}s")
    print(f"  Prompt TPS:         {prompt_tps:.1f} t/s")
    print(f"  Generation TPS:     {generation_tps:.1f} t/s")
    if peak_memory_gb > 0:
        print(f"  Peak memory:        {peak_memory_gb:.2f} GB")

    result = {
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
    if peak_memory_gb > 0:
        result["peak_memory_gb"] = peak_memory_gb

    return result


def run_batch_benchmark(
    base_url: str,
    api_key: str,
    model: str,
    batch_sizes: List[int],
    prompt_tokens: int = 2048,
    gen_tokens: int = 128,
    num_trials: int = 3,
) -> List[Dict]:
    """Run batch benchmark by sending concurrent requests to test continuous batching.

    Args:
        base_url: Server base URL
        api_key: API key
        model: Model name
        batch_sizes: List of batch sizes to test (concurrent requests)
        prompt_tokens: Approximate prompt tokens per request
        gen_tokens: Tokens to generate per request
        num_trials: Number of trials per batch size

    Returns:
        List of result dicts with batch_size, prompt_tps, generation_tps
    """
    import concurrent.futures
    import statistics

    import tiktoken

    # Generate a fixed prompt of approximately prompt_tokens length
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        # Use a repeated phrase to fill the token budget
        base_text = "The quick brown fox jumps over the lazy dog. "
        base_tokens = enc.encode(base_text)
        repeats = max(1, prompt_tokens // len(base_tokens))
        prompt_text = base_text * repeats
        # Trim to exact token count
        tokens = enc.encode(prompt_text)[:prompt_tokens]
        prompt_text = enc.decode(tokens)
    except Exception:
        # Fallback: approximate ~4 chars per token
        prompt_text = "The quick brown fox jumps over the lazy dog. " * (prompt_tokens // 10)

    def single_request():
        """Send one non-streaming request and return (prompt_tps, gen_tps, prompt_tokens, gen_tokens)."""
        import httpx

        resp = httpx.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": gen_tokens,
                "temperature": 0.7,
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=600,
        )
        data = resp.json()
        usage = data.get("usage", {})
        return {
            "prompt_tps": usage.get("prompt_tps", 0),
            "generation_tps": usage.get("generation_tps", 0),
            "prompt_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
            "generation_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
            "peak_memory": usage.get("peak_memory", 0),
        }

    batch_results = []

    for bs in batch_sizes:
        print(f"\n  Batch size {bs} ({num_trials} trials, ~{prompt_tokens} prompt tokens, {gen_tokens} gen tokens)...")

        # Warmup
        print("    Warmup...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as pool:
            list(pool.map(lambda _: single_request(), range(bs)))

        trial_prompt_tps = []
        trial_gen_tps = []
        trial_peak_mem = []

        for trial in range(num_trials):
            start = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as pool:
                futures = [pool.submit(single_request) for _ in range(bs)]
                responses = [f.result() for f in futures]
            wall_time = time.time() - start

            total_prompt_tok = sum(r["prompt_tokens"] for r in responses)
            total_gen_tok = sum(r["generation_tokens"] for r in responses)
            agg_prompt_tps = total_prompt_tok / wall_time if wall_time > 0 else 0
            agg_gen_tps = total_gen_tok / wall_time if wall_time > 0 else 0
            peak_mem = max((r["peak_memory"] for r in responses), default=0)

            trial_prompt_tps.append(agg_prompt_tps)
            trial_gen_tps.append(agg_gen_tps)
            if peak_mem > 0:
                trial_peak_mem.append(peak_mem)

            print(f"    Trial {trial + 1}: pp {agg_prompt_tps:.1f} tg {agg_gen_tps:.1f} t/s ({wall_time:.1f}s)")

        if trial_prompt_tps:
            avg_prompt = statistics.mean(trial_prompt_tps)
            avg_gen = statistics.mean(trial_gen_tps)
            result = {
                "batch_size": bs,
                "prompt_tps": round(avg_prompt, 2),
                "generation_tps": round(avg_gen, 2),
            }
            if trial_peak_mem:
                result["peak_memory_gb"] = round(max(trial_peak_mem), 3)
            else:
                result["peak_memory_gb"] = 0.0

            print(f"  Avg: pp {avg_prompt:.1f} tg {avg_gen:.1f} t/s")
            batch_results.append(result)

    return batch_results


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

    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8",
        help="Comma-separated batch sizes for concurrent request benchmark (default: 1,2,4,8)",
    )
    parser.add_argument(
        "--batch-prompt-tokens",
        type=int,
        default=2048,
        help="Approximate prompt tokens per request in batch benchmark (default: 2048)",
    )
    parser.add_argument(
        "--batch-gen-tokens",
        type=int,
        default=128,
        help="Tokens to generate per request in batch benchmark (default: 128)",
    )
    parser.add_argument(
        "--batch-trials",
        type=int,
        default=3,
        help="Number of trials per batch size (default: 3)",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Skip batch benchmark",
    )
    parser.add_argument(
        "--cold-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prepend a unique marker to every prompt to bust KV "
        "cache reuse, forcing cold prefill on every row (default: enabled; "
        "use --no-cold-prefill for cached/warm-reuse numbers)",
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
    print(
        f"Cold prefill: {'enabled (cache busted per prompt)' if args.cold_prefill else 'disabled (cache reuse allowed)'}"
    )

    context_files = common.find_context_files(args.contexts)
    if not context_files:
        return 1

    output_dir = common.create_output_directory("openai_compat", model, cold_prefill=args.cold_prefill)

    results = []
    benchmark_start = time.time()

    if args.cold_prefill:
        for ctx_file in context_files:
            print(f"\n{'=' * 50}")
            print(f"Benchmarking {ctx_file.name} ...")
            print(f"{'=' * 50}")

            result = common.run_benchmark_peak(
                run_benchmark,
                client,
                model,
                ctx_file,
                args.max_tokens,
                args.timeout,
                cold_prefill=args.cold_prefill,
                n_runs=args.runs,
            )
            if result:
                results.append(result)

                if args.save_responses:
                    resp_path = output_dir / f"response_{result['context_size']}.txt"
                    common.save_generated_text(result, model, resp_path, "OpenAI Compat")
    else:
        results = common.run_benchmark_peak_per_run(
            run_benchmark,
            context_files=context_files,
            n_runs=args.runs,
            client=client,
            model=model,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            cold_prefill=args.cold_prefill,
        )
        if args.save_responses:
            for result in results:
                resp_path = output_dir / f"response_{result['context_size']}.txt"
                common.save_generated_text(result, model, resp_path, "OpenAI Compat")

    total_benchmark_time = time.time() - benchmark_start

    if not results:
        print("\nNo successful benchmark results.")
        return 1

    # Run batch benchmark
    batch_results = None
    if not args.no_batch:
        batch_sizes = [int(s.strip()) for s in args.batch_sizes.split(",")]
        print(f"\nRunning batch benchmark (concurrent requests: {batch_sizes})...")
        try:
            batch_results = run_batch_benchmark(
                base_url,
                args.api_key,
                model,
                batch_sizes,
                prompt_tokens=args.batch_prompt_tokens,
                gen_tokens=args.batch_gen_tokens,
                num_trials=args.batch_trials,
            )
            if batch_results:
                print(f"\nBatch benchmark complete: {len(batch_results)} sizes tested")
            else:
                print("\nBatch benchmark produced no results")
        except Exception as e:
            print(f"\nBatch benchmark failed (continuing): {e}")

    has_memory = any(r.get("peak_memory_gb", 0) > 0 for r in results)
    common.save_all_outputs(
        results,
        output_dir,
        model,
        "OpenAI Compat",
        hardware_info,
        args,
        include_memory=has_memory,
        batch_results=batch_results,
    )

    common.print_benchmark_summary(
        results,
        model,
        "OpenAI Compat",
        hardware_info,
        output_dir,
        total_benchmark_time,
        batch_results=batch_results,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
