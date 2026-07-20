#!/usr/bin/env python3
"""
Benchmark script for the Unsloth Studio local inference server.

Unsloth Studio (http://127.0.0.1:8888) exposes an OpenAI-compatible API at
``/v1`` that requires a bearer token. Streaming chat completions carry a
top-level ``timings`` object (sibling of ``usage``) with server-reported
prefill/decode metrics in llama-server style:

    "timings": {
        "prompt_n": 20, "prompt_ms": 228.577, "prompt_per_second": 87.49,
        "predicted_n": 15, "predicted_ms": 149.95, "predicted_per_second": 100.03,
        "cache_n": 0
    }

Non-streaming responses omit ``timings`` (only standard OpenAI ``usage``), so
this engine always streams to capture server-reported TPS / TTFT / decode time.
The generic ``openai_benchmark.py`` does not read ``timings`` (it lives outside
``usage`` and uses different key names), hence the dedicated script.

Usage:
    python unsloth_benchmark.py
    python unsloth_benchmark.py unsloth/Qwen3.6-35B-A3B-GGUF
    python unsloth_benchmark.py unsloth/Qwen3.6-35B-A3B-GGUF \\
        --base-url http://127.0.0.1:8888/v1 --api-key sk-unsloth-...
    python unsloth_benchmark.py --contexts 2,4,8,16 --max-tokens 500
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
    """Create an OpenAI client pointed at the Unsloth Studio server."""
    return OpenAI(base_url=base_url, api_key=api_key)


def test_server_connection(client: OpenAI) -> bool:
    """Check that the server is reachable and returns a model list."""
    try:
        client.models.list()
        return True
    except Exception as e:
        print(f"Error connecting to Unsloth server: {e}")
        return False


def get_available_model(client: OpenAI) -> Optional[str]:
    """Return the first *loaded* model ID, falling back to the first listed.

    Unsloth reports many models, only one of which is ``loaded: true`` and able
    to serve requests. Prefer it so auto-detect targets a usable model.
    """
    try:
        models = list(client.models.list())
        if not models:
            return None
        for m in models:
            if getattr(m, "loaded", False):
                return m.id
        return models[0].id
    except Exception:
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
    """Benchmark a single context file against the Unsloth server.

    Streams so we can measure TTFT client-side and read the server-reported
    ``timings`` block from the final usage chunk. Returns a result dict on
    success, None on failure.
    """
    with open(context_file) as f:
        prompt = f.read()

    if cold_prefill:
        prompt = common.make_cache_buster() + prompt
    elif _run_idx is not None:
        prompt = common.make_cache_buster(run_idx=_run_idx) + prompt

    timings: Dict = {}

    def _capture_timings(chunk):
        # `timings` is a top-level sibling of `usage`; pydantic keeps it as
        # model_extra and exposes it as a plain attribute.
        t = getattr(chunk, "timings", None)
        if isinstance(t, dict) and t:
            timings.update(t)

    try:
        stream_result = common.stream_chat(
            client, model, prompt, max_tokens, temperature=0.7, timeout=timeout, chunk_hook=_capture_timings
        )
    except Exception as e:
        print(f"Error during benchmark: {e}")
        return None

    generated_text = stream_result["generated_text"]
    reasoning_text = stream_result["reasoning_text"]
    total_time = stream_result["total_time"]
    usage = stream_result["usage"]
    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0

    # Server-reported prefill/decode (ms -> s). prompt_ms is pure prefill time,
    # which is the time-to-first-token for a cold prompt; predicted_ms is the
    # decode window after the first token.
    server_ttft = timings.get("prompt_ms", 0) / 1000 if timings else 0.0
    server_decode = timings.get("predicted_ms", 0) / 1000 if timings else 0.0
    server_prompt_tps = timings.get("prompt_per_second", 0.0) if timings else 0.0
    server_gen_tps = timings.get("predicted_per_second", 0.0) if timings else 0.0
    cached_tokens = timings.get("cache_n", 0) if timings else 0

    # Token counts: prefer usage, fall back to timings counts.
    if prompt_tokens == 0 and timings:
        prompt_tokens = timings.get("prompt_n", 0)
    if completion_tokens == 0 and timings:
        completion_tokens = timings.get("predicted_n", 0)

    # TTFT / decode window: prefer server-reported values, then the
    # client-side anchors measured by stream_chat.
    ttft = server_ttft if server_ttft > 0 else stream_result["time_to_first_token"]
    generation_time = server_decode if server_decode > 0 else stream_result["decode_window"]

    # Fallback token counts from text.
    if prompt_tokens == 0:
        prompt_tokens = len(prompt.split())
    if completion_tokens == 0:
        completion_tokens = len(generated_text.split()) + len(reasoning_text.split())

    # TPS: prefer server-reported, else compute client-side. For decode TPS use
    # (N - 1) tokens since the first token's generation is captured in TTFT.
    prompt_tps = server_prompt_tps if server_prompt_tps > 0 else (prompt_tokens / ttft if ttft > 0 else 0.0)
    if server_gen_tps > 0:
        generation_tps = server_gen_tps
    elif generation_time > 0 and completion_tokens > 1:
        generation_tps = (completion_tokens - 1) / generation_time
    else:
        generation_tps = 0.0

    tps_source = "server" if (server_prompt_tps > 0 or server_gen_tps > 0) else "client"

    print(f"  Prompt tokens:      {prompt_tokens}")
    print(f"  Completion tokens:  {completion_tokens}")
    if cached_tokens:
        print(f"  Cached tokens:      {cached_tokens} (KV reuse)")
    if reasoning_text:
        print(f"  Reasoning chars:    {len(reasoning_text)} (counted toward decode window)")
    print(f"  TTFT:               {ttft:.3f}s")
    print(f"  Generation time:    {generation_time:.2f}s")
    print(f"  Total time:         {total_time:.2f}s")
    print(f"  Prompt TPS:         {prompt_tps:.1f} t/s ({tps_source}-side)")
    print(f"  Generation TPS:     {generation_tps:.1f} t/s ({tps_source}-side)")

    result = {
        "context_size": context_file.stem,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": completion_tokens,
        "time_to_first_token": ttft,
        "eval_duration": generation_time,
        "prompt_eval_duration": ttft,
        "total_time": total_time,
        "prompt_tps": prompt_tps,
        "generation_tps": generation_tps,
        "generated_text": generated_text,
    }
    if reasoning_text:
        result["reasoning_text"] = reasoning_text
    if cached_tokens:
        result["cached_tokens"] = cached_tokens

    return common.add_throughput_metrics(result, prompt_text=prompt)


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark the Unsloth Studio local inference server across context sizes"
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Model ID to use (auto-detected from server if omitted)",
    )
    parser.add_argument(
        "--model",
        dest="model_flag",
        default=None,
        help="Model ID (alternative to positional; takes precedence if both set)",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8888/v1",
        help="Base URL of the Unsloth Studio server (default: http://127.0.0.1:8888/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("UNSLOTH_API_KEY", "no-key"),
        help="API key (default: UNSLOTH_API_KEY env var or 'no-key')",
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
        print(f"Error: Cannot reach Unsloth server at {base_url}")
        print("Make sure Unsloth Studio is running and the base URL + API key are correct.")
        return 1
    print("Connected successfully.")

    # Resolve model name (--model flag wins over positional, then auto-detect)
    model = args.model_flag or args.model
    if not model:
        model = get_available_model(client)
        if not model:
            print("Error: No model specified and could not auto-detect one from the server.")
            return 1
        print(f"Auto-detected model: {model}")

    hardware_info = common.mark_client_hardware(common.get_hardware_info(), base_url)
    hardware_str = common.format_hardware_string(hardware_info)

    print(f"\nUnsloth Studio Benchmark")
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

    output_dir = common.create_output_directory("unsloth", model, cold_prefill=args.cold_prefill)

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
                    common.save_generated_text(result, model, resp_path, "Unsloth")
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
                common.save_generated_text(result, model, resp_path, "Unsloth")

    total_benchmark_time = time.time() - benchmark_start

    if not results:
        print("\nNo successful benchmark results.")
        return 1

    common.save_all_outputs(
        results,
        output_dir,
        model,
        "Unsloth",
        hardware_info,
        args,
    )

    common.print_benchmark_summary(
        results,
        model,
        "Unsloth",
        hardware_info,
        output_dir,
        total_benchmark_time,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
