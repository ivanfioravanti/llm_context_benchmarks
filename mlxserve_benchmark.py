#!/usr/bin/env python3
"""
Benchmark script for the mlx-serve inference server.

mlx-serve (default port 11234) exposes an OpenAI-compatible API at ``/v1``.
Streaming and non-streaming chat completions carry a top-level ``timings``
object (sibling of ``usage``) with server-reported prefill/decode metrics:

    "timings": {
        "prompt_n": 18, "cached_n": 0, "prompt_ms": 369.266,
        "prompt_per_second": 48.745, "predicted_n": 4,
        "predicted_ms": 141.151, "predicted_per_second": 28.339,
        "tokenize_ms": 1.892
    }

The generic ``openai_benchmark.py`` does not read ``timings`` (it lives outside
``usage`` and uses different key names), hence the dedicated script — which
mirrors ``unsloth_benchmark.py`` (identical timings schema) and adds batch
(concurrent request) inference support from ``openai_benchmark.py``.

Usage:
    python mlxserve_benchmark.py
    python mlxserve_benchmark.py Qwen3.6-27B-MTPLX-Optimized-Quality
    python mlxserve_benchmark.py --base-url http://127.0.0.1:11234/v1
    python mlxserve_benchmark.py --contexts 2,4,8,16 --max-tokens 500
    python mlxserve_benchmark.py --batch-sizes 1,2,4 --no-batch
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
    """Create an OpenAI client pointed at the mlx-serve server."""
    return OpenAI(base_url=base_url, api_key=api_key)


def test_server_connection(client: OpenAI) -> bool:
    """Check that the server is reachable and returns a model list."""
    try:
        client.models.list()
        return True
    except Exception as e:
        print(f"Error connecting to mlx-serve: {e}")
        return False


def get_available_model(client: OpenAI) -> Optional[str]:
    """Return the first *loaded* model ID, falling back to the first listed.

    mlx-serve reports models with a ``loaded`` boolean and ``state`` field;
    prefer a loaded/ready model so auto-detect targets a usable one.
    """
    try:
        models = list(client.models.list())
        if not models:
            return None
        for m in models:
            if getattr(m, "loaded", False) or getattr(m, "state", "") == "ready":
                return m.id
        return models[0].id
    except Exception:
        return None


def _parse_timings(obj) -> Dict:
    """Extract the ``timings`` dict from a chunk or response object.

    ``timings`` is a top-level sibling of ``usage``; pydantic keeps it as
    model_extra and exposes it as a plain attribute on the SDK object.
    For raw httpx JSON dicts, it's a top-level key.
    """
    if isinstance(obj, dict):
        t = obj.get("timings")
        return t if isinstance(t, dict) else {}
    t = getattr(obj, "timings", None)
    return t if isinstance(t, dict) else {}


def run_benchmark(
    client: OpenAI,
    model: str,
    context_file: Path,
    max_tokens: int = 128,
    timeout: int = 3600,
    cold_prefill: bool = True,
    _run_idx: Optional[int] = None,
) -> Optional[Dict]:
    """Benchmark a single context file against the mlx-serve server.

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
        # `timings` is a top-level sibling of `usage` in the final chunk.
        t = _parse_timings(chunk)
        if t:
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
    cached_tokens = timings.get("cached_n", 0) if timings else 0

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

    Unlike the generic openai_benchmark.py version, this reads mlx-serve's
    top-level ``timings`` block from non-streaming responses to get per-request
    decode TPS (``predicted_per_second``) for accurate aggregate decode rate.
    """
    import concurrent.futures
    import statistics

    import tiktoken

    # Generate a fixed prompt of approximately prompt_tokens length
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        base_text = "The quick brown fox jumps over the lazy dog. "
        base_tokens = enc.encode(base_text)
        repeats = max(1, prompt_tokens // len(base_tokens))
        prompt_text = base_text * repeats
        tokens = enc.encode(prompt_text)[:prompt_tokens]
        prompt_text = enc.decode(tokens)
    except Exception:
        prompt_text = "The quick brown fox jumps over the lazy dog. " * (prompt_tokens // 10)

    def single_request() -> Dict:
        """Send one non-streaming request; return metrics from usage + timings."""
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
        timings = data.get("timings", {})
        return {
            "prompt_tps": timings.get("prompt_per_second", 0.0),
            "generation_tps": timings.get("predicted_per_second", 0.0),
            "prompt_tokens": usage.get("prompt_tokens", timings.get("prompt_n", 0)),
            "generation_tokens": usage.get("completion_tokens", timings.get("predicted_n", 0)),
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
        trial_decode_tps = []  # pure decode rate summed across clients (from server timings)

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

            trial_prompt_tps.append(agg_prompt_tps)
            trial_gen_tps.append(agg_gen_tps)

            # Pure decode throughput: mlx-serve reports each request's own
            # predicted_per_second (decode phase only). Summed across N parallel
            # clients this is the aggregate decode rate.
            per_request_decode = [r["generation_tps"] for r in responses if r.get("generation_tps")]
            if len(per_request_decode) == len(responses):
                trial_decode_tps.append(sum(per_request_decode))

            print(f"    Trial {trial + 1}: pp {agg_prompt_tps:.1f} tg {agg_gen_tps:.1f} t/s ({wall_time:.1f}s)")

        if trial_prompt_tps:
            avg_prompt = statistics.mean(trial_prompt_tps)
            avg_gen = statistics.mean(trial_gen_tps)
            result = {
                "batch_size": bs,
                "prompt_tps": round(avg_prompt, 2),
                "generation_tps": round(avg_gen, 2),
            }
            if trial_decode_tps:
                avg_decode = statistics.mean(trial_decode_tps)
                result["decode_tps_total"] = round(avg_decode, 2)
                result["decode_tps_per_client"] = round(avg_decode / bs, 2)

            print(f"  Avg: pp {avg_prompt:.1f} tg {avg_gen:.1f} t/s")
            batch_results.append(result)

    return batch_results


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Benchmark the mlx-serve inference server across context sizes")
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
        default="http://127.0.0.1:11234/v1",
        help="Base URL of the mlx-serve server (default: http://127.0.0.1:11234/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MLXSERVE_API_KEY", "no-key"),
        help="API key (default: MLXSERVE_API_KEY env var or 'no-key')",
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
        print(f"Error: Cannot reach mlx-serve at {base_url}")
        print("Make sure mlx-serve is running and the base URL is correct.")
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

    print("\nmlx-serve Benchmark")
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

    output_dir = common.create_output_directory("mlx_serve", model, cold_prefill=args.cold_prefill)

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
                    common.save_generated_text(result, model, resp_path, "MLX-Serve")
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
                common.save_generated_text(result, model, resp_path, "MLX-Serve")

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

    common.save_all_outputs(
        results,
        output_dir,
        model,
        "MLX-Serve",
        hardware_info,
        args,
        batch_results=batch_results,
    )

    common.print_benchmark_summary(
        results,
        model,
        "MLX-Serve",
        hardware_info,
        output_dir,
        total_benchmark_time,
        batch_results=batch_results,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
