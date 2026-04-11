#!/usr/bin/env python3
"""Benchmark script for vMLX (MLX Studio) using OpenAI-compatible API."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from openai import OpenAI

import benchmark_common as common

VMLX_API_URL = "http://127.0.0.1:8001/v1"


def _make_cache_buster(run_idx: Optional[int] = None) -> str:
    """Generate a prefix to bust vMLX's prefix KV cache.

    vMLX uses a tiered prefix cache. When a new prompt shares a prefix with a
    previous request, cached tokens are reused, making prompt_eval_duration
    cover only the uncached delta while prompt_tokens reports the full length —
    inflating derived prompt t/s.

    When ``run_idx`` is provided, the buster is deterministic per run index:
    all calls within the same run share the same prefix (so KV cache carries
    over across context sizes), while different runs get different prefixes
    (so runs don't interfere). When ``run_idx`` is None, a random UUID is
    used for full cold-prefill busting. ~10 tokens of overhead per prompt.
    """
    if run_idx is not None:
        return f"[run-{run_idx}]\n"
    import uuid

    return f"[session-{uuid.uuid4().hex[:16]}]\n"


def ensure_endpoint(url: str) -> str:
    """Normalize a vMLX endpoint for OpenAI-compatible clients."""

    if not url:
        return VMLX_API_URL

    normalized = url.strip().rstrip("/")

    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]

    return normalized


def call_vmlx(
    client: OpenAI,
    request_model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
) -> Dict:
    """Send a non-streaming chat completion request to vMLX."""

    response = client.chat.completions.create(
        model=request_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
        stream=False,
    )

    return {
        "choices": [
            {
                "message": {
                    "content": response.choices[0].message.content or "",
                    "reasoning_content": getattr(response.choices[0].message, "reasoning_content", ""),
                }
            }
        ],
        "usage": response.usage.model_dump() if response.usage else {},
    }


def call_vmlx_streaming(
    client: OpenAI,
    request_model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
) -> Dict[str, object]:
    """Stream a chat completion from vMLX, capturing time-to-first-token."""

    message_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: Dict[str, int] = {}

    start_time = time.time()
    first_token_time: Optional[float] = None

    stream = client.chat.completions.create(
        model=request_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        chunk_choices = getattr(chunk, "choices", None)
        if chunk_choices:
            delta = chunk_choices[0].delta

            content = getattr(delta, "content", None)
            if content:
                if first_token_time is None:
                    first_token_time = time.time()
                message_parts.append(content)

            reasoning_delta = getattr(delta, "reasoning_content", None)
            if reasoning_delta:
                if first_token_time is None:
                    first_token_time = time.time()
                if isinstance(reasoning_delta, list):
                    for item in reasoning_delta:
                        if isinstance(item, dict):
                            text = item.get("text") or item.get("content")
                            if text:
                                reasoning_parts.append(text)
                        elif isinstance(item, str):
                            reasoning_parts.append(item)
                elif isinstance(reasoning_delta, str):
                    reasoning_parts.append(reasoning_delta)

        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage:
            if hasattr(chunk_usage, "model_dump"):
                usage = chunk_usage.model_dump()
            else:
                usage = {
                    "prompt_tokens": getattr(chunk_usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(chunk_usage, "completion_tokens", 0),
                    "total_tokens": getattr(chunk_usage, "total_tokens", 0),
                }

    total_time = time.time() - start_time
    prompt_eval_duration = (first_token_time - start_time) if first_token_time else 0.0

    # Extract cached tokens from prompt_tokens_details (vMLX reports KV cache
    # hits via this field). Prompt throughput should be based on fresh tokens
    # only — cached tokens skip prefill entirely, so including them inflates
    # prompt TPS by orders of magnitude when running sequential benchmarks
    # over files that share a common prefix.
    prompt_tokens_details = usage.get("prompt_tokens_details") or {}
    if isinstance(prompt_tokens_details, dict):
        cached_tokens = prompt_tokens_details.get("cached_tokens", 0) or 0
    else:
        cached_tokens = getattr(prompt_tokens_details, "cached_tokens", 0) or 0

    return {
        "generated_text": "".join(message_parts),
        "reasoning_text": "".join(reasoning_parts),
        "usage": usage,
        "total_time": total_time,
        "prompt_eval_duration": prompt_eval_duration,
        "cached_tokens": cached_tokens,
    }


def run_benchmark(
    model_name: str,
    context_file: Path,
    client: OpenAI,
    request_model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
    stream: bool = True,
    cold_prefill: bool = True,
    _run_idx: Optional[int] = None,
) -> Optional[Dict]:
    """Benchmark vMLX for a given context file."""
    print(f"Running benchmark for {context_file}...")

    with open(context_file, "r") as handle:
        prompt = handle.read()

    # Bust vMLX's KV cache by prepending a unique marker.
    # - cold_prefill: random UUID per call → every row is fully cold prefill
    # - _run_idx (no cold_prefill): deterministic per run index → same run
    #   shares a prefix (KV cache carries over), different runs are isolated
    # - neither: no buster → raw server KV cache behavior
    if cold_prefill:
        prompt = _make_cache_buster() + prompt
    elif _run_idx is not None:
        prompt = _make_cache_buster(run_idx=_run_idx) + prompt

    try:
        cached_tokens = 0
        if stream:
            stream_result = call_vmlx_streaming(
                client=client,
                request_model=request_model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout=timeout,
            )
            data = {
                "choices": [],
                "usage": stream_result.get("usage", {}),
            }
            generated_text = stream_result.get("generated_text", "")
            reasoning_text = stream_result.get("reasoning_text", "")
            total_time = float(stream_result.get("total_time", 0.0))
            prompt_eval_duration = float(stream_result.get("prompt_eval_duration", 0.0))
            cached_tokens = int(stream_result.get("cached_tokens", 0))
        else:
            start_time = time.time()
            data = call_vmlx(
                client=client,
                request_model=request_model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout=timeout,
            )
            total_time = time.time() - start_time
            prompt_eval_duration = 0.0
            choices = data.get("choices", [])
            message = choices[0].get("message", {}) if choices else {}
            generated_text = message.get("content", "")
            reasoning_text = message.get("reasoning_content", "")
    except Exception as exc:
        print(f"Error contacting vMLX API: {exc}")
        return None

    usage = data.get("usage", {})
    prompt_tokens = usage.get(
        "prompt_tokens",
        usage.get("prompt_tokens_total", usage.get("input_tokens", 0)),
    )
    generation_tokens = usage.get(
        "completion_tokens",
        usage.get("output_tokens", usage.get("response_tokens", 0)),
    )
    total_tokens = usage.get("total_tokens", prompt_tokens + generation_tokens)

    if generation_tokens <= 0 and total_tokens and prompt_tokens:
        inferred = total_tokens - prompt_tokens
        if inferred > 0:
            generation_tokens = inferred
    reasoning_tokens = usage.get("reasoning_tokens")

    if stream and not usage:
        print("Warning: Streamed response did not include token usage; throughput metrics may be zero.")

    # Prompt throughput is based on fresh (non-cached) tokens only.
    # Cached tokens skip prefill entirely, so including them inflates
    # prompt TPS by orders of magnitude when running sequential benchmarks
    # over files that share a common prefix.
    fresh_prompt_tokens = prompt_tokens - cached_tokens

    generation_duration = max(total_time - prompt_eval_duration, 0.0)
    eval_duration = generation_duration if generation_duration > 0 else total_time

    prompt_tps = (
        fresh_prompt_tokens / prompt_eval_duration if prompt_eval_duration > 0 and fresh_prompt_tokens > 0 else 0.0
    )
    generation_tps = generation_tokens / generation_duration if generation_duration and generation_duration > 0 else 0.0

    print(f"  Prompt tokens: {prompt_tokens}")
    if cached_tokens > 0:
        print(f"  Cached tokens: {cached_tokens} (excluded from prompt throughput)")
        print(f"  Fresh prompt tokens: {fresh_prompt_tokens}")
    print(f"  Generation tokens: {generation_tokens}")
    if reasoning_tokens:
        print(f"  Reasoning tokens: {reasoning_tokens}")
    print(f"  Total tokens: {total_tokens}")
    if prompt_eval_duration > 0:
        print(f"  Time to first token: {prompt_eval_duration:.2f}s")
        if fresh_prompt_tokens > 0:
            print(f"  Prompt throughput: {prompt_tps:.2f} tokens/sec (fresh tokens only)")
    print(f"  Generation throughput: {generation_tps:.2f} tokens/sec")
    print(f"  Total time: {total_time:.2f}s")

    result: Dict[str, object] = {
        "context_size": context_file.stem,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "fresh_prompt_tokens": fresh_prompt_tokens,
        "generation_tokens": generation_tokens,
        "prompt_tps": prompt_tps,
        "generation_tps": generation_tps,
        "total_time": total_time,
        "eval_duration": eval_duration,
        "prompt_eval_duration": prompt_eval_duration,
        "time_to_first_token": prompt_eval_duration,
        "generated_text": generated_text,
        "total_tokens": total_tokens,
    }

    if reasoning_tokens is not None:
        result["reasoning_tokens"] = reasoning_tokens
    if reasoning_text:
        result["reasoning_text"] = reasoning_text

    return result


def run_batch_benchmark(
    base_url: str,
    api_key: str,
    model: str,
    batch_sizes: list,
    prompt_tokens: int = 2048,
    gen_tokens: int = 128,
    num_trials: int = 3,
    timeout: int = 3600,
) -> list:
    """Run batch benchmark by sending concurrent requests to test continuous batching.

    vMLX supports continuous batching via mlx-lm's BatchGenerator. This measures
    aggregate throughput under N concurrent clients — the server analog of
    mlx_lm.batch_generate.

    Args:
        base_url: vMLX server base URL (ending in /v1)
        api_key: API key for authentication
        model: Model name
        batch_sizes: Concurrency levels to test
        prompt_tokens: Approximate prompt tokens per request
        gen_tokens: Tokens to generate per request
        num_trials: Trials per batch size (averaged)
        timeout: Per-request timeout

    Returns:
        List of result dicts with batch_size, prompt_tps, generation_tps, peak_memory_gb
    """
    import concurrent.futures
    import json
    import statistics

    import httpx

    # Build a synthetic prompt of approximately prompt_tokens length. Same
    # tiktoken cl100k_base encoding as the other batch benchmarks for
    # cross-provider comparability.
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        base_text = "The quick brown fox jumps over the lazy dog. "
        base_tokens = enc.encode(base_text)
        repeats = max(1, prompt_tokens // len(base_tokens))
        prompt_text = base_text * repeats
        tokens = enc.encode(prompt_text)[:prompt_tokens]
        prompt_text = enc.decode(tokens)
    except Exception:
        # Fallback: approximate ~4 chars per token
        prompt_text = "The quick brown fox jumps over the lazy dog. " * (prompt_tokens // 10)

    def single_request() -> dict:
        """Send one streaming request and return usage info and per-phase timing.

        Extracts cached_tokens from vMLX's prompt_tokens_details and computes
        fresh prompt tokens for accurate throughput measurement.
        """
        start = time.time()
        first_token_time = None

        # Prepend a cache buster so each concurrent request gets cold prefill.
        cache_buster = _make_cache_buster()

        with httpx.Client(timeout=timeout) as http:
            with http.stream(
                "POST",
                f"{base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": cache_buster + prompt_text}],
                    "max_tokens": gen_tokens,
                    "temperature": 0.7,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
                headers={"Authorization": f"Bearer {api_key}"},
            ) as resp:
                resp.raise_for_status()
                usage = {}
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[len("data: ") :]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except Exception:
                        continue

                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content and first_token_time is None:
                            first_token_time = time.time()

                    chunk_usage = chunk.get("usage")
                    if chunk_usage:
                        usage = chunk_usage

        total_time = time.time() - start
        ttft = (first_token_time - start) if first_token_time else 0.0
        decode_time = max(total_time - ttft, 0.0)

        # Extract cached tokens from vMLX's prompt_tokens_details.
        prompt_details = usage.get("prompt_tokens_details") or {}
        if isinstance(prompt_details, dict):
            cached_tokens = prompt_details.get("cached_tokens", 0) or 0
        else:
            cached_tokens = getattr(prompt_details, "cached_tokens", 0) or 0

        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "cached_tokens": cached_tokens,
            "generation_tokens": usage.get("completion_tokens", 0),
            "ttft": ttft,
            "decode_time": decode_time,
        }

    batch_results = []

    for bs in batch_sizes:
        print(
            f"\n  Batch size {bs} ({num_trials} trials, ~{prompt_tokens} prompt tokens, " f"{gen_tokens} gen tokens)..."
        )

        # Warmup at this batch size so the server has allocated all slots and
        # any one-time per-slot overhead is paid before we start timing.
        print("    Warmup...")
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as pool:
                list(pool.map(lambda _: single_request(), range(bs)))
        except Exception as e:
            print(f"    Warmup failed for batch size {bs}: {e} — skipping")
            continue

        trial_prompt_tps = []
        trial_gen_tps = []

        for trial in range(num_trials):
            try:
                start = time.time()
                with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as pool:
                    futures = [pool.submit(single_request) for _ in range(bs)]
                    responses = [f.result() for f in futures]
                wall_time = time.time() - start
            except Exception as e:
                print(f"    Trial {trial + 1} failed: {e}")
                continue

            total_prompt_tok = sum(r["prompt_tokens"] for r in responses)
            total_cached_tok = sum(r["cached_tokens"] for r in responses)
            total_fresh_tok = total_prompt_tok - total_cached_tok
            total_gen_tok = sum(r["generation_tokens"] for r in responses)
            # Split wall_time into prefill and decode portions using per-request
            # phase ratios. This gives correct aggregate throughput for both
            # serial and concurrent execution:
            #   Serial:  wall=N*req_time → split phase ≈ N*phase → tps ≈ per-req tps
            #   Concurrent: wall≈req_time → split phase ≈ phase → tps = batched tps
            sum_prefill = sum(r["ttft"] for r in responses)
            sum_decode = sum(r["decode_time"] for r in responses)
            sum_total = sum_prefill + sum_decode
            if sum_total > 0:
                prefill_wall = wall_time * (sum_prefill / sum_total)
                decode_wall = wall_time * (sum_decode / sum_total)
            else:
                prefill_wall = wall_time
                decode_wall = wall_time
            # Guard against zero from rounding
            prefill_wall = max(prefill_wall, 0.001)
            decode_wall = max(decode_wall, 0.001)
            # Prompt throughput uses fresh (non-cached) tokens only, matching
            # how run_benchmark() calculates it.
            agg_prompt_tps = total_fresh_tok / prefill_wall if total_fresh_tok > 0 else total_prompt_tok / prefill_wall
            agg_gen_tps = total_gen_tok / decode_wall

            trial_prompt_tps.append(agg_prompt_tps)
            trial_gen_tps.append(agg_gen_tps)

            cached_pct = (total_cached_tok / total_prompt_tok * 100) if total_prompt_tok > 0 else 0.0
            print(
                f"    Trial {trial + 1}: pp {agg_prompt_tps:.1f} tg {agg_gen_tps:.1f} t/s "
                f"(wall {wall_time:.1f}s, prefill {prefill_wall:.1f}s, decode {decode_wall:.1f}s"
                f"{f', cached {total_cached_tok} ({cached_pct:.0f}%)' if total_cached_tok > 0 else ''})"
            )

        if trial_prompt_tps:
            avg_prompt = statistics.mean(trial_prompt_tps)
            avg_gen = statistics.mean(trial_gen_tps)
            print(f"  Avg: pp {avg_prompt:.1f} tg {avg_gen:.1f} t/s")
            batch_results.append(
                {
                    "batch_size": bs,
                    "prompt_tps": round(avg_prompt, 2),
                    "generation_tps": round(avg_gen, 2),
                    "peak_memory_gb": 0.0,
                }
            )

    return batch_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run vMLX (MLX Studio) benchmarks using OpenAI-compatible API")
    parser.add_argument("model", help="Model id (e.g., mlx-community/Qwen3-8B-4bit)")

    common.setup_common_args(parser)

    parser.add_argument(
        "--api-key",
        help="API key (defaults to VMLX_API_KEY environment variable)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p value (default: 0.95)",
    )
    parser.add_argument(
        "--base-url",
        default=VMLX_API_URL,
        help="Override vMLX API endpoint (default: http://127.0.0.1:8001/v1)",
    )
    parser.add_argument(
        "--request-model",
        help="Model identifier sent to the API (defaults to the positional model)",
    )
    parser.add_argument(
        "--stream",
        dest="stream",
        action="store_true",
        help="Stream responses to measure time-to-first-token (default)",
    )
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Disable streaming responses (prompt TPS will be 0)",
    )
    parser.set_defaults(stream=True)

    parser.add_argument(
        "--cold-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prepend a unique marker to every prompt to bust vMLX's prefix "
        "KV cache, forcing cold prefill on every row (default: enabled; "
        "use --no-cold-prefill for cached/warm-reuse numbers)",
    )

    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8",
        help="Comma-separated batch sizes for concurrent-request benchmark (default: 1,2,4,8)",
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

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("VMLX_API_KEY")
    if not api_key:
        api_key = "local"
        print(
            "No API key provided; using placeholder key. Set --api-key or VMLX_API_KEY if the server requires authentication."
        )

    base_url = ensure_endpoint(args.base_url)
    request_model = args.request_model or args.model

    context_files = common.find_context_files(args.contexts)
    if not context_files:
        return 1

    print("\nCollecting hardware information...")
    hardware_info = common.get_hardware_info()
    hardware_str = common.format_hardware_string(hardware_info)
    print(f"Hardware: {hardware_str}")

    hardware_info["api_endpoint"] = base_url
    hardware_info["api_model"] = args.model
    if request_model != args.model:
        hardware_info["api_request_model"] = request_model

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception as exc:
        print(f"Error initializing vMLX client: {exc}")
        return 1

    print("\nConnection details:")
    print(f"Endpoint: {base_url}")
    print(f"Model: {args.model}")
    if request_model != args.model:
        print(f"Request model: {request_model}")
    print(f"Max tokens: {args.max_tokens}")
    print(
        f"Cold prefill: {'enabled (cache busted per prompt)' if args.cold_prefill else 'disabled (cache reuse allowed)'}"
    )

    output_dir = common.create_output_directory("vmlx", args.model, cold_prefill=args.cold_prefill)

    # Warmup run
    warmup_file = common.find_warmup_file()
    if warmup_file:
        print(f"\n{'=' * 50}")
        print(f"Warmup run (excluded from results): {warmup_file.name}")
        print(f"{'=' * 50}")
        run_benchmark(
            model_name=args.model,
            context_file=warmup_file,
            client=client,
            request_model=request_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            stream=args.stream,
            cold_prefill=args.cold_prefill,
        )
        print("Warmup complete.")
    else:
        print("Warning: 0.5k.txt not found, skipping warmup.")

    results = []
    benchmark_start = time.time()

    if args.cold_prefill:
        # Cold prefill: cache buster per-call, order doesn't matter.
        for context_file in context_files:
            print("\n" + "=" * 50)
            print(f"Benchmarking {context_file.name}...")
            print("=" * 50)

            result = common.run_benchmark_peak(
                run_benchmark,
                model_name=args.model,
                context_file=context_file,
                client=client,
                request_model=request_model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.timeout,
                stream=args.stream,
                cold_prefill=args.cold_prefill,
                n_runs=args.runs,
            )

            if result:
                results.append(result)
                if args.save_responses:
                    response_path = output_dir / f"response_{result['context_size']}.txt"
                    common.save_generated_text(result, args.model, response_path, "vMLX API")
    else:
        # Warm/cached: each run completes all context sizes before the next
        # starts, so KV cache accumulates within a run (run 1: 1k→2k→4k→…,
        # then run 2: 1k→2k→4k→…, etc.).  Deterministic per-run prefixes
        # keep runs cache-isolated.
        results = common.run_benchmark_peak_per_run(
            run_benchmark,
            context_files=context_files,
            n_runs=args.runs,
            model_name=args.model,
            client=client,
            request_model=request_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            stream=args.stream,
            cold_prefill=args.cold_prefill,
        )
        if args.save_responses:
            for result in results:
                response_path = output_dir / f"response_{result['context_size']}.txt"
                common.save_generated_text(result, args.model, response_path, "vMLX API")

    if not results:
        print("\nNo successful benchmark results")
        return 1

    total_benchmark_time = time.time() - benchmark_start

    # Run batch benchmark (concurrent requests to test continuous batching)
    batch_results = None
    if not args.no_batch:
        batch_sizes = [int(s.strip()) for s in args.batch_sizes.split(",")]
        print(f"\nRunning batch benchmark (concurrent requests: {batch_sizes})...")
        try:
            batch_results = run_batch_benchmark(
                base_url,
                api_key,
                request_model,
                batch_sizes,
                prompt_tokens=args.batch_prompt_tokens,
                gen_tokens=args.batch_gen_tokens,
                num_trials=args.batch_trials,
                timeout=args.timeout,
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
        args.model,
        "vMLX API",
        hardware_info,
        args,
        include_memory=True,
        batch_results=batch_results,
    )
    common.print_benchmark_summary(
        results,
        args.model,
        "vMLX API",
        hardware_info,
        output_dir,
        total_benchmark_time,
        batch_results=batch_results,
    )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
