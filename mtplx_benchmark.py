#!/usr/bin/env python3
"""Benchmark script for MTPLX (Apple Silicon, OpenAI-compatible).

MTPLX exposes a richer-than-OpenAI response: every chat/completions response
contains an ``mtplx_stats`` block with authoritative server-side timings
(``prefill_tok_s``, ``decode_tok_s``, ``ttft_s``, decode/verify/draft splits,
cache hit info, peak memory). We trust those over client-side timing.

For full cold-prefill numbers we also POST to ``/admin/cache/clear`` between
rows so the session bank doesn't carry KV across context sizes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import httpx

import benchmark_common as common

MTPLX_API_URL = "http://127.0.0.1:8000/v1"


def normalize_base_url(url: str) -> str:
    """Trim trailing slash and any ``/chat/completions`` suffix."""
    if not url:
        return MTPLX_API_URL
    normalized = url.strip().rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    return normalized


def server_root(base_url: str) -> str:
    """Return the server root (drops a trailing ``/v1`` if present)."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized[: -len("/v1")]
    return normalized


def test_server_connection(base_url: str, timeout: int = 10) -> Optional[Dict]:
    """Hit ``/health`` and return its JSON, or None on failure."""
    try:
        resp = httpx.get(f"{server_root(base_url)}/health", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"Error connecting to MTPLX at {base_url}: {exc}")
        return None


def list_models(base_url: str, timeout: int = 10) -> List[str]:
    """Return model IDs reported by ``/v1/models``."""
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]
    except Exception:
        return []


def clear_server_cache(base_url: str, timeout: int = 30) -> None:
    """Drop MTPLX's session bank so the next request gets a true cold prefill."""
    try:
        httpx.post(f"{server_root(base_url)}/admin/cache/clear", timeout=timeout)
    except Exception as exc:
        print(f"  Warning: cache clear failed: {exc}")


def call_mtplx(
    base_url: str,
    api_key: Optional[str],
    request_model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
    generation_mode: Optional[str] = None,
) -> Dict:
    """Send a non-streaming chat completion and return the parsed JSON."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": request_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    if generation_mode:
        payload["generation_mode"] = generation_mode

    resp = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def run_benchmark(
    model_name: str,
    context_file: Path,
    base_url: str,
    api_key: Optional[str],
    request_model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
    generation_mode: Optional[str] = None,
    cold_prefill: bool = True,
    clear_cache: bool = True,
    _run_idx: Optional[int] = None,
) -> Optional[Dict]:
    """Benchmark MTPLX for a single context file."""
    print(f"Running benchmark for {context_file}...")

    with open(context_file, "r") as handle:
        prompt = handle.read()

    if cold_prefill:
        prompt = common.make_cache_buster() + prompt
        if clear_cache:
            clear_server_cache(base_url)
    elif _run_idx is not None:
        prompt = common.make_cache_buster(run_idx=_run_idx) + prompt

    client_start = time.time()
    try:
        data = call_mtplx(
            base_url=base_url,
            api_key=api_key,
            request_model=request_model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            generation_mode=generation_mode,
        )
    except Exception as exc:
        print(f"Error contacting MTPLX API: {exc}")
        return None
    client_total = time.time() - client_start

    choices = data.get("choices", [])
    message = choices[0].get("message", {}) if choices else {}
    generated_text = message.get("content", "") or ""
    reasoning_text = message.get("reasoning_content", "") or ""

    usage = data.get("usage", {}) or {}
    stats = data.get("mtplx_stats", {}) or {}

    prompt_tokens = stats.get("prompt_tokens") or usage.get("prompt_tokens", 0) or 0
    generation_tokens = stats.get("completion_tokens") or usage.get("completion_tokens", 0) or 0
    cached_tokens = stats.get("cached_tokens", 0) or 0
    new_prefill_tokens = stats.get("new_prefill_tokens", prompt_tokens)

    prompt_eval_duration = stats.get("prompt_eval_time_s") or stats.get("ttft_s") or 0.0
    decode_elapsed = stats.get("decode_elapsed_s") or 0.0
    total_time = stats.get("request_elapsed_s") or stats.get("server_elapsed_s") or client_total
    ttft = stats.get("ttft_s") or prompt_eval_duration

    server_prefill_tps = stats.get("prefill_tok_s") or 0.0
    server_decode_tps = stats.get("decode_tok_s") or 0.0

    # Server stats are authoritative; only fall back to client-side math if missing.
    if server_prefill_tps > 0:
        prompt_tps = server_prefill_tps
    elif prompt_eval_duration > 0:
        prompt_tps = (new_prefill_tokens or prompt_tokens) / prompt_eval_duration
    else:
        prompt_tps = 0.0

    if server_decode_tps > 0:
        generation_tps = server_decode_tps
    elif decode_elapsed > 0:
        generation_tps = generation_tokens / decode_elapsed
    else:
        generation_tps = 0.0

    eval_duration = decode_elapsed if decode_elapsed > 0 else max(total_time - prompt_eval_duration, 0.0)

    peak_memory_bytes = stats.get("peak_memory_bytes", 0) or 0
    peak_memory_gb = peak_memory_bytes / (1024**3) if peak_memory_bytes else 0.0

    print(f"  Prompt tokens:       {prompt_tokens}")
    if cached_tokens:
        print(f"  Cached tokens:       {cached_tokens}")
    if new_prefill_tokens and new_prefill_tokens != prompt_tokens:
        print(f"  New prefill tokens:  {new_prefill_tokens}")
    print(f"  Generation tokens:   {generation_tokens}")
    print(f"  TTFT:                {ttft:.3f}s")
    print(f"  Prefill time:        {prompt_eval_duration:.3f}s")
    print(f"  Decode time:         {decode_elapsed:.3f}s")
    print(f"  Total time:          {total_time:.2f}s")
    print(f"  Prompt TPS:          {prompt_tps:.1f} t/s (server-side)")
    print(f"  Generation TPS:      {generation_tps:.1f} t/s (server-side)")
    if peak_memory_gb > 0:
        print(f"  Peak memory:         {peak_memory_gb:.2f} GB")

    accepted = stats.get("accepted_by_depth")
    if stats.get("mtp_depth") is not None:
        print(
            f"  MTP depth {stats.get('mtp_depth')}, verify_calls={stats.get('verify_calls')}, "
            f"drafted={stats.get('drafted_tokens', 0)}, accepted={stats.get('accepted_drafts', 0)}, "
            f"corrections={stats.get('correction_tokens', 0)}"
        )

    result: Dict[str, object] = {
        "context_size": context_file.stem,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "prompt_tps": prompt_tps,
        "generation_tps": generation_tps,
        "total_time": total_time,
        "eval_duration": eval_duration,
        "prompt_eval_duration": prompt_eval_duration,
        "time_to_first_token": ttft,
        "generated_text": generated_text,
        "peak_memory_gb": peak_memory_gb,
        "cached_tokens": cached_tokens,
        "new_prefill_tokens": new_prefill_tokens,
        "session_cache_hit": stats.get("session_cache_hit", False),
        "cache_miss_reason": stats.get("cache_miss_reason"),
    }

    # MTP-specific extras (kept under explicit keys so they survive CSV export
    # only if benchmark_common knows about them; otherwise they're available
    # in the raw JSON dump).
    for key in (
        "mtp_depth",
        "verify_calls",
        "drafted_tokens",
        "accepted_drafts",
        "rejected_drafts",
        "correction_tokens",
        "bonus_tokens",
        "verify_time_s",
        "draft_time_s",
        "accept_time_s",
        "repair_time_s",
        "sliding_decode_tok_s_first_32",
        "sliding_decode_tok_s_last_32",
        "generation_mode",
    ):
        if key in stats:
            result[key] = stats[key]
    if accepted is not None:
        result["accepted_by_depth"] = accepted

    if reasoning_text:
        result["reasoning_text"] = reasoning_text

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MTPLX benchmarks using the OpenAI-compatible API")
    parser.add_argument(
        "model",
        nargs="?",
        help="Model id (auto-detected from /v1/models if omitted)",
    )

    common.setup_common_args(parser)

    parser.add_argument(
        "--base-url",
        default=MTPLX_API_URL,
        help=f"MTPLX API endpoint (default: {MTPLX_API_URL})",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (MTPLX local servers usually don't need one)",
    )
    parser.add_argument(
        "--request-model",
        default=None,
        help="Model identifier sent to the API (defaults to the positional model)",
    )
    parser.add_argument(
        "--generation-mode",
        choices=["mtp", "ar"],
        default=None,
        help="Generation mode: 'mtp' (multi-token prediction, default on server) "
        "or 'ar' (autoregressive / no speculation). When omitted, the server's "
        "default is used.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6 — matches MTPLX speed profile)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p value (default: 0.95)",
    )
    parser.add_argument(
        "--cold-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prepend a unique marker to every prompt so MTPLX's session bank "
        "treats each call as a new prefix (default: enabled). Use --no-cold-prefill "
        "to allow session-cache reuse across rows.",
    )
    parser.add_argument(
        "--clear-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="POST /admin/cache/clear before each cold-prefill row (default: enabled). "
        "Has no effect when --no-cold-prefill is set.",
    )

    args = parser.parse_args()

    base_url = normalize_base_url(args.base_url)

    print(f"\nTesting connection to {base_url} ...")
    health = test_server_connection(base_url)
    if not health or not health.get("ok"):
        print(f"Error: MTPLX server not reachable at {base_url}")
        return 1

    server_model = health.get("model")
    print(f"Connected. Server model: {server_model}")
    print(f"Generation mode: {health.get('generation_mode')}  Profile: {(health.get('profile') or {}).get('name')}")

    model = args.model
    if not model:
        models = list_models(base_url)
        if not models:
            model = server_model
        else:
            model = models[0]
        if not model:
            print("Error: No model specified and could not auto-detect one from MTPLX.")
            return 1
        print(f"Auto-detected model: {model}")

    request_model = args.request_model or model

    context_files = common.find_context_files(args.contexts)
    if not context_files:
        return 1

    print("\nCollecting hardware information...")
    hardware_info = common.get_hardware_info()
    hardware_str = common.format_hardware_string(hardware_info)
    print(f"Hardware: {hardware_str}")
    hardware_info["api_endpoint"] = base_url
    hardware_info["api_model"] = model
    if request_model != model:
        hardware_info["api_request_model"] = request_model
    if health.get("generation_mode"):
        hardware_info["mtplx_generation_mode"] = health.get("generation_mode")
    if args.generation_mode:
        hardware_info["mtplx_forced_generation_mode"] = args.generation_mode
    if (health.get("profile") or {}).get("name"):
        hardware_info["mtplx_profile"] = health["profile"]["name"]

    print("\nConnection details:")
    print(f"Endpoint:    {base_url}")
    print(f"Model:       {model}")
    if request_model != model:
        print(f"Request model: {request_model}")
    print(f"Max tokens:  {args.max_tokens}")
    print(f"Gen mode:    {args.generation_mode or 'server default'}")
    print(
        f"Cold prefill: {'enabled (cache busted per prompt' if args.cold_prefill else 'disabled (cache reuse allowed'}"
        f"{', server cache cleared per row)' if (args.cold_prefill and args.clear_cache) else ')'}"
    )

    gen_mode_tag = f"-{args.generation_mode}" if args.generation_mode else f"-{health.get('generation_mode', 'mtp')}"
    output_dir = common.create_output_directory("mtplx", f"{model}{gen_mode_tag}", cold_prefill=args.cold_prefill)

    # Warmup
    warmup_file = common.find_warmup_file()
    if warmup_file:
        print(f"\n{'=' * 50}")
        print(f"Warmup run (excluded from results): {warmup_file.name}")
        print(f"{'=' * 50}")
        run_benchmark(
            model_name=model,
            context_file=warmup_file,
            base_url=base_url,
            api_key=args.api_key,
            request_model=request_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            generation_mode=args.generation_mode,
            cold_prefill=args.cold_prefill,
            clear_cache=args.clear_cache,
        )
        print("Warmup complete.")
    else:
        print("Warning: 0.5k.txt not found, skipping warmup.")

    results: List[Dict] = []
    benchmark_start = time.time()

    if args.cold_prefill:
        for context_file in context_files:
            print("\n" + "=" * 50)
            print(f"Benchmarking {context_file.name}...")
            print("=" * 50)

            result = common.run_benchmark_peak(
                run_benchmark,
                model_name=model,
                context_file=context_file,
                base_url=base_url,
                api_key=args.api_key,
                request_model=request_model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.timeout,
                generation_mode=args.generation_mode,
                cold_prefill=args.cold_prefill,
                clear_cache=args.clear_cache,
                n_runs=args.runs,
            )
            if result:
                results.append(result)
                if args.save_responses:
                    response_path = output_dir / f"response_{result['context_size']}.txt"
                    common.save_generated_text(result, model, response_path, "MTPLX API")
    else:
        results = common.run_benchmark_peak_per_run(
            run_benchmark,
            context_files=context_files,
            n_runs=args.runs,
            model_name=model,
            base_url=base_url,
            api_key=args.api_key,
            request_model=request_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            generation_mode=args.generation_mode,
            cold_prefill=args.cold_prefill,
            clear_cache=args.clear_cache,
        )
        if args.save_responses:
            for result in results:
                response_path = output_dir / f"response_{result['context_size']}.txt"
                common.save_generated_text(result, model, response_path, "MTPLX API")

    if not results:
        print("\nNo successful benchmark results")
        return 1

    total_benchmark_time = time.time() - benchmark_start

    has_memory = any(r.get("peak_memory_gb", 0) > 0 for r in results)
    common.save_all_outputs(
        results,
        output_dir,
        model,
        "MTPLX API",
        hardware_info,
        args,
        include_memory=has_memory,
    )
    common.print_benchmark_summary(
        results,
        model,
        "MTPLX API",
        hardware_info,
        output_dir,
        total_benchmark_time,
    )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
