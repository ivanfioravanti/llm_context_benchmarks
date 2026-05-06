#!/usr/bin/env python3
"""Benchmark script for dflash-mlx (Apple Silicon, OpenAI-compatible).

dflash-mlx exposes a standard OpenAI-compatible chat/completions endpoint.
Server-side timings are available via ``GET /metrics`` after each request,
under ``last_request``.  Token counts come from the response ``usage`` block
which is authoritative; ``/metrics`` fields supplement them when populated.

No ``/admin/cache/clear`` endpoint exists, so cold-prefill uses cache busters
only.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import httpx

import benchmark_common as common

DFLASH_API_URL = "http://127.0.0.1:8000/v1"


def normalize_base_url(url: str) -> str:
    """Trim trailing slash and any ``/chat/completions`` suffix."""
    if not url:
        return DFLASH_API_URL
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
        print(f"Error connecting to dflash-mlx at {base_url}: {exc}")
        return None


def get_metrics(base_url: str, timeout: int = 10) -> Optional[Dict]:
    """Fetch ``/metrics`` and return the parsed JSON, or None on failure."""
    try:
        resp = httpx.get(f"{server_root(base_url)}/metrics", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def list_models(base_url: str, timeout: int = 10) -> List[str]:
    """Return model IDs reported by ``/v1/models``."""
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]
    except Exception:
        return []


def call_dflash(
    base_url: str,
    api_key: Optional[str],
    request_model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
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

    resp = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def save_dflash_response(result: Dict, model_name: str, output_path: Path) -> None:
    """Save dflash response, including ``reasoning`` when ``content`` is empty.

    Reasoning models can hit ``max_tokens`` mid-thought and emit no final
    ``content``; ``benchmark_common.save_generated_text`` would write a
    blank file in that case, hiding the fact that the server actually
    produced output.
    """
    common.save_generated_text(result, model_name, output_path, "dflash-mlx API")
    reasoning_text = result.get("reasoning_text") or ""
    if not reasoning_text:
        return
    with open(output_path, "a") as handle:
        if not result.get("generated_text"):
            handle.write(
                "[no `content` returned — generation hit max_tokens during reasoning]\n\n"
            )
        handle.write("--- reasoning ---\n")
        handle.write(reasoning_text)


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
    cold_prefill: bool = True,
    _run_idx: Optional[int] = None,
) -> Optional[Dict]:
    """Benchmark dflash-mlx for a single context file."""
    print(f"Running benchmark for {context_file}...")

    with open(context_file, "r") as handle:
        prompt = handle.read()

    if cold_prefill:
        prompt = common.make_cache_buster() + prompt
    elif _run_idx is not None:
        prompt = common.make_cache_buster(run_idx=_run_idx) + prompt

    client_start = time.time()
    try:
        data = call_dflash(
            base_url=base_url,
            api_key=api_key,
            request_model=request_model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
        )
    except Exception as exc:
        print(f"Error contacting dflash-mlx API: {exc}")
        return None
    client_total = time.time() - client_start

    # Fetch server-side metrics immediately after the request
    metrics = get_metrics(base_url)
    last_req = (metrics or {}).get("last_request") or {}
    mem_info = (metrics or {}).get("memory") or {}

    choices = data.get("choices", [])
    message = choices[0].get("message", {}) if choices else {}
    generated_text = message.get("content", "") or ""
    reasoning_text = message.get("reasoning", "") or ""

    usage = data.get("usage", {}) or {}

    # Token counts from usage (authoritative)
    prompt_tokens = usage.get("prompt_tokens", 0) or 0
    generation_tokens = usage.get("completion_tokens", 0) or 0
    cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0

    # Server-side timings from /metrics last_request
    server_wall = last_req.get("wall_s") or client_total
    server_prefill_tps_physical = last_req.get("prefill_tok_s_physical") or 0.0
    server_prefill_tps_apparent = last_req.get("prefill_tok_s_apparent") or 0.0
    server_decode_tps = last_req.get("decode_tok_s") or 0.0
    prefill_s = last_req.get("prefill_s") or 0.0
    decode_s = last_req.get("decode_s") or 0.0
    mode_used = last_req.get("mode_used") or "unknown"

    # Derive timings
    ttft = prefill_s if prefill_s > 0 else 0.0
    total_time = server_wall if server_wall > 0 else client_total

    # Prompt TPS: prefer server-side, fall back to client math
    if server_prefill_tps_apparent > 0:
        prompt_tps = server_prefill_tps_apparent
    elif server_prefill_tps_physical > 0:
        prompt_tps = server_prefill_tps_physical
    elif prefill_s > 0 and prompt_tokens > 0:
        prompt_tps = prompt_tokens / prefill_s
    else:
        prompt_tps = 0.0

    # Generation TPS: prefer server-side, fall back to client math
    if server_decode_tps > 0:
        generation_tps = server_decode_tps
    elif decode_s > 0 and generation_tokens > 0:
        generation_tps = generation_tokens / decode_s
    elif total_time > ttft and generation_tokens > 0:
        generation_tps = generation_tokens / (total_time - ttft)
    else:
        generation_tps = 0.0

    prompt_eval_duration = prefill_s if prefill_s > 0 else ttft
    eval_duration = decode_s if decode_s > 0 else max(total_time - prompt_eval_duration, 0.0)

    peak_memory_gb = mem_info.get("mlx_peak_gb") or mem_info.get("rss_peak_gb") or 0.0

    print(f"  Prompt tokens:       {prompt_tokens}")
    if cached_tokens:
        print(f"  Cached tokens:       {cached_tokens}")
    print(f"  Generation tokens:   {generation_tokens}")
    print(f"  Mode used:           {mode_used}")
    print(f"  TTFT:                {ttft:.3f}s")
    print(f"  Prefill time:        {prefill_s:.3f}s")
    print(f"  Decode time:         {decode_s:.3f}s")
    print(f"  Total time:          {total_time:.2f}s")
    print(f"  Prompt TPS:          {prompt_tps:.1f} t/s")
    print(f"  Generation TPS:      {generation_tps:.1f} t/s")
    if peak_memory_gb > 0:
        print(f"  Peak memory:         {peak_memory_gb:.2f} GB")

    acceptance_rate = last_req.get("acceptance_rate")
    cycles = last_req.get("cycles")
    if cycles is not None:
        print(f"  DFlash cycles:       {cycles}")
    if acceptance_rate is not None:
        print(f"  Acceptance rate:     {acceptance_rate:.1%}")

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
        "mode_used": mode_used,
    }

    # DFlash-specific extras from /metrics
    for key in (
        "acceptance_rate",
        "cycles",
        "prefill_tok_s_physical",
        "prefill_tok_s_apparent",
        "decode_tok_s",
    ):
        val = last_req.get(key)
        if val is not None:
            result[key] = val

    if mem_info.get("rss_gb") is not None:
        result["rss_gb"] = mem_info["rss_gb"]
    if mem_info.get("mlx_active_gb") is not None:
        result["mlx_active_gb"] = mem_info["mlx_active_gb"]

    if reasoning_text:
        result["reasoning_text"] = reasoning_text

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dflash-mlx benchmarks using the OpenAI-compatible API")
    parser.add_argument(
        "model",
        nargs="?",
        help="Model id (auto-detected from /v1/models if omitted)",
    )

    common.setup_common_args(parser)

    parser.add_argument(
        "--base-url",
        default=DFLASH_API_URL,
        help=f"dflash-mlx API endpoint (default: {DFLASH_API_URL})",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (dflash-mlx local servers usually don't need one)",
    )
    parser.add_argument(
        "--request-model",
        default=None,
        help="Model identifier sent to the API (defaults to the positional model)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6)",
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
        help="Prepend a unique marker to every prompt so dflash's prefix cache "
        "treats each call as a new prefix (default: enabled). Use --no-cold-prefill "
        "to allow prefix-cache reuse across rows.",
    )

    args = parser.parse_args()

    base_url = normalize_base_url(args.base_url)

    print(f"\nTesting connection to {base_url} ...")
    health = test_server_connection(base_url)
    if not health or health.get("status") != "ok":
        print(f"Error: dflash-mlx server not reachable at {base_url}")
        return 1

    # Get server info from /metrics
    metrics = get_metrics(base_url)
    server_info = (metrics or {}).get("server") or {}
    server_model = server_info.get("model")
    server_draft = server_info.get("draft")
    server_version = server_info.get("version")
    server_profile = server_info.get("profile")

    print(f"Connected. Server model: {server_model}")
    if server_draft:
        print(f"Draft model: {server_draft}")
    if server_version:
        print(f"Version: {server_version}")
    if server_profile:
        print(f"Profile: {server_profile}")

    model = args.model
    if not model:
        models = list_models(base_url)
        if not models:
            model = server_model
        else:
            model = models[0]
        if not model:
            print("Error: No model specified and could not auto-detect one from dflash-mlx.")
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
    if server_draft:
        hardware_info["dflash_draft"] = server_draft
    if server_version:
        hardware_info["dflash_version"] = server_version
    if server_profile:
        hardware_info["dflash_profile"] = server_profile

    print("\nConnection details:")
    print(f"Endpoint:    {base_url}")
    print(f"Model:       {model}")
    if request_model != model:
        print(f"Request model: {request_model}")
    if server_draft:
        print(f"Draft model: {server_draft}")
    print(f"Max tokens:  {args.max_tokens}")
    print(
        f"Cold prefill: {'enabled (cache busted per prompt)' if args.cold_prefill else 'disabled (cache reuse allowed)'}"
    )

    output_dir = common.create_output_directory("dflash", model, cold_prefill=args.cold_prefill)

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
            cold_prefill=args.cold_prefill,
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
                cold_prefill=args.cold_prefill,
                n_runs=args.runs,
            )
            if result:
                results.append(result)
                if args.save_responses:
                    response_path = output_dir / f"response_{result['context_size']}.txt"
                    save_dflash_response(result, model, response_path)
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
            cold_prefill=args.cold_prefill,
        )
        if args.save_responses:
            for result in results:
                response_path = output_dir / f"response_{result['context_size']}.txt"
                save_dflash_response(result, model, response_path)

    if not results:
        print("\nNo successful benchmark results")
        return 1

    total_benchmark_time = time.time() - benchmark_start

    has_memory = any(r.get("peak_memory_gb", 0) > 0 for r in results)
    common.save_all_outputs(
        results,
        output_dir,
        model,
        "dflash-mlx API",
        hardware_info,
        args,
        include_memory=has_memory,
    )
    common.print_benchmark_summary(
        results,
        model,
        "dflash-mlx API",
        hardware_info,
        output_dir,
        total_benchmark_time,
    )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
