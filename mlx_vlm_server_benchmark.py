#!/usr/bin/env python3
"""Benchmark script for mlx-vlm running as an OpenAI-compatible HTTP server.

mlx-vlm exposes its server-side metrics inside every chat-completion's
``usage`` block (``prompt_tps``, ``generation_tps``, ``peak_memory``). There is
no dedicated ``/metrics`` endpoint — but those fields are authoritative and
preferred over client-side timing. Streaming is used so we can still measure
TTFT on the client.

For true cold-prefill numbers we POST ``/unload`` between rows: mlx-vlm
auto-loads the model on the next request (``get_cached_model`` in
``mlx_vlm/server.py``), so the next prompt sees a fresh process state.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import httpx

import benchmark_common as common

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"


def normalize_base_url(url: str) -> str:
    """Trim trailing slash and any ``/chat/completions`` suffix."""
    if not url:
        return DEFAULT_BASE_URL
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
        print(f"Error connecting to mlx-vlm server at {base_url}: {exc}")
        return None


def list_models(base_url: str, timeout: int = 10) -> List[str]:
    """Return model IDs reported by ``/v1/models``."""
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]
    except Exception:
        return []


def unload_server_model(base_url: str, timeout: int = 60) -> None:
    """POST ``/unload`` so the next request reloads the model from cold."""
    try:
        httpx.post(f"{server_root(base_url)}/unload", timeout=timeout)
    except Exception as exc:
        print(f"  Warning: /unload failed: {exc}")


def _coerce_usage(usage: Dict) -> Dict:
    """Pull token counts and server-reported metrics out of a usage block.

    mlx-vlm streams use ``input_tokens``/``output_tokens``; the non-streaming
    response uses OpenAI's ``prompt_tokens``/``completion_tokens``. Accept both.
    """
    if not usage:
        return {}
    prompt_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_tps": float(usage.get("prompt_tps") or 0.0),
        "generation_tps": float(usage.get("generation_tps") or 0.0),
        "peak_memory": float(usage.get("peak_memory") or 0.0),
    }


def call_mlx_vlm_streaming(
    base_url: str,
    api_key: Optional[str],
    request_model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
) -> Dict:
    """Stream a chat completion. Returns dict with text, usage, and TTFT."""
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": request_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }

    generated_text = ""
    last_usage: Dict = {}
    first_token_time: Optional[float] = None
    start = time.time()

    with httpx.stream(
        "POST",
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:") :].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices") or []
            if choices:
                delta = (choices[0].get("delta") or {}).get("content") or ""
                if delta:
                    if first_token_time is None:
                        first_token_time = time.time()
                    generated_text += delta

            usage = chunk.get("usage")
            if usage:
                last_usage = usage

    end = time.time()
    return {
        "generated_text": generated_text,
        "usage": last_usage,
        "ttft": (first_token_time - start) if first_token_time else 0.0,
        "total_time": end - start,
    }


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
    unload_between_rows: bool = True,
    _run_idx: Optional[int] = None,
) -> Optional[Dict]:
    """Benchmark mlx-vlm server for a single context file."""
    print(f"Running benchmark for {context_file}...")

    with open(context_file, "r") as handle:
        prompt = handle.read()

    if cold_prefill:
        prompt = common.make_cache_buster() + prompt
        if unload_between_rows:
            unload_server_model(base_url)
    elif _run_idx is not None:
        prompt = common.make_cache_buster(run_idx=_run_idx) + prompt

    try:
        data = call_mlx_vlm_streaming(
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
        print(f"Error contacting mlx-vlm server: {exc}")
        return None

    usage = _coerce_usage(data.get("usage") or {})
    generated_text = data.get("generated_text", "") or ""
    ttft = float(data.get("ttft") or 0.0)
    total_time = float(data.get("total_time") or 0.0)

    prompt_tokens = usage.get("prompt_tokens") or 0
    generation_tokens = usage.get("completion_tokens") or 0
    server_prompt_tps = usage.get("prompt_tps") or 0.0
    server_generation_tps = usage.get("generation_tps") or 0.0
    peak_memory_gb = usage.get("peak_memory") or 0.0

    # Server stats are authoritative; fall back to client math only if missing.
    if server_prompt_tps > 0:
        prompt_tps = server_prompt_tps
    elif ttft > 0 and prompt_tokens:
        prompt_tps = prompt_tokens / ttft
    else:
        prompt_tps = 0.0

    generation_time = max(total_time - ttft, 0.0)
    if server_generation_tps > 0:
        generation_tps = server_generation_tps
    elif generation_time > 0 and generation_tokens:
        generation_tps = generation_tokens / generation_time
    else:
        generation_tps = 0.0

    eval_duration = generation_time if generation_time > 0 else (
        generation_tokens / generation_tps if generation_tps > 0 else 0.0
    )
    prompt_eval_duration = ttft

    print(f"  Prompt tokens:      {prompt_tokens}")
    print(f"  Generation tokens:  {generation_tokens}")
    print(f"  TTFT:               {ttft:.3f}s")
    print(f"  Generation time:    {generation_time:.2f}s")
    print(f"  Total time:         {total_time:.2f}s")
    print(f"  Prompt TPS:         {prompt_tps:.1f} t/s (server-side)")
    print(f"  Generation TPS:     {generation_tps:.1f} t/s (server-side)")
    if peak_memory_gb > 0:
        print(f"  Peak memory:        {peak_memory_gb:.2f} GB")

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
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run benchmarks against mlx-vlm's OpenAI-compatible HTTP server"
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Model id (auto-detected from /v1/models if omitted)",
    )

    common.setup_common_args(parser)

    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"mlx-vlm server endpoint (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (mlx-vlm typically doesn't require one)",
    )
    parser.add_argument(
        "--request-model",
        default=None,
        help="Model identifier sent to the API (defaults to the positional model)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
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
        help="Prepend a unique marker to every prompt to bust prompt caching "
        "(default: enabled). Use --no-cold-prefill to allow cache reuse.",
    )
    parser.add_argument(
        "--unload-between-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="POST /unload before each cold-prefill row so mlx-vlm reloads the "
        "model from cold (default: enabled). Has no effect when "
        "--no-cold-prefill is set. Adds model-load overhead per row.",
    )

    args = parser.parse_args()

    base_url = normalize_base_url(args.base_url)

    print(f"\nTesting connection to {base_url} ...")
    health = test_server_connection(base_url)
    if not health or health.get("status") != "healthy":
        print(f"Error: mlx-vlm server not reachable at {base_url}")
        return 1

    server_model = health.get("loaded_model")
    print(f"Connected. Loaded model: {server_model or '(none yet — will load on first request)'}")

    model = args.model
    if not model:
        models = list_models(base_url)
        if models:
            model = models[0]
        elif server_model:
            model = server_model
        if not model:
            print("Error: No model specified and could not auto-detect one from mlx-vlm.")
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

    print("\nConnection details:")
    print(f"Endpoint:    {base_url}")
    print(f"Model:       {model}")
    if request_model != model:
        print(f"Request model: {request_model}")
    print(f"Max tokens:  {args.max_tokens}")
    cold_msg = (
        "enabled (cache buster per prompt"
        if args.cold_prefill
        else "disabled (cache reuse allowed"
    )
    if args.cold_prefill and args.unload_between_rows:
        cold_msg += ", /unload between rows)"
    else:
        cold_msg += ")"
    print(f"Cold prefill: {cold_msg}")

    output_dir = common.create_output_directory("mlx_vlm_server", model, cold_prefill=args.cold_prefill)

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
            unload_between_rows=args.unload_between_rows,
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
                unload_between_rows=args.unload_between_rows,
                n_runs=args.runs,
            )
            if result:
                results.append(result)
                if args.save_responses:
                    response_path = output_dir / f"response_{result['context_size']}.txt"
                    common.save_generated_text(result, model, response_path, "MLX-VLM Server")
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
            unload_between_rows=args.unload_between_rows,
        )
        if args.save_responses:
            for result in results:
                response_path = output_dir / f"response_{result['context_size']}.txt"
                common.save_generated_text(result, model, response_path, "MLX-VLM Server")

    if not results:
        print("\nNo successful benchmark results")
        return 1

    total_benchmark_time = time.time() - benchmark_start

    has_memory = any(r.get("peak_memory_gb", 0) > 0 for r in results)
    common.save_all_outputs(
        results,
        output_dir,
        model,
        "MLX-VLM Server",
        hardware_info,
        args,
        include_memory=has_memory,
    )
    common.print_benchmark_summary(
        results,
        model,
        "MLX-VLM Server",
        hardware_info,
        output_dir,
        total_benchmark_time,
    )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
