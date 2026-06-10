#!/usr/bin/env python3
"""Benchmark Apple Foundation Models Serve via its OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import httpx

import benchmark_common as common

AFMS_API_URL = "http://127.0.0.1:1976/v1"


def normalize_base_url(url: str) -> str:
    """Trim trailing slash and any chat-completions suffix."""
    if not url:
        return AFMS_API_URL
    normalized = url.strip().rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    return normalized


def server_root(base_url: str) -> str:
    """Return the server root, dropping a trailing /v1 when present."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized[: -len("/v1")]
    return normalized


def test_server_connection(base_url: str, timeout: int = 10) -> Optional[Dict]:
    """Hit /health and return parsed JSON, or None on failure."""
    try:
        resp = httpx.get(f"{server_root(base_url)}/health", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"Error connecting to Apple Foundation Models Serve at {base_url}: {exc}")
        return None


def list_models(base_url: str, timeout: int = 10) -> List[str]:
    """Return model IDs reported by /v1/models."""
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]
    except Exception:
        return []


def count_tokens(text: str) -> int:
    """Count tokens with the repo's context-file tokenizer fallback."""
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return len(text.split())


def call_afms_stream(
    base_url: str,
    api_key: Optional[str],
    request_model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> Dict:
    """Stream one chat completion and return text plus client-side timings."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": request_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    start_time = time.time()
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    generated_text = ""
    finish_reason = None

    with httpx.stream(
        "POST",
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith("data: "):
                line = line[len("data: ") :]
            if line == "[DONE]":
                break

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices", [])
            if not choices:
                continue

            choice = choices[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta", {}) or {}
            piece = delta.get("content") or ""
            if piece:
                now = time.time()
                if first_token_time is None:
                    first_token_time = now
                last_token_time = now
                generated_text += piece

    end_time = time.time()

    return {
        "generated_text": generated_text,
        "finish_reason": finish_reason,
        "total_time": end_time - start_time,
        "time_to_first_token": (first_token_time - start_time) if first_token_time else 0.0,
        "decode_window": (
            (last_token_time - first_token_time)
            if first_token_time and last_token_time and last_token_time > first_token_time
            else 0.0
        ),
    }


def run_benchmark(
    model_name: str,
    context_file: Path,
    base_url: str,
    api_key: Optional[str],
    request_model: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    cold_prefill: bool = True,
    _run_idx: Optional[int] = None,
) -> Optional[Dict]:
    """Benchmark Apple Foundation Models Serve for a single context file."""
    print(f"Running benchmark for {context_file}...")

    with open(context_file, "r") as handle:
        prompt = handle.read()

    if cold_prefill:
        prompt = common.make_cache_buster() + prompt
    elif _run_idx is not None:
        prompt = common.make_cache_buster(run_idx=_run_idx) + prompt

    try:
        data = call_afms_stream(
            base_url=base_url,
            api_key=api_key,
            request_model=request_model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
    except Exception as exc:
        print(f"Error contacting Apple Foundation Models Serve API: {exc}")
        return None

    generated_text = data["generated_text"]
    prompt_tokens = count_tokens(prompt)
    generation_tokens = count_tokens(generated_text)
    total_time = float(data["total_time"])
    ttft = float(data["time_to_first_token"])
    decode_window = float(data["decode_window"])
    generation_time = decode_window if decode_window > 0 else max(total_time - ttft, 0.0)

    prompt_tps = prompt_tokens / ttft if ttft > 0 else 0.0
    if generation_time > 0 and generation_tokens > 1:
        generation_tps = (generation_tokens - 1) / generation_time
    else:
        generation_tps = 0.0

    print(f"  Prompt tokens:       {prompt_tokens} (cl100k estimate)")
    print(f"  Generation tokens:   {generation_tokens} (cl100k estimate)")
    print(f"  Finish reason:       {data.get('finish_reason') or 'unknown'}")
    print(f"  TTFT:                {ttft:.3f}s")
    print(f"  Generation time:     {generation_time:.2f}s")
    print(f"  Total time:          {total_time:.2f}s")
    print(f"  Prompt TPS:          {prompt_tps:.1f} t/s (client-side)")
    print(f"  Generation TPS:      {generation_tps:.1f} t/s (client-side)")

    return {
        "context_size": context_file.stem,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "token_count_source": "cl100k_base_estimate",
        "prompt_tps": prompt_tps,
        "generation_tps": generation_tps,
        "total_time": total_time,
        "eval_duration": generation_time,
        "prompt_eval_duration": ttft,
        "time_to_first_token": ttft,
        "generated_text": generated_text,
        "finish_reason": data.get("finish_reason") or "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Apple Foundation Models Serve benchmarks using its OpenAI-compatible API"
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Model id as reported by /v1/models (auto-detected if omitted)",
    )
    common.setup_common_args(parser)
    parser.set_defaults(contexts="0.5,1,2")
    parser.add_argument(
        "--base-url",
        default=AFMS_API_URL,
        help=f"Apple Foundation Models Serve API endpoint (default: {AFMS_API_URL})",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key, if the server requires one",
    )
    parser.add_argument(
        "--request-model",
        default=None,
        help="Model identifier sent to the API (defaults to the positional model)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--cold-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prepend a unique marker to every prompt to avoid prefix cache reuse (default: enabled)",
    )

    args = parser.parse_args()
    base_url = normalize_base_url(args.base_url)

    print(f"\nTesting connection to {base_url} ...")
    health = test_server_connection(base_url, timeout=min(args.timeout, 10))
    if health is None:
        print(f"Error: Apple Foundation Models Serve is not reachable at {base_url}")
        return 1
    print(f"Connected. Health: {health.get('status', 'ok')}")

    models = list_models(base_url)
    model = args.model
    if not model:
        model = models[0] if models else None
        if not model:
            print("Error: No model specified and could not auto-detect one from /v1/models.")
            return 1
        print(f"Auto-detected model: {model}")
    elif models and model not in models:
        print(f"Warning: '{model}' not in /v1/models response. Available: {', '.join(models)}")

    request_model = args.request_model or model

    context_files = common.find_context_files(args.contexts)
    if not context_files:
        return 1

    print("\nCollecting hardware information...")
    hardware_info = common.get_hardware_info()
    hardware_str = common.format_hardware_string(hardware_info)
    hardware_info["api_endpoint"] = base_url
    hardware_info["api_model"] = model
    hardware_info["api_provider"] = "Apple Foundation Models Serve"
    hardware_info["token_count_source"] = "cl100k_base_estimate"
    if request_model != model:
        hardware_info["api_request_model"] = request_model

    print("\nConnection details:")
    print(f"Endpoint:    {base_url}")
    print(f"Model:       {model}")
    if request_model != model:
        print(f"Request model: {request_model}")
    if models:
        print(f"Models:      {', '.join(models)}")
    print(f"Hardware:    {hardware_str}")
    print(f"Max tokens:  {args.max_tokens} (sent as max_completion_tokens)")
    print(
        f"Cold prefill: {'enabled (cache busted per prompt)' if args.cold_prefill else 'disabled (cache reuse allowed)'}"
    )

    output_dir = common.create_output_directory("afms", model, cold_prefill=args.cold_prefill)

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
                timeout=args.timeout,
                cold_prefill=args.cold_prefill,
                n_runs=args.runs,
            )
            if result:
                results.append(result)
                if args.save_responses:
                    response_path = output_dir / f"response_{result['context_size']}.txt"
                    common.save_generated_text(result, model, response_path, "Apple Foundation Models Serve")
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
            timeout=args.timeout,
            cold_prefill=args.cold_prefill,
        )
        if args.save_responses:
            for result in results:
                response_path = output_dir / f"response_{result['context_size']}.txt"
                common.save_generated_text(result, model, response_path, "Apple Foundation Models Serve")

    if not results:
        print("\nNo successful benchmark results")
        return 1

    total_benchmark_time = time.time() - benchmark_start

    common.save_all_outputs(
        results,
        output_dir,
        model,
        "Apple Foundation Models Serve",
        hardware_info,
        args,
        include_memory=False,
    )
    common.print_benchmark_summary(
        results,
        model,
        "Apple Foundation Models Serve",
        hardware_info,
        output_dir,
        total_benchmark_time,
    )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
