#!/usr/bin/env python3
"""Benchmark script for vLLM OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

import benchmark_common as common

VLLM_API_URL = "http://localhost:8000"

_VLLM_PROMETHEUS_SUMMARY_METRICS = {
    "prompt_tokens_total": "prompt_tokens",
    "generation_tokens_total": "generation_tokens",
    "request_prompt_tokens": "prompt_tokens",
    "request_generation_tokens": "generation_tokens",
    "request_prefill_time_seconds": "prompt_eval_duration",
    "request_decode_time_seconds": "eval_duration",
    "time_to_first_token_seconds": "time_to_first_token",
    "e2e_request_latency_seconds": "total_time",
}


def ensure_endpoint(url: str) -> str:
    """Normalize an OpenAI-compatible endpoint to include `/v1`."""
    if not url:
        return VLLM_API_URL

    normalized = url.strip()
    if normalized.endswith("/v1/chat/completions"):
        normalized = normalized[: -len("/v1/chat/completions")]

    normalized = normalized.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"

    return normalized


def list_vllm_models(base_url: str, api_key: Optional[str] = None, timeout: int = 5) -> list[str]:
    """List model IDs served by vLLM via `/v1/models`."""
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.get(f"{base_url}/models", headers=headers, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return []

    names: list[str] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        if isinstance(model_id, str) and model_id:
            names.append(model_id)
    return names


def _to_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_numeric_time(value: Any, key: str) -> float:
    value_float = _to_float(value)
    if math.isnan(value_float):
        return value_float

    lower_key = key.lower()
    if "ms" in lower_key:
        return value_float / 1000.0
    return value_float


def _collect_timings(payload: Dict[str, Any], context_headers: Dict[str, str] = None) -> Dict[str, float]:
    """Pull timing fields from vLLM response payload and headers when available."""
    timings = {
        "prompt_eval_duration": math.nan,
        "eval_duration": math.nan,
        "time_to_first_token": math.nan,
    }

    def _merge_value(alias_map, candidate: str, value: Any):
        value_s = _extract_numeric_time(value, candidate)
        if not math.isnan(value_s) and math.isfinite(value_s):
            timings[alias_map[candidate]] = value_s

    # Common key aliases in vLLM/OpenAI responses.
    # We intentionally keep this permissive and key-substring based.
    alias_map = {
        "prompt_eval_duration": "prompt_eval_duration",
        "prompt_eval_time": "prompt_eval_duration",
        "prompt_processing": "prompt_eval_duration",
        "prompt_latency": "prompt_eval_duration",
        "prefill_time": "prompt_eval_duration",
        "prefill_duration": "prompt_eval_duration",
        "prefill_ms": "prompt_eval_duration",
        "generation_time": "eval_duration",
        "decode_time": "eval_duration",
        "decode_duration": "eval_duration",
        "decode_ms": "eval_duration",
        "eval_duration": "eval_duration",
        "time_to_first_token": "time_to_first_token",
        "time_to_first": "time_to_first_token",
        "ttft": "time_to_first_token",
        "ttft_ms": "time_to_first_token",
    }

    def _scan_object(obj: Any) -> None:
        if not isinstance(obj, dict):
            return

        for key, value in obj.items():
            if not isinstance(key, str):
                continue

            lower_key = key.lower()
            for alias, target in alias_map.items():
                if alias in lower_key:
                    _merge_value({alias: target}, alias, value)

            if isinstance(value, dict):
                _scan_object(value)

            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _scan_object(item)

    _scan_object(payload)

    # Some servers expose timing hints in headers.
    if context_headers:
        for key, value in context_headers.items():
            if not isinstance(key, str):
                continue

            lower_key = key.lower()
            if any(token in lower_key for token in ["ttft", "first", "time-to-first", "time_to_first"]):
                extracted = _extract_numeric_time(value, lower_key)
                if not math.isnan(extracted) and math.isfinite(extracted):
                    timings["time_to_first_token"] = extracted

            if any(token in lower_key for token in ["prefill", "prompt_eval", "prompt-time", "prompt_time"]):
                extracted = _extract_numeric_time(value, lower_key)
                if not math.isnan(extracted) and math.isfinite(extracted):
                    timings["prompt_eval_duration"] = extracted

            if any(token in lower_key for token in ["decode", "generation", "eval", "processing", "latency"]):
                extracted = _extract_numeric_time(value, lower_key)
                if not math.isnan(extracted) and math.isfinite(extracted):
                    timings["eval_duration"] = extracted

    return timings


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _estimate_token_count(text: str, model_name: str = "gpt-4o") -> int:
    """Estimate token count with tiktoken when available; fallback to simple heuristic."""
    if not text:
        return 0

    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback tokenizer for common OpenAI models.
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Very rough fallback: ~4 chars per token.
        if not text:
            return 0
        return max(1, len(text) // 4)


def _extract_usage_value(source: Dict[str, Any], candidates: Tuple[str, ...]) -> Optional[int]:
    for key in candidates:
        if key not in source:
            continue
        value = _safe_int(source.get(key), -1)
        if value >= 0:
            return value
    return None


def _parse_vllm_metrics(metrics_text: str, model_name: str) -> Dict[str, float]:
    """Parse selected vLLM Prometheus metrics for a given model."""
    if not metrics_text:
        return {}

    metric_line_re = re.compile(
        r"^(?P<name>vllm:[A-Za-z0-9_]+)(?P<suffix>_bucket|_sum|_count)?"
        r"\{(?P<labels>[^}]*)\}\s+(?P<value>-?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)$"
    )
    label_re = re.compile(r'([a-zA-Z0-9_]+)="(.*?)"')

    metrics: Dict[str, Dict[str, float]] = {}

    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        match = metric_line_re.match(line)
        if not match:
            continue

        name = match.group("name")
        suffix = match.group("suffix") or ""
        labels = {m.group(1): m.group(2) for m in label_re.finditer(match.group("labels"))}

        if labels.get("model_name") != model_name:
            continue

        value = _to_float(match.group("value"), math.nan)
        if math.isnan(value):
            continue

        metric_key = f"{name}{suffix}"
        metrics.setdefault(metric_key, {})
        # Keep the raw sample values as they are; caller computes deltas.
        metrics[metric_key]["value"] = value

    parsed: Dict[str, float] = {}
    for key, payload in metrics.items():
        parsed[key] = payload.get("value", math.nan)

    return parsed


def list_vllm_metrics(base_url: str, api_key: Optional[str] = None, timeout: int = 5) -> list[str]:
    """List model IDs served by vLLM via `/v1/models`."""
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.get(f"{base_url}/models", headers=headers, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return []

    names: list[str] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        if isinstance(model_id, str) and model_id:
            names.append(model_id)
    return names


def _read_vllm_metrics(
    base_url: str,
    model_name: str,
    api_key: Optional[str] = None,
    timeout: int = 5,
    debug: bool = False,
    debug_label: str | None = None,
) -> Dict[str, float]:
    """Read selected vLLM Prometheus counters/histograms for a model from `/metrics`."""
    headers: Dict[str, str] = {"Accept": "text/plain"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.get(f"{base_url}/metrics", headers=headers, timeout=timeout)
    response.raise_for_status()

    if debug:
        label = debug_label or f"vLLM metrics for {model_name}"
        print(f"\nDEBUG: {label}")
        print("-" * 80)
        print(response.text)
        print("-" * 80 + "\n")

    return _parse_vllm_metrics(response.text, model_name)


def _safe_metric_delta(before: Dict[str, float], after: Dict[str, float], key: str) -> float:
    before_value = before.get(key)
    after_value = after.get(key)

    if before_value is None or after_value is None:
        return math.nan

    if math.isnan(before_value) or math.isnan(after_value):
        return math.nan

    delta = after_value - before_value
    if delta < 0:
        return math.nan
    return delta


def _infer_from_vllm_metrics(
    metric_deltas: Dict[str, float],
    prompt_tokens: int,
    generation_tokens: int,
    total_tokens: int,
) -> Tuple[int, int, int, float, float, float]:
    """Use per-model vLLM counters/histograms to fill missing usage and timings."""
    inferred_prompt = _safe_int(
        (
            math.nan
            if math.isnan(metric_deltas.get("vllm:prompt_tokens_total", math.nan))
            else metric_deltas.get("vllm:prompt_tokens_total")
        ),
        0,
    )
    if inferred_prompt > 0 and prompt_tokens <= 0:
        prompt_tokens = inferred_prompt

    inferred_generation = _safe_int(
        (
            math.nan
            if math.isnan(metric_deltas.get("vllm:generation_tokens_total", math.nan))
            else metric_deltas.get("vllm:generation_tokens_total")
        ),
        0,
    )
    if inferred_generation > 0 and generation_tokens <= 0:
        generation_tokens = inferred_generation

    if total_tokens <= 0 and (prompt_tokens > 0 or inferred_generation > 0):
        total_tokens = prompt_tokens + generation_tokens

    prompt_eval = math.nan
    eval_duration = math.nan
    ttft = math.nan

    prefill_sum = metric_deltas.get("vllm:request_prefill_time_seconds_sum")
    prefill_count = metric_deltas.get("vllm:request_prefill_time_seconds_count")
    if not math.isnan(prefill_sum) and not math.isnan(prefill_count) and prefill_count > 0:
        prompt_eval = prefill_sum / prefill_count

    decode_sum = metric_deltas.get("vllm:request_decode_time_seconds_sum")
    decode_count = metric_deltas.get("vllm:request_decode_time_seconds_count")
    if not math.isnan(decode_sum) and not math.isnan(decode_count) and decode_count > 0:
        eval_duration = decode_sum / decode_count

    ttft_sum = metric_deltas.get("vllm:time_to_first_token_seconds_sum")
    ttft_count = metric_deltas.get("vllm:time_to_first_token_seconds_count")
    if not math.isnan(ttft_sum) and not math.isnan(ttft_count) and ttft_count > 0:
        ttft = ttft_sum / ttft_count

    return prompt_tokens, generation_tokens, total_tokens, prompt_eval, eval_duration, ttft


def _extract_usage_tokens(usage: Dict[str, Any]) -> Tuple[int, int, int]:
    if not isinstance(usage, dict) or not usage:
        return 0, 0, 0

    sources: list[Dict[str, Any]] = [usage]

    nested_usage = usage.get("usage")
    if isinstance(nested_usage, dict) and nested_usage is not usage:
        sources.append(nested_usage)

    usage_metadata = usage.get("usage_metadata")
    if isinstance(usage_metadata, dict):
        sources.append(usage_metadata)

    prompt_aliases = (
        "prompt_tokens",
        "input_tokens",
        "input_tokens_total",
        "prompt_tokens_total",
        "prompt_token_count",
        "prompt_count",
        "prefill_tokens",
    )
    generation_aliases = (
        "completion_tokens",
        "output_tokens",
        "response_tokens",
        "generated_tokens",
        "generation_tokens",
        "completion_token_count",
        "output_token_count",
        "generated_token_count",
    )
    total_aliases = (
        "total_tokens",
        "tokens",
        "total_token_count",
        "input_tokens_total",
    )

    prompt_tokens: Optional[int] = None
    generation_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    for source in sources:
        if prompt_tokens is None:
            prompt_tokens = _extract_usage_value(source, prompt_aliases)
        if generation_tokens is None:
            generation_tokens = _extract_usage_value(source, generation_aliases)
        if total_tokens is None:
            total_tokens = _extract_usage_value(source, total_aliases)

    prompt_tokens = prompt_tokens if prompt_tokens is not None else 0
    generation_tokens = generation_tokens if generation_tokens is not None else 0
    total_tokens = total_tokens if total_tokens is not None else 0

    if total_tokens and prompt_tokens == 0 and generation_tokens > 0:
        inferred = total_tokens - generation_tokens
        if inferred >= 0:
            prompt_tokens = inferred

    if total_tokens and generation_tokens == 0 and prompt_tokens > 0:
        inferred = total_tokens - prompt_tokens
        if inferred >= 0:
            generation_tokens = inferred

    return prompt_tokens, generation_tokens, total_tokens


def call_vllm(
    base_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
    api_key: Optional[str] = None,
    debug: bool = False,
) -> Tuple[Dict[str, Any], float, float]:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request_start = time.time()
    response = requests.post(
        f"{base_url}/chat/completions",
        json=payload,
        timeout=timeout,
        headers=headers,
    )
    request_time = time.time() - request_start
    response.raise_for_status()
    result = response.json()

    if debug:
        print("\n" + "=" * 60)
        print("DEBUG: Full vLLM response")
        print("=" * 60)
        print(json.dumps(result, indent=2))
        print("=" * 60 + "\n")

    timings = _collect_timings(result, response.headers)

    return result, request_time, timings["eval_duration"] if not math.isnan(timings["eval_duration"]) else request_time


def call_vllm_streaming(
    base_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
    api_key: Optional[str] = None,
    debug: bool = False,
) -> Tuple[Optional[str], Dict[str, Any], float, float, float]:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "stream_options": {
            "include_usage": True,
            "continuous_usage_stats": True,
        },
    }

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request_start = time.time()
    response = requests.post(
        f"{base_url}/chat/completions",
        json=payload,
        timeout=timeout,
        headers=headers,
        stream=True,
    )
    response.raise_for_status()

    timings = _collect_timings({}, response.headers)
    generated_text = ""
    usage: Dict[str, Any] = {}
    first_token_time = math.nan
    line_count = 0

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue

        text_line = raw_line.strip()
        if not text_line.startswith("data:"):
            continue

        payload_text = text_line[len("data:") :].strip()
        if payload_text == "[DONE]":
            break

        try:
            chunk = json.loads(payload_text)
        except json.JSONDecodeError:
            continue

        if debug:
            if line_count < 3:
                print(f"DEBUG stream chunk: {json.dumps(chunk)[:400]}")
            line_count += 1

        chunk_timings = _collect_timings(chunk)
        for key, value in chunk_timings.items():
            if math.isnan(value):
                continue
            if key == "time_to_first_token" and math.isnan(timings["time_to_first_token"]):
                timings["time_to_first_token"] = value

        choices = chunk.get("choices", [])
        if isinstance(choices, list) and choices:
            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            delta = first_choice.get("delta", {}) if isinstance(first_choice, dict) else {}
            if isinstance(delta, dict):
                token_piece = delta.get("content", "")
                if token_piece is None:
                    token_piece = ""

                if token_piece:
                    if math.isnan(first_token_time):
                        first_token_time = time.time() - request_start
                    generated_text += str(token_piece)

        chunk_usage = chunk.get("usage")
        if isinstance(chunk_usage, dict):
            usage.update(chunk_usage)

    # Some implementations may omit token usage in stream chunks.
    if not usage:
        nested_usage = chunk.get("metadata")
        if isinstance(nested_usage, dict):
            alt_usage = nested_usage.get("usage")
            if isinstance(alt_usage, dict):
                usage.update(alt_usage)

    total_time = time.time() - request_start
    return generated_text, usage, total_time, timings["eval_duration"], first_token_time


def run_benchmark(
    model_name: str,
    context_file: Path,
    base_url: str,
    api_key: Optional[str] = None,
    max_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    timeout: int = 300,
    stream: bool = True,
    use_vllm_metrics: bool = False,
    debug: bool = False,
    cold_prefill: bool = True,
    _run_idx: Optional[int] = None,
) -> Optional[Dict[str, object]]:
    with open(context_file, "r") as f:
        prompt = f.read()

    if cold_prefill:
        prompt = common.make_cache_buster() + prompt
    elif _run_idx is not None:
        prompt = common.make_cache_buster(run_idx=_run_idx) + prompt

    metrics_before: Dict[str, float] = {}
    if use_vllm_metrics:
        try:
            metrics_before = _read_vllm_metrics(
                base_url,
                model_name,
                api_key=api_key,
                timeout=timeout,
                debug=debug,
                debug_label=f"Before request metrics ({context_file.name})",
            )
        except requests.exceptions.RequestException:
            metrics_before = {}

    if stream:
        generated_text, usage, total_time, parsed_eval_duration, first_token_time = call_vllm_streaming(
            base_url,
            model_name,
            prompt,
            max_tokens,
            temperature,
            top_p,
            timeout,
            api_key=api_key,
            debug=debug,
        )
        timings = {
            "eval_duration": parsed_eval_duration,
            "prompt_eval_duration": math.nan,
            "time_to_first_token": first_token_time,
        }
    else:
        result, total_time, parsed_eval_duration = call_vllm(
            base_url,
            model_name,
            prompt,
            max_tokens,
            temperature,
            top_p,
            timeout,
            api_key=api_key,
            debug=debug,
        )
        timings = {
            "eval_duration": parsed_eval_duration,
            "time_to_first_token": math.nan,
            "prompt_eval_duration": math.nan,
        }
        usage = result.get("usage", {})
        generated_text = ""
        choices = result.get("choices", [])
        if isinstance(choices, list) and choices:
            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            if isinstance(first_choice, dict):
                message = first_choice.get("message", {})
                generated_text = message.get("content", "")

                stats_payload = result
                stats = _collect_timings(stats_payload)
                for key, value in stats.items():
                    if not math.isnan(value):
                        timings[key] = value

    metric_deltas: Dict[str, float] = {}
    if use_vllm_metrics and metrics_before:
        try:
            metrics_after = _read_vllm_metrics(
                base_url,
                model_name,
                api_key=api_key,
                timeout=timeout,
                debug=debug,
                debug_label=f"After request metrics ({context_file.name})",
            )
            for key in set(metrics_before.keys()) | set(metrics_after.keys()):
                delta = _safe_metric_delta(metrics_before, metrics_after, key)
                if not math.isnan(delta):
                    metric_deltas[key] = delta
        except requests.exceptions.RequestException:
            metric_deltas = {}

    prompt_tokens, generation_tokens, total_tokens = _extract_usage_tokens(usage)
    estimated_from_text = False

    if prompt_tokens <= 0:
        estimated = _estimate_token_count(prompt, model_name)
        if estimated > 0:
            prompt_tokens = estimated
            estimated_from_text = True

    if generation_tokens <= 0:
        estimated = _estimate_token_count(generated_text, model_name)
        if estimated > 0:
            generation_tokens = estimated
            estimated_from_text = True

    if total_tokens <= 0:
        total_tokens = prompt_tokens + generation_tokens
    elif generation_tokens == 0 and total_tokens and prompt_tokens:
        inferred_gen = total_tokens - prompt_tokens
        if inferred_gen > 0:
            generation_tokens = inferred_gen
            estimated_from_text = estimated_from_text or generation_tokens == 0

    prompt_eval_duration = timings.get("prompt_eval_duration", math.nan)
    eval_duration = timings.get("eval_duration", math.nan)
    time_to_first_token = timings.get("time_to_first_token", math.nan)

    if math.isnan(prompt_eval_duration) and not math.isnan(time_to_first_token):
        prompt_eval_duration = time_to_first_token

    if math.isnan(eval_duration):
        if math.isfinite(time_to_first_token) and time_to_first_token >= 0 and total_time > time_to_first_token:
            eval_duration = total_time - time_to_first_token
        else:
            eval_duration = total_time

    if math.isnan(prompt_eval_duration):
        prompt_tps = float("nan")
    elif prompt_eval_duration == 0:
        prompt_tps = 0.0
    else:
        prompt_tps = prompt_tokens / prompt_eval_duration if prompt_tokens else 0.0

    generation_tps = generation_tokens / eval_duration if eval_duration > 0 else 0.0

    if estimated_from_text:
        print("  Token counts were not returned by vLLM and were estimated from text length.")

    return {
        "context_size": context_file.stem,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "total_tokens": total_tokens,
        "prompt_eval_duration": prompt_eval_duration,
        "time_to_first_token": time_to_first_token,
        "eval_duration": eval_duration,
        "total_time": total_time,
        "prompt_tps": prompt_tps,
        "generation_tps": generation_tps,
        "generated_text": generated_text,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark vLLM OpenAI-compatible endpoint across context sizes")
    parser.add_argument("model", help="Model name configured in your vLLM server")

    common.setup_common_args(parser)

    parser.add_argument(
        "--base-url",
        default=VLLM_API_URL,
        help=f"vLLM base URL (default: {VLLM_API_URL})",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key for servers requiring Authorization header",
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
        "--stream",
        dest="stream",
        action="store_true",
        help="Measure time-to-first-token using streaming (default)",
    )
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Disable streaming and run non-stream requests",
    )
    parser.set_defaults(stream=True)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full JSON payloads from vLLM for troubleshooting",
    )
    parser.add_argument(
        "--metrics",
        dest="metrics",
        action="store_true",
        help="Use vLLM /metrics endpoint deltas for prompt/generation/timing stats when response usage is missing.",
    )
    parser.add_argument(
        "--no-metrics",
        dest="metrics",
        action="store_false",
        help="Disable vLLM /metrics sampling for stats.",
    )
    parser.set_defaults(metrics=False)
    parser.add_argument(
        "--cold-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prepend a unique marker to every prompt to bust KV "
        "cache reuse, forcing cold prefill on every row (default: enabled; "
        "use --no-cold-prefill for cached/warm-reuse numbers)",
    )

    args = parser.parse_args()

    base_url = ensure_endpoint(args.base_url)
    api_key = args.api_key

    context_files = common.find_context_files(args.contexts, args.context_dir)
    if not context_files:
        return 1

    print(f"Testing vLLM endpoint: {base_url}")
    try:
        available_models = list_vllm_models(base_url, api_key=api_key)
        if available_models:
            print(f"Available models from endpoint: {', '.join(available_models)}")
        else:
            print("Warning: /v1/models returned no models.")
    except requests.exceptions.RequestException as exc:
        print(f"Warning: could not query /v1/models ({exc}).")

    hardware_info = common.get_hardware_info()
    hardware_info["vllm_endpoint"] = base_url

    if args.stream:
        print("Stream mode enabled: using streamed chunks to estimate time-to-first-token")
    else:
        print("Stream mode disabled: time-to-first-token unavailable; using measured totals")

    if args.metrics:
        print("vLLM metrics fallback enabled: sampling /metrics between requests for missing usage/timings.")
    else:
        print("vLLM metrics fallback disabled.")

    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    print(
        f"Cold prefill: {'enabled (cache busted per prompt)' if args.cold_prefill else 'disabled (cache reuse allowed)'}"
    )
    print(f"Timeout: {args.timeout}s")

    output_dir = common.create_output_directory("vllm", args.model, args.output_dir, cold_prefill=args.cold_prefill)

    # Warmup run
    warmup_file = common.find_warmup_file()
    if warmup_file:
        print(f"\n{'=' * 50}")
        print(f"Warmup run (excluded from results): {warmup_file.name}")
        print(f"{'=' * 50}")
        try:
            run_benchmark(
                model_name=args.model,
                context_file=warmup_file,
                base_url=base_url,
                api_key=api_key,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.timeout,
                stream=args.stream,
                use_vllm_metrics=args.metrics,
                debug=args.debug,
            )
        except Exception:
            pass
        print("Warmup complete.")
    else:
        print("Warning: 0.5k.txt not found, skipping warmup.")

    results = []
    benchmark_start = time.time()

    if args.cold_prefill:
        for context_file in context_files:
            print("\n" + "=" * 50)
            print(f"Benchmarking {context_file.name}...")
            print("=" * 50)

            try:
                result = common.run_benchmark_peak(
                    run_benchmark,
                    model_name=args.model,
                    context_file=context_file,
                    base_url=base_url,
                    api_key=api_key,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    timeout=args.timeout,
                    stream=args.stream,
                    use_vllm_metrics=args.metrics,
                    debug=args.debug,
                    cold_prefill=args.cold_prefill,
                    n_runs=args.runs,
                )
            except requests.exceptions.HTTPError as e:
                message = str(e)
                if e.response is not None and e.response.status_code == 404:
                    try:
                        details = e.response.json()
                        err = details.get("error", {})
                        if isinstance(err, dict):
                            message = err.get("message", message)
                    except Exception:
                        pass

                    print(f"Request error: {message}")
                    print(f"HTTP 404 from vLLM endpoint: model '{args.model}' is not available.")
                    try:
                        available_models = list_vllm_models(base_url, api_key=api_key)
                        if available_models:
                            print("Available models:")
                            for model_name in available_models:
                                print(f"  - {model_name}")
                            print("Run again with one of the above model names.")
                        else:
                            print("No models were returned by /v1/models. Confirm your vLLM model is loaded.")
                    except Exception:
                        print("Could not retrieve /v1/models from the endpoint.")

                    result = None
                else:
                    print(f"Request error: {e}")
                    result = None
            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
                result = None
            except Exception as exc:
                print(f"Benchmark error: {exc}")
                result = None

            if result:
                results.append(result)

                print(f"\nResults for {context_file.name}:")
                print(f"  Prompt tokens: {result['prompt_tokens']}")
                print(f"  Generation tokens: {result['generation_tokens']}")
                if not math.isnan(result["prompt_eval_duration"]):
                    print(f"  Prompt eval duration: {result['prompt_eval_duration']:.2f}s")
                if not math.isnan(result["time_to_first_token"]):
                    print(f"  Time to first token: {result['time_to_first_token']:.2f}s")
                if not math.isnan(result["prompt_tps"]):
                    print(f"  Prompt TPS: {result['prompt_tps']:.2f}")
                else:
                    print("  Prompt TPS: n/a")
                print(f"  Generation TPS: {result['generation_tps']:.2f}")
                print(f"  Total time: {result['total_time']:.2f}s")

                if args.save_responses:
                    response_file = output_dir / f"response_{result['context_size']}.txt"
                    common.save_generated_text(
                        result,
                        args.model,
                        response_file,
                        framework="vLLM",
                    )
    else:
        results = common.run_benchmark_peak_per_run(
            run_benchmark,
            context_files=context_files,
            n_runs=args.runs,
            model_name=args.model,
            base_url=base_url,
            api_key=api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            stream=args.stream,
            use_vllm_metrics=args.metrics,
            debug=args.debug,
            cold_prefill=args.cold_prefill,
        )
        if args.save_responses:
            for result in results:
                response_file = output_dir / f"response_{result['context_size']}.txt"
                common.save_generated_text(
                    result,
                    args.model,
                    response_file,
                    framework="vLLM",
                )

    if not results:
        print("\nNo successful benchmark results")
        return 1

    total_benchmark_time = time.time() - benchmark_start
    common.save_all_outputs(
        results,
        output_dir,
        args.model,
        "vLLM",
        hardware_info,
        args,
    )
    common.print_benchmark_summary(
        results,
        args.model,
        "vLLM",
        hardware_info,
        output_dir,
        total_benchmark_time,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
