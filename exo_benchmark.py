#!/usr/bin/env python3
"""Benchmark script for Exo using an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from openai import OpenAI

import benchmark_common as common

EXO_API_URL = "http://0.0.0.0:52415"


def ensure_endpoint(url: str) -> str:
    """Normalize an OpenAI-compatible endpoint to include /v1."""

    if not url:
        return EXO_API_URL

    normalized = url.strip()

    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]

    normalized = normalized.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"

    return normalized


def call_exo(
    client: OpenAI,
    request_model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
) -> Dict:
    """Send a non-streaming chat completion request to Exo."""

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
                    "reasoning_content": getattr(
                        response.choices[0].message, "reasoning_content", ""
                    ),
                }
            }
        ],
        "usage": response.usage.model_dump() if response.usage else {},
    }


def count_tokens_tiktoken(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens using tiktoken. Falls back to word-based estimate if unavailable."""
    if not text:
        return 0
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Use cl100k_base as fallback (used by GPT-3.5/4)
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        # Fallback: rough estimate of ~4 chars per token
        return max(1, len(text) // 4)


def call_exo_streaming(
    client: OpenAI,
    request_model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
) -> Dict[str, object]:
    """Stream a chat completion from Exo, capturing time-to-first-token."""

    message_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: Dict[str, int] = {}
    token_count: int = 0  # Count tokens client-side like dashboard does

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
                token_count += 1  # Each chunk typically = 1 token

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
                                token_count += 1
                        elif isinstance(item, str):
                            reasoning_parts.append(item)
                            token_count += 1
                elif isinstance(reasoning_delta, str):
                    reasoning_parts.append(reasoning_delta)
                    token_count += 1

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
    generated_text = "".join(message_parts)
    reasoning_text = "".join(reasoning_parts)

    # If API didn't provide usage, calculate client-side.
    if not usage or usage.get("completion_tokens", 0) == 0:
        prompt_tokens = count_tokens_tiktoken(prompt, request_model)
        completion_tokens = count_tokens_tiktoken(
            generated_text + reasoning_text, request_model
        )
        if completion_tokens == 0 and token_count > 0:
            # Fall back to chunk count when tokenizer isn't available.
            completion_tokens = token_count
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    return {
        "generated_text": generated_text,
        "reasoning_text": reasoning_text,
        "usage": usage,
        "total_time": total_time,
        "prompt_eval_duration": prompt_eval_duration,
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
) -> Optional[Dict]:
    """Benchmark Exo for a given context file."""
    print(f"Running benchmark for {context_file}...")

    with open(context_file, "r") as handle:
        prompt = handle.read()

    try:
        if stream:
            stream_result = call_exo_streaming(
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
        else:
            start_time = time.time()
            data = call_exo(
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
        print(f"Error contacting Exo API: {exc}")
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
        print("Note: API did not include token usage; using client-side token counting.")

    generation_duration = max(total_time - prompt_eval_duration, 0.0)
    eval_duration = generation_duration if generation_duration > 0 else total_time

    prompt_tps = (
        prompt_tokens / prompt_eval_duration if prompt_eval_duration and prompt_eval_duration > 0 else 0.0
    )
    generation_tps = (
        generation_tokens / eval_duration if eval_duration and eval_duration > 0 else 0.0
    )

    print(f"  Prompt tokens: {prompt_tokens}")
    print(f"  Generation tokens: {generation_tokens}")
    if reasoning_tokens:
        print(f"  Reasoning tokens: {reasoning_tokens}")
    print(f"  Total tokens: {total_tokens}")
    if prompt_eval_duration > 0:
        print(f"  Time to first token: {prompt_eval_duration:.2f}s")
        print(f"  Prompt throughput: {prompt_tps:.2f} tokens/sec")
    print(f"  Generation throughput: {generation_tps:.2f} tokens/sec")
    print(f"  Total time: {total_time:.2f}s")

    result: Dict[str, object] = {
        "context_size": context_file.stem,
        "prompt_tokens": prompt_tokens,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Exo benchmarks using OpenAI-compatible API")
    parser.add_argument("model", help="Model id (e.g., local-model)")

    common.setup_common_args(parser)

    parser.add_argument(
        "--api-key",
        help="API key (defaults to EXO_API_KEY/OPENAI_API_KEY; uses placeholder if unset)",
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
        default=EXO_API_URL,
        help="Override Exo API endpoint (default: http://0.0.0.0:52415)",
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

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("EXO_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = "local"
        print("No API key provided; using a placeholder key for local Exo server.")

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
        print(f"Error initializing Exo client: {exc}")
        return 1

    print("\nConnection details:")
    print(f"Endpoint: {base_url}")
    print(f"Model: {args.model}")
    if request_model != args.model:
        print(f"Request model: {request_model}")
    print(f"Max tokens: {args.max_tokens}")

    output_dir = common.create_output_directory("exo", args.model)

    results = []
    benchmark_start = time.time()

    for context_file in context_files:
        print("\n" + "=" * 50)
        print(f"Benchmarking {context_file.name}...")
        print("=" * 50)

        result = run_benchmark(
            model_name=args.model,
            context_file=context_file,
            client=client,
            request_model=request_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            stream=args.stream,
        )

        if result:
            results.append(result)
            if args.save_responses:
                response_path = output_dir / f"response_{result['context_size']}.txt"
                common.save_generated_text(result, args.model, response_path, "Exo API")

    if not results:
        print("\nNo successful benchmark results")
        return 1

    total_benchmark_time = time.time() - benchmark_start

    common.save_all_outputs(results, output_dir, args.model, "Exo API", hardware_info, args)
    common.print_benchmark_summary(
        results,
        args.model,
        "Exo API",
        hardware_info,
        output_dir,
        total_benchmark_time,
    )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
