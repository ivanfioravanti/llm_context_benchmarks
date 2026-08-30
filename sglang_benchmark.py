#!/usr/bin/env python3
"""
Benchmark script for SGLang servers (OpenAI-compatible API).

SGLang (https://github.com/sgl-project/sglang) serves an OpenAI-compatible
API at ``/v1``. This engine always streams so TTFT can be measured
client-side, and reads every stat SGLang reports on the wire:

* final ``usage`` chunk (``stream_options.include_usage``):
  ``prompt_tokens``, ``completion_tokens``, ``total_tokens``,
  ``reasoning_tokens`` (SGLang-specific top-level field) and
  ``prompt_tokens_details.cached_tokens`` — radix-cache hit tokens, needs
  the server started with ``--enable-cache-report``
* a final ``sglext`` chunk (SGLang response extension, requested via the
  ``return_cached_tokens_details`` / ``return_spec_tokens_details`` body
  flags): ``cached_tokens_details`` (KV cache hit breakdown:
  device/host/storage) and ``spec_tokens_details`` (speculative decoding /
  MTP accept-rate and accept-length counters)
* ``reasoning_content`` deltas when the server runs a reasoning parser

SGLang reports no server-side prefill/decode timing split on the
OpenAI-compatible API (the native ``/generate`` ``meta_info.e2e_latency``
is non-streaming only), so prefill time is proxied by client-side TTFT and
the decode window by the first→last token span — the same convention as
the generic ``openai_benchmark.py`` engine.

Usage:
    python sglang_benchmark.py
    python sglang_benchmark.py Qwen/Qwen3-8B
    python sglang_benchmark.py Qwen/Qwen3-8B --base-url http://dgx1.local:8888/v1
    python sglang_benchmark.py --contexts 2,4,8,16 --max-tokens 500
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import httpx
from openai import OpenAI

import benchmark_common as common

# Stats requested via extra_body on every call. Pydantic servers ignore
# unknown request fields, so older SGLang builds without these flags are
# unaffected; on builds that support them the sglext chunk carries the
# KV-cache breakdown and speculative-decoding counters.
SGLANG_EXT_BODY = {
    "return_cached_tokens_details": True,
    "return_spec_tokens_details": True,
}


def build_client(base_url: str, api_key: str) -> OpenAI:
    """Create an OpenAI client pointed at the SGLang server."""
    return OpenAI(base_url=base_url, api_key=api_key)


def test_server_connection(client: OpenAI) -> bool:
    """Check that the server is reachable and returns a model list."""
    try:
        client.models.list()
        return True
    except Exception as e:
        print(f"Error connecting to SGLang server: {e}")
        return False


def get_available_model(client: OpenAI) -> Optional[str]:
    """Return the first model ID listed by the server."""
    try:
        models = list(client.models.list())
        if not models:
            return None
        return models[0].id
    except Exception:
        return None


def fetch_server_info(base_url: str, api_key: str) -> Dict:
    """Fetch ``/get_server_info`` for the banner and hardware metadata.

    Keys vary across SGLang versions; every field is read defensively and
    the whole call is best-effort — an unreachable endpoint just means no
    server metadata, never a failed run.
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    try:
        response = httpx.get(
            f"{root}/get_server_info",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        response.raise_for_status()
        info = response.json()
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


def server_info_line(server_info: Dict) -> str:
    """One-line summary of the served deployment for the banner."""
    bits = []
    if server_info.get("version"):
        bits.append(f"v{server_info['version']}")
    parallel = []
    for key, label in (("tp_size", "tp"), ("dp_size", "dp"), ("ep_size", "ep")):
        value = server_info.get(key)
        if isinstance(value, int) and value > 1:
            parallel.append(f"{label}{value}")
    if parallel:
        bits.append("/".join(parallel))
    if server_info.get("context_len"):
        bits.append(f"ctx {server_info['context_len']}")
    if server_info.get("max_total_num_tokens"):
        bits.append(f"kv {server_info['max_total_num_tokens']}")
    return ", ".join(bits)


def _to_int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def run_benchmark(
    client: OpenAI,
    model: str,
    context_file: Path,
    max_tokens: int = 128,
    timeout: int = 3600,
    cold_prefill: bool = True,
    _run_idx: Optional[int] = None,
) -> Optional[Dict]:
    """Benchmark a single context file against the SGLang server.

    Streams so we can measure TTFT client-side and read ``usage`` /
    ``sglext`` from the final chunks. Returns a result dict on success,
    None on failure.
    """
    with open(context_file) as f:
        prompt = f.read()

    if cold_prefill:
        prompt = common.make_cache_buster() + prompt
    elif _run_idx is not None:
        prompt = common.make_cache_buster(run_idx=_run_idx) + prompt

    sglext: Dict = {}

    def _capture_sglext(chunk):
        # SGLang emits the extension stats as a top-level `sglext` object on
        # a late chunk with empty choices; the OpenAI SDK keeps unknown
        # fields as model_extra attributes.
        ext = getattr(chunk, "sglext", None)
        if isinstance(ext, dict) and ext:
            sglext.update(ext)

    try:
        stream_result = common.stream_chat(
            client,
            model,
            prompt,
            max_tokens,
            temperature=0.0,
            timeout=timeout,
            chunk_hook=_capture_sglext,
            extra_body=SGLANG_EXT_BODY,
        )
    except Exception as e:
        print(f"Error during benchmark: {e}")
        return None

    generated_text = stream_result["generated_text"]
    reasoning_text = stream_result["reasoning_text"]
    total_time = stream_result["total_time"]
    usage = stream_result["usage"]
    prompt_tokens = _to_int(usage.get("prompt_tokens"))
    completion_tokens = _to_int(usage.get("completion_tokens"))

    # SGLang puts radix-cache hits in prompt_tokens_details.cached_tokens
    # (needs --enable-cache-report); very old builds exposed a top-level
    # usage.cached_tokens. Accept both.
    prompt_details = usage.get("prompt_tokens_details") or {}
    cached_tokens = _to_int(prompt_details.get("cached_tokens")) or _to_int(usage.get("cached_tokens"))

    # Reasoning tokens: SGLang-specific top-level usage field; other shapes
    # (completion_tokens_details.reasoning_tokens) kept as fallback.
    reasoning_tokens = _to_int(usage.get("reasoning_tokens"))
    if not reasoning_tokens:
        reasoning_tokens = _to_int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))

    # KV cache hit breakdown (device / host / storage) from the sglext chunk.
    cache_details = sglext.get("cached_tokens_details") or {}
    cached_device = _to_int(cache_details.get("device"))
    cached_host = _to_int(cache_details.get("host"))
    cached_storage = _to_int(cache_details.get("storage"))

    # Speculative-decoding (MTP/EAGLE) counters from the sglext chunk.
    spec = sglext.get("spec_tokens_details") or {}
    if isinstance(spec, list):  # n>1 servers emit one entry per choice
        spec = spec[0] if spec else {}
    spec = spec if isinstance(spec, dict) else {}

    # TTFT / decode window: SGLang streams the first token after prefill, so
    # the client-side anchors are the honest proxies.
    ttft = stream_result["time_to_first_token"]
    generation_time = stream_result["decode_window"]

    # Fallback token counts from text.
    if prompt_tokens == 0:
        prompt_tokens = len(prompt.split())
    if completion_tokens == 0:
        completion_tokens = len(generated_text.split()) + len(reasoning_text.split())

    prompt_tps = prompt_tokens / ttft if ttft > 0 else 0.0
    if generation_time > 0 and completion_tokens > 1:
        generation_tps = (completion_tokens - 1) / generation_time
    else:
        generation_tps = 0.0

    # Delivery-burst guard. SGLang streams every decode step
    # (stream_interval=1), so a window implying >4000 t/s single-stream on a
    # real token count is not decode speed — the chunks were generated
    # server-side and flushed in one read (network/scheduler stall). Such a
    # run measures the flush, not the model; discard it the way
    # run_benchmark_peak discards degenerate runs.
    if completion_tokens >= 16 and 0 < generation_time < completion_tokens / 4000.0:
        print(
            f"  Rejecting burst-delivered run: {completion_tokens} tokens in "
            f"{generation_time * 1000:.1f}ms ({generation_tps:.0f} t/s implied) — "
            "chunks arrived in a single flush, not streamed decode"
        )
        return None

    print(f"  Prompt tokens:      {prompt_tokens}")
    print(f"  Completion tokens:  {completion_tokens}")
    if reasoning_tokens:
        print(f"  Reasoning tokens:   {reasoning_tokens}")
    if cached_tokens:
        line = f"  Cached tokens:      {cached_tokens} (radix cache)"
        parts = []
        if cached_device:
            parts.append(f"device {cached_device}")
        if cached_host:
            parts.append(f"host {cached_host}")
        if cached_storage:
            parts.append(f"storage {cached_storage}")
        if parts:
            line += f" [{', '.join(parts)}]"
        print(line)
    if spec.get("spec_accept_rate"):
        print(
            f"  Spec decode:        accept rate {spec['spec_accept_rate']:.2f}, "
            f"accept length {spec.get('spec_accept_length', 0):.2f}"
        )
    if reasoning_text:
        print(f"  Reasoning chars:    {len(reasoning_text)} (counted toward decode window)")
    print(f"  TTFT:               {ttft:.3f}s")
    print(f"  Generation time:    {generation_time:.2f}s")
    print(f"  Total time:         {total_time:.2f}s")
    print(f"  Prompt TPS:         {prompt_tps:.1f} t/s")
    print(f"  Generation TPS:     {generation_tps:.1f} t/s")

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
    if reasoning_tokens:
        result["reasoning_tokens"] = reasoning_tokens
    if cached_tokens:
        result["cached_tokens"] = cached_tokens
        if cached_device:
            result["cached_device_tokens"] = cached_device
        if cached_host:
            result["cached_host_tokens"] = cached_host
        if cached_storage:
            result["cached_storage_tokens"] = cached_storage
    if spec.get("spec_accept_rate"):
        result["spec_accept_rate"] = spec["spec_accept_rate"]
    if spec.get("spec_accept_length"):
        result["spec_accept_length"] = spec["spec_accept_length"]

    return common.add_throughput_metrics(result, prompt_text=prompt)


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Benchmark an SGLang OpenAI-compatible server across context sizes")
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
        default=os.environ.get("SGLANG_BASE_URL", "http://dgx1.local:8888/v1"),
        help="Base URL of the SGLang server (default: http://dgx1.local:8888/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("SGLANG_API_KEY", "no-key"),
        help="API key (default: SGLANG_API_KEY env var or 'no-key')",
    )
    parser.add_argument(
        "--cold-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prepend a unique marker to every prompt to bust the radix "
        "cache, forcing cold prefill on every row (default: enabled; "
        "use --no-cold-prefill for cached/warm-reuse numbers)",
    )

    common.setup_common_args(parser)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    client = build_client(base_url, args.api_key)

    print(f"Testing connection to {base_url} ...")
    if not test_server_connection(client):
        print(f"Error: Cannot reach SGLang server at {base_url}")
        print("Make sure SGLang is running and the base URL + API key are correct.")
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

    server_info = fetch_server_info(base_url, args.api_key)
    deployment = server_info_line(server_info)

    hardware_info = common.mark_client_hardware(common.get_hardware_info(), base_url)
    if server_info:
        # Persist served-deployment metadata next to the client hardware.
        for key in ("version", "context_len", "max_total_num_tokens", "tp_size", "dp_size", "ep_size"):
            value = server_info.get(key)
            if value is not None:
                hardware_info[f"sglang_{key}"] = value
    hardware_str = common.format_hardware_string(hardware_info)

    print(f"\nSGLang Server Benchmark")
    print(f"Server:     {base_url}" + (f" ({deployment})" if deployment else ""))
    print(f"Model:      {model}")
    print(f"Hardware:   {hardware_str}")
    print(f"Max tokens: {args.max_tokens}")
    print(
        f"Cold prefill: {'enabled (cache busted per prompt)' if args.cold_prefill else 'disabled (cache reuse allowed)'}"
    )

    context_files = common.find_context_files(args.contexts, context_type=args.context_type)
    if not context_files:
        return 1

    output_dir = common.create_output_directory(
        "sglang", model, cold_prefill=args.cold_prefill, context_type=args.context_type
    )

    results = []
    benchmark_start = time.time()

    if args.cold_prefill:
        for i, ctx_file in enumerate(context_files):
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
                    common.save_generated_text(result, model, resp_path, "SGLang")

            common.cooldown_after_context(ctx_file, is_last=i == len(context_files) - 1)
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
                common.save_generated_text(result, model, resp_path, "SGLang")

    total_benchmark_time = time.time() - benchmark_start

    if not results:
        print("\nNo successful benchmark results.")
        return 1

    common.save_all_outputs(
        results,
        output_dir,
        model,
        "SGLang",
        hardware_info,
        args,
    )

    common.print_benchmark_summary(
        results,
        model,
        "SGLang",
        hardware_info,
        output_dir,
        total_benchmark_time,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
