#!/usr/bin/env python3
"""
Benchmark script for LM Studio local server.

Uses LM Studio's native `/api/v0/chat/completions` endpoint which reports
`time_to_first_token`, `generation_time`, and `tokens_per_second` directly
in the response's `stats` field. This avoids the streaming-parser issues
that affect the generic openai-compat path with models that emit thinking/
reasoning tokens (e.g. Qwen3.5) — the non-streaming response carries the
full timing data up front.

Requirements:
    - LM Studio 0.3+ with the local server running
    - Native API (`/api/v0`) enabled (Developer tab → Start Server)

Usage:
    python lmstudio_benchmark.py <model-id>
    python lmstudio_benchmark.py <model-id> --base-url http://localhost:1234
    python lmstudio_benchmark.py <model-id> --contexts 2,4,8,16 --max-tokens 128
"""

import argparse
import os
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import requests

import benchmark_common as common

DEFAULT_BASE_URL = "http://localhost:1234"

# LM Studio exposes two native REST APIs:
#
#   /api/v0/chat/completions  — legacy chat-completions shape. Same
#       request/response schema as OpenAI's chat/completions, with a
#       `stats` field carrying library-measured TTFT and TPS. Stable and
#       recommended for benchmarking: simple schema, well-understood.
#
#   /api/v1/chat              — OpenAI **Responses API** shape, NOT a
#       v0+ upgrade. Uses typed `input`/`output` arrays and different
#       field names (`input_tokens`, `total_output_tokens`,
#       `reasoning_output_tokens`, `time_to_first_token_seconds`,
#       `tokens_per_second`). Exposes the thinking/reasoning token split
#       explicitly. Adapter not implemented yet — it's a separate request
#       builder + response parser, not just a path rewrite.
#
# Default to v0. `--api-version v1` will error out with a clear message
# until the Responses-API adapter is written.
DEFAULT_API_VERSION = "v0"
SUPPORTED_API_VERSIONS = ("v0", "v1")


def _make_cache_buster(run_idx: Optional[int] = None) -> str:
    """Generate a prefix to bust LM Studio's prompt cache.

    LM Studio caches KV state across requests when prompts share a prefix.

    When ``run_idx`` is provided, the buster is deterministic per run index:
    all calls within the same run share the same prefix (so KV cache carries
    over across context sizes), while different runs get different prefixes
    (so runs don't interfere). When ``run_idx`` is None, a random UUID is
    used for full cold-prefill busting. ~10 tokens of overhead per prompt.
    """
    if run_idx is not None:
        return f"[run-{run_idx}]\n"
    return f"[session-{uuid.uuid4().hex[:16]}]\n"


def test_server_connection(base_url: str, timeout: int = 5) -> bool:
    """Check that the LM Studio server is reachable."""
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=timeout)
        return resp.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server: {e}")
        return False


def list_models(base_url: str, timeout: int = 5) -> List[str]:
    """Return the model IDs currently loaded in LM Studio."""
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=timeout)
        if resp.status_code != 200:
            return []
        data = resp.json()
        return [m.get("id", "") for m in data.get("data", []) if m.get("id")]
    except Exception:
        return []


def _send_request(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
    api_version: str = DEFAULT_API_VERSION,
) -> Optional[Dict]:
    """Send one non-streaming chat completion to LM Studio's native API.

    Targets `/api/{api_version}/chat/completions`. LM Studio 0.4.0+ uses
    `/api/v1`, older builds use `/api/v0`. Returns the parsed JSON response,
    or None on error. The caller is responsible for extracting stats and
    usage from the response.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": False,
    }
    try:
        resp = requests.post(
            f"{base_url}/api/{api_version}/chat/completions",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"  Error during request: {e}")
        return None


def run_benchmark(
    base_url: str,
    model: str,
    context_file: Path,
    max_tokens: int = 128,
    cold_prefill: bool = True,
    timeout: int = 3600,
    api_version: str = DEFAULT_API_VERSION,
    _run_idx: Optional[int] = None,
) -> Optional[Dict]:
    """Benchmark a single context file against LM Studio's native API.

    Args:
        base_url: LM Studio server URL (e.g. http://localhost:1234)
        model: Model ID as reported by /v1/models
        context_file: Path to the context .txt file
        max_tokens: Max generation tokens
        cold_prefill: Prepend a unique prefix to bust KV cache reuse
        timeout: Per-request timeout in seconds

    Returns:
        A result dict matching the common schema, or None on failure.
    """
    print(f"Running benchmark for {context_file}...")

    with open(context_file) as f:
        prompt = f.read()

    if cold_prefill:
        prompt = _make_cache_buster() + prompt
    elif _run_idx is not None:
        prompt = _make_cache_buster(run_idx=_run_idx) + prompt

    start_time = time.time()
    result = _send_request(base_url, model, prompt, max_tokens, timeout, api_version)
    total_time = time.time() - start_time

    if result is None:
        return None

    # Extract generated text. LM Studio puts thinking-mode output in
    # `reasoning_content` and leaves `content` empty, so fall back to it
    # whenever content is missing — otherwise --save-responses would save
    # empty files for reasoning-heavy generations.
    choices = result.get("choices") or []
    generated_text = ""
    if choices:
        message = choices[0].get("message") or {}
        generated_text = message.get("content") or message.get("reasoning_content") or ""

    # Extract usage (prompt_tokens / completion_tokens). Try multiple field
    # names — LM Studio versions vary on what they populate.
    usage = result.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    generation_tokens = (
        usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("generation_tokens") or 0
    )

    # Extract LM Studio native stats. Try multiple possible field locations
    # since LM Studio has moved these around between versions — stats could
    # live under `stats`, `runtime`, or directly on the response envelope.
    stats = result.get("stats") or result.get("runtime") or result.get("metrics") or {}

    def _stat(name_candidates):
        """Look up the first non-zero value for any of the candidate keys."""
        for key in name_candidates:
            if key in stats and stats[key]:
                try:
                    return float(stats[key])
                except (TypeError, ValueError):
                    continue
            # Some versions expose stats at the top level of the response
            if key in result and result[key]:
                try:
                    return float(result[key])
                except (TypeError, ValueError):
                    continue
        return 0.0

    server_ttft = _stat(["time_to_first_token", "time_to_first_token_seconds", "ttft", "ttft_seconds"])
    server_generation_time = _stat(["generation_time", "generation_time_seconds", "eval_duration"])
    server_generation_tps = _stat(["tokens_per_second", "generation_tps", "eval_tps", "tps"])
    server_prompt_tps = _stat(["prompt_tps", "prompt_tokens_per_second", "prompt_eval_tps"])

    # If we got prompt_tokens but no prompt_eval_count, that's fine — use
    # prompt_tokens. Refuse to fabricate a count via word splitting if both
    # are missing.
    if prompt_tokens == 0:
        print("  ERROR: LM Studio did not report prompt_tokens. Skipping this row.")
        return None

    # Tokenizer-independent truncation check. Context files are generated
    # with tiktoken cl100k_base, but LM Studio's loaded model uses its own
    # tokenizer (typically 10-20% fewer tokens for English text), so an
    # expected-count check would false-positive. Compare against a char-
    # to-token lower bound instead: if the server processed fewer tokens
    # than file_chars / 10, something is actually being dropped.
    file_chars = len(prompt)
    min_plausible_tokens = file_chars // 10
    if prompt_tokens < min_plausible_tokens:
        print(
            f"  WARNING: processed {prompt_tokens} prompt tokens from a "
            f"{file_chars}-char file — well below the ~{min_plausible_tokens} "
            f"tokens expected. LM Studio likely truncated the prompt (context "
            f"length cap in the loaded-model settings). Results for this row "
            f"are INVALID."
        )

    # Prefer server-reported metrics where present, fall back to client-side
    # derivations. Three levels of fallback for each metric:
    #   1. Server-reported field (most accurate — library-measured)
    #   2. Derived from a sibling server field (e.g. generation_time - ttft)
    #   3. Wall-clock approximation from total_time (coarsest, but non-zero)

    ttft = server_ttft

    # Last-resort TTFT: if the server reported a generation_tps and we know
    # how many tokens it generated, we can back out the decode duration and
    # subtract from total_time to estimate TTFT. Only useful when other
    # signals failed.
    if ttft <= 0 and server_generation_tps > 0 and generation_tokens > 0:
        implied_decode = generation_tokens / server_generation_tps
        ttft = max(total_time - implied_decode, 0.0)
    # Even-more-last-resort: if we know nothing about decode timing, fall
    # back to attributing total_time to TTFT for zero-generation responses
    # (e.g. warmup with max_tokens=1 that produced only a special token).
    if ttft <= 0 and generation_tokens == 0:
        ttft = total_time

    prompt_eval_duration = ttft

    # Prompt TPS: prefer server-reported, fall back to prompt_tokens / ttft.
    # The old experimental script hardcoded this to 0; the approximation is
    # noisier than ollama's library-measured number but far more useful.
    if server_prompt_tps > 0:
        prompt_tps = server_prompt_tps
    elif prompt_eval_duration > 0:
        prompt_tps = prompt_tokens / prompt_eval_duration
    else:
        prompt_tps = 0.0

    # Generation TPS: prefer server-reported, fall back to
    # generation_tokens / decode_time. Verified from LM Studio responses:
    # `stats.generation_time` is the decode-only duration (tokens / gen_tps),
    # NOT total end-to-end wall time, so we use it directly without
    # subtracting TTFT.
    if server_generation_tps > 0:
        generation_tps = server_generation_tps
    else:
        decode_time = server_generation_time if server_generation_time > 0 else max(total_time - ttft, 0.0)
        generation_tps = generation_tokens / decode_time if decode_time > 0 else 0.0

    # Pure decode duration (eval_duration field in the result dict).
    # Same note: server_generation_time is already decode-only.
    eval_duration = server_generation_time if server_generation_time > 0 else max(total_time - ttft, 0.0)

    print(f"  Prompt: {prompt_tokens} tokens in {prompt_eval_duration:.2f}s = {prompt_tps:.1f} t/s")
    print(f"  Generation: {generation_tokens} tokens in {eval_duration:.2f}s = {generation_tps:.1f} t/s")
    if ttft > 0:
        print(f"  Time to first token: {ttft:.2f}s")
    print(f"  Total time: {total_time:.2f}s")

    return {
        "context_size": context_file.stem,
        "prompt_tokens": prompt_tokens,
        "prompt_tps": prompt_tps,
        "generation_tokens": generation_tokens,
        "generation_tps": generation_tps,
        "total_time": total_time,
        "eval_duration": eval_duration,
        "prompt_eval_duration": prompt_eval_duration,
        "time_to_first_token": ttft,
        "generated_text": generated_text,
    }


def run_batch_benchmark(
    base_url: str,
    model: str,
    batch_sizes: List[int],
    prompt_tokens: int = 2048,
    gen_tokens: int = 128,
    num_trials: int = 3,
    cold_prefill: bool = True,
    timeout: int = 3600,
    api_version: str = DEFAULT_API_VERSION,
) -> List[Dict]:
    """Run batch benchmark by firing concurrent requests at LM Studio.

    LM Studio 0.3+ supports multiple concurrent chat-completion requests. This
    measures aggregate throughput under N concurrent clients — the LM Studio
    analog of the OLLAMA_NUM_PARALLEL / mlx_lm.server `decode-concurrency`
    numbers we capture for the other providers.

    NOTE: LM Studio's concurrency behavior depends on the per-model loader
    settings in the UI ("Parallel Inference" or similar). If the server
    serializes requests, batch_size > 1 numbers will look the same as
    batch_size 1 times N — that's a UI config issue, not a benchmark bug.

    Args:
        base_url: LM Studio server URL
        model: Model ID
        batch_sizes: Concurrency levels to test
        prompt_tokens: Approximate prompt tokens per request
        gen_tokens: Tokens to generate per request
        num_trials: Trials per batch size (averaged)
        cold_prefill: Prepend a unique cache-buster per request
        timeout: Per-request timeout

    Returns:
        List of result dicts with batch_size, prompt_tps, generation_tps,
        peak_memory_gb (=0, LM Studio's API doesn't report memory).
    """
    import concurrent.futures

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
        prompt_text = "The quick brown fox jumps over the lazy dog. " * (prompt_tokens // 10)

    def single_request() -> Dict:
        """Send one non-streaming request and return usage info and per-phase timing."""
        body = (_make_cache_buster() + prompt_text) if cold_prefill else prompt_text
        result = _send_request(base_url, model, body, gen_tokens, timeout, api_version)
        if result is None:
            return {"prompt_tokens": 0, "generation_tokens": 0, "ttft": 0.0, "decode_time": 0.0}
        usage = result.get("usage") or {}
        prompt_tok = usage.get("prompt_tokens") or 0
        gen_tok = usage.get("completion_tokens") or 0

        # Extract per-request timing from LM Studio's stats field.
        stats = result.get("stats") or result.get("runtime") or result.get("metrics") or {}

        def _stat(name_candidates):
            for key in name_candidates:
                if key in stats and stats[key]:
                    try:
                        return float(stats[key])
                    except (TypeError, ValueError):
                        continue
                if key in result and result[key]:
                    try:
                        return float(result[key])
                    except (TypeError, ValueError):
                        continue
            return 0.0

        ttft = _stat(["time_to_first_token", "time_to_first_token_seconds", "ttft", "ttft_seconds"])
        gen_time = _stat(["generation_time", "generation_time_seconds", "eval_duration"])
        # If TTFT is missing but we have generation TPS + token count, back it out.
        if ttft <= 0 and gen_time <= 0:
            gen_tps = _stat(["tokens_per_second", "generation_tps", "eval_tps", "tps"])
            if gen_tps > 0 and gen_tok > 0:
                gen_time = gen_tok / gen_tps

        return {
            "prompt_tokens": prompt_tok,
            "generation_tokens": gen_tok,
            "ttft": ttft,
            "decode_time": gen_time,
        }

    batch_results = []

    for bs in batch_sizes:
        print(
            f"\n  Batch size {bs} ({num_trials} trials, ~{prompt_tokens} prompt tokens, " f"{gen_tokens} gen tokens)..."
        )

        # Per-batch-size warmup so the model has allocated all parallel
        # slots and paid any per-slot cost before timing.
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
            agg_prompt_tps = total_prompt_tok / prefill_wall
            agg_gen_tps = total_gen_tok / decode_wall

            trial_prompt_tps.append(agg_prompt_tps)
            trial_gen_tps.append(agg_gen_tps)

            print(
                f"    Trial {trial + 1}: pp {agg_prompt_tps:.1f} tg {agg_gen_tps:.1f} t/s "
                f"(wall {wall_time:.1f}s, prefill {prefill_wall:.1f}s, decode {decode_wall:.1f}s)"
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
                    # LM Studio's API doesn't report memory; stub for schema
                    "peak_memory_gb": 0.0,
                }
            )

    return batch_results


def main() -> int:
    """Main function to run LM Studio benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark LM Studio local server via its native /api/v0 endpoint")
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Model ID as reported by /v1/models (auto-detected if omitted)",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"LM Studio server URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--api-version",
        default=DEFAULT_API_VERSION,
        choices=SUPPORTED_API_VERSIONS,
        help=f"LM Studio native REST API version to target (default: {DEFAULT_API_VERSION}). "
        f"v0 uses the chat/completions shape and is stable. v1 is LM Studio's "
        f"OpenAI Responses API implementation with a completely different schema "
        f"(typed input/output arrays) — not yet supported by this benchmark.",
    )

    parser.add_argument(
        "--cold-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prepend a unique marker to every prompt to bust LM Studio's KV "
        "cache reuse, forcing cold prefill on every row (default: enabled; "
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
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per context size; peak score is kept (default: 3)",
    )

    common.setup_common_args(parser)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    # Guard rail: v1 uses a completely different (Responses API) schema
    # than our existing chat/completions request builder. Refuse with a
    # clear explanation rather than producing mysterious 404s or empty
    # result dicts.
    if args.api_version == "v1":
        print(
            "Error: --api-version v1 is not supported yet.\n\n"
            "LM Studio 0.4.0's /api/v1/chat uses the OpenAI Responses API schema\n"
            "(typed `input`/`output` arrays, `input_tokens`/`total_output_tokens`,\n"
            "`reasoning_output_tokens`), NOT the chat/completions shape this script\n"
            "is built around. Supporting it requires a separate request builder and\n"
            "response parser.\n\n"
            "For now, use --api-version v0 (the default). v0 is still fully\n"
            "functional in LM Studio 0.4.0+ and gives us everything we need for\n"
            "benchmarking: prompt_tokens, completion_tokens, time_to_first_token,\n"
            "tokens_per_second, and generation_time."
        )
        return 1

    # Test server connection
    print(f"Testing connection to LM Studio server at {base_url}...")
    if not test_server_connection(base_url):
        print(f"Error: cannot reach LM Studio server at {base_url}")
        print("Ensure LM Studio is running with a model loaded and the server started.")
        print("LM Studio → Developer → Start Server (typically port 1234).")
        return 1
    print("Connected successfully.")

    # Resolve model ID
    model_id = args.model
    available_models = list_models(base_url)
    if not model_id:
        if not available_models:
            print("Error: no models loaded in LM Studio and none provided on CLI.")
            return 1
        model_id = available_models[0]
        print(f"Auto-detected model: {model_id}")
    else:
        if available_models and model_id not in available_models:
            print(f"Warning: '{model_id}' not in /v1/models response. Available:")
            for m in available_models:
                print(f"  - {m}")
            print("Attempting anyway...")

    # Hardware info
    print("\nCollecting hardware information...")
    hardware_info = common.get_hardware_info()
    hardware_str = common.format_hardware_string(hardware_info)
    print(f"Hardware: {hardware_str}")
    print(f"Model: {model_id}")
    print(f"Max tokens: {args.max_tokens}")
    print(
        f"Cold prefill: {'enabled (cache busted per prompt)' if args.cold_prefill else 'disabled (cache reuse allowed)'}"
    )
    print(f"API version: /api/{args.api_version}/")

    # Find context files
    context_files = common.find_context_files(args.contexts)
    if not context_files:
        return 1

    # Output directory
    output_dir = common.create_output_directory("lmstudio", model_id, cold_prefill=args.cold_prefill)

    # Warmup. Use a realistic max_tokens (not 1) so thinking-mode models
    # actually emit content tokens — with max_tokens=1 a Qwen3.5-style
    # model may generate only the opening `<think>` special token, which
    # LM Studio reports as completion_tokens=0 and makes the diagnostic
    # output misleading.
    warmup_file = common.find_warmup_file()
    if warmup_file:
        print(f"\n{'=' * 50}")
        print(f"Warmup run (excluded from results): {warmup_file.name}")
        print(f"{'=' * 50}")
        run_benchmark(
            base_url,
            model_id,
            warmup_file,
            max_tokens=16,
            cold_prefill=args.cold_prefill,
            timeout=args.timeout,
            api_version=args.api_version,
        )
        print("Warmup complete.")
    else:
        print("Warning: 0.5k.txt not found, skipping warmup.")

    # Context sweep
    start_time = time.time()
    results = []
    if args.cold_prefill:
        for file in context_files:
            print(f"\n{'=' * 50}")
            print(f"Benchmarking {file.name}...")
            print(f"{'=' * 50}")

            result = common.run_benchmark_peak(
                run_benchmark,
                base_url,
                model_id,
                file,
                max_tokens=args.max_tokens,
                cold_prefill=args.cold_prefill,
                timeout=args.timeout,
                api_version=args.api_version,
                n_runs=args.runs,
            )
            if result:
                results.append(result)

                if args.save_responses:
                    resp_path = output_dir / f"response_{result['context_size']}.txt"
                    common.save_generated_text(result, model_id, resp_path, "LM Studio")
    else:
        results = common.run_benchmark_peak_per_run(
            run_benchmark,
            context_files=context_files,
            n_runs=args.runs,
            base_url=base_url,
            model=model_id,
            max_tokens=args.max_tokens,
            cold_prefill=args.cold_prefill,
            timeout=args.timeout,
            api_version=args.api_version,
        )
        if args.save_responses:
            for result in results:
                resp_path = output_dir / f"response_{result['context_size']}.txt"
                common.save_generated_text(result, model_id, resp_path, "LM Studio")

    total_benchmark_time = time.time() - start_time

    if not results:
        print("\nNo successful benchmark results")
        return 1

    # Batch benchmark
    batch_results = None
    if not args.no_batch:
        batch_sizes = [int(s.strip()) for s in args.batch_sizes.split(",")]
        print(f"\nRunning batch benchmark (concurrent requests: {batch_sizes})...")
        print(
            "  NOTE: LM Studio's concurrency depends on the per-model loader "
            "settings in the UI. If trials at bs>1 run at the same speed as bs=1, "
            "the server is serializing requests — enable parallel inference in "
            "the model's load-time settings."
        )
        try:
            batch_results = run_batch_benchmark(
                base_url,
                model_id,
                batch_sizes,
                prompt_tokens=args.batch_prompt_tokens,
                gen_tokens=args.batch_gen_tokens,
                num_trials=args.batch_trials,
                cold_prefill=args.cold_prefill,
                timeout=args.timeout,
                api_version=args.api_version,
            )
            if batch_results:
                print(f"\nBatch benchmark complete: {len(batch_results)} sizes tested")
            else:
                print("\nBatch benchmark produced no results")
        except Exception as e:
            print(f"\nBatch benchmark failed (continuing): {e}")

    # Save all outputs
    common.save_all_outputs(
        results,
        output_dir,
        model_id,
        "LM Studio",
        hardware_info,
        args,
        batch_results=batch_results,
    )

    # Print summary
    common.print_benchmark_summary(
        results,
        model_id,
        "LM Studio",
        hardware_info,
        output_dir,
        total_benchmark_time,
        batch_results=batch_results,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
