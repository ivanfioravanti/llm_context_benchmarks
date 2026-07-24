#!/usr/bin/env python3
"""
Benchmark script for any OpenAI-compatible API endpoint.

Connects to any server that implements the OpenAI Chat Completions API
and benchmarks its performance across different context sizes.

Supports vLLM, llama.cpp server, Ollama, LM Studio, text-generation-webui,
and any other OpenAI-compatible server.

Usage:
    # Against a local vLLM / llama.cpp / Ollama server:
    python openai_benchmark.py --model meta-llama/Llama-3.1-8B-Instruct
    python openai_benchmark.py --model llama3.2 --base-url http://localhost:11434/v1

    # Against a remote server with an API key:
    python openai_benchmark.py --model gpt-4o --base-url https://api.openai.com/v1 --api-key sk-...

    # Custom contexts and token budget:
    python openai_benchmark.py --model mistral --contexts 2,4,8,16 --max-tokens 500
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

import benchmark_common as common


def build_client(base_url: str, api_key: str) -> OpenAI:
    """Create an OpenAI client pointed at the given base URL."""
    return OpenAI(base_url=base_url, api_key=api_key)


def test_server_connection(client: OpenAI) -> bool:
    """Check that the server is reachable and returns a model list."""
    try:
        client.models.list()
        return True
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return False


def measure_endpoint_latency(client: OpenAI, samples: int = 5) -> Dict[str, float]:
    """Measure warm authenticated round-trip latency against ``/v1/models``.

    ``test_server_connection`` runs first and warms DNS, TLS, and the client's
    connection pool. These samples therefore estimate the recurring endpoint
    overhead present in each benchmark request rather than one-time setup cost.
    """
    latencies = []
    for _ in range(max(samples, 0)):
        start = time.perf_counter()
        try:
            client.models.list()
        except Exception as exc:
            print(f"Warning: endpoint latency probe failed: {exc}")
            break
        latencies.append(time.perf_counter() - start)

    if not latencies:
        return {"mean_s": 0.0, "median_s": 0.0, "min_s": 0.0, "max_s": 0.0, "samples": 0}
    return {
        "mean_s": statistics.fmean(latencies),
        "median_s": statistics.median(latencies),
        "min_s": min(latencies),
        "max_s": max(latencies),
        "samples": len(latencies),
    }


def get_available_model(client: OpenAI) -> Optional[str]:
    """Return the first model ID reported by the server, or None."""
    try:
        models = list(client.models.list())
        if models:
            return models[0].id
    except Exception:
        pass
    return None


class HostMemorySampler:
    """Samples system RAM while a request runs. On Apple Silicon the unified
    memory is also the GPU memory, so for a locally hosted server this is the
    closest thing to server RAM/VRAM a client can observe. Only meaningful
    when the endpoint runs on this machine (see common.is_local_base_url)."""

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.peak_bytes = 0
        self._stop = None
        self._thread = None

    def __enter__(self):
        import threading

        import psutil

        self._stop = threading.Event()

        def sample():
            while not self._stop.is_set():
                used = psutil.virtual_memory().used
                if used > self.peak_bytes:
                    self.peak_bytes = used
                self._stop.wait(self.interval)

        self.peak_bytes = psutil.virtual_memory().used
        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)
        return False

    @property
    def peak_gb(self) -> float:
        return round(self.peak_bytes / (1024**3), 2)


# Set from main() when the endpoint is local; run_benchmark/run_batch_benchmark
# then record host_memory_gb (peak system RAM during the request).
SAMPLE_HOST_MEMORY = False


def run_benchmark(
    client: OpenAI,
    model: str,
    context_file: Path,
    max_tokens: int = 128,
    timeout: int = 3600,
    temperature: float = 1.0,
    endpoint_latency: Optional[Dict[str, float]] = None,
    cold_prefill: bool = True,
    _run_idx: Optional[int] = None,
) -> Optional[Dict]:
    """Benchmark a single context file against the OpenAI-compatible endpoint.

    Uses streaming so we can measure time-to-first-token (TTFT) accurately.

    Returns a result dict on success, None on failure.
    """
    with open(context_file) as f:
        prompt = f.read()

    if cold_prefill:
        prompt = common.make_cache_buster() + prompt
    elif _run_idx is not None:
        prompt = common.make_cache_buster(run_idx=_run_idx) + prompt

    host_mem_sampler = HostMemorySampler() if SAMPLE_HOST_MEMORY else None
    if host_mem_sampler:
        host_mem_sampler.__enter__()
    try:
        stream_result = common.stream_chat(client, model, prompt, max_tokens, temperature=temperature, timeout=timeout)
    except Exception as e:
        print(f"Error during benchmark: {e}")
        return None
    finally:
        if host_mem_sampler:
            host_mem_sampler.__exit__()

    generated_text = stream_result["generated_text"]
    reasoning_text = stream_result["reasoning_text"]
    total_time = stream_result["total_time"]

    usage_extra = stream_result["usage"]
    prompt_tokens = usage_extra.get("prompt_tokens") or usage_extra.get("input_tokens") or 0
    completion_tokens = usage_extra.get("completion_tokens") or usage_extra.get("output_tokens") or 0

    # Extract server-reported stats; key names vary by server.
    # TPS keys: mlx-vlm uses *_tps, oMLX uses *_tokens_per_second,
    # mtplx uses prefill_tps/decode_tps.
    server_prompt_tps = (
        usage_extra.get("prompt_tps")
        or usage_extra.get("prompt_tokens_per_second")
        or usage_extra.get("prefill_tps")
        or 0.0
    )
    server_generation_tps = (
        usage_extra.get("generation_tps")
        or usage_extra.get("generation_tokens_per_second")
        or usage_extra.get("decode_tps")
        or 0.0
    )
    server_prompt_eval = (
        usage_extra.get("prompt_eval_duration") or usage_extra.get("prefill_time_s") or usage_extra.get("ttft_s") or 0.0
    )
    server_gen_duration = usage_extra.get("generation_duration") or usage_extra.get("decode_time_s") or 0.0
    peak_memory_gb = usage_extra.get("peak_memory", 0.0) or 0.0

    # TTFT is always the end-to-end client observation. For prompt throughput,
    # prefer server timing; otherwise remove the measured warm endpoint
    # round-trip baseline from TTFT. Both raw and adjusted values are retained.
    client_ttft = stream_result["time_to_first_token"]
    latency_stats = endpoint_latency or {}
    endpoint_latency_s = latency_stats.get("mean_s", 0.0) or 0.0
    latency_adjusted_prompt_time = client_ttft - endpoint_latency_s if client_ttft > endpoint_latency_s else client_ttft
    prompt_duration = server_prompt_eval if server_prompt_eval > 0 else latency_adjusted_prompt_time
    generation_time = server_gen_duration if server_gen_duration > 0 else stream_result["decode_window"]

    # Fallback: approximate prompt token count from word count
    if prompt_tokens == 0:
        prompt_tokens = len(prompt.split())

    # Fallback: approximate completion tokens from generated text + reasoning
    if completion_tokens == 0:
        completion_tokens = len(generated_text.split()) + len(reasoning_text.split())

    # Prefer server-reported TPS when available; otherwise compute client-side.
    # For decode TPS, use (N - 1) tokens since the first token's generation is
    # already captured in TTFT, leaving N-1 inter-token intervals.
    prompt_tps_e2e = prompt_tokens / client_ttft if client_ttft > 0 else 0.0
    prompt_tps_latency_adjusted = (
        prompt_tokens / latency_adjusted_prompt_time if latency_adjusted_prompt_time > 0 else 0.0
    )
    prompt_tps = (
        server_prompt_tps
        if server_prompt_tps > 0
        else (prompt_tokens / prompt_duration if prompt_duration > 0 else 0.0)
    )
    if server_generation_tps > 0:
        generation_tps = server_generation_tps
    elif generation_time > 0 and completion_tokens > 1:
        generation_tps = (completion_tokens - 1) / generation_time
    else:
        generation_tps = 0.0

    if server_prompt_tps > 0:
        prompt_tps_source = "server"
    elif server_prompt_eval > 0:
        prompt_tps_source = "server timing"
    elif endpoint_latency_s > 0:
        prompt_tps_source = "client, latency-adjusted"
    else:
        prompt_tps_source = "client, end-to-end"
    generation_tps_source = "server" if server_generation_tps > 0 else "client"

    print(f"  Prompt tokens:      {prompt_tokens}")
    print(f"  Completion tokens:  {completion_tokens}")
    if reasoning_text:
        print(f"  Reasoning chars:    {len(reasoning_text)} (counted toward decode window)")
    print(f"  TTFT (end-to-end):  {client_ttft:.3f}s")
    if endpoint_latency_s > 0 and server_prompt_eval <= 0:
        print(f"  Endpoint latency:   {endpoint_latency_s * 1000:.1f}ms average")
        print(f"  Prompt time adj.:   {latency_adjusted_prompt_time:.3f}s")
    print(f"  Generation time:    {generation_time:.2f}s")
    print(f"  Total time:         {total_time:.2f}s")
    print(f"  Prompt TPS:         {prompt_tps:.1f} t/s ({prompt_tps_source})")
    if endpoint_latency_s > 0 and server_prompt_tps <= 0 and prompt_tps_e2e > 0:
        print(f"  Prompt TPS e2e:     {prompt_tps_e2e:.1f} t/s (unadjusted)")
    print(f"  Generation TPS:     {generation_tps:.1f} t/s ({generation_tps_source}-side)")
    if peak_memory_gb > 0:
        print(f"  Peak memory:        {peak_memory_gb:.2f} GB")

    result = {
        "context_size": context_file.stem,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": completion_tokens,
        "time_to_first_token": client_ttft,
        "eval_duration": generation_time,
        "prompt_eval_duration": prompt_duration,
        "total_time": total_time,
        "prompt_tps": prompt_tps,
        "prompt_tps_e2e": prompt_tps_e2e,
        "prompt_tps_latency_adjusted": prompt_tps_latency_adjusted,
        "generation_tps": generation_tps,
        "generated_text": generated_text,
    }
    if endpoint_latency_s > 0:
        result.update(
            {
                "endpoint_latency_ms": endpoint_latency_s * 1000,
                "endpoint_latency_median_ms": (latency_stats.get("median_s", 0.0) or 0.0) * 1000,
                "endpoint_latency_min_ms": (latency_stats.get("min_s", 0.0) or 0.0) * 1000,
                "endpoint_latency_max_ms": (latency_stats.get("max_s", 0.0) or 0.0) * 1000,
                "endpoint_latency_samples": latency_stats.get("samples", 0),
            }
        )
    if reasoning_text:
        result["reasoning_text"] = reasoning_text
    if peak_memory_gb > 0:
        result["peak_memory_gb"] = peak_memory_gb
    if host_mem_sampler:
        result["host_memory_gb"] = host_mem_sampler.peak_gb
        print(f"  Host memory peak:   {host_mem_sampler.peak_gb:.2f} GB (system RAM, local server)")

    return common.add_throughput_metrics(result, prompt_text=prompt)


def run_batch_benchmark(
    base_url: str,
    api_key: str,
    model: str,
    batch_sizes: List[int],
    prompt_tokens: int = 2048,
    gen_tokens: int = 128,
    num_trials: int = 3,
    temperature: float = 1.0,
    endpoint_latency_s: float = 0.0,
) -> List[Dict]:
    """Run batch benchmark by sending concurrent requests to test continuous batching.

    Args:
        base_url: Server base URL
        api_key: API key
        model: Model name
        batch_sizes: List of batch sizes to test (concurrent requests)
        prompt_tokens: Approximate prompt tokens per request
        gen_tokens: Tokens to generate per request
        num_trials: Number of trials per batch size

    Returns:
        List of result dicts with batch_size, prompt_tps, generation_tps
    """
    import concurrent.futures
    import statistics

    import tiktoken

    # Generate a fixed prompt of approximately prompt_tokens length
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        # Use a repeated phrase to fill the token budget
        base_text = "The quick brown fox jumps over the lazy dog. "
        base_tokens = enc.encode(base_text)
        repeats = max(1, prompt_tokens // len(base_tokens))
        prompt_text = base_text * repeats
        # Trim to exact token count
        tokens = enc.encode(prompt_text)[:prompt_tokens]
        prompt_text = enc.decode(tokens)
    except Exception:
        # Fallback: approximate ~4 chars per token
        prompt_text = "The quick brown fox jumps over the lazy dog. " * (prompt_tokens // 10)

    import httpx

    http_client = httpx.Client(
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=600,
    )

    def single_request():
        """Send one non-streaming request and return (prompt_tps, gen_tps, prompt_tokens, gen_tokens)."""
        resp = http_client.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": gen_tokens,
                "temperature": temperature,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        return {
            "prompt_tps": usage.get("prompt_tps", 0),
            "generation_tps": usage.get("generation_tps", 0),
            "prompt_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
            "generation_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
            "peak_memory": usage.get("peak_memory", 0),
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
        trial_prompt_tps_e2e = []
        trial_gen_tps_e2e = []
        trial_peak_mem = []
        trial_decode_tps = []  # pure decode rate summed across clients (from server usage)

        trial_host_mem = []
        for trial in range(num_trials):
            start = time.time()
            host_mem_sampler = HostMemorySampler() if SAMPLE_HOST_MEMORY else None
            if host_mem_sampler:
                host_mem_sampler.__enter__()
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as pool:
                    futures = [pool.submit(single_request) for _ in range(bs)]
                    responses = [f.result() for f in futures]
            finally:
                if host_mem_sampler:
                    host_mem_sampler.__exit__()
                    trial_host_mem.append(host_mem_sampler.peak_gb)
            wall_time = time.time() - start

            total_prompt_tok = sum(r["prompt_tokens"] for r in responses)
            total_gen_tok = sum(r["generation_tokens"] for r in responses)
            adjusted_wall_time = wall_time
            if endpoint_latency_s > 0 and wall_time > endpoint_latency_s:
                adjusted_wall_time = wall_time - endpoint_latency_s
            agg_prompt_tps_e2e = total_prompt_tok / wall_time if wall_time > 0 else 0
            agg_gen_tps_e2e = total_gen_tok / wall_time if wall_time > 0 else 0
            agg_prompt_tps = total_prompt_tok / adjusted_wall_time if adjusted_wall_time > 0 else 0
            agg_gen_tps = total_gen_tok / adjusted_wall_time if adjusted_wall_time > 0 else 0
            peak_mem = max((r["peak_memory"] for r in responses), default=0)

            trial_prompt_tps.append(agg_prompt_tps)
            trial_gen_tps.append(agg_gen_tps)
            trial_prompt_tps_e2e.append(agg_prompt_tps_e2e)
            trial_gen_tps_e2e.append(agg_gen_tps_e2e)
            if peak_mem > 0:
                trial_peak_mem.append(peak_mem)

            # Pure decode throughput: the server reports each request's own
            # generation_tps (decode phase only). Summed across the N parallel
            # clients this is the aggregate decode rate, independent of how
            # long the prompt phase took.
            per_request_decode = [r["generation_tps"] for r in responses if r.get("generation_tps")]
            if len(per_request_decode) == len(responses):
                trial_decode_tps.append(sum(per_request_decode))

            latency_note = " latency-adjusted" if adjusted_wall_time != wall_time else ""
            print(
                f"    Trial {trial + 1}: pp {agg_prompt_tps:.1f} tg {agg_gen_tps:.1f} t/s "
                f"({wall_time:.1f}s e2e{latency_note})"
            )

        if trial_prompt_tps:
            avg_prompt = statistics.mean(trial_prompt_tps)
            avg_gen = statistics.mean(trial_gen_tps)
            result = {
                "batch_size": bs,
                "prompt_tps": round(avg_prompt, 2),
                "generation_tps": round(avg_gen, 2),
                "prompt_tps_e2e": round(statistics.mean(trial_prompt_tps_e2e), 2),
                "generation_tps_e2e": round(statistics.mean(trial_gen_tps_e2e), 2),
            }
            if endpoint_latency_s > 0:
                result["endpoint_latency_ms"] = round(endpoint_latency_s * 1000, 3)
            if trial_decode_tps:
                avg_decode = statistics.mean(trial_decode_tps)
                result["decode_tps_total"] = round(avg_decode, 2)
                result["decode_tps_per_client"] = round(avg_decode / bs, 2)
            if trial_host_mem:
                result["host_memory_gb"] = round(max(trial_host_mem), 2)
            if trial_peak_mem:
                result["peak_memory_gb"] = round(max(trial_peak_mem), 3)
            else:
                result["peak_memory_gb"] = 0.0

            print(f"  Avg: pp {avg_prompt:.1f} tg {avg_gen:.1f} t/s")
            batch_results.append(result)

    http_client.close()
    return batch_results


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark any OpenAI-compatible API endpoint across different context sizes"
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Model name/ID to use (auto-detected from server if omitted)",
    )
    parser.add_argument(
        "--model",
        dest="model_flag",
        default=None,
        help="Model name/ID (alternative to positional; takes precedence if both set)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080/v1",
        help="Base URL of the OpenAI-compatible server (default: http://localhost:8080/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "no-key"),
        help="API key (default: OPENAI_API_KEY env var or 'no-key' for local servers)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0; required by some hosted models, including Kimi K3)",
    )
    parser.add_argument(
        "--latency-adjustment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Measure warm endpoint latency and remove it from client-side prompt timing (default: enabled)",
    )
    parser.add_argument(
        "--latency-samples",
        type=int,
        default=5,
        help="Warm /models round trips used for endpoint-latency estimation (default: 5)",
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

    # host RAM/VRAM sampling is only meaningful when the server is local
    global SAMPLE_HOST_MEMORY
    SAMPLE_HOST_MEMORY = common.is_local_base_url(base_url)
    if SAMPLE_HOST_MEMORY:
        print("Local endpoint detected — sampling host memory (RAM/VRAM) during requests.")
    client = build_client(base_url, args.api_key)

    print(f"Testing connection to {base_url} ...")
    if not test_server_connection(client):
        print(f"Error: Cannot reach server at {base_url}")
        print("Make sure the server is running and the base URL is correct.")
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

    if model.lower().startswith("kimi-k3") and args.temperature != 1.0:
        print(f"Kimi K3 requires temperature 1.0; overriding requested value {args.temperature}.")
        args.temperature = 1.0

    endpoint_latency = None
    if args.latency_adjustment and args.latency_samples > 0:
        print(f"Measuring warm endpoint latency ({args.latency_samples} samples) ...")
        endpoint_latency = measure_endpoint_latency(client, args.latency_samples)
        if endpoint_latency["samples"]:
            print(
                f"Endpoint latency: {endpoint_latency['mean_s'] * 1000:.1f}ms average "
                f"({endpoint_latency['min_s'] * 1000:.1f}–{endpoint_latency['max_s'] * 1000:.1f}ms, "
                f"n={endpoint_latency['samples']})"
            )
        else:
            print("Endpoint latency unavailable; client statistics will remain end-to-end.")

    hardware_info = common.mark_client_hardware(common.get_hardware_info(), base_url)
    hardware_str = common.format_hardware_string(hardware_info)

    print(f"\nOpenAI-compatible Endpoint Benchmark")
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

    output_dir = common.create_output_directory("openai_compat", model, cold_prefill=args.cold_prefill)

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
                temperature=args.temperature,
                endpoint_latency=endpoint_latency,
                cold_prefill=args.cold_prefill,
                n_runs=args.runs,
            )
            if result:
                results.append(result)

                if args.save_responses:
                    resp_path = output_dir / f"response_{result['context_size']}.txt"
                    common.save_generated_text(result, model, resp_path, "OpenAI Compat")
    else:
        results = common.run_benchmark_peak_per_run(
            run_benchmark,
            context_files=context_files,
            n_runs=args.runs,
            client=client,
            model=model,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            temperature=args.temperature,
            endpoint_latency=endpoint_latency,
            cold_prefill=args.cold_prefill,
        )
        if args.save_responses:
            for result in results:
                resp_path = output_dir / f"response_{result['context_size']}.txt"
                common.save_generated_text(result, model, resp_path, "OpenAI Compat")

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
                temperature=args.temperature,
                endpoint_latency_s=(endpoint_latency or {}).get("mean_s", 0.0),
            )
            if batch_results:
                print(f"\nBatch benchmark complete: {len(batch_results)} sizes tested")
            else:
                print("\nBatch benchmark produced no results")
        except Exception as e:
            print(f"\nBatch benchmark failed (continuing): {e}")

    has_memory = any(r.get("peak_memory_gb", 0) > 0 for r in results)
    common.save_all_outputs(
        results,
        output_dir,
        model,
        "OpenAI Compat",
        hardware_info,
        args,
        include_memory=has_memory,
        batch_results=batch_results,
    )

    common.print_benchmark_summary(
        results,
        model,
        "OpenAI Compat",
        hardware_info,
        output_dir,
        total_benchmark_time,
        batch_results=batch_results,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
