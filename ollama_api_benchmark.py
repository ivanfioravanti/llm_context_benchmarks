#!/usr/bin/env python3
"""
Benchmark script for Ollama using the Python API.

This script benchmarks Ollama models using the official Python API for programmatic access.

Usage:
    python ollama_api_benchmark.py llama3.2
    python ollama_api_benchmark.py gpt-oss:20b --contexts 2,4,8,16 --max-tokens 500
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import ollama

import benchmark_common as common


def _derive_num_ctx(context_file: Path, max_tokens: int) -> tuple[int, int]:
    """Derive num_ctx and expected prompt token count from a context filename.

    Parses the stem (e.g. '64k' -> 65536, '0.5k' -> 512) and adds headroom for
    generated tokens. Returns (num_ctx, expected_prompt_tokens).
    """
    ctx_k = float(context_file.stem.rstrip("k"))
    expected_prompt_tokens = int(ctx_k * 1024)
    num_ctx = expected_prompt_tokens + max_tokens + 256
    return num_ctx, expected_prompt_tokens


def _make_cache_buster() -> str:
    """Generate a unique prefix to bust Ollama's prompt cache.

    Ollama/llama.cpp automatically reuses the KV cache whenever a new prompt
    shares a prefix with the previous request in the same slot. That makes
    prompt_eval_duration report only the *uncached delta* while
    prompt_eval_count still reports the full prompt length, inflating the
    derived prompt t/s. Prepending a per-call UUID prefix forces a prefix
    miss so every row is cold prefill. ~10 tokens of overhead per prompt.
    """
    import uuid

    return f"[session-{uuid.uuid4().hex[:16]}]\n"


def run_benchmark(
    model_name: str,
    context_file: Path,
    max_tokens: int = 128,
    cold_prefill: bool = True,
    _run_idx: Optional[int] = None,
) -> Optional[Dict]:
    """Run Ollama benchmark for a given context file.

    Args:
        model_name: Name of the Ollama model
        context_file: Path to the context file
        max_tokens: Maximum tokens to generate

    Returns:
        Dictionary with benchmark results or None if failed
    """
    print(f"Running benchmark for {context_file}...")

    # Read the prompt from file
    with open(context_file) as f:
        prompt = f.read()

    # Bust Ollama's prompt cache by prepending a unique marker. Without this,
    # the KV cache from a previous warmup or benchmark row that shares a
    # prefix will be reused, inflating prompt_tps (prompt_eval_count covers
    # the full prompt but prompt_eval_duration covers only the uncached delta).
    if cold_prefill or _run_idx is not None:
        prompt = _make_cache_buster() + prompt

    # Size the KV cache to fit this context file (plus generation headroom).
    # Without this, Ollama silently caps num_ctx at its default (2048), which
    # turns every large-context row into a measurement of ~2k prefill.
    num_ctx, expected_prompt_tokens = _derive_num_ctx(context_file, max_tokens)

    # Start timing
    start_time = time.time()

    try:
        # Run the model
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            options={
                "num_predict": max_tokens,
                "num_ctx": num_ctx,
            },
            keep_alive="10m",
        )

        # Calculate total time
        total_time = time.time() - start_time

        # Parse the response
        generated_text = response.get("response", "")
        generation_tokens = response.get("eval_count", 0)
        eval_duration = response.get("eval_duration", 0) / 1e9  # ns -> s
        prompt_eval_duration = response.get("prompt_eval_duration", 0) / 1e9  # ns -> s
        prompt_eval_count = response.get("prompt_eval_count")

        # Refuse to fabricate a prompt token count. If Ollama did not report
        # one, the row is unreliable and should be skipped rather than
        # falling back to whitespace-word counting.
        if prompt_eval_count is None:
            print(
                "  ERROR: Ollama did not return prompt_eval_count. "
                "Skipping this row (cannot compute reliable metrics)."
            )
            return None

        # Detect silent truncation. Ollama will quietly cap prompts when
        # num_ctx exceeds the model's architectural max or the user's
        # OLLAMA_CONTEXT_LENGTH env var. We CANNOT compare against the
        # filename token count because context files are generated with
        # cl100k_base (tiktoken) while the model uses its own tokenizer —
        # qwen/llama3-family tokenizers typically yield 10-20% fewer tokens
        # on English text, which would trip a naive expected-count check.
        # Instead use a tokenizer-independent char-to-token lower bound:
        # English text tokenizes to ~3-5 chars/token, so if Ollama processed
        # fewer tokens than file_chars / 10 it's genuinely dropping content.
        file_chars = len(prompt)
        min_plausible_tokens = file_chars // 10
        if prompt_eval_count < min_plausible_tokens:
            print(
                f"  WARNING: processed {prompt_eval_count} prompt tokens from a "
                f"{file_chars}-char file — well below the ~{min_plausible_tokens} "
                f"tokens expected. Ollama likely truncated the prompt (model max "
                f"context or OLLAMA_CONTEXT_LENGTH cap). Results for this row are "
                f"INVALID."
            )

        # Calculate tokens per second
        prompt_tps = (
            prompt_eval_count / prompt_eval_duration if prompt_eval_duration > 0 else 0
        )
        generation_tps = generation_tokens / eval_duration if eval_duration > 0 else 0

        # Debug logging
        print(
            f"  Prompt: {prompt_eval_count} tokens in {prompt_eval_duration:.2f}s = {prompt_tps:.1f} t/s"
        )
        print(
            f"  Generation: {generation_tokens} tokens in {eval_duration:.2f}s = {generation_tps:.1f} t/s"
        )
        if prompt_eval_duration > 0:
            print(f"  Time to first token: {prompt_eval_duration:.2f}s")
        print(f"  Total time: {total_time:.2f}s (num_ctx={num_ctx})")

        return {
            "context_size": context_file.stem,
            "prompt_tokens": prompt_eval_count,
            "prompt_tps": prompt_tps,
            "generation_tokens": generation_tokens,
            "generation_tps": generation_tps,
            "total_time": total_time,
            "eval_duration": eval_duration,
            "prompt_eval_duration": prompt_eval_duration,
            # Matches mlx_benchmark semantics: library-reported prefill duration.
            "time_to_first_token": prompt_eval_duration,
            "generated_text": generated_text,
            "num_ctx": num_ctx,
        }

    except Exception as e:
        print(f"Error running benchmark: {e}")
        return None


def check_model_available(model_name: str) -> bool:
    """Check if the model is available in Ollama.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        True if model is available, False otherwise
    """
    try:
        # Try to list models and check if our model is there
        response = ollama.list()

        # The response is a dictionary with 'models' key
        models = response.get("models", [])

        # Extract model names - the model object has a 'model' attribute
        model_names = []
        for model_obj in models:
            # Access the model attribute directly if it's an object
            if hasattr(model_obj, "model"):
                name = model_obj.model
            elif isinstance(model_obj, dict):
                # If it's a dict, try to get the 'model' key
                name = model_obj.get("model", str(model_obj))
            else:
                # Fallback to string representation
                name = str(model_obj)
            model_names.append(name)

        # Check exact match (including tags)
        if model_name in model_names:
            print(f"Found model: {model_name}")
            return True

        # Also check without considering digest (for models like "gpt-oss:20b")
        for available_model in model_names:
            if (
                available_model == model_name
                or available_model.split(":")[0] == model_name.split(":")[0]
            ):
                print(f"Found model: {available_model}")
                return True

        print(f"Model '{model_name}' not found. Available models:")
        for model in model_names:
            print(f"  - {model}")
        return False

    except Exception as e:
        print(f"Error checking model availability: {e}")
        print(f"Attempting to use model anyway...")
        return True  # Try to proceed anyway


def run_batch_benchmark(
    model_name: str,
    batch_sizes: List[int],
    prompt_tokens: int = 2048,
    gen_tokens: int = 128,
    num_trials: int = 3,
    cold_prefill: bool = True,
) -> List[Dict]:
    """Run batch benchmark by firing concurrent requests at the Ollama server.

    Ollama has no batched-forward-pass API. Instead, its server (llama.cpp under
    the hood) performs continuous batching of concurrent HTTP requests, up to
    OLLAMA_NUM_PARALLEL slots. This benchmark measures aggregate throughput
    under N concurrent clients — the Ollama analog of MLX's batch_generate.

    IMPORTANT: OLLAMA_NUM_PARALLEL must be >= max(batch_sizes) in the Ollama
    server's environment, or the server will queue requests serially and the
    measured numbers will reflect latency under contention, not batched
    throughput. This env var is read by the daemon at startup, not by the
    client.

    Args:
        model_name: Ollama model name
        batch_sizes: Concurrency levels to test
        prompt_tokens: Approximate prompt tokens per request
        gen_tokens: Tokens to generate per request
        num_trials: Trials per batch size (averaged)

    Returns:
        List of result dicts with batch_size, prompt_tps, generation_tps, peak_memory_gb
    """
    import concurrent.futures
    import statistics

    # Size num_ctx for a single slot (+ headroom for generation).
    num_ctx = prompt_tokens + gen_tokens + 256

    # Build a synthetic prompt of approximately prompt_tokens length. Uses the
    # same tiktoken cl100k_base encoding as openai_benchmark so the two engines'
    # batch numbers are directly comparable.
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

    def single_request() -> Dict:
        """Send one non-streaming request and return token usage and per-phase timing."""
        body = (_make_cache_buster() + prompt_text) if cold_prefill else prompt_text
        resp = ollama.generate(
            model=model_name,
            prompt=body,
            options={
                "num_predict": gen_tokens,
                "num_ctx": num_ctx,
            },
            keep_alive="10m",
        )
        prompt_eval_duration = resp.get("prompt_eval_duration", 0) / 1e9  # ns -> s
        eval_duration = resp.get("eval_duration", 0) / 1e9  # ns -> s
        return {
            "prompt_tokens": resp.get("prompt_eval_count", 0),
            "generation_tokens": resp.get("eval_count", 0),
            "prompt_eval_duration": prompt_eval_duration,
            "eval_duration": eval_duration,
        }

    batch_results = []

    for bs in batch_sizes:
        print(
            f"\n  Batch size {bs} ({num_trials} trials, ~{prompt_tokens} prompt tokens, "
            f"{gen_tokens} gen tokens)..."
        )

        # Warmup at this batch size so the server has allocated all slots and
        # any one-time per-slot overhead is paid before we start timing.
        print("    Warmup...")
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as pool:
                list(pool.map(lambda _: single_request(), range(bs)))
        except Exception as e:
            print(f"    Warmup failed for batch size {bs}: {e} — skipping this size")
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
            sum_prefill = sum(r["prompt_eval_duration"] for r in responses)
            sum_decode = sum(r["eval_duration"] for r in responses)
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
                    # Ollama doesn't report peak memory via the API.
                    "peak_memory_gb": 0.0,
                }
            )

    return batch_results


def main() -> int:
    """Main function to run Ollama API benchmarks."""
    parser = argparse.ArgumentParser(
        description="Run Ollama benchmarks using Python API"
    )
    parser.add_argument(
        "model", help="Ollama model name (e.g., llama3.2, mistral)"
    )

    parser.add_argument(
        "--cold-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prepend a unique marker to every prompt to bust Ollama's KV "
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

    # Add common arguments
    common.setup_common_args(parser)

    args = parser.parse_args()

    # Check if model is available
    if not check_model_available(args.model):
        print(f"\nPlease pull the model first with: ollama pull {args.model}")
        return 1

    # Create output directory using common function
    output_dir = common.create_output_directory("ollama_api", args.model, cold_prefill=args.cold_prefill)

    # Find context files using common module
    context_files = common.find_context_files(args.contexts)
    if not context_files:
        return 1

    # Get hardware information
    print("\nCollecting hardware information...")
    hardware_info = common.get_hardware_info()
    hardware_str = common.format_hardware_string(hardware_info)
    print(f"Hardware: {hardware_str}")
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Cold prefill: {'enabled (cache busted per prompt)' if args.cold_prefill else 'disabled (cache reuse allowed)'}")

    # Warmup run — max_tokens=1 is enough to load weights and hit the code
    # path; mirrors mlx_benchmark's warmup (see mlx_benchmark.py:749).
    warmup_file = common.find_warmup_file()
    if warmup_file:
        print(f"\n{'=' * 50}")
        print(f"Warmup run (excluded from results): {warmup_file.name}")
        print(f"{'=' * 50}")
        run_benchmark(args.model, warmup_file, max_tokens=1, cold_prefill=args.cold_prefill)
        print("Warmup complete.")
    else:
        print("Warning: 0.5k.txt not found, skipping warmup.")

    # Run benchmarks
    start_time = time.time()
    results = []
    for file in context_files:
        print(f"\n{'=' * 50}")
        print(f"Benchmarking {file.name}...")
        print(f"{'=' * 50}")

        result = common.run_benchmark_peak(run_benchmark, args.model, file, args.max_tokens, cold_prefill=args.cold_prefill, n_runs=args.runs)
        if result:
            results.append(result)

            # Save the generated text if requested
            if args.save_responses:
                output_filename = output_dir / f"response_{result['context_size']}.txt"
                common.save_generated_text(
                    result, args.model, output_filename, "Ollama API"
                )
    
    total_benchmark_time = time.time() - start_time

    if not results:
        print("\nNo successful benchmark results")
        return 1

    # Run batch benchmark (concurrent-request continuous batching)
    batch_results = None
    if not args.no_batch:
        batch_sizes = [int(s.strip()) for s in args.batch_sizes.split(",")]
        max_bs = max(batch_sizes)
        print(f"\nRunning batch benchmark (concurrent requests: {batch_sizes})...")

        # Try to warn about OLLAMA_NUM_PARALLEL misconfiguration. We can only
        # see it if it's set in the current shell — on macOS the daemon is
        # typically launched via launchctl and inherits its own environment,
        # so the absence of the var here doesn't prove anything. Hence a
        # soft NOTE rather than a hard error.
        num_parallel = os.environ.get("OLLAMA_NUM_PARALLEL")
        if num_parallel is not None:
            try:
                if int(num_parallel) < max_bs:
                    print(
                        f"  WARNING: OLLAMA_NUM_PARALLEL={num_parallel} < max batch size {max_bs}. "
                        f"The server will queue requests serially — numbers will be misleading."
                    )
            except ValueError:
                pass
        else:
            print(
                f"  NOTE: OLLAMA_NUM_PARALLEL is not set in this shell. Ensure the Ollama "
                f"daemon was started with OLLAMA_NUM_PARALLEL >= {max_bs}, otherwise the "
                f"server will queue requests and report latency, not batched throughput."
            )

        try:
            batch_results = run_batch_benchmark(
                args.model,
                batch_sizes,
                prompt_tokens=args.batch_prompt_tokens,
                gen_tokens=args.batch_gen_tokens,
                num_trials=args.batch_trials,
                cold_prefill=args.cold_prefill,
            )
            if batch_results:
                print(f"\nBatch benchmark complete: {len(batch_results)} sizes tested")
            else:
                print("\nBatch benchmark produced no results")
        except Exception as e:
            print(f"\nBatch benchmark failed (continuing): {e}")

    # Save all outputs using common function
    common.save_all_outputs(
        results,
        output_dir,
        args.model,
        "Ollama API",
        hardware_info,
        args,
        batch_results=batch_results,
    )

    # Print summary using common function
    common.print_benchmark_summary(
        results,
        args.model,
        "Ollama API",
        hardware_info,
        output_dir,
        total_benchmark_time,
        batch_results=batch_results,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())