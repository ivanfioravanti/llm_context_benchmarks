#!/usr/bin/env python3
"""
Benchmark script for Ollama using the CLI.

This script benchmarks Ollama models using the command-line interface for direct testing.

Usage:
    python ollama_cli_benchmark.py llama3.2
    python ollama_cli_benchmark.py gpt-oss:20b --contexts 2,4,8,16 --max-tokens 500
"""

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import benchmark_common as common


def _derive_num_ctx(context_file: Path, max_tokens: int) -> int:
    """Derive num_ctx for a context filename (stem like '64k' -> 65536)."""
    ctx_k = float(context_file.stem.rstrip("k"))
    expected_prompt_tokens = int(ctx_k * 1024)
    return expected_prompt_tokens + max_tokens + 256


def _create_temp_model(base_model: str, num_ctx: int, num_predict: int) -> str:
    """Create a temporary ollama model that overrides num_ctx and num_predict.

    The `ollama run` CLI has no flags for num_ctx or num_predict. The only way
    to set them is via a Modelfile. We create a tiny Modelfile that points
    FROM the base model and layers the parameter overrides on top — ollama
    create is fast for this because it doesn't copy weights, just adds a
    parameter layer to the manifest.
    """
    # Deterministic tag per (base, num_ctx, num_predict) so repeated benchmark
    # runs reuse the same temp model instead of accumulating stale ones.
    key = f"{base_model}:{num_ctx}:{num_predict}".encode()
    tag = "benchmark-tmp-" + hashlib.md5(key).hexdigest()[:10]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".modelfile", delete=False) as f:
        f.write(f"FROM {base_model}\n")
        f.write(f"PARAMETER num_ctx {num_ctx}\n")
        f.write(f"PARAMETER num_predict {num_predict}\n")
        modelfile_path = f.name

    try:
        result = subprocess.run(
            ["ollama", "create", tag, "-f", modelfile_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ollama create failed for temp model '{tag}': {result.stderr.strip()}")
    finally:
        try:
            os.unlink(modelfile_path)
        except OSError:
            pass

    return tag


def _remove_temp_model(tag: str) -> None:
    """Best-effort removal of a temporary ollama model."""
    try:
        subprocess.run(
            ["ollama", "rm", tag],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        pass


def _make_cache_buster(run_idx: Optional[int] = None) -> str:
    """Generate a prefix to bust Ollama's prompt cache.

    Ollama reuses KV cache whenever a new prompt shares a prefix with the
    previous request.

    When ``run_idx`` is provided, the buster is deterministic per run index:
    all calls within the same run share the same prefix (so KV cache carries
    over across context sizes), while different runs get different prefixes
    (so runs don't interfere). When ``run_idx`` is None, a random UUID is
    used for full cold-prefill busting. ~10 tokens of overhead per prompt.
    """
    if run_idx is not None:
        return f"[run-{run_idx}]\n"
    import uuid

    return f"[session-{uuid.uuid4().hex[:16]}]\n"


def parse_ollama_output(output: str) -> Dict:
    """Parse the verbose output from ollama run command.

    Args:
        output: Raw output from ollama CLI

    Returns:
        Dictionary with parsed metrics
    """
    # Remove ANSI escape sequences and terminal control codes
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    output = ansi_escape.sub("", output)

    metrics = {}

    # Parse total duration
    total_match = re.search(r"total duration:\s+([\d.]+)([a-z]+)", output)
    if total_match:
        value = float(total_match.group(1))
        unit = total_match.group(2)
        if unit == "ms":
            metrics["total_duration"] = value / 1000
        elif unit == "s":
            metrics["total_duration"] = value
        else:
            metrics["total_duration"] = value

    # Parse load duration
    load_match = re.search(r"load duration:\s+([\d.]+)([a-z]+)", output)
    if load_match:
        value = float(load_match.group(1))
        unit = load_match.group(2)
        if unit == "ms":
            metrics["load_duration"] = value / 1000
        elif unit == "s":
            metrics["load_duration"] = value
        else:
            metrics["load_duration"] = value

    # Parse prompt eval count
    prompt_count_match = re.search(r"prompt eval count:\s+(\d+)", output)
    if prompt_count_match:
        metrics["prompt_eval_count"] = int(prompt_count_match.group(1))

    # Parse prompt eval duration
    prompt_dur_match = re.search(r"prompt eval duration:\s+([\d.]+)([a-z]+)", output)
    if prompt_dur_match:
        value = float(prompt_dur_match.group(1))
        unit = prompt_dur_match.group(2)
        if unit == "ms":
            metrics["prompt_eval_duration"] = value / 1000
        elif unit == "s":
            metrics["prompt_eval_duration"] = value
        else:
            metrics["prompt_eval_duration"] = value

    # Parse prompt eval rate
    prompt_rate_match = re.search(r"prompt eval rate:\s+([\d.]+)\s+tokens/s", output)
    if prompt_rate_match:
        metrics["prompt_eval_rate"] = float(prompt_rate_match.group(1))

    # Parse eval count (generation tokens) - looking for line without "prompt" prefix
    eval_count_match = re.search(r"^eval count:\s+(\d+)", output, re.MULTILINE)
    if eval_count_match:
        metrics["eval_count"] = int(eval_count_match.group(1))

    # Parse eval duration (generation time) - looking for line without "prompt" prefix
    eval_dur_match = re.search(r"^eval duration:\s+([\d.]+)([a-z]+)", output, re.MULTILINE)
    if eval_dur_match:
        value = float(eval_dur_match.group(1))
        unit = eval_dur_match.group(2)
        if unit == "ms":
            metrics["eval_duration"] = value / 1000
        elif unit == "s":
            metrics["eval_duration"] = value
        else:
            metrics["eval_duration"] = value

    # Parse eval rate (generation tokens per second) - looking for line without "prompt" prefix
    eval_rate_match = re.search(r"^eval rate:\s+([\d.]+)\s+tokens/s", output, re.MULTILINE)
    if eval_rate_match:
        metrics["eval_rate"] = float(eval_rate_match.group(1))

    return metrics


def extract_generated_text(stdout: str, stderr: str, prompt: str) -> str:
    """Extract the generated text from the ollama output.

    With --verbose flag:
    - The generated text appears in stdout
    - The metrics appear in stderr

    Args:
        stdout: Standard output from ollama
        stderr: Standard error from ollama
        prompt: The original prompt

    Returns:
        The generated text
    """
    # The model's response should be the entire stdout
    # since metrics go to stderr with --verbose
    generated_text = stdout.strip()

    # If the stdout starts with the prompt (sometimes ollama echoes it),
    # remove it to get just the generated text
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt) :].strip()

    return generated_text


def run_cli_benchmark(
    model_name: str,
    context_file: Path,
    cold_prefill: bool = True,
    timeout: int = 3600,
    _run_idx: Optional[int] = None,
) -> Optional[Dict]:
    """Run Ollama benchmark using CLI for a given context file.

    Args:
        model_name: Name of the Ollama model (expected to be a temp model with
            num_ctx and num_predict parameters baked in via Modelfile)
        context_file: Path to the context file
        cold_prefill: If True, prepend a unique cache-buster to the prompt so
            Ollama's KV cache reuse doesn't inflate prompt-processing numbers
        timeout: subprocess timeout in seconds

    Returns:
        Dictionary with benchmark results or None if failed
    """
    print(f"Running CLI benchmark for {context_file}...")

    # Read the prompt from file
    with open(context_file) as f:
        prompt = f.read()

    # Bust Ollama's prompt cache by prepending a unique marker so no two
    # prompts share a prefix. Adds ~10 tokens of overhead.
    if cold_prefill:
        prompt = _make_cache_buster() + prompt
    elif _run_idx is not None:
        prompt = _make_cache_buster(run_idx=_run_idx) + prompt

    # Pipe the prompt via stdin instead of passing it as argv. The old
    # `ollama run model --verbose "prompt"` form hit ARG_MAX (~256KB on macOS)
    # for anything above ~60k tokens. stdin has no size limit.
    cmd = ["ollama", "run", model_name, "--verbose"]

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        total_wall_time = time.time() - start_time

        # Parse metrics from stderr (`--verbose` sends timings to stderr)
        metrics = parse_ollama_output(result.stderr)

        # Extract generated text from stdout
        generated_text = extract_generated_text(result.stdout, result.stderr, prompt)

        # Derive prompt_eval_rate from duration + count if not in the regex output
        if (
            "prompt_eval_rate" not in metrics
            and metrics.get("prompt_eval_duration", 0) > 0
            and "prompt_eval_count" in metrics
        ):
            metrics["prompt_eval_rate"] = metrics["prompt_eval_count"] / metrics["prompt_eval_duration"]

        # Same for eval_rate
        if "eval_rate" not in metrics and metrics.get("eval_duration", 0) > 0 and "eval_count" in metrics:
            metrics["eval_rate"] = metrics["eval_count"] / metrics["eval_duration"]

        # Refuse to fabricate a prompt token count. If Ollama did not report
        # one, the row is unreliable — return None rather than pretending
        # len(prompt.split()) is a token count.
        if "prompt_eval_count" not in metrics:
            print("  ERROR: Ollama did not report prompt_eval_count. Skipping row.")
            if result.stderr:
                print(f"  stderr excerpt: {result.stderr[:500]}")
            return None

        if "eval_count" not in metrics or "eval_rate" not in metrics:
            print("  ERROR: Failed to parse required metrics from output")
            print(f"  Parsed metrics: {metrics}")
            return None

        prompt_eval_count = metrics["prompt_eval_count"]
        prompt_eval_duration = metrics.get("prompt_eval_duration", 0)

        # Tokenizer-independent truncation detection. Don't compare against
        # the filename expectation — context files are generated with tiktoken
        # cl100k_base and the model's native tokenizer (e.g. qwen) typically
        # produces 10-20% fewer tokens for the same English text, which would
        # trip a naive expected-count check. Use a char-to-token lower bound:
        # if Ollama processed fewer tokens than file_chars / 10, it's
        # genuinely dropping content.
        file_chars = len(prompt)
        min_plausible_tokens = file_chars // 10
        if prompt_eval_count < min_plausible_tokens:
            print(
                f"  WARNING: processed {prompt_eval_count} prompt tokens from a "
                f"{file_chars}-char file — well below the ~{min_plausible_tokens} "
                f"tokens expected. Ollama likely truncated the prompt. Results "
                f"for this row are INVALID."
            )

        print(f"  Prompt: {prompt_eval_count} tokens at " f"{metrics.get('prompt_eval_rate', 0):.1f} t/s")
        print(f"  Generation: {metrics.get('eval_count', 0)} tokens at " f"{metrics.get('eval_rate', 0):.1f} t/s")
        if prompt_eval_duration > 0:
            print(f"  Time to first token: {prompt_eval_duration:.2f}s")
        print(f"  Total wall time: {total_wall_time:.2f}s")

        return {
            "context_size": context_file.stem,
            "prompt_tokens": prompt_eval_count,
            "prompt_tps": metrics.get("prompt_eval_rate", 0),
            "generation_tokens": metrics.get("eval_count", 0),
            "generation_tps": metrics.get("eval_rate", 0),
            "total_time": total_wall_time,
            "eval_duration": metrics.get("eval_duration", 0),
            "prompt_eval_duration": prompt_eval_duration,
            # Matches mlx_benchmark semantics: library-reported prefill duration.
            "time_to_first_token": prompt_eval_duration,
            "generated_text": generated_text,
            "wall_time": total_wall_time,
        }

    except subprocess.TimeoutExpired:
        print(f"Timeout running benchmark for {context_file}")
        return None
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
        # Use ollama list command
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            print(f"Error checking model availability: {result.stderr}")
            return False

        # Parse the output
        lines = result.stdout.strip().split("\n")

        # Skip header line
        if len(lines) > 1:
            for line in lines[1:]:
                # Split by whitespace and get the first column (model name)
                if line.strip():
                    parts = line.split()
                    if parts:
                        available_model = parts[0]
                        # Check for exact match or base model match
                        if available_model == model_name:
                            print(f"Found model: {model_name}")
                            return True
                        # Check without tag
                        if ":" in model_name and ":" in available_model:
                            if available_model.split(":")[0] == model_name.split(":")[0]:
                                print(f"Found model: {available_model}")
                                return True

        print(f"Model '{model_name}' not found in Ollama")
        print("Available models:")
        if len(lines) > 1:
            for line in lines[1:]:
                if line.strip():
                    parts = line.split()
                    if parts:
                        print(f"  - {parts[0]}")
        return False

    except Exception as e:
        print(f"Error checking model availability: {e}")
        print(f"Attempting to use model anyway...")
        return True  # Try to proceed anyway


def run_batch_benchmark(
    temp_model: str,
    batch_sizes: List[int],
    prompt_tokens: int = 2048,
    gen_tokens: int = 128,
    num_trials: int = 3,
    cold_prefill: bool = True,
    timeout: int = 3600,
) -> List[Dict]:
    """Run batch benchmark by firing concurrent `ollama run` subprocesses.

    This is the CLI analog of run_batch_benchmark in ollama_api_benchmark. Each
    "concurrent request" is a separate `ollama run` subprocess that talks to
    the same daemon, so the server-side continuous batching (controlled by
    OLLAMA_NUM_PARALLEL) kicks in the same way it does for the API path —
    just with per-subprocess startup overhead (~50-200ms) bolted on top.

    IMPORTANT: the daemon must be started with OLLAMA_NUM_PARALLEL >=
    max(batch_sizes), otherwise requests are queued serially.

    Args:
        temp_model: Ollama model tag (must already be created with num_ctx
            and num_predict parameters baked in)
        batch_sizes: Concurrency levels to test
        prompt_tokens: Approximate prompt tokens per request
        gen_tokens: Tokens to generate per request
        num_trials: Trials per batch size (averaged)
        cold_prefill: Prepend a unique cache-buster to each request's prompt
        timeout: Per-subprocess timeout in seconds

    Returns:
        List of result dicts with batch_size, prompt_tps, generation_tps, peak_memory_gb
    """
    import concurrent.futures
    import statistics

    # Build a synthetic prompt of approximately prompt_tokens length. Same
    # tiktoken encoding as ollama_api_benchmark and openai_benchmark, so the
    # batch numbers across engines are directly comparable.
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
        """Send one non-streaming request via `ollama run` and parse metrics."""
        body = (_make_cache_buster() + prompt_text) if cold_prefill else prompt_text
        result = subprocess.run(
            ["ollama", "run", temp_model, "--verbose"],
            input=body,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        metrics = parse_ollama_output(result.stderr)
        return {
            "prompt_tokens": metrics.get("prompt_eval_count", 0),
            "generation_tokens": metrics.get("eval_count", 0),
        }

    batch_results = []

    for bs in batch_sizes:
        print(
            f"\n  Batch size {bs} ({num_trials} trials, ~{prompt_tokens} prompt tokens, " f"{gen_tokens} gen tokens)..."
        )

        # Per-batch-size warmup so all slots are allocated before timing
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
            # Aggregate throughput across all concurrent requests divided by
            # the wall clock for the whole batch.
            agg_prompt_tps = total_prompt_tok / wall_time if wall_time > 0 else 0
            agg_gen_tps = total_gen_tok / wall_time if wall_time > 0 else 0

            trial_prompt_tps.append(agg_prompt_tps)
            trial_gen_tps.append(agg_gen_tps)

            print(f"    Trial {trial + 1}: pp {agg_prompt_tps:.1f} tg {agg_gen_tps:.1f} t/s " f"({wall_time:.1f}s)")

        if trial_prompt_tps:
            avg_prompt = statistics.mean(trial_prompt_tps)
            avg_gen = statistics.mean(trial_gen_tps)
            print(f"  Avg: pp {avg_prompt:.1f} tg {avg_gen:.1f} t/s")
            batch_results.append(
                {
                    "batch_size": bs,
                    "prompt_tps": round(avg_prompt, 2),
                    "generation_tps": round(avg_gen, 2),
                    "peak_memory_gb": 0.0,  # Not available via Ollama API
                }
            )

    return batch_results


def main() -> int:
    """Main function to run Ollama CLI benchmarks."""
    parser = argparse.ArgumentParser(description="Run Ollama benchmarks using command-line interface")
    parser.add_argument("model", help="Ollama model name (e.g., llama3.2, mistral)")

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
    output_dir = common.create_output_directory("ollama_cli", args.model, cold_prefill=args.cold_prefill)

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
    print(
        f"Cold prefill: {'enabled (cache busted per prompt)' if args.cold_prefill else 'disabled (cache reuse allowed)'}"
    )

    # Ollama run has no --num-ctx or --num-predict flag, so we create a
    # temporary model via Modelfile that bakes in the parameters we need.
    # Size num_ctx for the LARGEST context we'll run + headroom, so the
    # same temp model serves every row.
    warmup_file = common.find_warmup_file()
    all_files = context_files + ([warmup_file] if warmup_file else [])
    max_num_ctx = max(_derive_num_ctx(f, args.max_tokens) for f in all_files)

    print(f"\nCreating temporary model with num_ctx={max_num_ctx}, num_predict={args.max_tokens}...")
    try:
        temp_main = _create_temp_model(args.model, num_ctx=max_num_ctx, num_predict=args.max_tokens)
        print(f"  Created: {temp_main}")
    except Exception as e:
        print(f"Failed to create temp model: {e}")
        return 1

    temp_batch = None
    exit_code = 0

    try:
        # Warmup run (excluded from results)
        if warmup_file:
            print(f"\n{'=' * 50}")
            print(f"Warmup run (excluded from results): {warmup_file.name}")
            print(f"{'=' * 50}")
            run_cli_benchmark(temp_main, warmup_file, cold_prefill=args.cold_prefill, timeout=args.timeout)
            print("Warmup complete.")
        else:
            print("Warning: 0.5k.txt not found, skipping warmup.")

        # Run benchmarks
        start_time = time.time()
        results = []
        if args.cold_prefill:
            for file in context_files:
                print(f"\n{'=' * 50}")
                print(f"Benchmarking {file.name}...")
                print(f"{'=' * 50}")

                result = common.run_benchmark_peak(
                    run_cli_benchmark,
                    temp_main,
                    file,
                    cold_prefill=args.cold_prefill,
                    timeout=args.timeout,
                    n_runs=args.runs,
                )
                if result:
                    results.append(result)

                    if args.save_responses:
                        output_filename = output_dir / f"response_{result['context_size']}.txt"
                        common.save_generated_text(result, args.model, output_filename, "Ollama CLI")
        else:
            results = common.run_benchmark_peak_per_run(
                run_cli_benchmark,
                context_files=context_files,
                n_runs=args.runs,
                model_name=temp_main,
                cold_prefill=args.cold_prefill,
                timeout=args.timeout,
            )
            if args.save_responses:
                for result in results:
                    output_filename = output_dir / f"response_{result['context_size']}.txt"
                    common.save_generated_text(result, args.model, output_filename, "Ollama CLI")

        total_benchmark_time = time.time() - start_time

        if not results:
            print("\nNo successful benchmark results")
            exit_code = 1
            return exit_code

        # Run batch benchmark (concurrent-request continuous batching)
        batch_results = None
        if not args.no_batch:
            batch_sizes = [int(s.strip()) for s in args.batch_sizes.split(",")]
            max_bs = max(batch_sizes)
            print(f"\nRunning batch benchmark (concurrent requests: {batch_sizes})...")

            # OLLAMA_NUM_PARALLEL guardrail — read by the daemon at startup,
            # not per request. We can only see the var if it's set in our
            # shell; on macOS launchctl daemons this may not be visible.
            num_parallel = os.environ.get("OLLAMA_NUM_PARALLEL")
            if num_parallel is not None:
                try:
                    if int(num_parallel) < max_bs:
                        print(
                            f"  WARNING: OLLAMA_NUM_PARALLEL={num_parallel} < max batch size "
                            f"{max_bs}. Server will queue requests serially — numbers will be "
                            f"misleading."
                        )
                except ValueError:
                    pass
            else:
                print(
                    f"  NOTE: OLLAMA_NUM_PARALLEL is not set in this shell. Ensure the Ollama "
                    f"daemon was started with OLLAMA_NUM_PARALLEL >= {max_bs}, or the server "
                    f"will queue requests and report latency, not batched throughput."
                )

            # Batch uses different prompt/gen token budgets — make a second
            # temp model sized for it. Same cleanup path via the outer finally.
            batch_num_ctx = args.batch_prompt_tokens + args.batch_gen_tokens + 256
            try:
                temp_batch = _create_temp_model(
                    args.model,
                    num_ctx=batch_num_ctx,
                    num_predict=args.batch_gen_tokens,
                )
                print(f"  Created batch temp model: {temp_batch}")
            except Exception as e:
                print(f"  Failed to create batch temp model: {e} — skipping batch benchmark")
                temp_batch = None

            if temp_batch:
                try:
                    batch_results = run_batch_benchmark(
                        temp_batch,
                        batch_sizes,
                        prompt_tokens=args.batch_prompt_tokens,
                        gen_tokens=args.batch_gen_tokens,
                        num_trials=args.batch_trials,
                        cold_prefill=args.cold_prefill,
                        timeout=args.timeout,
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
            "Ollama CLI",
            hardware_info,
            args,
            batch_results=batch_results,
        )

        # Print summary using common function
        common.print_benchmark_summary(
            results,
            args.model,
            "Ollama CLI",
            hardware_info,
            output_dir,
            total_benchmark_time,
            batch_results=batch_results,
        )

        return 0

    finally:
        # Always clean up temp models, even on failure / KeyboardInterrupt
        print("\nCleaning up temporary models...")
        _remove_temp_model(temp_main)
        if temp_batch:
            _remove_temp_model(temp_batch)


if __name__ == "__main__":
    sys.exit(main())
