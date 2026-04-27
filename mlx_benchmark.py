#!/usr/bin/env python3
"""
Benchmark script for MLX framework on Apple Silicon.

This script runs benchmarks using MLX-LM for efficient inference on Apple Silicon Macs.
The model is loaded once and reused across all benchmark runs.

Usage:
    python mlx_benchmark.py mlx-community/Qwen3-0.6B-4bit
    python mlx_benchmark.py mlx-community/Qwen3-0.6B-4bit --kv-bit 4 --max-tokens 500
"""

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import benchmark_common as common


def safe_duration(tokens: int, tokens_per_sec: float) -> float:
    """Safely calculate duration from token count and throughput."""
    if tokens <= 0 or tokens_per_sec <= 0:
        return 0.0
    return float(tokens) / float(tokens_per_sec)


def _count_prompt_tokens(prepared_prompt, tokenizer) -> int:
    """Count the number of tokens in a prepared prompt."""
    return len(_ensure_token_ids(prepared_prompt, tokenizer))


def _ensure_token_ids(prepared_prompt, tokenizer) -> List[int]:
    """Ensure the prepared prompt is returned as a list of token IDs."""
    if isinstance(prepared_prompt, list):
        return prepared_prompt
    return tokenizer.encode(prepared_prompt, add_special_tokens=False)


def _verify_prefix_match(prev_tokens: List[int], curr_tokens: List[int]) -> bool:
    """Verify that prev_tokens is a strict prefix of curr_tokens."""
    if len(prev_tokens) > len(curr_tokens):
        return False
    return curr_tokens[: len(prev_tokens)] == prev_tokens


def _find_common_prefix_length(prev_tokens: List[int], curr_tokens: List[int]) -> int:
    """Find the length of the longest common prefix between two token sequences.

    BPE tokenizers may produce different token boundaries at the text boundary
    between context files, so a strict text prefix doesn't guarantee a strict
    token prefix. This function finds how many tokens actually match.
    """
    min_len = min(len(prev_tokens), len(curr_tokens))
    for i in range(min_len):
        if prev_tokens[i] != curr_tokens[i]:
            return i
    return min_len


def _trim_cache_to_prompt_length(cache, prompt_length: int) -> None:
    """Reset cache offset to prompt_length, discarding generated tokens.

    After generation the cache has prompt_length + generated_length entries.
    Resetting offset means the next incremental prefill will overwrite the
    stale generated-token positions.
    """
    for c in cache:
        if hasattr(c, "offset"):
            c.offset = prompt_length


def _show_prefill_progress(num_tokens: int, stop_event: threading.Event) -> None:
    """Display a live elapsed-time indicator while prefill is running."""
    start = time.time()
    while not stop_event.wait(0.2):
        elapsed = time.time() - start
        sys.stdout.write(f"\r  Prefilling {num_tokens} tokens... {elapsed:.1f}s")
        sys.stdout.flush()


def load_model(model_url: str, trust_remote_code: bool = False) -> Tuple:
    """Load an MLX model and tokenizer.

    Args:
        model_url: MLX model URL (e.g., mlx-community/Qwen3-0.6B-4bit)
        trust_remote_code: Allow running custom model/tokenizer code

    Returns:
        Tuple of (model, tokenizer, config)
    """
    import mlx_lm
    from mlx_lm.utils import load_model as _load_model
    from mlx_lm.utils import load_tokenizer

    # Ensure PreTrainedConfig exposes max_position_embeddings for model types
    # not yet registered in transformers (e.g. deepseek_v4)
    from transformers import PreTrainedConfig
    if not hasattr(PreTrainedConfig, "max_position_embeddings"):
        PreTrainedConfig.max_position_embeddings = None

    model_config = {}
    tokenizer_config = {}
    if trust_remote_code:
        model_config["trust_remote_code"] = True
        tokenizer_config["trust_remote_code"] = True

    # Use load_model directly so we can pass strict=False to skip
    # mismatched weights (e.g. vision_tower params in VLM checkpoints)
    model_path = mlx_lm.utils._download(model_url)
    model, config = _load_model(model_path, lazy=False, strict=False, model_config=model_config)
    tokenizer = load_tokenizer(model_path, tokenizer_config, eos_token_ids=config.get("eos_token_id", None))

    return model, tokenizer, config


def prepare_prompt(
    prompt_text: str,
    tokenizer,
    ignore_chat_template: bool = False,
    chat_template_config: Optional[str] = None,
) -> Union[str, List[int]]:
    """Prepare prompt input for mlx_lm.stream_generate.

    Mirrors mlx_lm.generate CLI behavior: when a tokenizer has a chat template,
    wrap the raw text as a user message and add the generation prompt.
    """
    has_chat_template = bool(
        getattr(tokenizer, "has_chat_template", False) or getattr(tokenizer, "chat_template", None) is not None
    )
    if ignore_chat_template or not has_chat_template:
        return prompt_text

    template_kwargs = {}
    if chat_template_config:
        template_kwargs = json.loads(chat_template_config)

    messages = [{"role": "user", "content": prompt_text}]
    templated_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        continue_final_message=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    return tokenizer.encode(templated_prompt, add_special_tokens=False)


def run_benchmark(
    model,
    tokenizer,
    context_file: Path,
    kv_bit: Optional[int] = None,
    max_tokens: int = 128,
    max_kv_size: Optional[int] = None,
    ignore_chat_template: bool = False,
    chat_template_config: Optional[str] = None,
    _run_idx: Optional[int] = None,
) -> Optional[Dict]:
    """Run MLX benchmark for a given context file.

    Args:
        model: Loaded MLX model
        tokenizer: Loaded tokenizer
        context_file: Path to the context file
        kv_bit: KV cache bit size (optional)
        max_tokens: Maximum tokens to generate
        max_kv_size: Maximum KV cache size in tokens (optional)
        ignore_chat_template: If true, skip tokenizer chat template wrapping
        chat_template_config: JSON config passed to apply_chat_template

    Returns:
        Dictionary with benchmark results or None if failed
    """
    import mlx_lm
    from mlx_lm.models.cache import make_prompt_cache

    print(f"Running benchmark for {context_file}...")

    try:
        with open(context_file, "r") as f:
            prompt = f.read()

        prepared_prompt = prepare_prompt(
            prompt,
            tokenizer,
            ignore_chat_template=ignore_chat_template,
            chat_template_config=chat_template_config,
        )

        kwargs = {}
        if kv_bit is not None:
            kwargs["kv_bits"] = kv_bit
        if max_kv_size is not None:
            kwargs["max_kv_size"] = max_kv_size

        # Build the cache externally so we can read its size after generation.
        # mlx_lm.stream_generate would otherwise create one internally and we
        # would never get a handle to it. Behaviour is identical either way.
        prompt_cache = make_prompt_cache(model, max_kv_size=max_kv_size)

        # Reset peak memory before each run to get per-context-size measurement
        import mlx.core as mx

        mx.reset_peak_memory()

        # Count prompt tokens for the progress indicator
        num_prompt_tokens = _count_prompt_tokens(prepared_prompt, tokenizer)

        # Start live prefill progress indicator
        prefill_done = threading.Event()
        progress_thread = threading.Thread(
            target=_show_prefill_progress,
            args=(num_prompt_tokens, prefill_done),
            daemon=True,
        )
        progress_thread.start()

        start_time = time.time()
        first_token_received = False
        token_count = 0

        last_response = None
        generated_text = ""
        for response in mlx_lm.stream_generate(
            model,
            tokenizer,
            prompt=prepared_prompt,
            max_tokens=max_tokens,
            prompt_cache=prompt_cache,
            **kwargs,
        ):
            if not first_token_received:
                first_token_received = True
                prefill_done.set()
                progress_thread.join()
                prefill_time = time.time() - start_time
                pp_tps = num_prompt_tokens / prefill_time if prefill_time > 0 else 0
                sys.stdout.write(
                    f"\r  Prefill: {num_prompt_tokens} tokens in " f"{prefill_time:.2f}s ({pp_tps:.0f} t/s)\n"
                )
                sys.stdout.flush()

            last_response = response
            generated_text += response.text
            token_count += 1
            pct = min(token_count * 100 // max_tokens, 100)
            sys.stdout.write(f"\r  Generating: {token_count}/{max_tokens} ({pct}%)")
            sys.stdout.flush()

        # End generation progress line
        if first_token_received:
            sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            # No tokens were generated — stop the prefill indicator
            prefill_done.set()
            progress_thread.join()
            sys.stdout.write("\n")
            sys.stdout.flush()

        total_wall_time = time.time() - start_time

        if last_response is None:
            print(f"No response generated for {context_file}")
            return None

        prompt_tokens = last_response.prompt_tokens
        prompt_tps = last_response.prompt_tps
        generation_tokens = last_response.generation_tokens
        generation_tps = last_response.generation_tps
        peak_memory_gb = last_response.peak_memory
        prompt_eval_duration = safe_duration(prompt_tokens, prompt_tps)
        kv_cache_gb = common.kv_cache_bytes(prompt_cache) / 1e9

        print(f"  Prompt: {prompt_tokens} tokens, {prompt_tps:.3f} tokens-per-sec")
        print(f"  Generation: {generation_tokens} tokens, {generation_tps:.3f} tokens-per-sec")
        print(f"  Peak memory: {peak_memory_gb:.3f} GB")
        print(f"  KV cache: {kv_cache_gb:.3f} GB")
        if prompt_eval_duration > 0:
            print(f"  Time to first token: {prompt_eval_duration:.2f}s")
        print(f"  Total wall time: {total_wall_time:.2f}s")

        return {
            "context_size": Path(context_file).stem,
            "prompt_tokens": prompt_tokens,
            "prompt_tps": prompt_tps,
            "generation_tokens": generation_tokens,
            "generation_tps": generation_tps,
            "peak_memory_gb": peak_memory_gb,
            "kv_cache_gb": kv_cache_gb,
            "total_time": total_wall_time,
            "generated_text": generated_text,
            "prompt_eval_duration": prompt_eval_duration,
            "time_to_first_token": prompt_eval_duration,
        }

    except Exception as e:
        print(f"Error running benchmark: {e}")
        return None


def run_cached_benchmark(
    model,
    tokenizer,
    context_files: List[Path],
    kv_bit: Optional[int] = None,
    max_tokens: int = 128,
    max_kv_size: Optional[int] = None,
    ignore_chat_template: bool = False,
    chat_template_config: Optional[str] = None,
) -> Optional[List[Dict]]:
    """Run cached KV cache benchmark with incremental prefill.

    Encodes the largest context file once and slices the token array to create
    prompts for each context size. This guarantees perfect token-level prefix
    alignment, which is not possible with BPE tokenizers when encoding
    text-prefix files independently (BPE merges characters differently depending
    on surrounding context).
    """
    import mlx.core as mx
    import mlx_lm
    from mlx_lm.models.cache import RotatingKVCache, make_prompt_cache

    if max_kv_size is not None:
        print(
            "  WARNING: --max-kv-size uses RotatingKVCache which is incompatible "
            "with cached KV cache benchmarking. Skipping cached phase."
        )
        return None

    # Read and encode the largest context file once
    with open(context_files[-1], "r") as f:
        largest_raw_text = f.read()
    largest_prepared = prepare_prompt(
        largest_raw_text,
        tokenizer,
        ignore_chat_template=ignore_chat_template,
        chat_template_config=chat_template_config,
    )
    full_tokens = _ensure_token_ids(largest_prepared, tokenizer)
    total_text_len = len(largest_raw_text)

    print(f"  Encoded largest file ({context_files[-1].name}): {len(full_tokens)} tokens")

    # Compute token-level slice points proportional to each file's text length
    token_targets = []
    for context_file in context_files:
        with open(context_file, "r") as f:
            raw_text = f.read()
        if total_text_len > 0:
            target = round(len(full_tokens) * len(raw_text) / total_text_len)
        else:
            target = len(full_tokens)
        # Ensure monotonic increase and don't exceed full tokens
        target = max(target, token_targets[-1] if token_targets else 0)
        target = min(target, len(full_tokens))
        token_targets.append(target)

    prompt_cache = make_prompt_cache(model)
    results = []
    prev_count = 0

    for i, context_file in enumerate(context_files):
        target_count = token_targets[i]
        print(f"\n{'=' * 50}")
        print(f"Cached benchmark: {context_file.name} ({i + 1}/{len(context_files)})...")
        print(f"{'=' * 50}")

        try:
            curr_tokens = full_tokens[:target_count]
            delta_tokens = curr_tokens[prev_count:]
            num_delta = len(delta_tokens)
            total_prompt_tokens = len(curr_tokens)
            cached_token_count = total_prompt_tokens - num_delta

            print(
                f"  Total prompt tokens: {total_prompt_tokens}, "
                f"Delta tokens: {num_delta}, "
                f"Cached tokens: {cached_token_count}"
            )

            kwargs = {}
            if kv_bit is not None:
                kwargs["kv_bits"] = kv_bit

            mx.reset_peak_memory()

            if num_delta == 0:
                print("  No delta tokens (same as previous). Skipping.")
                prev_count = target_count
                continue

            # Progress indicator
            prefill_done = threading.Event()
            progress_thread = threading.Thread(
                target=_show_prefill_progress,
                args=(num_delta, prefill_done),
                daemon=True,
            )
            progress_thread.start()

            start_time = time.time()
            first_token_received = False
            token_count = 0
            ttft = 0.0

            last_response = None
            generated_text = ""
            for response in mlx_lm.stream_generate(
                model,
                tokenizer,
                prompt=delta_tokens,
                max_tokens=max_tokens,
                prompt_cache=prompt_cache,
                **kwargs,
            ):
                if not first_token_received:
                    first_token_received = True
                    ttft = time.time() - start_time
                    prefill_done.set()
                    progress_thread.join()
                    pp_tps = num_delta / ttft if ttft > 0 else 0
                    sys.stdout.write(
                        f"\r  Incremental prefill: {num_delta} tokens in " f"{ttft:.2f}s ({pp_tps:.0f} t/s)\n"
                    )
                    sys.stdout.flush()

                last_response = response
                generated_text += response.text
                token_count += 1
                pct = min(token_count * 100 // max_tokens, 100)
                sys.stdout.write(f"\r  Generating: {token_count}/{max_tokens} ({pct}%)")
                sys.stdout.flush()

            if first_token_received:
                sys.stdout.write("\n")
                sys.stdout.flush()
            else:
                prefill_done.set()
                progress_thread.join()
                sys.stdout.write("\n")
                sys.stdout.flush()

            total_wall_time = time.time() - start_time

            if last_response is None:
                print(f"  No response generated for {context_file}")
                prev_count = target_count
                continue

            generation_tokens = last_response.generation_tokens
            generation_tps = last_response.generation_tps
            peak_memory_gb = last_response.peak_memory
            kv_cache_gb = common.kv_cache_bytes(prompt_cache) / 1e9

            prompt_eval_duration = ttft if ttft > 0 else safe_duration(num_delta, last_response.prompt_tps)
            incremental_prompt_tps = num_delta / prompt_eval_duration if prompt_eval_duration > 0 else 0

            print(f"  Prompt: {total_prompt_tokens} total, {num_delta} delta tokens")
            print(f"  Incremental prompt TPS: {incremental_prompt_tps:.3f} tokens-per-sec")
            print(f"  Generation: {generation_tokens} tokens, {generation_tps:.3f} tokens-per-sec")
            print(f"  Peak memory: {peak_memory_gb:.3f} GB")
            print(f"  KV cache: {kv_cache_gb:.3f} GB")
            if prompt_eval_duration > 0:
                print(f"  Time to first token: {prompt_eval_duration:.2f}s")
            print(f"  Total wall time: {total_wall_time:.2f}s")

            # Trim cache to remove generated tokens, keeping only prompt entries
            _trim_cache_to_prompt_length(prompt_cache, total_prompt_tokens)

            # Skip the first context (no prior cache prefix — identical to a cold prefill)
            if cached_token_count > 0:
                generation_duration = max(total_wall_time - prompt_eval_duration, 0.0)
                results.append(
                    {
                        "context_size": Path(context_file).stem,
                        "prompt_tokens": total_prompt_tokens,
                        "delta_tokens": num_delta,
                        "cached_tokens": cached_token_count,
                        "prompt_tps": last_response.prompt_tps,
                        "incremental_prompt_tps": incremental_prompt_tps,
                        "generation_tokens": generation_tokens,
                        "generation_tps": generation_tps,
                        "peak_memory_gb": peak_memory_gb,
                        "kv_cache_gb": kv_cache_gb,
                        "total_time": total_wall_time,
                        "eval_duration": generation_duration,
                        "generated_text": generated_text,
                        "prompt_eval_duration": prompt_eval_duration,
                        "time_to_first_token": prompt_eval_duration,
                        "cached": True,
                    }
                )

            prev_count = target_count

        except Exception as e:
            print(f"  Error in cached benchmark for {context_file}: {e}")
            import traceback

            traceback.print_exc()
            prompt_cache = make_prompt_cache(model)
            prev_count = token_targets[i]  # Use target, not actual
            continue

    return results if results else None


def run_batch_benchmark(
    model,
    tokenizer,
    batch_sizes: List[int],
    prompt_tokens: int = 2048,
    gen_tokens: int = 128,
    num_trials: int = 3,
    vocab_size: Optional[int] = None,
) -> List[Dict]:
    """Run batch benchmark measuring throughput at different batch sizes.

    Args:
        model: Loaded MLX model
        tokenizer: Loaded tokenizer
        batch_sizes: List of batch sizes to test
        prompt_tokens: Number of prompt tokens per sequence
        gen_tokens: Number of tokens to generate
        num_trials: Number of trials per batch size (takes median)
        vocab_size: Vocabulary size to sample synthetic prompts from

    Returns:
        List of result dicts with batch_size, prompt_tps, generation_tps, peak_memory_gb
    """
    import mlx.core as mx
    from mlx_lm import batch_generate, stream_generate
    from mlx_lm.models.cache import make_prompt_cache

    if vocab_size is None:
        vocab_size = tokenizer.vocab_size
    batch_results = []

    # Match mlx_lm.benchmark: disable EOS to avoid early stopping on random prompts.
    _raw_eos = getattr(tokenizer, "eos_token_ids", set())
    original_eos = {_raw_eos} if isinstance(_raw_eos, int) else set(_raw_eos)
    restored_via_private_attr = hasattr(tokenizer, "_eos_token_ids")
    if restored_via_private_attr:
        tokenizer._eos_token_ids = set()
    else:
        tokenizer.eos_token_ids = set()
    try:
        for bs in batch_sizes:
            print(
                f"\n  Batch size {bs} ({num_trials} trials, {prompt_tokens} prompt tokens, {gen_tokens} gen tokens)..."
            )
            mx.reset_peak_memory()

            # Generate prompts once (same for all trials) - like mlx_lm.benchmark
            prompts = mx.random.randint(0, vocab_size, (bs, prompt_tokens)).tolist()

            # Warmup run
            print("    Warmup...")
            if bs == 1:
                for response in stream_generate(model, tokenizer, prompts[0], max_tokens=gen_tokens):
                    pass
            else:
                batch_generate(model, tokenizer, prompts=prompts, max_tokens=gen_tokens)

            # Actual trials - reuse same prompts
            trial_prompt_tps = []
            trial_gen_tps = []
            trial_kv_bytes = []

            for trial in range(num_trials):
                if bs == 1:
                    # Build the cache externally so we can read its size after.
                    trial_cache = make_prompt_cache(model)
                    last_response = None
                    for response in stream_generate(
                        model, tokenizer, prompts[0], max_tokens=gen_tokens, prompt_cache=trial_cache
                    ):
                        last_response = response
                    if last_response is not None:
                        trial_prompt_tps.append(last_response.prompt_tps)
                        trial_gen_tps.append(last_response.generation_tps)
                        trial_kv_bytes.append(common.kv_cache_bytes(trial_cache))
                        print(
                            f"    Trial {trial + 1}: pp {last_response.prompt_tps:.1f} tg {last_response.generation_tps:.1f} t/s"
                        )
                else:
                    # batch_generate exposes the per-prompt caches via
                    # return_prompt_caches=True so we can sum nbytes across the
                    # whole batch (matches what peak_memory_gb measures).
                    resp = batch_generate(
                        model,
                        tokenizer,
                        prompts=prompts,
                        max_tokens=gen_tokens,
                        return_prompt_caches=True,
                    )
                    trial_prompt_tps.append(resp.stats.prompt_tps)
                    trial_gen_tps.append(resp.stats.generation_tps)
                    if resp.caches:
                        trial_kv_bytes.append(sum(common.kv_cache_bytes(c) for c in resp.caches))
                    print(
                        f"    Trial {trial + 1}: pp {resp.stats.prompt_tps:.1f} tg {resp.stats.generation_tps:.1f} t/s"
                    )

            if trial_prompt_tps:
                avg_prompt_tps = statistics.mean(trial_prompt_tps)
                avg_gen_tps = statistics.mean(trial_gen_tps)
                peak_mem = mx.get_peak_memory() / 1e9
                avg_kv_gb = (statistics.mean(trial_kv_bytes) / 1e9) if trial_kv_bytes else 0.0

                print(
                    f"  Avg: pp {avg_prompt_tps:.1f} tg {avg_gen_tps:.1f} t/s, "
                    f"peak mem {peak_mem:.2f} GB, kv cache {avg_kv_gb:.2f} GB"
                )

                batch_results.append(
                    {
                        "batch_size": bs,
                        "prompt_tps": round(avg_prompt_tps, 2),
                        "generation_tps": round(avg_gen_tps, 2),
                        "peak_memory_gb": round(peak_mem, 3),
                        "kv_cache_gb": round(avg_kv_gb, 3),
                    }
                )
    finally:
        if restored_via_private_attr:
            tokenizer._eos_token_ids = original_eos
        else:
            tokenizer.eos_token_ids = original_eos

    return batch_results


def check_mlx_installed() -> bool:
    """Check if MLX-LM is installed."""
    try:
        import mlx_lm

        return True
    except ImportError:
        return False


def main() -> int:
    """Main function to run MLX benchmarks."""
    parser = argparse.ArgumentParser(description="Run MLX benchmarks on context files")
    parser.add_argument("model", help="MLX model URL (e.g., mlx-community/Qwen3-0.6B-4bit)")
    parser.add_argument(
        "--kv-bit",
        type=int,
        default=None,
        help="KV cache bit size (optional, e.g., 4 or 8)",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="KV cache size in tokens (optional, e.g., 4096)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow running custom model/tokenizer code when loading (HF)",
    )
    parser.add_argument(
        "--ignore-chat-template",
        action="store_true",
        help="Use raw prompt text instead of tokenizer chat template",
    )
    parser.add_argument(
        "--chat-template-config",
        default=None,
        help="JSON config passed to tokenizer.apply_chat_template",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8,16,32",
        help="Comma-separated batch sizes for batch benchmark (default: 1,2,4,8,16,32)",
    )
    parser.add_argument(
        "--batch-prompt-tokens",
        type=int,
        default=2048,
        help="Number of prompt tokens per sequence in batch benchmark (default: 2048)",
    )
    parser.add_argument(
        "--batch-gen-tokens",
        type=int,
        default=128,
        help="Number of tokens to generate per sequence in batch benchmark (default: 128)",
    )
    parser.add_argument(
        "--batch-trials",
        type=int,
        default=3,
        help="Number of trials per batch size, takes median (default: 3)",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        dest="no_batch",
        help="Skip batch benchmark",
    )
    # Backward-compatible alias for older invocations.
    parser.add_argument(
        "--no-batch-benchmark",
        action="store_true",
        dest="no_batch",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-perplexity",
        action="store_true",
        help="Skip perplexity computation",
    )
    parser.add_argument(
        "--cached",
        action="store_true",
        help="Run cached KV cache benchmark (incremental prefill) after cold benchmarks",
    )

    # Add common arguments
    common.setup_common_args(parser)

    args = parser.parse_args()

    # Check if MLX-LM is installed
    if not check_mlx_installed():
        print("MLX-LM is not installed. Please install it with: pip install mlx-lm")
        return 1

    # Extract model name from URL
    model_name = args.model.rstrip("/").split("/")[-1]

    # Create output directory using common function
    output_dir = common.create_output_directory("mlx", model_name, cold_prefill=True)

    # Find context files using common module
    context_files = common.find_context_files(args.contexts)
    if not context_files:
        return 1

    # Load model once
    print(f"\nLoading model: {args.model}...")
    load_start = time.time()
    try:
        model, tokenizer, model_config = load_model(args.model, args.trust_remote_code)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return 1
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.1f}s")

    # Get hardware information
    print("\nCollecting hardware information...")
    hardware_info = common.get_hardware_info()
    hardware_str = common.format_hardware_string(hardware_info)
    print(f"Hardware: {hardware_str}")
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    if args.kv_bit:
        print(f"KV cache bits: {args.kv_bit}")
    if args.max_kv_size:
        print(f"Max KV size: {args.max_kv_size}")
    if args.ignore_chat_template:
        print("Chat template: disabled (raw prompt)")
    else:
        print("Chat template: enabled when tokenizer provides one")

    # Warmup run using the smallest context file
    warmup_file = context_files[0]
    print(f"\nWarmup run ({warmup_file.name}, max_tokens=1)...")
    import mlx_lm

    warmup_kwargs = {}
    if args.kv_bit is not None:
        warmup_kwargs["kv_bits"] = args.kv_bit
    if args.max_kv_size is not None:
        warmup_kwargs["max_kv_size"] = args.max_kv_size

    try:
        with open(warmup_file, "r") as f:
            warmup_prompt = f.read()
        warmup_prepared_prompt = prepare_prompt(
            warmup_prompt,
            tokenizer,
            ignore_chat_template=args.ignore_chat_template,
            chat_template_config=args.chat_template_config,
        )
        for _ in mlx_lm.stream_generate(model, tokenizer, prompt=warmup_prepared_prompt, max_tokens=1, **warmup_kwargs):
            pass
        print("Warmup complete (result discarded)")
    except Exception as e:
        print(f"Warmup failed (continuing anyway): {e}")

    perplexity = None
    perplexity_data = None
    if args.no_perplexity:
        print("\nSkipping perplexity (--no-perplexity)")
    else:
        # Compute perplexity
        print("\nComputing perplexity...")
        try:
            import mlx.core as mx
            from mlx_lm.perplexity import eval_ppl, load_data

            np_seed = 123
            import numpy as np_rng

            np_rng.random.seed(np_seed)
            mx.random.seed(np_seed)

            ppl_num_samples = 256
            ppl_seq_length = 512
            ppl_dataset = "allenai/tulu-3-sft-mixture"
            data = load_data(tokenizer, ppl_dataset, num_samples=ppl_num_samples, sequence_length=ppl_seq_length)
            ppl, ppl_se = eval_ppl(model, data, batch_size=8)
            perplexity = float(ppl)
            perplexity_data = {
                "perplexity": perplexity,
                "std_error": float(ppl_se),
                "dataset": ppl_dataset,
                "num_samples": ppl_num_samples,
                "sequence_length": ppl_seq_length,
            }
            print(f"Perplexity: {perplexity:.2f} (±{float(ppl_se):.2f})")
        except Exception as e:
            print(f"Perplexity computation failed (continuing): {e}")

    # Run batch benchmark
    batch_results = None
    if not args.no_batch:
        batch_sizes = [int(s.strip()) for s in args.batch_sizes.split(",")]
        print(f"\nRunning batch benchmark (sizes: {batch_sizes})...")
        model_vocab_size = (
            model_config.get("vocab_size")
            or model_config.get("text_config", {}).get("vocab_size")
            or tokenizer.vocab_size
        )

        # Set seed for reproducibility (like mlx_lm.benchmark)
        import mlx.core as mx

        mx.random.seed(0)
        try:
            batch_results = run_batch_benchmark(
                model,
                tokenizer,
                batch_sizes,
                prompt_tokens=args.batch_prompt_tokens,
                gen_tokens=args.batch_gen_tokens,
                num_trials=args.batch_trials,
                vocab_size=model_vocab_size,
            )
            if batch_results:
                print(f"\nBatch benchmark complete: {len(batch_results)} sizes tested")
            else:
                print("\nBatch benchmark produced no results")
        except Exception as e:
            print(f"\nBatch benchmark failed (continuing): {e}")

    # Run benchmarks
    start_time = time.time()
    results = []
    for file in context_files:
        print(f"\n{'=' * 50}")
        print(f"Benchmarking {file.name}...")
        print(f"{'=' * 50}")

        result = common.run_benchmark_peak(
            run_benchmark,
            model,
            tokenizer,
            file,
            args.kv_bit,
            args.max_tokens,
            args.max_kv_size,
            args.ignore_chat_template,
            args.chat_template_config,
            n_runs=args.runs,
        )
        if result:
            results.append(result)

            # Save the generated text if requested
            if args.save_responses:
                output_filename = output_dir / f"response_{result['context_size']}.txt"
                common.save_generated_text(result, args.model, output_filename, "MLX")

    total_benchmark_time = time.time() - start_time

    if not results:
        print("\nNo successful benchmark results")
        return 1

    # Run cached KV cache benchmark if requested
    cached_results = None
    if args.cached:
        print(f"\n{'#' * 60}")
        print("# CACHED KV CACHE BENCHMARK")
        print(f"{'#' * 60}")
        print("Processing context files smallest-to-largest with KV cache reuse...")

        cached_start = time.time()
        cached_results = run_cached_benchmark(
            model,
            tokenizer,
            context_files,
            args.kv_bit,
            args.max_tokens,
            args.max_kv_size,
            args.ignore_chat_template,
            args.chat_template_config,
        )
        cached_benchmark_time = time.time() - cached_start

        if cached_results:
            print(f"\nCached benchmark complete: {len(cached_results)} files tested in {cached_benchmark_time:.1f}s")

            if args.save_responses:
                for result in cached_results:
                    output_filename = output_dir / f"response_{result['context_size']}_cached.txt"
                    common.save_generated_text(result, args.model, output_filename, "MLX (cached)")
        else:
            print("\nCached benchmark produced no results")

    # Save all outputs using common function
    common.save_all_outputs(
        results,
        output_dir,
        model_name,
        "MLX",
        hardware_info,
        args,
        include_memory=True,
        perplexity=perplexity,
        perplexity_data=perplexity_data,
        batch_results=batch_results,
        cached_results=cached_results,
    )

    # Print summary using common function
    common.print_benchmark_summary(
        results,
        model_name,
        "MLX",
        hardware_info,
        output_dir,
        total_benchmark_time,
        perplexity=perplexity,
        batch_results=batch_results,
        cached_results=cached_results,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
