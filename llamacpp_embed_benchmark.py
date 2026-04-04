#!/usr/bin/env python3
"""Benchmark script for embedded llama.cpp runtime via llama-cpp-python."""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import benchmark_common as common


def check_llama_cpp_installed() -> bool:
    """Check if llama-cpp-python is installed."""
    try:
        import llama_cpp  # noqa: F401

        return True
    except ImportError:
        return False


def load_model(
    model_path: str,
    n_ctx: int,
    n_gpu_layers: int,
    batch_size: int,
    threads: Optional[int],
):
    """Load llama.cpp model through llama-cpp-python."""
    from llama_cpp import Llama

    model_kwargs = {
        "model_path": model_path,
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "n_batch": batch_size,
        "verbose": False,
    }
    if threads is not None:
        model_kwargs["n_threads"] = threads

    return Llama(**model_kwargs)


def extract_timings_from_response(response: Dict) -> Tuple[float, float]:
    """Extract prompt and generation timing (seconds) if available."""
    timings = response.get("timings")
    if not isinstance(timings, dict):
        return 0.0, 0.0

    prompt_ms = timings.get("prompt_ms", timings.get("prompt_eval_ms", 0.0))
    predict_ms = timings.get("predicted_ms", timings.get("eval_ms", 0.0))

    prompt_s = float(prompt_ms or 0.0) / 1000.0
    predict_s = float(predict_ms or 0.0) / 1000.0
    return prompt_s, predict_s


def run_benchmark(
    llm,
    context_file: Path,
    max_tokens: int = 128,
    timeout: int = 3600,
    seed: int = 0,
) -> Optional[Dict]:
    """Run embedded llama.cpp benchmark for one context file."""
    del timeout  # currently not enforced by llama-cpp-python completion call

    with open(context_file, "r") as f:
        prompt = f.read()

    start_time = time.time()
    try:
        response = llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_k=40,
            top_p=0.95,
            echo=False,
            seed=seed,
        )
    except Exception as e:
        print(f"Error during benchmark: {e}")
        return None

    total_time = time.time() - start_time

    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    generation_tokens = int(usage.get("completion_tokens", 0) or 0)

    if prompt_tokens == 0:
        prompt_tokens = len(llm.tokenize(prompt.encode("utf-8"), add_bos=True))

    choices = response.get("choices", []) if isinstance(response, dict) else []
    choice = choices[0] if choices else {}
    generated_text = choice.get("text", "")

    prompt_time, predict_time = extract_timings_from_response(response)

    prompt_tps = (prompt_tokens / prompt_time) if prompt_time > 0 else 0.0
    if predict_time > 0 and generation_tokens > 0:
        generation_tps = generation_tokens / predict_time
    elif total_time > 0 and generation_tokens > 0:
        # Fallback when no split timings are returned by binding.
        generation_tps = generation_tokens / total_time
    else:
        generation_tps = 0.0

    return {
        "context_size": context_file.stem,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "prompt_time": prompt_time,
        "eval_duration": predict_time if predict_time > 0 else total_time,
        "total_time": total_time,
        "prompt_tps": prompt_tps,
        "generation_tps": generation_tps,
        "generated_text": generated_text,
        "prompt_eval_duration": prompt_time,
        "time_to_first_token": prompt_time,
        "wall_time": total_time,
    }


def main() -> int:
    """Main function to run embedded llama.cpp benchmarks."""
    parser = argparse.ArgumentParser(
        description="Benchmark llama.cpp embedded runtime across different context sizes"
    )
    parser.add_argument(
        "model",
        help="Path to local GGUF model file",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=65536,
        help="Context window to allocate in llama.cpp (default: 65536)",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU (-1 = all layers, default)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="CPU threads for llama.cpp (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Prompt processing batch size for llama.cpp (default: 2048)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sampling seed (default: 0)",
    )

    common.setup_common_args(parser)
    args = parser.parse_args()

    model_path = str(Path(args.model).expanduser())
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        return 1

    if not check_llama_cpp_installed():
        print("llama-cpp-python is not installed.")
        print("Install it with: pip install llama-cpp-python")
        print("For Apple Silicon Metal builds, use CMAKE args per llama-cpp-python docs.")
        return 1

    print("Loading embedded llama.cpp runtime...")
    print(f"Model path: {model_path}")
    print(f"n_ctx: {args.n_ctx}")
    print(f"n_gpu_layers: {args.n_gpu_layers}")
    print(f"batch size: {args.batch_size}")
    print(f"threads: {args.threads if args.threads is not None else 'auto'}")

    try:
        llm = load_model(
            model_path=model_path,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            batch_size=args.batch_size,
            threads=args.threads,
        )
    except Exception as e:
        print(f"Failed to load GGUF model: {e}")
        return 1

    model_name = Path(model_path).name

    output_dir = common.create_output_directory("llamacpp_embed", model_name, args.output_dir)

    context_files = common.find_context_files(args.contexts, args.context_dir)
    if not context_files:
        return 1

    print("\nCollecting hardware information...")
    hardware_info = common.get_hardware_info()
    hardware_info["llamacpp_mode"] = "embedded"
    hardware_info["model_path"] = model_path
    hardware_info["n_ctx"] = args.n_ctx
    hardware_info["n_gpu_layers"] = args.n_gpu_layers
    hardware_info["batch_size"] = args.batch_size
    hardware_info["threads"] = args.threads if args.threads is not None else "auto"

    hardware_str = common.format_hardware_string(hardware_info)
    print(f"Hardware: {hardware_str}")
    print(f"Model: {model_name}")
    print(f"Max tokens: {args.max_tokens}")

    # Warmup run
    warmup_file = common.find_warmup_file()
    if warmup_file:
        print(f"\n{'=' * 50}")
        print(f"Warmup run (excluded from results): {warmup_file.name}")
        print(f"{'=' * 50}")
        run_benchmark(llm=llm, context_file=warmup_file, max_tokens=args.max_tokens, timeout=args.timeout, seed=args.seed)
        print("Warmup complete.")
    else:
        print("Warning: 0.5k.txt not found, skipping warmup.")

    results = []
    start_time = time.time()
    for context_file in context_files:
        print(f"\n{'=' * 50}")
        print(f"Benchmarking {context_file.name}...")
        print(f"{'=' * 50}")

        result = run_benchmark(
            llm=llm,
            context_file=context_file,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            seed=args.seed,
        )

        if not result:
            print(f"Failed to benchmark {context_file.name}")
            continue

        results.append(result)
        print(f"\nResults for {context_file.name}:")
        print(f"  Prompt tokens: {result['prompt_tokens']}")
        print(f"  Generated tokens: {result['generation_tokens']}")
        if result["prompt_time"] > 0:
            print(f"  Prompt time: {result['prompt_time']:.2f}s")
        print(f"  Generation time: {result['eval_duration']:.2f}s")
        print(f"  Total time: {result['total_time']:.2f}s")
        if result["prompt_tps"] > 0:
            print(f"  Prompt TPS: {result['prompt_tps']:.1f} tokens/sec")
        print(f"  Generation TPS: {result['generation_tps']:.1f} tokens/sec")

        if args.save_responses:
            response_file = output_dir / f"response_{context_file.stem}.txt"
            common.save_generated_text(
                result,
                model_name,
                response_file,
                framework="llama.cpp Embed [EXPERIMENTAL]",
            )

    total_benchmark_time = time.time() - start_time

    if not results:
        print("\nNo successful benchmark results")
        return 1

    common.save_all_outputs(
        results,
        output_dir,
        model_name,
        "llama.cpp Embed [EXPERIMENTAL]",
        hardware_info,
        args,
    )

    common.print_benchmark_summary(
        results,
        model_name,
        "llama.cpp Embed [EXPERIMENTAL]",
        hardware_info,
        output_dir,
        total_benchmark_time,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
