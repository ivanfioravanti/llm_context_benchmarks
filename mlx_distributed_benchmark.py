#!/usr/bin/env python3
"""
Benchmark script for MLX distributed inference via mlx.launch.

This script runs benchmarks using mlx_lm.generate launched with mlx.launch,
which makes it behave similarly to mlx_benchmark.py while using distributed
execution (for example with the JACCL backend).
"""

import argparse
import re
import subprocess
import sys
import time
import math
from pathlib import Path
from typing import Dict, List, Optional

import benchmark_common as common


def get_env_value(env_list: List[str], key: str) -> Optional[str]:
    prefix = f"{key}="
    for item in env_list:
        if item.startswith(prefix):
            return item.split("=", 1)[1]
    return None


def set_env_value(env_list: List[str], key: str, value: str) -> List[str]:
    prefix = f"{key}="
    updated: List[str] = []
    replaced = False
    for item in env_list:
        if item.startswith(prefix):
            updated.append(f"{key}={value}")
            replaced = True
        else:
            updated.append(item)
    if not replaced:
        updated.append(f"{key}={value}")
    return updated


def resolve_sharded_script(script_arg: str) -> Optional[Path]:
    """Resolve sharded_generate.py from explicit path or installed mlx_lm package."""
    candidate = Path(script_arg).expanduser()
    candidates = [candidate]

    if not candidate.is_absolute():
        candidates.append(Path.cwd() / candidate)

    try:
        import mlx_lm

        candidates.append(
            Path(mlx_lm.__file__).resolve().parent / "examples" / "sharded_generate.py"
        )
    except Exception:
        pass

    for path in candidates:
        if path.exists():
            return path.resolve()

    return None


def run_benchmark(
    model_url: str,
    context_file: Path,
    backend: str,
    hostfile: str,
    sharded_script: str,
    launcher: str = "mlx.launch",
    launch_env: Optional[List[str]] = None,
    pipeline: bool = False,
    max_tokens: int = 200,
    timeout: int = 1800,
) -> Optional[Dict]:
    """Run one distributed MLX benchmark for a context file."""
    print(f"Running distributed benchmark for {context_file}...")
    script_path = Path(sharded_script).expanduser()

    cmd = [launcher, "--backend", backend]
    if launch_env:
        for env_var in launch_env:
            cmd.extend(["--env", env_var])
    cmd.extend(
        [
            "--hostfile",
            hostfile,
            str(script_path),
            "--prompt",
            "-",
            "--model",
            model_url,
            "--max-tokens",
            str(max_tokens),
        ]
    )

    if pipeline:
        cmd.append("--pipeline")

    start_time = time.time()

    try:
        with open(context_file, "r") as handle:
            process = subprocess.Popen(
                cmd,
                stdin=handle,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            lines: List[str] = []
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    lines.append(line)
                return_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                print(f"Timeout running benchmark for {context_file}")
                return None

        total_wall_time = time.time() - start_time

        if return_code != 0:
            print("Error running distributed benchmark command.")
            return None

        output = "".join(lines)

        prompt_match = re.search(
            r"Prompt:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec",
            output,
        )
        gen_match = re.search(
            r"Generation:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec",
            output,
        )
        memory_match = re.search(r"Peak memory:\s*([\d.]+)\s*GB", output)

        if not prompt_match or not gen_match:
            print(f"Failed to parse MLX distributed output for {context_file}")
            return None

        # Best-effort extraction of generated text before the metrics section.
        metrics_start = output.find("\n==========\nPrompt:")
        if metrics_start != -1:
            generated_text = output[:metrics_start].strip()
        else:
            generated_text = output

        print(
            f"  Prompt: {prompt_match.group(1)} tokens, {prompt_match.group(2)} tokens-per-sec"
        )
        print(
            f"  Generation: {gen_match.group(1)} tokens, {gen_match.group(2)} tokens-per-sec"
        )
        if memory_match:
            print(f"  Peak memory: {memory_match.group(1)} GB")
        print(f"  Total wall time: {total_wall_time:.2f}s")

        parsed: Dict[str, object] = {
            "context_size": Path(context_file).stem,
            "prompt_tokens": int(prompt_match.group(1)),
            "prompt_tps": float(prompt_match.group(2)),
            "generation_tokens": int(gen_match.group(1)),
            "generation_tps": float(gen_match.group(2)),
            "total_time": total_wall_time,
            "eval_duration": math.nan,
            "prompt_eval_duration": math.nan,
            "time_to_first_token": math.nan,
            "generated_text": generated_text,
        }
        if memory_match:
            parsed["peak_memory_gb"] = float(memory_match.group(1))

        return parsed

    except Exception as exc:
        print(f"Error running benchmark: {exc}")
        return None


def check_mlx_installed() -> bool:
    """Check if mlx_lm is installed."""
    try:
        import mlx_lm  # noqa: F401

        return True
    except ImportError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run distributed MLX benchmarks on context files via mlx.launch"
    )
    parser.add_argument("model", help="MLX model path or repo")
    parser.add_argument(
        "--backend",
        default="jaccl",
        help="Distributed backend for mlx.launch (default: jaccl)",
    )
    parser.add_argument(
        "--hostfile",
        required=True,
        help="Path to mlx.launch hostfile JSON",
    )
    parser.add_argument(
        "--sharded-script",
        default="mlx_lm/examples/sharded_generate.py",
        help="Path to mlx_lm/examples/sharded_generate.py",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment variable for mlx.launch, KEY=VALUE (repeatable)",
    )
    parser.add_argument(
        "--launcher",
        default="mlx.launch",
        help="Launcher command (default: mlx.launch)",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use pipeline parallelism (passed to sharded_generate.py)",
    )
    parser.add_argument(
        "--fallback-fast-synch-off",
        dest="fallback_fast_synch_off",
        action="store_true",
        help=(
            "If a context run fails with MLX_METAL_FAST_SYNCH=1, retry once with "
            "MLX_METAL_FAST_SYNCH=0 (default: enabled)"
        ),
    )
    parser.add_argument(
        "--no-fallback-fast-synch-off",
        dest="fallback_fast_synch_off",
        action="store_false",
        help="Disable automatic retry with MLX_METAL_FAST_SYNCH=0",
    )
    parser.set_defaults(fallback_fast_synch_off=True)

    common.setup_common_args(parser)
    args = parser.parse_args()

    if not check_mlx_installed():
        print("MLX-LM is not installed. Please install it with: pip install mlx-lm")
        return 1

    resolved_script = resolve_sharded_script(args.sharded_script)
    if not resolved_script:
        print(f"Error: Cannot find sharded script from '{args.sharded_script}'.")
        print(
            "Provide --sharded-script /path/to/mlx_lm/examples/sharded_generate.py"
        )
        return 1

    model_name = args.model.split("/")[-1]
    output_dir = common.create_output_directory("mlx-distributed", model_name, args.output_dir)

    context_files = common.find_context_files(args.contexts, args.context_dir)
    if not context_files:
        return 1

    print("\nCollecting hardware information...")
    hardware_info = common.get_hardware_info()
    hardware_str = common.format_hardware_string(hardware_info)
    print(f"Hardware: {hardware_str}")
    print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")
    print(f"Hostfile: {args.hostfile}")
    print(f"Sharded script: {resolved_script}")
    print(f"Max tokens: {args.max_tokens}")
    if args.env:
        print(f"Launch env: {', '.join(args.env)}")
    if args.pipeline:
        print("Parallel mode: pipeline")

    # Warmup run
    warmup_file = common.find_warmup_file()
    if warmup_file:
        print(f"\n{'=' * 50}")
        print(f"Warmup run (excluded from results): {warmup_file.name}")
        print(f"{'=' * 50}")
        run_benchmark(
            model_url=args.model,
            context_file=warmup_file,
            backend=args.backend,
            hostfile=args.hostfile,
            sharded_script=str(resolved_script),
            launcher=args.launcher,
            launch_env=args.env,
            pipeline=args.pipeline,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        print("Warmup complete.")
    else:
        print("Warning: 0.5k.txt not found, skipping warmup.")

    start_time = time.time()
    results = []

    for context_file in context_files:
        print(f"\n{'=' * 50}")
        print(f"Benchmarking {context_file.name}...")
        print(f"{'=' * 50}")

        result = run_benchmark(
            model_url=args.model,
            context_file=context_file,
            backend=args.backend,
            hostfile=args.hostfile,
            sharded_script=str(resolved_script),
            launcher=args.launcher,
            launch_env=args.env,
            pipeline=args.pipeline,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        if (
            not result
            and args.fallback_fast_synch_off
            and get_env_value(args.env, "MLX_METAL_FAST_SYNCH") == "1"
        ):
            print(
                "Fast sync run failed; retrying this context with MLX_METAL_FAST_SYNCH=0..."
            )
            fallback_env = set_env_value(args.env, "MLX_METAL_FAST_SYNCH", "0")
            result = run_benchmark(
                model_url=args.model,
                context_file=context_file,
                backend=args.backend,
                hostfile=args.hostfile,
                sharded_script=str(resolved_script),
                launcher=args.launcher,
                launch_env=fallback_env,
                pipeline=args.pipeline,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            if result:
                result["fast_synch_fallback"] = 1
        if result:
            results.append(result)

            if args.save_responses:
                output_filename = output_dir / f"response_{result['context_size']}.txt"
                common.save_generated_text(
                    result, args.model, output_filename, "MLX Distributed"
                )

    total_benchmark_time = time.time() - start_time

    if not results:
        print("\nNo successful benchmark results")
        return 1

    include_memory = any("peak_memory_gb" in item for item in results)
    common.save_all_outputs(
        results,
        output_dir,
        model_name,
        "MLX Distributed",
        hardware_info,
        args,
        include_memory=include_memory,
    )
    common.print_benchmark_summary(
        results,
        model_name,
        "MLX Distributed",
        hardware_info,
        output_dir,
        total_benchmark_time,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
