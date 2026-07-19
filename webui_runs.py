"""Subprocess run manager for the web UI: launches benchmark scripts,
captures their output live and claims result folders."""

import json
import os
import re
import subprocess
import threading
import time
import uuid
from datetime import datetime

from fastapi import HTTPException

from webui_common import OUTPUT_DIR, ROOT, RUN_META_FILE

PROGRESS_RE = re.compile(r"Benchmarking\s+([\d.]+k)\.txt")
BATCH_PROGRESS_RE = re.compile(r"^\s*Batch size\s+(\d+)\s*\(")
BATCH_COMPLETE_RE = re.compile(r"^\s*Batch benchmark complete:\s+(\d+)\s+sizes tested")
# Warmup / decode failures across engines, e.g.:
#   "Warmup failed for batch size 8: ... — skipping"
#   "Failed to decode random prompts: ... — skipping batch size 4"
BATCH_SKIP_RE = re.compile(
    r"(?:Warmup failed for batch size|skipping batch size)\s+(\d+)",
    re.IGNORECASE,
)
GEN_TPS_RES = [
    re.compile(r"Generation:\s+\d+\s+tokens\s+in\s+[\d.]+s\s+=\s+([\d.]+)\s*t/s"),
    re.compile(r"Generation TPS:\s+([\d.]+)"),
    re.compile(r"generation_tps:\s+([\d.]+)"),
]
PROMPT_TPS_RES = [
    re.compile(r"Prompt:\s+\d+\s+tokens\s+in\s+[\d.]+s\s+=\s+([\d.]+)\s*t/s"),
    re.compile(r"Prompt TPS:\s+([\d.]+)"),
]
TTFT_RES = [
    re.compile(r"Time to first token:\s+([\d.]+)s"),
    re.compile(r"TTFT:\s+([\d.]+)s"),
]

SECRET_FLAGS = {"--api-key"}


def batch_sizes_from_argv(argv: list) -> list[int]:
    """Return the effective batch sweep, or none when this run skips it."""
    if "--no-batch" in argv:
        return []

    value = None
    for index, arg in enumerate(argv):
        if arg == "--batch-sizes" and index + 1 < len(argv):
            value = argv[index + 1]
        elif arg.startswith("--batch-sizes="):
            value = arg.split("=", 1)[1]
    if value is None:
        return []
    try:
        return [int(size.strip()) for size in value.split(",") if size.strip()]
    except ValueError:
        return []


def redact_argv(argv: list) -> list:
    out, hide = [], False
    for arg in argv:
        out.append("***" if hide else arg)
        hide = arg in SECRET_FLAGS
    return out


class BenchmarkRun:
    def __init__(self, run_id, kind, engine_id, tag, model, label, endpoint_name, argv, contexts):
        self.id = run_id
        self.kind = kind  # "benchmark" | "ctxgen"
        self.engine = engine_id
        self.tag = tag
        self.model = model
        self.label = label
        self.endpoint_name = endpoint_name
        self.argv = argv
        self.contexts = contexts
        self.status = "starting"
        self.returncode = None
        self.started = time.time()
        self.finished = None
        self.log_lines = []
        self.lock = threading.Lock()
        self.proc = None
        self.stop_requested = False
        self.current_context = None
        self.contexts_done = 0
        self.batch_sizes = batch_sizes_from_argv(argv)
        self.current_batch_size = None
        self.current_batch_index = None
        self.batch_sizes_done = 0
        self.batch_skipped = []  # indices into batch_sizes that failed/skipped
        self.phase = None
        self.live = {}
        self.result_folders = []
        self.error = None

    def snapshot(self):
        with self.lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "engine": self.engine,
                "model": self.model,
                "label": self.label,
                "endpoint": self.endpoint_name,
                "command": " ".join(redact_argv(self.argv)),
                "status": self.status,
                "returncode": self.returncode,
                "started": self.started,
                "finished": self.finished,
                "elapsed": (self.finished or time.time()) - self.started,
                "contexts": self.contexts,
                "current_context": self.current_context,
                "contexts_done": self.contexts_done,
                "batch_sizes": self.batch_sizes,
                "current_batch_size": self.current_batch_size,
                "current_batch_index": self.current_batch_index,
                "batch_sizes_done": self.batch_sizes_done,
                "batch_skipped": list(self.batch_skipped),
                "phase": self.phase,
                "live": dict(self.live),
                "result_folders": list(self.result_folders),
                "error": self.error,
                "log_length": len(self.log_lines),
            }

    def log_slice(self, offset):
        with self.lock:
            return self.log_lines[offset:], len(self.log_lines)


class RunManager:
    def __init__(self):
        self.runs = {}
        self.lock = threading.Lock()

    def start(self, kind, engine_id, tag, model, label, endpoint_name, argv, contexts):
        run = BenchmarkRun(uuid.uuid4().hex[:12], kind, engine_id, tag, model, label, endpoint_name, argv, contexts)
        with self.lock:
            self.runs[run.id] = run
        thread = threading.Thread(target=self._execute, args=(run,), daemon=True)
        thread.start()
        return run

    def get(self, run_id) -> BenchmarkRun:
        with self.lock:
            run = self.runs.get(run_id)
        if run is None:
            raise HTTPException(404, f"Unknown run '{run_id}'")
        return run

    def list_runs(self):
        with self.lock:
            runs = list(self.runs.values())
        return sorted((r.snapshot() for r in runs), key=lambda r: r["started"], reverse=True)

    def stop(self, run_id):
        run = self.get(run_id)
        with run.lock:
            run.stop_requested = True
            proc = run.proc
        if proc and proc.poll() is None:
            proc.terminate()
            threading.Timer(10, lambda: proc.poll() is None and proc.kill()).start()
        return run

    def _execute(self, run: BenchmarkRun):
        OUTPUT_DIR.mkdir(exist_ok=True)
        before = {p.name for p in OUTPUT_DIR.iterdir() if p.is_dir()}
        try:
            # PYTHONUNBUFFERED: without it the child buffers stdout when piped,
            # so the UI only sees output in 8 KB bursts instead of live lines.
            proc = subprocess.Popen(
                run.argv,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except Exception as exc:
            with run.lock:
                run.status = "failed"
                run.error = str(exc)
                run.finished = time.time()
            return

        with run.lock:
            run.proc = proc
            run.status = "running"

        seen_contexts = set()
        for line in proc.stdout:
            line = line.rstrip("\n")
            with run.lock:
                run.log_lines.append(line)
                match = PROGRESS_RE.search(line)
                if match:
                    ctx = match.group(1)
                    if run.current_batch_index is not None:
                        run.batch_sizes_done = max(run.batch_sizes_done, run.current_batch_index + 1)
                        run.current_batch_size = None
                        run.current_batch_index = None
                    if run.current_context and run.current_context not in seen_contexts:
                        seen_contexts.add(run.current_context)
                    run.current_context = ctx
                    run.contexts_done = len(seen_contexts)
                    run.phase = "context"
                batch_match = BATCH_PROGRESS_RE.search(line)
                if batch_match:
                    batch_size = int(batch_match.group(1))
                    if run.current_context and run.current_context not in seen_contexts:
                        seen_contexts.add(run.current_context)
                        run.contexts_done = len(seen_contexts)
                    run.current_context = None
                    if run.current_batch_index is not None:
                        run.batch_sizes_done = max(run.batch_sizes_done, run.current_batch_index + 1)
                    run.current_batch_index = next(
                        (
                            index
                            for index in range(run.batch_sizes_done, len(run.batch_sizes))
                            if run.batch_sizes[index] == batch_size
                        ),
                        None,
                    )
                    # Fall back to any matching chip if the remaining-range lookup misses
                    # (e.g. duplicate sizes or a size already marked done).
                    if run.current_batch_index is None:
                        try:
                            run.current_batch_index = run.batch_sizes.index(batch_size)
                        except ValueError:
                            run.current_batch_index = None
                    run.current_batch_size = batch_size
                    run.phase = "batch"
                skip_match = BATCH_SKIP_RE.search(line)
                if skip_match and run.batch_sizes:
                    skipped_size = int(skip_match.group(1))
                    idx = run.current_batch_index
                    if idx is None or run.batch_sizes[idx] != skipped_size:
                        idx = next(
                            (
                                index
                                for index in range(len(run.batch_sizes))
                                if run.batch_sizes[index] == skipped_size and index not in run.batch_skipped
                            ),
                            None,
                        )
                    if idx is not None:
                        if idx not in run.batch_skipped:
                            run.batch_skipped.append(idx)
                        run.batch_sizes_done = max(run.batch_sizes_done, idx + 1)
                        if run.current_batch_index == idx:
                            run.current_batch_size = None
                            run.current_batch_index = None
                batch_complete_match = BATCH_COMPLETE_RE.search(line)
                if batch_complete_match:
                    # Sweep finished — advance past every planned size. Skipped
                    # indices stay in batch_skipped so chips don't look successful.
                    if run.current_batch_index is not None:
                        run.batch_sizes_done = max(run.batch_sizes_done, run.current_batch_index + 1)
                    run.batch_sizes_done = max(run.batch_sizes_done, len(run.batch_sizes))
                    run.current_batch_size = None
                    run.current_batch_index = None
                    run.phase = None
                live_updated = False
                for regex in GEN_TPS_RES:
                    m = regex.search(line)
                    if m:
                        run.live["generation_tps"] = float(m.group(1))
                        live_updated = True
                        break
                for regex in PROMPT_TPS_RES:
                    m = regex.search(line)
                    if m:
                        run.live["prompt_tps"] = float(m.group(1))
                        live_updated = True
                        break
                for regex in TTFT_RES:
                    m = regex.search(line)
                    if m:
                        run.live["ttft"] = float(m.group(1))
                        live_updated = True
                        break
                if live_updated and run.phase:
                    run.live["source"] = run.phase
                    if run.phase == "batch" and run.current_batch_size is not None:
                        run.live["source_batch_size"] = run.current_batch_size
                    elif run.phase == "context" and run.current_context:
                        run.live["source_context"] = run.current_context
        proc.wait()

        folders = []
        if run.kind == "benchmark":
            try:
                after = {p.name for p in OUTPUT_DIR.iterdir() if p.is_dir()}
                prefix = f"benchmark_{run.tag}_"
                for name in sorted(after - before):
                    if not name.startswith(prefix):
                        continue
                    meta_path = OUTPUT_DIR / name / RUN_META_FILE
                    if meta_path.exists():
                        continue  # claimed by a concurrent run
                    meta = {
                        "run_id": run.id,
                        "engine_id": run.engine,
                        "label": run.label or "",
                        "endpoint": run.endpoint_name or "",
                        "created": datetime.now().isoformat(timespec="seconds"),
                    }
                    meta_path.write_text(json.dumps(meta, indent=2))
                    folders.append(name)
            except OSError:
                pass

        with run.lock:
            run.returncode = proc.returncode
            run.finished = time.time()
            run.result_folders = folders
            run.current_context = None
            run.current_batch_size = None
            run.current_batch_index = None
            run.phase = None
            if run.stop_requested:
                run.status = "stopped"
            elif proc.returncode == 0:
                run.status = "done"
                run.contexts_done = len(run.contexts)
                run.batch_sizes_done = len(run.batch_sizes)
            else:
                run.status = "failed"
