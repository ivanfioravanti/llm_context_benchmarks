"""Subprocess run manager for the web UI: launches benchmark scripts,
captures their output live and claims result folders."""

import json
import os
import re
import shlex
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
    re.compile(r"Generation:\s+\d+\s+tokens\s+in\s+[\d.]+s\s+=\s+([\d.]+)\s*t/s"),  # ollama-api, lmstudio
    re.compile(r"Generation TPS:\s+([\d.]+)"),  # mtplx, mlxserve, dflash, llamacpp, openai, ...
    re.compile(r"generation_tps:\s+([\d.]+)"),  # run_benchmark_peak per-run line
    re.compile(r"Generation throughput:\s+([\d.]+)\s*tokens/sec"),  # omlx, deepseek, grok, exo, vmlx
    re.compile(r"Generation:\s+\d+\s+tokens\s+at\s+([\d.]+)\s*t/s"),  # ollama-cli
    re.compile(r"\btg\s+([\d.]+)\s*t/s"),  # batch trial / peak lines: "pp 900.0 tg 55.0 t/s"
]
PROMPT_TPS_RES = [
    re.compile(r"Prompt:\s+\d+\s+tokens\s+in\s+[\d.]+s\s+=\s+([\d.]+)\s*t/s"),  # ollama-api, lmstudio
    re.compile(r"Prompt TPS:\s+([\d.]+)"),  # mtplx, mlxserve, dflash, llamacpp, openai, ...
    re.compile(r"Prompt throughput:\s+([\d.]+)\s*tokens/sec"),  # omlx, deepseek, grok, exo, vmlx, mlx-vlm
    re.compile(r"\bpp\s+([\d.]+)\s+tg"),  # batch trial / peak lines: "pp 900.0 tg 55.0 t/s"
    re.compile(r"Prompt:\s+\d+\s+tokens,\s+([\d.]+)\s*tokens-per-sec"),  # mlx, mlx-distributed
    re.compile(r"Prompt:\s+\d+\s+tokens\s+at\s+([\d.]+)\s*t/s"),  # ollama-cli
    re.compile(r"prompt_tps:\s+([\d.]+)"),  # run_benchmark_peak merged-peak line
]
TTFT_RES = [
    re.compile(r"Time to first token:\s+([\d.]+)s"),
    re.compile(r"TTFT:\s+([\d.]+)s"),
]
TOTAL_TIME_RES = [
    re.compile(r"Total(?: wall)? time:\s+([\d.]+)s"),
]
PEAK_MEM_RES = [
    re.compile(r"[Pp]eak mem(?:ory)?:\s+([\d.]+)\s*GB"),
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


def format_command(argv: list, secret_placeholder: str = "***") -> str:
    """Return a shell-safe command while replacing secret flag values."""
    parts = []
    hide_next = False
    for arg in argv:
        if hide_next:
            parts.append(secret_placeholder if secret_placeholder.startswith("$") else shlex.quote(secret_placeholder))
            hide_next = False
            continue
        if arg in SECRET_FLAGS:
            parts.append(shlex.quote(arg))
            hide_next = True
            continue
        matching_flag = next((flag for flag in SECRET_FLAGS if arg.startswith(flag + "=")), None)
        if matching_flag:
            value = secret_placeholder if secret_placeholder.startswith("$") else shlex.quote(secret_placeholder)
            parts.append(f"{shlex.quote(matching_flag)}={value}")
            continue
        parts.append(shlex.quote(arg))
    return " ".join(parts)


class BenchmarkRun:
    def __init__(
        self,
        run_id,
        kind,
        engine_id,
        tag,
        model,
        label,
        endpoint_name,
        argv,
        contexts,
        endpoint_hardware="",
        settings=None,
    ):
        self.id = run_id
        self.kind = kind  # "benchmark" | "ctxgen"
        self.engine = engine_id
        self.tag = tag
        self.model = model
        self.label = label
        self.endpoint_name = endpoint_name
        self.endpoint_hardware = endpoint_hardware
        self.worker = ""  # remote worker name ("" for local subprocess runs)
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
        # original /api/runs start payload, replayed by the Results »rerun« button
        self.settings = settings
        self.error = None
        # per-context / per-batch metrics accumulated from console output;
        # entries are {"context": "2k", ...} or {"batch_size": 8, ...} with
        # whatever metric lines the engine has printed for them so far
        self.progress = []
        self._seen_contexts = set()
        self.remote = False

    def snapshot(self):
        with self.lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "engine": self.engine,
                "model": self.model,
                "label": self.label,
                "worker": self.worker,
                "command": format_command(self.argv),
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
                "progress": [dict(entry) for entry in self.progress],
            }

    def log_slice(self, offset):
        with self.lock:
            return self.log_lines[offset:], len(self.log_lines)

    def ingest_line(self, line: str):
        """Parse one benchmark stdout line into live progress state.

        Shared by the local subprocess loop in RunManager._execute and the
        worker log-upload endpoint, so remote runs show identical live cards.
        """
        with self.lock:
            self.log_lines.append(line)
            match = PROGRESS_RE.search(line)
            if match:
                ctx = match.group(1)
                if self.current_batch_index is not None:
                    self.batch_sizes_done = max(self.batch_sizes_done, self.current_batch_index + 1)
                    self.current_batch_size = None
                    self.current_batch_index = None
                if self.current_context and self.current_context not in self._seen_contexts:
                    self._seen_contexts.add(self.current_context)
                self.current_context = ctx
                self.contexts_done = len(self._seen_contexts)
                self.phase = "context"
                if not self.progress or self.progress[-1].get("context") != ctx:
                    self.progress.append({"context": ctx})
            batch_match = BATCH_PROGRESS_RE.search(line)
            if batch_match:
                batch_size = int(batch_match.group(1))
                if self.current_context and self.current_context not in self._seen_contexts:
                    self._seen_contexts.add(self.current_context)
                    self.contexts_done = len(self._seen_contexts)
                self.current_context = None
                if self.current_batch_index is not None:
                    self.batch_sizes_done = max(self.batch_sizes_done, self.current_batch_index + 1)
                self.current_batch_index = next(
                    (
                        index
                        for index in range(self.batch_sizes_done, len(self.batch_sizes))
                        if self.batch_sizes[index] == batch_size
                    ),
                    None,
                )
                # Fall back to any matching chip if the remaining-range lookup misses
                # (e.g. duplicate sizes or a size already marked done).
                if self.current_batch_index is None:
                    try:
                        self.current_batch_index = self.batch_sizes.index(batch_size)
                    except ValueError:
                        self.current_batch_index = None
                self.current_batch_size = batch_size
                self.phase = "batch"
                if not self.progress or self.progress[-1].get("batch_size") != batch_size:
                    self.progress.append({"batch_size": batch_size})
            skip_match = BATCH_SKIP_RE.search(line)
            if skip_match and self.batch_sizes:
                skipped_size = int(skip_match.group(1))
                idx = self.current_batch_index
                if idx is None or self.batch_sizes[idx] != skipped_size:
                    idx = next(
                        (
                            index
                            for index in range(len(self.batch_sizes))
                            if self.batch_sizes[index] == skipped_size and index not in self.batch_skipped
                        ),
                        None,
                    )
                if idx is not None:
                    if idx not in self.batch_skipped:
                        self.batch_skipped.append(idx)
                    self.batch_sizes_done = max(self.batch_sizes_done, idx + 1)
                    if self.current_batch_index == idx:
                        self.current_batch_size = None
                        self.current_batch_index = None
            batch_complete_match = BATCH_COMPLETE_RE.search(line)
            if batch_complete_match:
                # Sweep finished — advance past every planned size. Skipped
                # indices stay in batch_skipped so chips don't look successful.
                if self.current_batch_index is not None:
                    self.batch_sizes_done = max(self.batch_sizes_done, self.current_batch_index + 1)
                self.batch_sizes_done = max(self.batch_sizes_done, len(self.batch_sizes))
                self.current_batch_size = None
                self.current_batch_index = None
                self.phase = None
            live_updated = False
            for regex in GEN_TPS_RES:
                m = regex.search(line)
                if m:
                    value = float(m.group(1))
                    self.live["generation_tps"] = value
                    if self.progress:
                        self.progress[-1]["generation_tps"] = value
                    live_updated = True
                    break
            for regex in PROMPT_TPS_RES:
                m = regex.search(line)
                if m:
                    value = float(m.group(1))
                    self.live["prompt_tps"] = value
                    if self.progress:
                        self.progress[-1]["prompt_tps"] = value
                    live_updated = True
                    break
            for regex in TTFT_RES:
                m = regex.search(line)
                if m:
                    value = float(m.group(1))
                    self.live["ttft"] = value
                    if self.progress:
                        self.progress[-1]["time_to_first_token"] = value
                    live_updated = True
                    break
            for regex in TOTAL_TIME_RES:
                m = regex.search(line)
                if m:
                    if self.progress:
                        self.progress[-1]["total_time"] = float(m.group(1))
                    break
            for regex in PEAK_MEM_RES:
                m = regex.search(line)
                if m:
                    if self.progress:
                        self.progress[-1]["peak_memory_gb"] = float(m.group(1))
                    break
            if live_updated and self.phase:
                self.live["source"] = self.phase
                if self.phase == "batch" and self.current_batch_size is not None:
                    self.live["source_batch_size"] = self.current_batch_size
                elif self.phase == "context" and self.current_context:
                    self.live["source_context"] = self.current_context


class RunManager:
    def __init__(self):
        self.runs = {}
        self.lock = threading.Lock()
        # run_id -> {worker_folder_name: master_folder_name}
        self.remote_folders = {}
        # Worker mode: {"master": url, "token": str, "name": str} — when set,
        # every locally launched benchmark run is mirrored to a master WebUI
        # (benchmark-webui --master ...).
        self.mirror = None

    def start(
        self,
        kind,
        engine_id,
        tag,
        model,
        label,
        endpoint_name,
        argv,
        contexts,
        endpoint_hardware="",
        settings=None,
    ):
        run = BenchmarkRun(
            uuid.uuid4().hex[:12],
            kind,
            engine_id,
            tag,
            model,
            label,
            endpoint_name,
            argv,
            contexts,
            endpoint_hardware=endpoint_hardware,
            settings=settings,
        )
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

    def delete(self, run_id):
        run = self.get(run_id)
        # stop an in-flight run first so its subprocess doesn't outlive the card
        if run.status in ("starting", "running"):
            self.stop(run_id)
        with self.lock:
            self.runs.pop(run_id, None)
            self.remote_folders.pop(run_id, None)
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

        mirror_session = None
        if self.mirror and run.kind == "benchmark":
            mirror_session = self._start_mirror(run, proc, before)

        for line in proc.stdout:
            run.ingest_line(line.rstrip("\n"))
            if mirror_session:
                mirror_session.buffer_line(line.rstrip("\n"))
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
                        "endpoint_hardware": run.endpoint_hardware or "",
                        "created": datetime.now().isoformat(timespec="seconds"),
                    }
                    if run.settings:
                        meta["settings"] = run.settings
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

        if mirror_session:
            mirror_session.close(proc.returncode, stopped=run.stop_requested)

    def _start_mirror(self, run: BenchmarkRun, proc, before: set):
        """Mirror a locally launched run to the master configured in self.mirror."""
        from webui_worker import MirrorSession, WorkerClient

        cfg = self.mirror
        client = WorkerClient(
            master=cfg["master"],
            worker=cfg.get("name") or "webui-worker",
            token=cfg.get("token") or "",
            engine=run.engine,
            model=run.model,
            label=run.label,
            argv=run.argv,
            contexts=run.contexts,
            settings=run.settings,
            folder_prefix=f"benchmark_{run.tag}_" if run.tag else "",
        )
        client.try_register()
        session = MirrorSession(client, proc, before, on_master_stop=lambda: self.stop(run.id), output_dir=OUTPUT_DIR)
        session.start()
        return session

    # -- remote workers ----------------------------------------------------

    def start_remote(
        self,
        engine_id,
        tag,
        model,
        label,
        worker,
        hardware,
        argv,
        contexts,
        settings=None,
    ):
        """Register a run executed on a worker machine (no local subprocess).

        The run appears in the UI immediately with status "running"; the worker
        then streams log lines and uploads result files via /api/worker/*.
        """
        run = BenchmarkRun(
            uuid.uuid4().hex[:12],
            "benchmark",
            engine_id,
            tag,
            model,
            label,
            worker,
            argv,
            contexts,
            endpoint_hardware=hardware,
            settings=settings,
        )
        run.worker = worker
        run.remote = True
        run.status = "running"
        with self.lock:
            self.runs[run.id] = run
            self.remote_folders.setdefault(run.id, {})
        return run

    def claim_remote_folder(self, run: BenchmarkRun, worker_folder: str) -> str:
        """Map a worker-side result folder to a fresh master-side folder name.

        Idempotent per (run, worker folder): the first upload claims the name,
        later uploads for the same folder reuse it. A name collision (same
        worker machine and second, or a leftover folder) gets a numeric suffix.
        """
        with self.lock:
            claimed = self.remote_folders.get(run.id, {}).get(worker_folder)
            if claimed:
                return claimed
            taken = set(self.remote_folders.get(run.id, {}))
            for run_folders in self.remote_folders.values():
                taken.update(run_folders.values())
            if OUTPUT_DIR.is_dir():
                taken.update(p.name for p in OUTPUT_DIR.iterdir() if p.is_dir())
            name = worker_folder
            n = 2
            while name in taken:
                name = f"{worker_folder}_{n}"
                n += 1
            self.remote_folders.setdefault(run.id, {})[worker_folder] = name
        folder = OUTPUT_DIR / name
        folder.mkdir(parents=True, exist_ok=True)
        meta = {
            "run_id": run.id,
            "engine_id": run.engine,
            "label": run.label or "",
            "endpoint": run.worker or "",
            "worker": run.worker or "",
            "endpoint_hardware": run.endpoint_hardware or "",
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        if run.settings:
            meta["settings"] = run.settings
        (folder / RUN_META_FILE).write_text(json.dumps(meta, indent=2))
        with run.lock:
            if name not in run.result_folders:
                run.result_folders.append(name)
        return name

    def finish_remote(self, run: BenchmarkRun, returncode: int, stopped: bool = False) -> BenchmarkRun:
        """Finalize a remote run when its worker reports the process exit."""
        with run.lock:
            run.returncode = returncode
            run.finished = time.time()
            run.current_context = None
            run.current_batch_size = None
            run.current_batch_index = None
            run.phase = None
            if run.stop_requested or stopped:
                run.status = "stopped"
            elif returncode == 0:
                run.status = "done"
                run.contexts_done = len(run.contexts)
                run.batch_sizes_done = len(run.batch_sizes)
            else:
                run.status = "failed"
        return run
