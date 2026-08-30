#!/usr/bin/env python3
"""
Worker client for the LLM context benchmark WebUI.

Runs a benchmark locally (same syntax as `uv run benchmark -- ...`) and mirrors
everything to a master WebUI in realtime:

- stdout lines are streamed as they are produced, so the master's live run
  cards (progress chips, live tPS, log view) work exactly as for local runs;
- every file the engine writes into output/ is uploaded as soon as it appears,
  and re-uploaded when it changes (a final sync at the end covers the rest).

The local result folder is untouched — the master copy is additive. If the
master is unreachable the benchmark keeps running and the worker retries in
the background; a master restarted mid-run gets a fresh card and a full log
replay. Stop requests made in the master UI terminate the local benchmark.

Usage:
    uv run benchmark-worker --master http://192.168.1.10:8321 -- mlx mlx-community/Qwen3-0.6B-4bit
    uv run benchmark-worker --master http://nas.local:8321 --token s3cret -- ollama-api gpt-oss:20b
    uv run benchmark-worker --master http://192.168.1.10:8321 -- openai my-model \\
        --base-url http://127.0.0.1:8000/v1 --contexts 2,4,8
"""

import argparse
import os
import platform
import subprocess
import sys
import threading
import time

import requests

from benchmark import get_available_engines
from webui_common import OUTPUT_DIR, ROOT, RUN_META_FILE, SUMMARY_CACHE_FILE
from webui_engines import get_engine_catalog

FLUSH_SECONDS = 1.0
MAX_LINES_PER_POST = 5000
POLL_SECONDS = 3.0
REGISTER_RETRY_SECONDS = 5.0
KILL_AFTER_TERMINATE_SECONDS = 10
LOG_DRAIN_SECONDS = 30
# Master-side bookkeeping files: the master writes its own copies, so
# uploading the worker's would clobber them.
MIRROR_SKIP_FILES = {RUN_META_FILE, SUMMARY_CACHE_FILE}


def note(msg: str):
    """Worker status goes to stderr so the benchmark's stdout stays clean."""
    print(f"[worker] {msg}", file=sys.stderr, flush=True)


def hardware_string() -> str:
    return f"{platform.system()} {platform.machine()}".strip()


def parse_positional(command: list) -> tuple:
    """Derive (engine, model, contexts) from the tokens after `--`."""
    engine = command[0]
    model = ""
    if len(command) > 1 and not command[1].startswith("-"):
        model = command[1]

    contexts = []
    if "--contexts" in command:
        i = command.index("--contexts")
        if i + 1 < len(command):
            contexts = [c.strip() for c in command[i + 1].split(",") if c.strip()]
    if not contexts:
        default = get_engine_catalog().get(engine, {}).get("default_contexts") or "0.5,1,2,4,8,16,32"
        contexts = [c.strip() for c in default.split(",") if c.strip()]
    return engine, model, contexts


class WorkerClient:
    """Streams logs and result files of one local benchmark to a master WebUI."""

    def __init__(self, master, worker, token, engine, model, label, argv, contexts, settings, folder_prefix=""):
        self.base = master.rstrip("/")
        self.worker = worker
        self.headers = {"x-worker-token": token} if token else {}
        self.session = requests.Session()
        self.engine = engine
        self.model = model
        self.label = label
        self.argv = argv
        self.contexts = contexts
        self.settings = settings
        self.folder_prefix = folder_prefix  # only mirror folders with this prefix ("benchmark_<tag>_")
        self.run_id = None
        self.stop_requested = False
        self.lines = []  # every benchmark stdout line, in order
        self.sent_upto = 0
        self.sent_files = {}  # absolute path str -> (mtime_ns, size) last uploaded
        self.folder_map = {}  # local folder name -> master folder name
        self.lock = threading.Lock()
        self._last_register_attempt = 0.0

    # -- registration ------------------------------------------------------

    def try_register(self, force: bool = False) -> bool:
        """Announce the run to the master. Non-fatal: retried while running."""
        now = time.time()
        if self.run_id or (not force and now - self._last_register_attempt < REGISTER_RETRY_SECONDS):
            return bool(self.run_id)
        self._last_register_attempt = now
        payload = {
            "engine": self.engine,
            "model": self.model,
            "label": self.label,
            "worker": self.worker,
            "hardware": hardware_string(),
            "argv": self.argv,
            "contexts": self.contexts,
            "settings": self.settings,
        }
        try:
            r = self.session.post(f"{self.base}/api/worker/register", json=payload, headers=self.headers, timeout=10)
            if r.status_code == 401:
                note("master rejected the worker token — mirroring disabled")
                return False
            r.raise_for_status()
        except requests.RequestException as exc:
            note(f"master unreachable ({exc.__class__.__name__}) — will retry")
            return False
        self.run_id = r.json()["run_id"]
        with self.lock:
            self.sent_upto = 0  # fresh master card replays the full log
        note(f"mirroring to {self.base} (run {self.run_id})")
        return True

    def _lost_run(self):
        """Master forgot this run (restart/deletion): re-attach next cycle."""
        note("master lost this run — re-attaching")
        self.run_id = None
        # Files went to the dead run's folders; the next registration must
        # re-upload everything to the fresh master-side run.
        self.sent_files.clear()
        self.folder_map.clear()

    # -- log streaming -----------------------------------------------------

    def buffer_line(self, line: str):
        with self.lock:
            self.lines.append(line)

    def flush_logs(self):
        if not self.run_id and not self.try_register():
            return
        with self.lock:
            batch = self.lines[self.sent_upto : self.sent_upto + MAX_LINES_PER_POST]
        if not batch:
            return
        try:
            r = self.session.post(
                f"{self.base}/api/worker/runs/{self.run_id}/logs",
                json={"lines": batch},
                headers=self.headers,
                timeout=15,
            )
            if r.status_code == 404:
                self._lost_run()
                return
            r.raise_for_status()
            with self.lock:
                self.sent_upto += len(batch)
            if r.json().get("stop"):
                self.stop_requested = True
        except requests.RequestException as exc:
            note(f"log upload failed ({exc.__class__.__name__}) — retrying")

    def poll_stop(self):
        if not self.run_id:
            return
        try:
            r = self.session.get(f"{self.base}/api/worker/runs/{self.run_id}/poll", headers=self.headers, timeout=10)
            if r.status_code == 404:
                self._lost_run()
                return
            r.raise_for_status()
            if r.json().get("stop"):
                self.stop_requested = True
        except requests.RequestException:
            pass  # next cycle retries; stopping is best-effort

    # -- file mirroring ----------------------------------------------------

    def upload_file(self, folder, path) -> bool:
        """Upload one file from a local result folder; True when current."""
        try:
            st = path.stat()
        except OSError:
            return False
        sig = (st.st_mtime_ns, st.st_size)
        if self.sent_files.get(str(path)) == sig:
            return True
        if not self.run_id and not self.try_register():
            return False
        rel = path.relative_to(folder).as_posix()
        try:
            with open(path, "rb") as fh:
                r = self.session.post(
                    f"{self.base}/api/worker/runs/{self.run_id}/file",
                    params={"folder": folder.name, "name": rel},
                    data=fh,
                    headers=self.headers,
                    timeout=120,
                )
            if r.status_code == 404:
                self._lost_run()
                return False
            r.raise_for_status()
        except (requests.RequestException, OSError) as exc:
            note(f"file upload failed for {rel} ({exc.__class__.__name__}) — retrying")
            return False
        self.sent_files[str(path)] = sig
        self.folder_map[folder.name] = r.json()["folder"]
        return True

    def mirror_folders(self, before: set, output_dir=None):
        """Upload new/changed files from result folders created by this run."""
        out = output_dir if output_dir is not None else OUTPUT_DIR
        if not out.is_dir():
            return
        for folder in sorted(out.iterdir()):
            if not folder.is_dir() or folder.name in before:
                continue
            if not folder.name.startswith("benchmark_"):
                continue
            if self.folder_prefix and not folder.name.startswith(self.folder_prefix):
                continue  # created by a different engine's concurrent run
            for path in sorted(folder.rglob("*")):
                if not path.is_file():
                    continue
                if path.relative_to(folder).parts[0] in MIRROR_SKIP_FILES:
                    continue
                self.upload_file(folder, path)

    def uploaded_count(self) -> int:
        return len(self.sent_files)

    # -- completion --------------------------------------------------------

    def finish(self, returncode: int, before: set, stopped: bool = False) -> bool:
        """Report completion; after a master restart, re-attach and re-sync once."""
        for _ in range(2):
            if not self.run_id and not self.try_register(force=True):
                note("cannot report completion — master unreachable; local results kept")
                return False
            try:
                r = self.session.post(
                    f"{self.base}/api/worker/runs/{self.run_id}/finish",
                    json={"returncode": returncode, "stopped": bool(stopped)},
                    headers=self.headers,
                    timeout=15,
                )
                if r.status_code == 404:
                    self._lost_run()
                    continue
                r.raise_for_status()
            except requests.RequestException as exc:
                note(f"finish failed ({exc.__class__.__name__}) — local results kept")
                return False
            folders = ", ".join(self.folder_map.values()) or "no result folder"
            note(f"done (rc={returncode}) — mirrored {self.uploaded_count()} files: {folders}")
            return True
        note("could not report completion after re-attach; local results kept")
        return False


class MirrorSession:
    """Background mirroring for one running benchmark subprocess.

    Owns the flusher thread: streams buffered log lines to the master, polls
    it for stop requests, and uploads new/changed result files. Used by the
    benchmark-worker CLI and by a worker WebUI mirroring its launched runs
    (``benchmark-webui --master ...``).
    """

    def __init__(self, client: WorkerClient, proc, before: set, on_master_stop=None, output_dir=None):
        self.client = client
        self.proc = proc
        self.before = before
        self.on_master_stop = on_master_stop
        self.output_dir = output_dir if output_dir is not None else OUTPUT_DIR
        self.finished = threading.Event()  # set when the child's stdout is drained
        self._thread = None

    def buffer_line(self, line: str):
        self.client.buffer_line(line)

    def start(self):
        self._thread = threading.Thread(target=self._flusher, daemon=True)
        self._thread.start()

    def _stop_local(self):
        if self.on_master_stop is not None:
            self.on_master_stop()
            return
        self.proc.terminate()
        threading.Timer(KILL_AFTER_TERMINATE_SECONDS, lambda: self.proc.poll() is None and self.proc.kill()).start()

    def _flusher(self):
        last_poll = 0.0
        drain_deadline = None
        while True:
            self.client.flush_logs()
            if time.time() - last_poll >= POLL_SECONDS:
                self.client.poll_stop()
                last_poll = time.time()
            if self.client.stop_requested and self.proc.poll() is None:
                note("stop requested in master UI — terminating benchmark")
                self._stop_local()
            try:
                self.client.mirror_folders(self.before, self.output_dir)
            except OSError:
                pass
            if self.finished.is_set():
                with self.client.lock:
                    pending = self.client.sent_upto < len(self.client.lines)
                if not pending:
                    break
                if drain_deadline is None:
                    drain_deadline = time.time() + LOG_DRAIN_SECONDS
                if time.time() > drain_deadline:
                    note("giving up on flushing remaining logs (master unreachable)")
                    break
            self.finished.wait(FLUSH_SECONDS)

    def close(self, returncode: int, stopped: bool = False) -> bool:
        """Final file sweep, then report completion to the master."""
        self.finished.set()
        if self._thread:
            self._thread.join(timeout=60)
        try:
            self.client.mirror_folders(self.before, self.output_dir)
        except OSError:
            pass
        return self.client.finish(returncode, self.before, stopped=stopped)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark-worker",
        description="Run a benchmark locally and mirror it live to a master WebUI.",
        epilog="Everything after `--` is passed to the benchmark dispatcher, "
        "e.g.: benchmark-worker --master http://host:8321 -- mlx model",
    )
    parser.add_argument("--master", required=True, help="Master WebUI base URL, e.g. http://192.168.1.10:8321")
    parser.add_argument("--worker", default=platform.node(), help="Worker name shown in the UI (default: hostname)")
    parser.add_argument(
        "--token",
        default=os.environ.get("BENCHMARK_WORKER_TOKEN", ""),
        help="Shared secret when the master requires one (env: BENCHMARK_WORKER_TOKEN)",
    )
    parser.add_argument("--label", default="", help="Label for the run card in the UI")
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Benchmark dispatcher tokens: <engine> [model] [options] (after `--`)",
    )
    args = parser.parse_args()

    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("missing benchmark command — put the dispatcher tokens after `--`")
    engines = get_available_engines()
    if command[0] not in engines:
        parser.error(f"unknown engine '{command[0]}' (see: uv run benchmark -- --list-engines)")

    engine, model, contexts = parse_positional(command)
    argv = [sys.executable, str(ROOT / "benchmark.py"), *command]
    tag = get_engine_catalog().get(engine, {}).get("tag", "")
    client = WorkerClient(
        master=args.master,
        worker=args.worker,
        token=args.token,
        engine=engine,
        model=model or "(auto)",
        label=args.label,
        argv=argv,
        contexts=contexts,
        settings={
            "engine": engine,
            "model": model,
            "contexts": ",".join(contexts),
            "extra_args": " ".join(command[2:] if model else command[1:]),
        },
        folder_prefix=f"benchmark_{tag}_" if tag else "",
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    before = {p.name for p in OUTPUT_DIR.iterdir() if p.is_dir()}
    client.try_register()

    note(f"starting: {' '.join(command)}")
    proc = subprocess.Popen(
        argv,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    session = MirrorSession(client, proc, before)
    returncode = None
    interrupted = False

    def reader():
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            session.buffer_line(line.rstrip("\n"))
        session.finished.set()

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()
    session.start()

    try:
        proc.wait()
        reader_thread.join(timeout=30)
        returncode = proc.returncode
    except KeyboardInterrupt:
        note("interrupted — shutting down")
        interrupted = True
        returncode = 130
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=KILL_AFTER_TERMINATE_SECONDS)
            except subprocess.TimeoutExpired:
                proc.kill()
    finally:
        session.finished.set()
        session.close(returncode, stopped=interrupted)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
