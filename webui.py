"""
Web UI for the LLM context benchmark toolkit.

Serves a single-page app that can launch any registered benchmark engine,
watch runs live, manage named endpoints, browse saved results in output/
and build interactive comparisons across any set of runs.

Companion modules: webui_common (paths), webui_engines (engine catalog +
command construction), webui_runs (subprocess run manager).

Usage:
    uv run benchmark-webui                # http://127.0.0.1:8321
    uv run benchmark-webui --port 9000 --host 0.0.0.0
    python webui.py --no-open
"""

import argparse
import hmac
import json
import os
import platform
import re
import shutil
import sys
import threading
import time
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import benchmark_common
from compare_benchmarks import parse_benchmark_folder
from webui_common import (
    CONTEXT_FILE_RE,
    ENDPOINTS_FILE,
    FOLDER_TS_RE,
    OUTPUT_DIR,
    ROOT,
    RUN_META_FILE,
    STATIC_DIR,
    SUMMARY_CACHE_FILE,
    is_apple_silicon,
)
from webui_engines import build_command, cached_mlx_models, engine_available, get_engine_catalog
from webui_runs import RunManager, format_command

run_manager = RunManager()


# ---------------------------------------------------------------------------
# Endpoints store
# ---------------------------------------------------------------------------


def load_endpoints() -> list:
    if ENDPOINTS_FILE.exists():
        try:
            os.chmod(ENDPOINTS_FILE, 0o600)  # may contain API keys
            return json.loads(ENDPOINTS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_endpoints(endpoints: list):
    # owner-only: the file may contain API keys
    fd = os.open(str(ENDPOINTS_FILE), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, json.dumps(endpoints, indent=2).encode())
    finally:
        os.close(fd)


def normalize_base_url(base_url: str, engine_id: str, api_key: str = "") -> tuple[str, bool]:
    """Append /v1 to a path-less base URL when the engine expects an OpenAI-style
    /v1 root and the server actually answers there (probed, never guessed).

    Returns (url, corrected). URLs that already carry a path, engines that
    handle the prefix themselves (lmstudio, exo, mtplx, ...) and unreachable
    servers are left untouched.
    """
    url = (base_url or "").strip().rstrip("/")
    # engines whose script wants the /v1 root advertise that via their default
    info = get_engine_catalog().get(engine_id) or {}
    if not url or not (info.get("default_base_url") or "").rstrip("/").endswith("/v1"):
        return base_url, False
    from urllib.parse import urlparse

    if (urlparse(url).path or "").strip("/"):
        return base_url, False
    headers = {"User-Agent": "context-bench-webui"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        import httpx

        if httpx.get(url + "/models", timeout=3.0, headers=headers).status_code < 400:
            return base_url, False  # server genuinely serves at the root
        if httpx.get(url + "/v1/models", timeout=3.0, headers=headers).status_code < 400:
            return url + "/v1", True
    except Exception:
        pass  # unreachable right now — keep what the user entered
    return base_url, False


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def read_run_meta(folder: Path) -> dict:
    meta_path = folder / RUN_META_FILE
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def context_sort_key(ctx: str) -> float:
    try:
        return float(str(ctx).rstrip("k"))
    except ValueError:
        return 0.0


def folder_timestamp(name: str):
    m = FOLDER_TS_RE.search(name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").isoformat(timespec="seconds")
        except ValueError:
            pass
    return None


def folder_data_sig(folder: Path) -> str:
    """Fingerprint of files that feed a result summary (not label/meta)."""
    parts = []
    for name in ("benchmark_results.csv", "hardware_info.json", "batch_benchmark.json"):
        path = folder / name
        if path.is_file():
            st = path.stat()
            parts.append(f"{name}:{st.st_mtime_ns}:{st.st_size}")
    for path in sorted(folder.glob("*.png")):
        st = path.stat()
        parts.append(f"{path.name}:{st.st_mtime_ns}:{st.st_size}")
    return "|".join(parts)


def apply_meta_to_summary(summary: dict, meta: dict) -> dict:
    """Overlay live webui_run.json fields onto a cached/parsed summary."""
    out = dict(summary)
    out["label"] = meta.get("label") or ""
    out["endpoint"] = meta.get("endpoint") or ""
    out["endpoint_hardware"] = meta.get("endpoint_hardware") or ""
    # Endpoint hardware (user-provided) beats the locally captured chip string.
    out["machine"] = out["endpoint_hardware"] or out.get("machine") or ""
    return out


def build_result_summary(folder: Path, parsed: dict) -> dict:
    """Build the archive-list summary from an already-parsed folder (no meta)."""
    rows = sorted(parsed["results"], key=lambda r: context_sort_key(r.get("context_size", "0")))
    hw = parsed["hardware_info"] or {}
    gen_series = [
        [context_sort_key(r["context_size"]), r.get("generation_tps")]
        for r in rows
        if isinstance(r.get("generation_tps"), float)
    ]
    gen_values = [v for _, v in gen_series]
    charts = sorted(p.name for p in folder.glob("*.png"))
    return {
        "folder": folder.name,
        "engine": parsed["engine"],
        "model": parsed["model"],
        "cache_mode": parsed["cache_mode"],
        "context_type": parsed.get("context_type", ""),
        "timestamp": folder_timestamp(folder.name),
        "machine": hw.get("chip") or hw.get("machine_label") or hw.get("processor") or "",
        "hardware": benchmark_common.format_hardware_string(hw) if hw else "",
        "contexts": [r.get("context_size") for r in rows],
        "peak_generation_tps": max(gen_values) if gen_values else None,
        "gen_series": gen_series,
        "columns": sorted({k for r in rows for k in r.keys()}),
        "has_batch": bool(parsed["batch_data"]),
        "charts": charts,
    }


def summarize_result_folder(folder: Path, parsed: dict | None = None):
    """List/detail summary for a result folder. Cached in webui_summary.json
    keyed by a fingerprint of the CSV/hardware/batch/chart files; label and
    endpoint fields are always overlaid from webui_run.json so renames stay cheap.
    Pass ``parsed`` to skip a second parse when the caller already has it.
    """
    meta = read_run_meta(folder)
    sig = folder_data_sig(folder)
    cache_path = folder / SUMMARY_CACHE_FILE
    if parsed is None and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text())
            if cached.get("sig") == sig and isinstance(cached.get("summary"), dict):
                return apply_meta_to_summary(cached["summary"], meta)
        except (json.JSONDecodeError, OSError, TypeError):
            pass

    if parsed is None:
        parsed, _ = parse_benchmark_folder(folder)
    if not parsed:
        return None
    summary = build_result_summary(folder, parsed)
    try:
        cache_path.write_text(json.dumps({"sig": sig, "summary": summary}, indent=2))
    except OSError:
        pass
    return apply_meta_to_summary(summary, meta)


def resolve_result_folder(name: str) -> Path:
    if "/" in name or "\\" in name or name.startswith(".") or not name.startswith("benchmark_"):
        raise HTTPException(400, "Invalid result folder name")
    folder = OUTPUT_DIR / name
    if not folder.is_dir():
        raise HTTPException(404, f"Result folder '{name}' not found")
    return folder


# ---------------------------------------------------------------------------
# App / routes
# ---------------------------------------------------------------------------

app = FastAPI(title="LLM Context Bench", docs_url=None, redoc_url=None)


def in_container() -> bool:
    """True when the server runs inside a container (Docker/Podman)."""
    return (
        os.environ.get("CONTEXT_BENCH_CONTAINER") == "1"
        or Path("/.dockerenv").exists()
        or Path("/run/.containerenv").exists()
    )


@app.get("/api/meta")
def api_meta():
    catalog = get_engine_catalog()
    engines = []
    for engine_id, info in catalog.items():
        engines.append(
            {
                "id": engine_id,
                "label": info["label"],
                "description": info["description"],
                "example": info["example"],
                "model": info["model"],
                "connection": info["connection"],
                "default_base_url": info.get("default_base_url", ""),
                "default_contexts": info.get("default_contexts", "0.5,1,2,4,8,16,32"),
                "cold_prefill": info["cold_prefill"],
                "local_mlx": info["local_mlx"],
                "available": engine_available(info),
                "options": info["options"],
            }
        )
    context_files = []
    for path in sorted(ROOT.glob("*.txt")):
        m = CONTEXT_FILE_RE.match(path.stem)
        if m:
            context_files.append({"size": float(m.group(1)), "name": path.stem, "file": path.name})
    context_files.sort(key=lambda c: c["size"])
    source_files = sorted(p.name for p in ROOT.glob("*.txt") if not CONTEXT_FILE_RE.match(p.stem))
    hw = benchmark_common.get_hardware_info()
    return {
        "engines": engines,
        "in_container": in_container(),
        "mlx_available": is_apple_silicon(),
        "hardware": hw,
        "hardware_string": benchmark_common.format_hardware_string(hw),
        "context_files": context_files,
        "source_files": source_files,
    }


@app.get("/api/endpoints")
def api_endpoints_list():
    return load_endpoints()


@app.post("/api/endpoints")
def api_endpoints_create(payload: dict):
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Endpoint name is required")
    endpoints = load_endpoints()
    entry = {
        "id": uuid.uuid4().hex[:10],
        "name": name,
        "engine": payload.get("engine") or "",
        "model": payload.get("model") or "",
        "base_url": payload.get("base_url") or "",
        "api_key": payload.get("api_key") or "",
        "host": payload.get("host") or "",
        "port": payload.get("port") or "",
        "hardware": payload.get("hardware") or "",
        "notes": payload.get("notes") or "",
    }
    entry["base_url"], corrected = normalize_base_url(entry["base_url"], entry["engine"], entry["api_key"])
    endpoints.append(entry)
    save_endpoints(endpoints)
    return {**entry, "base_url_corrected": corrected}


@app.put("/api/endpoints/{endpoint_id}")
def api_endpoints_update(endpoint_id: str, payload: dict):
    endpoints = load_endpoints()
    for entry in endpoints:
        if entry["id"] == endpoint_id:
            for key in ("name", "engine", "model", "base_url", "api_key", "host", "port", "hardware", "notes"):
                if key in payload:
                    entry[key] = payload[key]
            if not (entry.get("name") or "").strip():
                raise HTTPException(400, "Endpoint name is required")
            entry["base_url"], corrected = normalize_base_url(
                entry.get("base_url") or "", entry.get("engine") or "", entry.get("api_key") or ""
            )
            save_endpoints(endpoints)
            return {**entry, "base_url_corrected": corrected}
    raise HTTPException(404, "Endpoint not found")


@app.post("/api/models")
def api_models(payload: dict):
    """Discover available models on an inference server (OpenAI /v1/models or Ollama /api/tags)."""
    engine = payload.get("engine") or ""
    engine_info = get_engine_catalog().get(engine) or {}
    connection_kind = engine_info.get("connection")
    base_url = (payload.get("base_url") or "").strip()
    host = (payload.get("host") or "").strip()
    port = payload.get("port") or ""
    api_key = (payload.get("api_key") or "").strip()

    if engine in ("ollama-api", "ollama-cli"):
        url, flavor = "http://127.0.0.1:11434/api/tags", "ollama"
    elif connection_kind == "hostport" and host:
        url, flavor = f"http://{host}:{port or 8080}/v1/models", "openai"
    elif base_url:
        trimmed = base_url.rstrip("/")
        url = trimmed + "/models" if trimmed.endswith("/v1") else trimmed + "/v1/models"
        flavor = "openai"
    elif host:
        url, flavor = f"http://{host}:{port or 8080}/v1/models", "openai"
    else:
        return {"models": [], "detail": "No connection configured"}

    headers = {"User-Agent": "context-bench-webui"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        import httpx

        response = httpx.get(url, timeout=5.0, headers=headers)
        response.raise_for_status()
        data = response.json()
        if flavor == "ollama":
            models = [m.get("name") or m.get("model") for m in data.get("models", [])]
        else:
            models = [m.get("id") for m in data.get("data", [])]
        models = sorted({m for m in models if m})
        return {"models": models}
    except Exception as exc:
        return {"models": [], "detail": f"{exc.__class__.__name__}: {exc}"}


@app.get("/api/cached-models")
def api_cached_models(engine: str = ""):
    """Local HF-cache models for an engine that has no server (e.g. MLX)."""
    if engine == "mlx":
        return {"models": cached_mlx_models()}
    return {"models": []}


@app.post("/api/endpoints/{endpoint_id}/ping")
def api_endpoints_ping(endpoint_id: str):
    endpoint = next((e for e in load_endpoints() if e["id"] == endpoint_id), None)
    if endpoint is None:
        raise HTTPException(404, "Endpoint not found")
    url = (endpoint.get("base_url") or "").strip().rstrip("/")
    if url.endswith("/v1"):
        # OpenAI-style servers don't serve the bare /v1 root; probe the real route
        url += "/models"
    if not url and endpoint.get("host"):
        url = f"http://{endpoint['host']}:{endpoint.get('port') or 8080}/health"
    if not url:
        return {"ok": None, "detail": "No URL configured — local engine"}
    started = time.time()
    try:
        import httpx

        response = httpx.get(url, timeout=3.0, headers={"User-Agent": "context-bench-webui"})
        latency_ms = round((time.time() - started) * 1000)
        # any HTTP response means the server is up; auth errors etc. still count
        return {"ok": True, "status": response.status_code, "latency_ms": latency_ms}
    except Exception as exc:
        return {"ok": False, "detail": exc.__class__.__name__, "latency_ms": round((time.time() - started) * 1000)}


@app.delete("/api/endpoints/{endpoint_id}")
def api_endpoints_delete(endpoint_id: str):
    endpoints = load_endpoints()
    remaining = [e for e in endpoints if e["id"] != endpoint_id]
    if len(remaining) == len(endpoints):
        raise HTTPException(404, "Endpoint not found")
    save_endpoints(remaining)
    return {"ok": True}


@app.get("/api/runs")
def api_runs_list():
    return run_manager.list_runs()


@app.post("/api/command-preview")
def api_command_preview(payload: dict):
    """Build the exact launch command without exposing a configured API key."""
    engine_id = payload.get("engine")
    if engine_id not in get_engine_catalog():
        raise HTTPException(400, f"Unknown engine '{engine_id}'")
    argv, _ = build_command(engine_id, payload)
    return {"command": format_command(argv, secret_placeholder="$OPENAI_API_KEY")}


@app.post("/api/runs")
def api_runs_start(payload: dict):
    engine_id = payload.get("engine")
    catalog = get_engine_catalog()
    if engine_id not in catalog:
        raise HTTPException(400, f"Unknown engine '{engine_id}'")

    endpoint_name = ""
    endpoint_hardware = ""
    endpoint_id = payload.get("endpoint_id")
    if endpoint_id:
        endpoint = next((e for e in load_endpoints() if e["id"] == endpoint_id), None)
        if endpoint:
            endpoint_name = endpoint["name"]
            endpoint_hardware = endpoint.get("hardware") or ""

    # safety net for endpoints saved while their server was offline: probe a
    # path-less base URL once more and self-heal the stored endpoint
    connection = payload.get("connection") or {}
    base_url = (connection.get("base_url") or "").strip()
    if base_url:
        corrected_url, corrected = normalize_base_url(base_url, engine_id, (connection.get("api_key") or "").strip())
        if corrected:
            connection["base_url"] = corrected_url
            payload["connection"] = connection
            if endpoint_id:
                endpoints = load_endpoints()
                for entry in endpoints:
                    if entry["id"] == endpoint_id and (entry.get("base_url") or "").strip().rstrip(
                        "/"
                    ) == base_url.rstrip("/"):
                        entry["base_url"] = corrected_url
                        save_endpoints(endpoints)
                        break

    argv, contexts = build_command(engine_id, payload)
    label = (payload.get("label") or "").strip() or endpoint_name
    run = run_manager.start(
        "benchmark",
        engine_id,
        catalog[engine_id]["tag"],
        (payload.get("model") or "").strip() or "(auto)",
        label,
        endpoint_name,
        argv,
        contexts,
        endpoint_hardware=endpoint_hardware,
        settings=payload,
    )
    return run.snapshot()


@app.get("/api/results/{name}/settings")
def api_results_settings(name: str):
    """Return the stored launch settings of a saved run (for the »rerun« button)."""
    folder = resolve_result_folder(name)
    settings = read_run_meta(folder).get("settings")
    if not settings:
        raise HTTPException(404, "No saved settings for this run")
    return settings
    return run.snapshot()


@app.post("/api/source-files")
async def api_source_upload(request: Request, name: str):
    """Accept a raw .txt upload as a new source text for generate-context-files."""
    safe = Path(name).name
    if not safe.lower().endswith(".txt"):
        raise HTTPException(400, "Only .txt files are supported")
    if CONTEXT_FILE_RE.match(Path(safe).stem):
        raise HTTPException(400, "That name collides with generated context files ({size}k.txt)")
    body = await request.body()
    if not body:
        raise HTTPException(400, "The uploaded file is empty")
    if len(body) > 100 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 100 MB)")
    (ROOT / safe).write_bytes(body)
    return {"ok": True, "name": safe}


@app.post("/api/context-files")
def api_context_files_generate(payload: dict):
    source = (payload.get("source") or "").strip()
    sizes = (payload.get("sizes") or "").strip()
    if not source:
        raise HTTPException(400, "Source file is required")
    source_path = ROOT / Path(source).name
    if not source_path.exists():
        raise HTTPException(404, f"Source file '{source}' not found")
    argv = [sys.executable, str(ROOT / "generate_context_files.py"), str(source_path)]
    if sizes:
        argv += ["--sizes", sizes]
    run = run_manager.start(
        "ctxgen",
        "context-files",
        "",
        source_path.name,
        "Context files",
        "",
        argv,
        [s.strip() for s in sizes.split(",") if s.strip()],
    )
    return run.snapshot()


@app.get("/api/runs/{run_id}")
def api_runs_get(run_id: str, offset: int = 0):
    run = run_manager.get(run_id)
    lines, next_offset = run.log_slice(max(0, offset))
    snapshot = run.snapshot()
    snapshot["log"] = lines
    snapshot["next_offset"] = next_offset
    return snapshot


@app.post("/api/runs/{run_id}/stop")
def api_runs_stop(run_id: str):
    return run_manager.stop(run_id).snapshot()


@app.delete("/api/runs/{run_id}")
def api_runs_delete(run_id: str):
    run_manager.delete(run_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Worker ingest (remote machines running `benchmark-worker` push runs here)
# ---------------------------------------------------------------------------

# Optional shared secret; set via --worker-token or BENCHMARK_WORKER_TOKEN.
WORKER_TOKEN = os.environ.get("BENCHMARK_WORKER_TOKEN", "")

WORKER_FOLDER_RE = re.compile(r"^benchmark_[A-Za-z0-9._+-]+$")
WORKER_FILE_RE = re.compile(r"^[A-Za-z0-9._+-]+$")
MAX_UPLOAD_BYTES = 512 * 1024 * 1024


def require_worker_token(request: Request):
    if WORKER_TOKEN:
        provided = request.headers.get("x-worker-token", "")
        if not hmac.compare_digest(provided, WORKER_TOKEN):
            raise HTTPException(401, "Invalid worker token")


@app.post("/api/worker/register", dependencies=[Depends(require_worker_token)])
def api_worker_register(payload: dict):
    """A worker announces a benchmark it is starting locally."""
    engine_id = payload.get("engine") or ""
    if engine_id not in get_engine_catalog():
        raise HTTPException(400, f"Unknown engine '{engine_id}'")
    argv = payload.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
        raise HTTPException(400, "argv must be a non-empty list of strings")
    contexts = payload.get("contexts")
    if not isinstance(contexts, list):
        contexts = []
    worker = (payload.get("worker") or "worker").strip() or "worker"
    run = run_manager.start_remote(
        engine_id,
        get_engine_catalog()[engine_id]["tag"],
        (payload.get("model") or "").strip() or "(auto)",
        (payload.get("label") or "").strip() or worker,
        worker,
        (payload.get("hardware") or "").strip(),
        argv,
        [str(c) for c in contexts],
        settings=payload.get("settings") or None,
    )
    return {"run_id": run.id, "status": run.status}


@app.post("/api/worker/runs/{run_id}/logs", dependencies=[Depends(require_worker_token)])
def api_worker_logs(run_id: str, payload: dict):
    """Append a batch of benchmark stdout lines from a worker."""
    run = run_manager.get(run_id)
    lines = payload.get("lines")
    if not isinstance(lines, list):
        raise HTTPException(400, "lines must be a list of strings")
    for line in lines[-5000:]:
        run.ingest_line(str(line))
    return {"ok": True, "log_length": len(run.log_lines), "stop": run.stop_requested}


@app.get("/api/worker/runs/{run_id}/poll", dependencies=[Depends(require_worker_token)])
def api_worker_poll(run_id: str):
    """Cheap status poll so the worker can honor UI stop requests."""
    run = run_manager.get(run_id)
    return {"stop": bool(run.stop_requested)}


@app.post("/api/worker/runs/{run_id}/file", dependencies=[Depends(require_worker_token)])
async def api_worker_file(run_id: str, request: Request, folder: str, name: str):
    """Store one result file from a worker into its master-side run folder."""
    run = run_manager.get(run_id)
    if not WORKER_FOLDER_RE.match(folder):
        raise HTTPException(400, "Invalid folder name")
    parts = name.split("/")
    if len(parts) > 3 or not parts[-1] or not all(WORKER_FILE_RE.match(p) for p in parts):
        raise HTTPException(400, "Invalid file name")
    body = await request.body()
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large")
    master_name = run_manager.claim_remote_folder(run, folder)
    dest = OUTPUT_DIR / master_name
    for part in parts[:-1]:
        dest = dest / part
    dest.mkdir(parents=True, exist_ok=True)
    (dest / parts[-1]).write_bytes(body)
    return {"ok": True, "folder": master_name}


@app.post("/api/worker/runs/{run_id}/finish", dependencies=[Depends(require_worker_token)])
def api_worker_finish(run_id: str, payload: dict):
    """A worker reports its benchmark process exited."""
    run = run_manager.get(run_id)
    rc = payload.get("returncode")
    try:
        rc = int(rc)
    except (TypeError, ValueError):
        rc = 1
    return run_manager.finish_remote(run, rc, stopped=bool(payload.get("stopped"))).snapshot()


@app.get("/api/results")
def api_results_list():
    if not OUTPUT_DIR.is_dir():
        return []
    summaries = []
    for folder in sorted(OUTPUT_DIR.iterdir()):
        if folder.is_dir() and folder.name.startswith("benchmark_"):
            try:
                summary = summarize_result_folder(folder)
            except Exception as exc:  # a single broken folder must not kill the list
                summary = {"folder": folder.name, "error": str(exc)}
            if summary:
                summaries.append(summary)
    summaries.sort(key=lambda s: s.get("timestamp") or "", reverse=True)
    return summaries


@app.get("/api/results/{name}")
def api_results_detail(name: str):
    folder = resolve_result_folder(name)
    parsed, _ = parse_benchmark_folder(folder)
    if not parsed:
        raise HTTPException(404, f"'{name}' has no benchmark_results.csv")
    summary = summarize_result_folder(folder, parsed=parsed)
    files = sorted(p.name for p in folder.iterdir() if p.is_file())
    return {
        "summary": summary,
        "results": sorted(parsed["results"], key=lambda r: context_sort_key(r.get("context_size", "0"))),
        "hardware_info": parsed["hardware_info"],
        "batch_data": parsed["batch_data"],
        "perplexity_data": parsed["perplexity_data"],
        "cached_results": parsed["cached_results"],
        "files": files,
    }


@app.patch("/api/results/{name}")
def api_results_label(name: str, payload: dict):
    folder = resolve_result_folder(name)
    meta = read_run_meta(folder)
    meta["label"] = (payload.get("label") or "").strip()
    (folder / RUN_META_FILE).write_text(json.dumps(meta, indent=2))
    return {"ok": True, "label": meta["label"]}


@app.delete("/api/results/{name}")
def api_results_delete(name: str):
    folder = resolve_result_folder(name)
    shutil.rmtree(folder)
    return {"ok": True}


app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR), check_dir=False), name="output")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def revalidate_app_assets(request: Request, call_next):
    """App JS/CSS changes with every update; no-cache makes browsers revalidate
    instead of heuristically reusing stale files (ETag 304s keep it cheap)."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


def main():
    global WORKER_TOKEN
    parser = argparse.ArgumentParser(description="Web UI for the LLM context benchmark toolkit")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8321, help="Bind port (default: 8321)")
    parser.add_argument("--no-open", action="store_true", help="Don't open the browser on start")
    parser.add_argument(
        "--worker-token",
        default=WORKER_TOKEN,
        help="Shared secret required by /api/worker/* upload endpoints "
        "(env: BENCHMARK_WORKER_TOKEN; empty = no auth)",
    )
    parser.add_argument(
        "--master",
        default="",
        help="Run as a worker: mirror every benchmark launched here to a master WebUI at this URL",
    )
    parser.add_argument(
        "--master-token",
        default=os.environ.get("BENCHMARK_MASTER_TOKEN", ""),
        help="Token to authenticate to the master (env: BENCHMARK_MASTER_TOKEN)",
    )
    parser.add_argument(
        "--worker-name",
        default=platform.node(),
        help="Worker name shown at the master (default: hostname)",
    )
    args = parser.parse_args()

    if args.master:
        run_manager.mirror = {"master": args.master, "token": args.master_token, "name": args.worker_name}

    OUTPUT_DIR.mkdir(exist_ok=True)
    url = f"http://{args.host}:{args.port}"
    print(f"LLM Context Bench UI on {url}")
    if WORKER_TOKEN:
        print("Worker ingest enabled (token auth)")
    else:
        print("Worker ingest enabled (no token — LAN use only)")
    if args.master:
        print(f"Worker mode: mirroring launched runs to {args.master} as '{args.worker_name}'")
    if not args.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
