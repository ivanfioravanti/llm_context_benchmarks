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
import json
import os
import shutil
import sys
import threading
import time
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
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
    is_apple_silicon,
)
from webui_engines import build_command, engine_available, get_engine_catalog
from webui_runs import RunManager

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


def summarize_result_folder(folder: Path):
    parsed, _ = parse_benchmark_folder(folder)
    if not parsed:
        return None
    meta = read_run_meta(folder)
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
        "label": meta.get("label") or "",
        "endpoint": meta.get("endpoint") or "",
        "timestamp": folder_timestamp(folder.name),
        "machine": hw.get("chip") or hw.get("machine_label") or hw.get("processor") or "",
        "hardware": benchmark_common.format_hardware_string(hw) if hw else "",
        "contexts": [r.get("context_size") for r in rows],
        "peak_generation_tps": max(gen_values) if gen_values else None,
        "gen_series": gen_series,
        "columns": sorted({k for r in rows for k in r.keys()}),
        "has_batch": bool(parsed["batch_data"]),
        "has_perplexity": bool(parsed["perplexity_data"]),
        "has_cached": bool(parsed["cached_results"]),
        "charts": charts,
    }


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
        "notes": payload.get("notes") or "",
    }
    endpoints.append(entry)
    save_endpoints(endpoints)
    return entry


@app.put("/api/endpoints/{endpoint_id}")
def api_endpoints_update(endpoint_id: str, payload: dict):
    endpoints = load_endpoints()
    for entry in endpoints:
        if entry["id"] == endpoint_id:
            for key in ("name", "engine", "model", "base_url", "api_key", "host", "port", "notes"):
                if key in payload:
                    entry[key] = payload[key]
            if not (entry.get("name") or "").strip():
                raise HTTPException(400, "Endpoint name is required")
            save_endpoints(endpoints)
            return entry
    raise HTTPException(404, "Endpoint not found")


@app.post("/api/models")
def api_models(payload: dict):
    """Discover available models on an inference server (OpenAI /v1/models or Ollama /api/tags)."""
    engine = payload.get("engine") or ""
    base_url = (payload.get("base_url") or "").strip()
    host = (payload.get("host") or "").strip()
    port = payload.get("port") or ""
    api_key = (payload.get("api_key") or "").strip()

    if engine in ("ollama-api", "ollama-cli"):
        url, flavor = "http://127.0.0.1:11434/api/tags", "ollama"
    elif host:
        url, flavor = f"http://{host}:{port or 8080}/v1/models", "openai"
    elif base_url:
        trimmed = base_url.rstrip("/")
        url = trimmed + "/models" if trimmed.endswith("/v1") else trimmed + "/v1/models"
        flavor = "openai"
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


@app.post("/api/endpoints/{endpoint_id}/ping")
def api_endpoints_ping(endpoint_id: str):
    endpoint = next((e for e in load_endpoints() if e["id"] == endpoint_id), None)
    if endpoint is None:
        raise HTTPException(404, "Endpoint not found")
    url = (endpoint.get("base_url") or "").strip()
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


@app.post("/api/runs")
def api_runs_start(payload: dict):
    engine_id = payload.get("engine")
    catalog = get_engine_catalog()
    if engine_id not in catalog:
        raise HTTPException(400, f"Unknown engine '{engine_id}'")

    endpoint_name = ""
    endpoint_id = payload.get("endpoint_id")
    if endpoint_id:
        endpoint = next((e for e in load_endpoints() if e["id"] == endpoint_id), None)
        if endpoint:
            endpoint_name = endpoint["name"]

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
    )
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
    summary = summarize_result_folder(folder)
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


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


def main():
    parser = argparse.ArgumentParser(description="Web UI for the LLM context benchmark toolkit")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8321, help="Bind port (default: 8321)")
    parser.add_argument("--no-open", action="store_true", help="Don't open the browser on start")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    url = f"http://{args.host}:{args.port}"
    print(f"LLM Context Bench UI on {url}")
    if not args.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
