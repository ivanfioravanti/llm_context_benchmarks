"""Engine catalog and benchmark-command construction for the web UI."""

import json
import sys

from fastapi import HTTPException

from webui_common import ROOT, is_apple_silicon


def _batch_opts(sizes: str = "1,2,4,8") -> list:
    return [
        {
            "key": "batch",
            "type": "invflag",
            "flag": "--no-batch",
            "label": "Batch sweep",
            "default": True,
            "help": "Also run the continuous-batch benchmark",
        },
        {"key": "batch_sizes", "type": "str", "flag": "--batch-sizes", "label": "Batch sizes", "default": sizes},
        {
            "key": "batch_prompt_tokens",
            "type": "int",
            "flag": "--batch-prompt-tokens",
            "label": "Batch prompt tokens",
            "default": 2048,
        },
        {
            "key": "batch_gen_tokens",
            "type": "int",
            "flag": "--batch-gen-tokens",
            "label": "Batch gen tokens",
            "default": 128,
        },
        {"key": "batch_trials", "type": "int", "flag": "--batch-trials", "label": "Batch trials", "default": 3},
    ]


def _sampling_opts(temperature: float = 0.7, stream: bool = True) -> list:
    opts = [
        {
            "key": "temperature",
            "type": "float",
            "flag": "--temperature",
            "label": "Temperature",
            "default": temperature,
        },
        {"key": "top_p", "type": "float", "flag": "--top-p", "label": "Top-p", "default": 0.95},
    ]
    if stream:
        opts.append({"key": "stream", "type": "optbool", "flag": "--stream", "label": "Streaming", "default": True})
    return opts


def _request_model_opt() -> dict:
    return {
        "key": "request_model",
        "type": "str",
        "flag": "--request-model",
        "label": "Request model override",
        "default": "",
        "help": "Model name sent in the API request (defaults to the model field)",
    }


def get_engine_catalog() -> dict:
    """Full engine catalog with everything the UI needs to build run forms."""
    catalog = {
        "ollama-api": {
            "label": "Ollama API",
            "script": "ollama_api_benchmark.py",
            "tag": "ollama_api",
            "description": "Ollama Python API (local daemon)",
            "example": "gpt-oss:20b",
            "model": "required",
            "connection": None,
            "cold_prefill": True,
            "local_mlx": False,
            "options": _batch_opts(),
        },
        "ollama-cli": {
            "label": "Ollama CLI",
            "script": "ollama_cli_benchmark.py",
            "tag": "ollama_cli",
            "description": "Ollama CLI with verbose output",
            "example": "llama3.2",
            "model": "required",
            "connection": None,
            "cold_prefill": True,
            "local_mlx": False,
            "options": _batch_opts(),
        },
        "llamacpp": {
            "label": "llama.cpp",
            "script": "llamacpp_benchmark.py",
            "tag": "llamacpp",
            "description": "llama.cpp server via HTTP API",
            "example": "gpt-oss:20b",
            "model": "required",
            "connection": "hostport",
            "cold_prefill": True,
            "local_mlx": False,
            "options": [
                {
                    "key": "server_hardware",
                    "type": "str",
                    "flag": "--server-hardware",
                    "label": "Server hardware label",
                    "default": "",
                },
                {
                    "key": "server_memory_gb",
                    "type": "float",
                    "flag": "--server-memory-gb",
                    "label": "Server memory (GB)",
                    "default": None,
                },
                {
                    "key": "server_cores",
                    "type": "int",
                    "flag": "--server-cores",
                    "label": "Server cores",
                    "default": None,
                },
            ],
        },
        "lmstudio": {
            "label": "LM Studio",
            "script": "lmstudio_benchmark.py",
            "tag": "lmstudio",
            "description": "LM Studio local server",
            "example": "local-model",
            "model": "auto",
            "connection": "base_url",
            "default_base_url": "http://localhost:1234",
            "cold_prefill": True,
            "local_mlx": False,
            "options": _batch_opts(),
        },
        "exo": {
            "label": "Exo",
            "script": "exo_benchmark.py",
            "tag": "exo",
            "description": "Exo OpenAI-compatible endpoint",
            "example": "local-model",
            "model": "required",
            "connection": "base_url",
            "default_base_url": "http://0.0.0.0:52415",
            "cold_prefill": False,
            "local_mlx": False,
            "options": [_request_model_opt()] + _sampling_opts(),
        },
        "openai": {
            "label": "OpenAI-compatible",
            "script": "openai_benchmark.py",
            "tag": "openai_compat",
            "description": "Generic OpenAI-compatible endpoint (vLLM, llama.cpp, ...)",
            "example": "local-model",
            "model": "auto",
            "connection": "base_url",
            "default_base_url": "http://localhost:8080/v1",
            "cold_prefill": True,
            "local_mlx": False,
            "options": [
                {
                    "key": "temperature",
                    "type": "float",
                    "flag": "--temperature",
                    "label": "Temperature",
                    "default": 1.0,
                    "help": "Some hosted models, including Kimi K3, require exactly 1.0",
                },
                {
                    "key": "latency_adjustment",
                    "type": "optbool",
                    "flag": "--latency-adjustment",
                    "label": "Adjust for endpoint latency",
                    "default": True,
                    "help": "Measure warm authenticated round trips and remove their average from client prompt timing",
                },
                {
                    "key": "latency_samples",
                    "type": "int",
                    "flag": "--latency-samples",
                    "label": "Latency samples",
                    "default": 5,
                    "help": "Number of warm /models requests used to estimate endpoint overhead",
                },
            ]
            + _batch_opts(),
        },
        "unsloth": {
            "label": "Unsloth",
            "script": "unsloth_benchmark.py",
            "tag": "unsloth",
            "description": "Unsloth Studio local server (OpenAI-compatible + server timings)",
            "example": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "model": "auto",
            "connection": "base_url",
            "default_base_url": "http://127.0.0.1:8888/v1",
            "cold_prefill": True,
            "local_mlx": False,
            "options": [],
        },
        "vllm": {
            "label": "vLLM",
            "script": "vllm_benchmark.py",
            "tag": "vllm",
            "description": "vLLM server (streaming + /metrics + continuous batch)",
            "example": "Qwen/Qwen3-8B",
            "model": "required",
            "connection": "base_url",
            "default_base_url": "http://127.0.0.1:8000/v1",
            "cold_prefill": True,
            "local_mlx": False,
            "options": _sampling_opts()
            + [
                {"key": "metrics", "type": "optbool", "flag": "--metrics", "label": "Scrape /metrics", "default": True},
                {
                    "key": "metrics_base_url",
                    "type": "str",
                    "flag": "--metrics-base-url",
                    "label": "Metrics base URL",
                    "default": "",
                },
            ]
            + _batch_opts(),
        },
        "deepseek": {
            "label": "DeepSeek",
            "script": "deepseek_benchmark.py",
            "tag": "deepseek",
            "description": "DeepSeek cloud API",
            "example": "deepseek-chat",
            "model": "required",
            "connection": "base_url",
            "default_base_url": "https://api.deepseek.com/v1",
            "cold_prefill": True,
            "local_mlx": False,
            "options": [_request_model_opt()] + _sampling_opts(),
        },
        "grok": {
            "label": "Grok",
            "script": "grok_benchmark.py",
            "tag": "grok",
            "description": "xAI Grok API",
            "example": "grok-3",
            "model": "required",
            "connection": "base_url",
            "default_base_url": "https://api.x.ai/v1",
            "cold_prefill": True,
            "local_mlx": False,
            "options": [_request_model_opt()] + _sampling_opts(),
        },
        "afms": {
            "label": "Apple Foundation",
            "script": "apple_foundation_benchmark.py",
            "tag": "afms",
            "description": "Apple Foundation Models Serve endpoint",
            "example": "system",
            "model": "auto",
            "connection": "base_url",
            "default_base_url": "http://127.0.0.1:1976/v1",
            "default_contexts": "0.5,1,2",
            "cold_prefill": True,
            "local_mlx": False,
            "options": [
                _request_model_opt(),
                {
                    "key": "temperature",
                    "type": "float",
                    "flag": "--temperature",
                    "label": "Temperature",
                    "default": 0.0,
                },
            ],
        },
        "mlx": {
            "label": "MLX",
            "script": "mlx_benchmark.py",
            "tag": "mlx",
            "description": "MLX framework (Apple Silicon only)",
            "example": "mlx-community/Qwen3-0.6B-4bit",
            "model": "required",
            "connection": None,
            "cold_prefill": False,
            "local_mlx": True,
            "options": [
                {"key": "kv_bit", "type": "int", "flag": "--kv-bit", "label": "KV cache bits", "default": None},
                {
                    "key": "max_kv_size",
                    "type": "int",
                    "flag": "--max-kv-size",
                    "label": "Max KV size (tokens)",
                    "default": None,
                },
                {
                    "key": "trust_remote_code",
                    "type": "flag",
                    "flag": "--trust-remote-code",
                    "label": "Trust remote code",
                    "default": False,
                },
                {
                    "key": "ignore_chat_template",
                    "type": "flag",
                    "flag": "--ignore-chat-template",
                    "label": "Ignore chat template",
                    "default": False,
                },
                {
                    "key": "cached",
                    "type": "flag",
                    "flag": "--cached",
                    "label": "Cached-prefill sweep",
                    "default": False,
                },
                {
                    "key": "perplexity",
                    "type": "invflag",
                    "flag": "--no-perplexity",
                    "label": "Perplexity",
                    "default": True,
                },
                {
                    "key": "kl_capture",
                    "type": "invflag",
                    "flag": "--no-kl-capture",
                    "label": "KL capture",
                    "default": True,
                },
            ]
            + _batch_opts("1,2,4,8,16"),
        },
        "mlx-vlm": {
            "label": "MLX-VLM",
            "script": "mlx_vlm_benchmark.py",
            "tag": "mlx_vlm",
            "description": "MLX-VLM vision-language models (Apple Silicon)",
            "example": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
            "model": "required",
            "connection": None,
            "cold_prefill": False,
            "local_mlx": True,
            "options": [
                {"key": "kv_bits", "type": "float", "flag": "--kv-bits", "label": "KV cache bits", "default": None},
                {
                    "key": "kv_quant_scheme",
                    "type": "choice",
                    "flag": "--kv-quant-scheme",
                    "label": "KV quant scheme",
                    "default": "",
                    "choices": ["", "uniform", "turboquant"],
                },
                {
                    "key": "max_kv_size",
                    "type": "int",
                    "flag": "--max-kv-size",
                    "label": "Max KV size (tokens)",
                    "default": None,
                },
                {
                    "key": "trust_remote_code",
                    "type": "flag",
                    "flag": "--trust-remote-code",
                    "label": "Trust remote code",
                    "default": False,
                },
            ]
            + _batch_opts("1,2,4,8,16"),
        },
        "mlx-vlm-server": {
            "label": "MLX-VLM Server",
            "script": "mlx_vlm_server_benchmark.py",
            "tag": "mlx_vlm_server",
            "description": "mlx-vlm HTTP server (OpenAI-compatible)",
            "example": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
            "model": "auto",
            "connection": "base_url",
            "cold_prefill": True,
            "local_mlx": False,
            "options": [_request_model_opt()]
            + _sampling_opts(stream=False)
            + [
                {
                    "key": "unload_between_rows",
                    "type": "optbool",
                    "flag": "--unload-between-rows",
                    "label": "Unload between rows",
                    "default": True,
                },
            ],
        },
        "omlx": {
            "label": "oMLX",
            "script": "omlx_benchmark.py",
            "tag": "omlx",
            "description": "oMLX inference server (OpenAI-compatible)",
            "example": "gemma-4-26b-a4b-it-4bit",
            "model": "required",
            "connection": "base_url",
            "default_base_url": "http://127.0.0.1:8000/v1",
            "cold_prefill": True,
            "local_mlx": False,
            "options": [_request_model_opt()] + _sampling_opts() + _batch_opts(),
        },
        "mtplx": {
            "label": "MTPLX",
            "script": "mtplx_benchmark.py",
            "tag": "mtplx",
            "description": "MTPLX inference server (OpenAI-compatible)",
            "example": "mtplx-qwen36-27b-optimized-speed",
            "model": "auto",
            "connection": "base_url",
            "cold_prefill": True,
            "local_mlx": False,
            "options": [
                _request_model_opt(),
                {
                    "key": "generation_mode",
                    "type": "choice",
                    "flag": "--generation-mode",
                    "label": "Generation mode",
                    "default": "",
                    "choices": ["", "mtp", "ar"],
                },
                {
                    "key": "clear_cache",
                    "type": "optbool",
                    "flag": "--clear-cache",
                    "label": "Clear cache",
                    "default": True,
                },
            ]
            + _sampling_opts(temperature=0.6)
            + _batch_opts(),
        },
        "dflash-mlx": {
            "label": "DFlash MLX",
            "script": "dflash_benchmark.py",
            "tag": "dflash",
            "description": "dflash-mlx speculative decoding (OpenAI-compatible)",
            "example": "Qwen/Qwen3.6-27B",
            "model": "auto",
            "connection": "base_url",
            "cold_prefill": True,
            "local_mlx": False,
            "options": [_request_model_opt()] + _sampling_opts(temperature=0.6, stream=False),
        },
        "vmlx": {
            "label": "vMLX",
            "script": "vmlx_benchmark.py",
            "tag": "vmlx",
            "description": "vMLX / MLX Studio server (OpenAI-compatible)",
            "example": "mlx-community/Qwen3-8B-4bit",
            "model": "required",
            "connection": "base_url",
            "default_base_url": "http://127.0.0.1:8001/v1",
            "cold_prefill": True,
            "local_mlx": False,
            "options": [_request_model_opt()] + _sampling_opts() + _batch_opts(),
        },
        "mlx-serve": {
            "label": "MLX-Serve",
            "script": "mlxserve_benchmark.py",
            "tag": "mlx_serve",
            "description": "mlx-serve server (OpenAI-compatible + server timings + batch)",
            "example": "Qwen3.6-27B-MTPLX-Optimized-Quality",
            "model": "auto",
            "connection": "base_url",
            "default_base_url": "http://127.0.0.1:11234/v1",
            "cold_prefill": True,
            "local_mlx": False,
            "options": _batch_opts(),
        },
        "paroquant": {
            "label": "Paroquant",
            "script": "paroquant_benchmark.py",
            "tag": "paroquant",
            "description": "Paroquant quantized inference (MLX, Apple Silicon)",
            "example": "my-org/My-Model-PQ",
            "model": "required",
            "connection": None,
            "cold_prefill": False,
            "local_mlx": True,
            "options": [
                {
                    "key": "ignore_chat_template",
                    "type": "flag",
                    "flag": "--ignore-chat-template",
                    "label": "Ignore chat template",
                    "default": False,
                },
                {
                    "key": "perplexity",
                    "type": "invflag",
                    "flag": "--no-perplexity",
                    "label": "Perplexity",
                    "default": True,
                },
            ]
            + _batch_opts("1,2,4,8,16"),
        },
        "mlx-distributed": {
            "label": "MLX Distributed",
            "script": "mlx_distributed_benchmark.py",
            "tag": "mlx-distributed",
            "description": "MLX distributed generate via mlx.launch (e.g. JACCL)",
            "example": "/path/to/model",
            "model": "required",
            "connection": None,
            "cold_prefill": False,
            "local_mlx": True,
            "options": [
                {
                    "key": "hostfile",
                    "type": "str",
                    "flag": "--hostfile",
                    "label": "Hostfile (JSON)",
                    "default": "",
                    "required": True,
                },
                {"key": "backend", "type": "str", "flag": "--backend", "label": "Backend", "default": "jaccl"},
                {"key": "launcher", "type": "str", "flag": "--launcher", "label": "Launcher", "default": "mlx.launch"},
                {
                    "key": "sharded_script",
                    "type": "str",
                    "flag": "--sharded-script",
                    "label": "Sharded script",
                    "default": "mlx_lm/examples/sharded_generate.py",
                },
                {
                    "key": "pipeline",
                    "type": "flag",
                    "flag": "--pipeline",
                    "label": "Pipeline parallelism",
                    "default": False,
                },
            ],
        },
    }
    return dict(sorted(catalog.items()))


def engine_available(info: dict) -> bool:
    if info.get("local_mlx") and not is_apple_silicon():
        return False
    return True


def cached_mlx_models() -> list:
    """Local HF-cache models the MLX (mlx_lm) engine can load.

    Walks the cache via huggingface_hub (honors HF_HOME) and keeps only
    MLX-format text models: a tokenizer + config.json, no GGUF weights, and
    either an MLX ``quantization`` block in config.json or an ``mlx`` marker in
    the repo id. Vanilla HF transformers repos (Qwen3-0.6B, gpt2, gemma, ...)
    are excluded because mlx_lm needs MLX-converted weights.
    """
    # ponytail: heuristic, not a load attempt. Name markers drop unambiguous
    # non-text MLX repos (whisper/TTS/flux/multimodal) that mlx_lm can't load;
    # mlx-vlm has its own engine. Upgrade to probing mlx_lm.load if needed.
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return []

    non_text = ("whisper", "tts", "asr", "flux", "mflux", "multimodal")
    models = []
    for repo in scan_cache_dir().repos:
        if str(repo.repo_type) != "model":
            continue
        repo_lower = repo.repo_id.lower()
        if any(m in repo_lower for m in non_text):
            continue
        names = set()
        config_path = None
        for rev in repo.revisions:
            for f in rev.files:
                names.add(f.file_name)
                if f.file_name == "config.json" and config_path is None:
                    config_path = getattr(f, "file_path", None)
        if "config.json" not in names or any(n.lower().endswith(".gguf") for n in names):
            continue
        if not any(n in names for n in ("tokenizer.json", "tokenizer_config.json")):
            continue
        is_mlx = "mlx" in repo_lower
        if not is_mlx and config_path:
            try:
                with open(config_path) as fh:
                    is_mlx = isinstance(json.load(fh).get("quantization"), dict)
            except Exception:
                pass
        if is_mlx:
            models.append(repo.repo_id)
    return sorted(models)


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------


def build_command(engine_id: str, payload: dict) -> tuple:
    """Build the argv for a benchmark run. Returns (argv, contexts_list)."""
    catalog = get_engine_catalog()
    info = catalog.get(engine_id)
    if info is None:
        raise HTTPException(400, f"Unknown engine '{engine_id}'")
    if not engine_available(info):
        raise HTTPException(400, f"Engine '{engine_id}' requires Apple Silicon (MLX) and is disabled on this machine")

    script = ROOT / info["script"]
    if not script.exists():
        raise HTTPException(500, f"Benchmark script {info['script']} not found")

    argv = [sys.executable, str(script)]

    model = (payload.get("model") or "").strip()
    if model:
        argv.append(model)
    elif info["model"] == "required":
        raise HTTPException(400, f"Engine '{engine_id}' requires a model")

    contexts = (payload.get("contexts") or info.get("default_contexts") or "0.5,1,2,4,8,16,32").strip()
    argv += ["--contexts", contexts]
    argv += ["--max-tokens", str(int(payload.get("max_tokens") or 128))]
    argv += ["--runs", str(int(payload.get("runs") or 2))]
    argv += ["--timeout", str(int(payload.get("timeout") or 3600))]
    if payload.get("save_responses"):
        argv.append("--save-responses")

    if info["cold_prefill"] and payload.get("cold_prefill") is False:
        argv.append("--no-cold-prefill")

    connection = payload.get("connection") or {}
    if info["connection"] == "base_url":
        base_url = (connection.get("base_url") or "").strip()
        if base_url:
            argv += ["--base-url", base_url]
        api_key = (connection.get("api_key") or "").strip()
        if api_key:
            argv += ["--api-key", api_key]
        if engine_id == "lmstudio" and connection.get("api_version"):
            argv += ["--api-version", connection["api_version"]]
    elif info["connection"] == "hostport":
        if connection.get("host"):
            argv += ["--host", str(connection["host"])]
        if connection.get("port"):
            argv += ["--port", str(int(connection["port"]))]

    options = payload.get("options") or {}
    for opt in info["options"]:
        key = opt["key"]
        value = options.get(key, opt.get("default"))
        if opt.get("required") and (value is None or str(value).strip() == ""):
            raise HTTPException(400, f"Option '{opt['label']}' is required for engine '{engine_id}'")
        if value is None or (isinstance(value, str) and value.strip() == ""):
            continue
        kind = opt["type"]
        if kind == "flag":
            if value:
                argv.append(opt["flag"])
        elif kind == "invflag":
            if not value:
                argv.append(opt["flag"])
        elif kind == "optbool":
            if bool(value) != opt.get("default", True):
                flag = opt["flag"]
                argv.append(flag if value else flag.replace("--", "--no-", 1))
        elif kind in ("int",):
            argv += [opt["flag"], str(int(value))]
        elif kind in ("float",):
            argv += [opt["flag"], str(float(value))]
        else:
            argv += [opt["flag"], str(value)]

    extra = (payload.get("extra_args") or "").strip()
    if extra:
        argv += extra.split()

    context_list = [c.strip() for c in contexts.split(",") if c.strip()]
    return argv, context_list
