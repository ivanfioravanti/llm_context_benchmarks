"""Shared constants and small helpers for the web UI modules."""

import platform
import re
from pathlib import Path

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output"
STATIC_DIR = ROOT / "webui_static"
ENDPOINTS_FILE = ROOT / "webui_endpoints.json"
RUN_META_FILE = "webui_run.json"

CONTEXT_FILE_RE = re.compile(r"^(\d+(?:\.\d+)?)k$")
FOLDER_TS_RE = re.compile(r"_(\d{8}_\d{6})$")


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"
