#!/usr/bin/env python3
"""Build a deterministic real-code corpus for code-context benchmark files.

Concatenates whole ``.py`` files from a source tree (e.g. the CPython stdlib)
into a single ``code_corpus.txt`` used by ``generate_context_files.py
--context-type code``. Content is never repeated: a repeated prefix would
corrupt prefill measurements and inflate speculative-decoding acceptance.
"""

import argparse
import shlex
import sys
from datetime import datetime
from pathlib import Path

import tiktoken

SKIP_DIR_COMPONENTS = {"test", "tests", "__pycache__"}

FILE_MARKER = "# ---- {rel} ----\n"


def collect_source_files(source_dir: Path) -> list[Path]:
    """Return every .py file under source_dir in a deterministic order.

    Plain string sort of POSIX paths; directory components named
    test/tests/__pycache__ are skipped entirely.
    """
    files = []
    for f in source_dir.rglob("*.py"):
        rel_parts = f.relative_to(source_dir).parts
        if any(part in SKIP_DIR_COMPONENTS for part in rel_parts[:-1]):
            continue
        files.append(f)
    return sorted(files, key=lambda f: f.relative_to(source_dir).as_posix())


def read_chunk(path: Path, source_dir: Path) -> str:
    """Return the marker line plus the file text, newline-terminated."""
    rel = path.relative_to(source_dir).as_posix()
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    if text and not text.endswith("\n"):
        text += "\n"
    return FILE_MARKER.format(rel=rel) + text


def main():
    parser = argparse.ArgumentParser(
        description="Build a deterministic code corpus from a tree of .py files (no repetition)."
    )
    parser.add_argument("--source-dir", required=True, help="Directory of .py files to concatenate")
    parser.add_argument("--output", default="code_corpus.txt", help="Output corpus file (default: code_corpus.txt)")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=260000,
        help="Stop appending whole files once the corpus reaches this many tokens (default: 260000)",
    )
    parser.add_argument(
        "--encoding",
        default="cl100k_base",
        help="Tiktoken encoding for token counting (default: cl100k_base)",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        print(f"Error: source dir '{args.source_dir}' is not a directory")
        sys.exit(1)

    try:
        encoding = tiktoken.get_encoding(args.encoding)
    except Exception as e:
        print(f"Error: invalid encoding '{args.encoding}': {e}")
        sys.exit(1)

    files = collect_source_files(source_dir)
    if not files:
        print(f"Error: no .py files found under '{args.source_dir}'")
        sys.exit(1)
    print(f"Found {len(files)} candidate .py files under {source_dir}")

    # Append whole files until the corpus reaches --max-tokens. Per-file
    # incremental counts are exact as long as no BPE token merges across a
    # chunk boundary; the final full encode below verifies the exact total and
    # keeps appending in the (rare) case boundary merges shaved it below target.
    body_parts = []
    incremental_tokens = 0
    included = 0
    idx = 0
    while idx < len(files):
        if incremental_tokens >= args.max_tokens:
            exact_tokens = len(encoding.encode("".join(body_parts)))
            if exact_tokens >= args.max_tokens:
                incremental_tokens = exact_tokens
                break
            incremental_tokens = exact_tokens
        body_parts.append(read_chunk(files[idx], source_dir))
        incremental_tokens += len(encoding.encode(body_parts[-1]))
        included += 1
        idx += 1
        if included % 250 == 0:
            print(f"  {included} files, {incremental_tokens} tokens...")

    body = "".join(body_parts)
    exact_tokens = len(encoding.encode(body))
    if exact_tokens < args.max_tokens:
        print(
            f"Error: source exhausted at {exact_tokens} tokens (< {args.max_tokens} requested). "
            "Content is never repeated; pass a larger --source-dir."
        )
        sys.exit(1)

    header = (
        "# Code corpus for context benchmark files\n"
        f"# Source dir: {source_dir.resolve()}\n"
        f"# Files included: {included}\n"
        f"# Content tokens ({args.encoding}, excluding this header): {exact_tokens}\n"
        f"# Built: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# Command: {shlex.join(sys.argv)}\n"
    )

    output_path = Path(args.output)
    output_path.write_text(header + body, encoding="utf-8")

    total_tokens = len(encoding.encode(header + body))
    print(f"Wrote {output_path} ({included} files)")
    print(f"Final token count: {total_tokens} (content: {exact_tokens})")


if __name__ == "__main__":
    main()
