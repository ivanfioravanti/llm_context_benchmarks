"""Capture top-K logprobs from MLX and llama.cpp for KL-divergence comparison.

Capture is done once per benchmark run, on a fixed reference text (default:
the first ~4000 chars of ``2k.txt``). The output is a single ``logprobs.json``
file in the run's output directory. Comparison and KL computation happen later
in ``compare_benchmarks.py``.

Both runs being compared must use the same tokenizer family — KL is computed
on display-string tokens, so different tokenizers produce non-aligned
distributions and the comparison will be meaningless.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

# Cap how much of the reference file we read, to keep tokenization/forward-pass
# cheap. 4000 chars comfortably yields >512 tokens for any common tokenizer.
REF_MAX_CHARS = 4000
DEFAULT_NUM_POSITIONS = 512
DEFAULT_TOP_K = 64
DEFAULT_REF_FILE = "2k.txt"


def _read_ref_text(ref_file: Path, max_chars: int = REF_MAX_CHARS) -> str:
    with open(ref_file) as f:
        return f.read()[:max_chars]


def capture_logprobs_mlx(
    model,
    tokenizer,
    ref_file: Path,
    num_positions: int = DEFAULT_NUM_POSITIONS,
    top_k: int = DEFAULT_TOP_K,
) -> Dict:
    """Run one forward pass and record top-K logprobs at each position.

    The model's per-position output distribution depends only on the prefix, so
    a single ``model(input_ids)`` call gives us all positions at once.
    """
    import mlx.core as mx
    import numpy as np

    text = _read_ref_text(ref_file)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) < num_positions + 1:
        num_positions = max(1, len(token_ids) - 1)
    token_ids = token_ids[: num_positions + 1]

    arr = mx.array(token_ids)[None, :]
    logits = model(arr)[0].astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(logprobs)
    lp = np.array(logprobs)[:num_positions]

    idx = np.argpartition(-lp, top_k - 1, axis=-1)[:, :top_k]
    vals = np.take_along_axis(lp, idx, axis=-1)
    order = np.argsort(-vals, axis=-1)
    idx = np.take_along_axis(idx, order, axis=-1)
    vals = np.take_along_axis(vals, order, axis=-1)

    positions = []
    for i in range(num_positions):
        top_strs = [tokenizer.decode([int(t)]) for t in idx[i]]
        positions.append(
            {
                "top_tokens": top_strs,
                "top_logprobs": [float(v) for v in vals[i]],
            }
        )

    return {
        "ref_file": Path(ref_file).name,
        "num_positions": num_positions,
        "top_k": top_k,
        "vocab_size": int(lp.shape[-1]),
        "input_tokens": [tokenizer.decode([int(t)]) for t in token_ids[:num_positions]],
        "positions": positions,
    }


def capture_logprobs_llamacpp(
    server_url: str,
    ref_file: Path,
    num_positions: int = DEFAULT_NUM_POSITIONS,
    top_k: int = DEFAULT_TOP_K,
    timeout: int = 600,
) -> Dict:
    """Capture top-K logprobs via the OpenAI-compatible /v1/completions endpoint.

    Uses ``echo: true`` + ``logprobs: K`` + ``max_tokens: 0`` to get per-token
    logprobs across the prompt without generating anything. Requires a recent
    llama.cpp server build with the OpenAI-compat endpoint.
    """
    import requests

    text = _read_ref_text(ref_file)

    payload = {
        "prompt": text,
        "max_tokens": 0,
        "echo": True,
        "logprobs": top_k,
        "temperature": 0,
    }

    resp = requests.post(f"{server_url}/v1/completions", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    choice = data["choices"][0]
    lp = choice.get("logprobs") or {}
    top_field = lp.get("top_logprobs")
    if not top_field:
        raise RuntimeError(
            "llama.cpp server did not return top_logprobs. "
            "Need a build that supports OpenAI-compat echo+logprobs."
        )

    tokens_field = lp.get("tokens", [])
    cap = min(num_positions, len(top_field))
    top_field = top_field[:cap]
    tokens_field = tokens_field[:cap]

    positions = []
    for top_dict in top_field:
        items = sorted(top_dict.items(), key=lambda kv: -kv[1])
        positions.append(
            {
                "top_tokens": [k for k, _ in items],
                "top_logprobs": [float(v) for _, v in items],
            }
        )

    return {
        "ref_file": Path(ref_file).name,
        "num_positions": cap,
        "top_k": top_k,
        "vocab_size": None,
        "input_tokens": tokens_field,
        "positions": positions,
    }


def save_logprobs(data: Dict, output_path: Path) -> None:
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_logprobs(path: Path) -> Optional[Dict]:
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def compute_kl_divergence(baseline: Dict, target: Dict) -> Dict:
    """Compute mean per-position KL(baseline || target) over top-K approximations.

    For each position, takes the union of the two top-K token sets. Tokens
    missing from a side are assigned a floor logprob 5 nats below that side's
    worst observed value, then both sides are renormalized over the union via
    log-sum-exp. This is the standard top-K KL approximation.
    """
    base_pos = baseline["positions"]
    tgt_pos = target["positions"]
    n = min(len(base_pos), len(tgt_pos))

    warnings: List[str] = []
    if baseline.get("ref_file") != target.get("ref_file"):
        warnings.append(
            f"ref_file mismatch: baseline={baseline.get('ref_file')}, target={target.get('ref_file')}"
        )

    if n == 0:
        return {
            "mean_kl": float("nan"),
            "per_position_kl": [],
            "num_positions": 0,
            "warnings": warnings + ["no overlapping positions"],
        }

    per_pos: List[float] = []
    for i in range(n):
        bp = base_pos[i]
        tp = tgt_pos[i]
        b_map = dict(zip(bp["top_tokens"], bp["top_logprobs"]))
        t_map = dict(zip(tp["top_tokens"], tp["top_logprobs"]))
        union = list(set(b_map.keys()) | set(t_map.keys()))

        b_floor = min(bp["top_logprobs"]) - 5.0
        t_floor = min(tp["top_logprobs"]) - 5.0

        b_lp = [b_map.get(tok, b_floor) for tok in union]
        t_lp = [t_map.get(tok, t_floor) for tok in union]

        b_max = max(b_lp)
        b_sum = b_max + math.log(sum(math.exp(x - b_max) for x in b_lp))
        b_lp_norm = [x - b_sum for x in b_lp]

        t_max = max(t_lp)
        t_sum = t_max + math.log(sum(math.exp(x - t_max) for x in t_lp))
        t_lp_norm = [x - t_sum for x in t_lp]

        kl = 0.0
        for blp, tlp in zip(b_lp_norm, t_lp_norm):
            kl += math.exp(blp) * (blp - tlp)
        per_pos.append(kl)

    return {
        "mean_kl": sum(per_pos) / len(per_pos),
        "per_position_kl": per_pos,
        "num_positions": n,
        "warnings": warnings,
    }
