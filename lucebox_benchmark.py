#!/usr/bin/env python3
"""Benchmark a Lucebox server.

Lucebox speaks the OpenAI chat API and reports per-request prefill/decode
timings in ``usage.timings``; openai_benchmark already reads those, so this is
just the generic engine with Lucebox's tag, label, and default port.
"""

import sys

import openai_benchmark


def main() -> int:
    return openai_benchmark.main(tag="lucebox", name="Lucebox", default_base_url="http://localhost:8081/v1")


if __name__ == "__main__":
    sys.exit(main())
