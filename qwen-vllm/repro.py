#!/usr/bin/env python3
"""Check the local vLLM server for output corruption with long agent-style prompts.

Sends a ~21k-token prompt (this repo's files) followed by multi-turn follow-ups,
like opencode does, and flags responses that break down mid-generation. MTP
speculative decoding corrupted ~15-20% of these rounds; short prompts never did.

Usage: repro.py [rounds] [cache|nocache]
Environment: BENCH_URL (default http://localhost:8000) and BENCH_MODEL (default
Qwen3.8-27B), as for bench.py, so it can check the llamacpp-bench servers too.
  cache:   identical prefix every round (prefix cache hits, like opencode)
  nocache: new random prefix every round (forces a full prefill)
Exits 1 if any round was corrupted.
"""
import json
import os
import re
import sys
import time
import urllib.request
import uuid
from pathlib import Path

URL = os.environ.get("BENCH_URL", "http://localhost:8000") + "/v1/chat/completions"
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.8-27B")
ROOT = Path(__file__).resolve().parent
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MODE = sys.argv[2] if len(sys.argv) > 2 else "cache"
FILES = [
    "monitoring/grafana/dashboards/vllm.json", "docker-compose.yml", "bench.py",
    "monitoring/xpu_exporter.py", "monitoring/gen_dashboard.py",
    # Copy of an unrelated script, so the prompt keeps the content of earlier runs.
    "repro-fixtures/gen_image.py",
]
ASKS = [
    "Write a Python function that calls an HTTP endpoint /generateImage with urllib, "
    "posting JSON with an X-Api-Key header and returning the parsed JSON response, with retries.",
    "Now add type hints and a docstring, and handle HTTP 429 with exponential backoff.",
    "Now write pytest tests for it using unittest.mock to patch urlopen.",
    "Now refactor it into a small class with a configurable base_url and timeout.",
    "Summarize in plain English what the code you wrote does, in about 150 words.",
]
CONTEXT = "\n\n".join(f"### {f}\n```\n{(ROOT / f).read_text()}\n```" for f in FILES)
SYSTEM = "You are a coding agent. Answer with concise code.\n\n" + CONTEXT


def corrupt(text):
    """Newline floods, replacement chars, stray non-ASCII scripts, leaked special
    tokens, a word or short phrase looping 15+ times on one line, or an early stop."""
    return bool(re.search(r"\n\s*\n(\s*\n){8,}|�|[Ā-ſЀ-ӿ‎]|<(?:pad|unused\d*|eos|bos)>"
                          r"|(?P<loop>\w[^\n]{1,19}?)(?P=loop){14,}", text)
                or len(text) < 400)


def chat(messages):
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": 700,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"] or "", data["usage"], time.perf_counter() - start


if __name__ == "__main__":
    messages = [{"role": "system", "content": SYSTEM}]
    bad = 0
    for i in range(ROUNDS):
        if MODE == "nocache":
            messages[0]["content"] = f"[session {uuid.uuid4()}]\n" + SYSTEM
        messages.append({"role": "user", "content": ASKS[i % len(ASKS)]})
        text, usage, elapsed = chat(messages)
        bad += corrupt(text)
        print(f"round {i + 1}: prompt {usage['prompt_tokens']:5d} tok  out {usage['completion_tokens']:3d} tok  "
              f"{elapsed:4.0f}s  {'CORRUPT' if corrupt(text) else 'ok'}")
        if corrupt(text):
            print("   tail:", repr(text[-200:]))
        messages.append({"role": "assistant", "content": text})
    print(f"corrupted rounds: {bad}/{ROUNDS}")
    sys.exit(1 if bad else 0)
