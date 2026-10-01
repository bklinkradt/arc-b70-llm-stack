#!/usr/bin/env python3
"""Measure decode speed of the local vLLM server (stdlib only).

Usage: bench.py [runs] [prose|code] [concurrency] [temperature]

BENCH_URL (default http://localhost:8000) and BENCH_MODEL (default Qwen3.8-27B)
point it at another OpenAI-compatible server, such as ../llamacpp-bench.

Use temperature 0 to compare configurations: sampling at 0.7 changes how many
MTP drafts are accepted from run to run, which swings decode speed by 20+ tok/s.
"""
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE_URL = os.environ.get("BENCH_URL", "http://localhost:8000")
URL = f"{BASE_URL}/v1/chat/completions"
METRICS_URL = f"{BASE_URL}/metrics"
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.8-27B")
PROMPTS = {
    "prose": "Write a detailed, 600-word explanation of how a CPU cache hierarchy works.",
    "code": (
        "Refactor this Python into a well-typed module with a dataclass, docstrings, "
        "input validation and pytest unit tests. Output only code.\n\n"
        "def process(data):\n"
        "    out = []\n"
        "    for d in data:\n"
        "        if d['qty'] > 0 and d['price'] is not None:\n"
        "            total = d['qty'] * d['price']\n"
        "            if d.get('discount'):\n"
        "                total = total - total * d['discount']\n"
        "            out.append({'id': d['id'], 'total': round(total, 2)})\n"
        "    return sorted(out, key=lambda x: x['total'], reverse=True)\n"
    ),
}
MAX_TOKENS = 1024
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
PROMPT = PROMPTS[sys.argv[2] if len(sys.argv) > 2 else "prose"]
# Simultaneous requests per run; >1 checks the engine survives overlapping requests.
CONCURRENCY = int(sys.argv[3]) if len(sys.argv) > 3 else 1
TEMPERATURE = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7


def running_requests():
    """Requests the server is running right now (other clients skew the numbers)."""
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=5) as resp:
            return sum(float(line.split()[-1]) for line in resp.read().decode().splitlines()
                       if line.startswith("vllm:num_requests_running"))
    except OSError:
        return 0


def run():
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    tokens = 0
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("choices") and first is None:
                first = time.perf_counter()
            if chunk.get("usage"):
                tokens = chunk["usage"]["completion_tokens"]
    end = time.perf_counter()
    decode_tps = (tokens - 1) / (end - first)
    print(f"TTFT {first - start:5.2f}s  tokens {tokens:4d}  decode {decode_tps:5.1f} tok/s")
    return decode_tps


if __name__ == "__main__":
    if busy := running_requests():
        print(f"warning: {busy:.0f} other request(s) running; results will be low")
    with ThreadPoolExecutor(CONCURRENCY) as pool:
        results = list(pool.map(lambda _: run(), range(RUNS * CONCURRENCY)))
    print(f"avg decode: {sum(results) / len(results):.1f} tok/s")
