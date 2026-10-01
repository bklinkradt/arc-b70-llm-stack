#!/usr/bin/env python3
"""Measure prefill speed of the local vLLM server (stdlib only).

Usage: prefill.py [words...]   (default 4000 16000 60000, about 4k / 17k / 63k tokens)

Each prompt starts with a random number, so the prefix cache can't skip any of it, and
asks for one output token, so the time is almost all prefill. One warm-up request per
size, then the average of three. BENCH_URL as in bench.py.
"""
import json
import os
import random
import sys
import time
import urllib.request

URL = os.environ.get("BENCH_URL", "http://localhost:8000") + "/v1/chat/completions"
WORDS = ("the cache line is evicted when a new block arrives and the tag does not "
         "match any way in the set").split()


def run(n_words):
    rnd = random.Random(time.time_ns())
    text = f"[{rnd.random()}] " + " ".join(rnd.choice(WORDS) for _ in range(n_words))
    body = json.dumps({
        "model": os.environ.get("BENCH_MODEL", "Qwen3.8-27B"),
        "messages": [{"role": "user", "content": text + "\nSummarise in one word."}],
        "max_tokens": 1,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    start = time.perf_counter()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        tokens = json.load(resp)["usage"]["prompt_tokens"]
    return tokens, time.perf_counter() - start


if __name__ == "__main__":
    for n in map(int, sys.argv[1:] or ["4000", "16000", "60000"]):
        run(n)
        results = [run(n) for _ in range(3)]
        tokens = results[0][0]
        avg = sum(t for _, t in results) / len(results)
        print(f"prompt {tokens:6d} tok  time {avg:6.2f}s  prefill {tokens / avg:7.0f} tok/s")
