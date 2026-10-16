#!/usr/bin/env python3
"""Measure a long session growing turn by turn on the local vLLM server (stdlib only).

Usage: context.py [words per turn] [total words]   (default 15000 235000, about
16k tokens a turn up to ~250k)

Like a coding-agent session: each turn appends a block of text to one conversation,
which keeps the model's earlier replies, so the prefix cache covers everything but the
new block. Prints, per turn, the prompt size, time to first token (prefill of the new
block) and decode speed at that depth. Then sends the final prompt once more with a
different first line, so nothing is cached, for the cold prefill time. BENCH_URL as in
bench.py.
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
CHUNK = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
TOTAL = int(sys.argv[2]) if len(sys.argv) > 2 else 235000
MAX_TOKENS = 200


def run(messages):
    """Stream one reply; returns (reply, prompt tokens, TTFT s, decode tok/s)."""
    body = json.dumps({
        "model": os.environ.get("BENCH_MODEL", "Qwen3.8-27B"),
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    reply, usage = [], {}
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                text = choice.get("delta", {}).get("content")
                if text:
                    first = first or time.perf_counter()
                    reply.append(text)
    end = time.perf_counter()
    first = first or end
    out = usage.get("completion_tokens", 0)
    decode = (out - 1) / (end - first) if out > 1 and end > first else 0
    return "".join(reply), usage.get("prompt_tokens", 0), first - start, decode


def block(rnd, n):
    return " ".join(rnd.choice(WORDS) for _ in range(n))


if __name__ == "__main__":
    rnd = random.Random(time.time_ns())
    messages = []
    words = 0
    print(f"{'turn':>4} {'prompt tok':>10} {'TTFT s':>7} {'decode tok/s':>12}")
    while words + CHUNK <= TOTAL:
        text = block(rnd, CHUNK)
        if not messages:
            text = f"[{rnd.random()}] " + text
        messages.append({"role": "user",
                         "content": text + "\nSummarise this block in about 100 words."})
        words += CHUNK
        reply, tokens, ttft, decode = run(messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"{len(messages) // 2:4d} {tokens:10d} {ttft:7.2f} {decode:12.1f}", flush=True)

    # Same conversation with a new first line: a full prefill of the last prompt.
    messages[0]["content"] = f"[{rnd.random()}] " + messages[0]["content"].split("] ", 1)[1]
    _, tokens, ttft, decode = run(messages[:-1])
    print(f"cold {tokens:10d} {ttft:7.2f} {decode:12.1f}  "
          f"(full prefill {tokens / ttft:.0f} tok/s)")
