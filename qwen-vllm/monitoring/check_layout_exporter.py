#!/usr/bin/env python3
"""Self-check for layout_exporter.py; touches nothing live.

Runs the exporter as a plain process against a fake docker-api that answers the
recorded `docker inspect` JSON in fixtures/inspect/ (trimmed, secrets redacted) and a
temporary copy of layouts.conf, then switches the fake cluster through a few states
and checks each scrape:

    python3 qwen-vllm/monitoring/check_layout_exporter.py      # prints a sample scrape
"""
import copy
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures", "inspect")
LAYOUTS = os.path.join(HERE, "..", "..", "gateway", "llama-swap", "layouts", "layouts.conf")

INSPECT = {f[:-5]: json.load(open(os.path.join(FIXTURES, f)))
           for f in os.listdir(FIXTURES) if f.endswith(".json")}
running = set()   # what the fake cluster runs
failing = set()   # containers whose inspect answers 500


class FakeDockerApi(BaseHTTPRequestHandler):
    def do_GET(self):
        m = re.fullmatch(r"(?:/v[0-9.]+)?/containers/([\w.-]+)/json", self.path)
        if not m or m.group(1) not in INSPECT:
            self.send_error(404)
            return
        name = m.group(1)
        if name in failing:
            self.send_error(500)
            return
        data = copy.deepcopy(INSPECT[name])
        data["State"]["Running"] = name in running
        data["State"]["Status"] = "running" if name in running else "exited"
        if "Health" in data["State"]:
            data["State"]["Health"]["Status"] = "healthy" if name in running else "unhealthy"
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def parse(text):
    """Metric lines -> list of (name, {labels}, value)."""
    out = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"(\w+)(?:\{(.*)\})? (\S+)", line)
        assert m, f"bad line: {line!r}"
        labels = dict(re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', m.group(2) or ""))
        out.append((m.group(1), labels, float(m.group(3))))
    return out


def find(samples, name, **want):
    return [(lb, v) for n, lb, v in samples if n == name and all(lb.get(k) == w for k, w in want.items())]


def main():
    api = ThreadingHTTPServer(("127.0.0.1", 0), FakeDockerApi)
    threading.Thread(target=api.serve_forever, daemon=True).start()
    layouts = tempfile.mkdtemp(prefix="layouts-")
    shutil.copy(LAYOUTS, layouts)
    port = free_port()
    env = dict(os.environ, LAYOUTS_DIR=layouts, PORT=str(port), POLL_S="0.2",
               DOCKER_HOST=f"tcp://127.0.0.1:{api.server_address[1]}")
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "layout_exporter.py")], env=env)
    failures = []

    def scrape():
        time.sleep(0.6)  # > two polls
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
            text = r.read().decode()
        return text, parse(text)

    def state(desc, mode, run, fail=(), checks=()):
        if mode is None:
            if os.path.exists(os.path.join(layouts, "mode")):
                os.remove(os.path.join(layouts, "mode"))
        else:
            with open(os.path.join(layouts, "mode"), "w") as f:
                f.write(mode + "\n")
        running.clear(); running.update(run)
        failing.clear(); failing.update(fail)
        text, s = scrape()
        for what, ok in checks(s) if callable(checks) else checks:
            if not ok:
                failures.append(f"{desc}: {what}")
        print(f"ok  {desc}" if not any(f.startswith(desc + ":") for f in failures) else f"FAIL {desc}")
        return text, s

    def eff(s):
        (lb, v), = find(s, "local_serve_effective_layout")
        return lb["layout"], lb["configured"], v

    def cards(s, card):
        return sorted(lb["container"] for lb, v in find(s, "local_serve_card_container", card=card))

    def running_of(s, c):
        (lb, v), = find(s, "local_serve_container_running", container=c)
        return lb, v

    try:
        for _ in range(50):  # wait for the exporter to listen
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=1)
                break
            except OSError:
                time.sleep(0.1)

        state("split, as configured", "split", {"qwen-vllm", "qwen-vllm-2"}, checks=lambda s: [
            ("effective split/1", eff(s) == ("split", "split", 1)),
            ("layout_info split", find(s, "local_serve_layout_info", layout="split") != []),
            ("card 0 = qwen-vllm", cards(s, "0") == ["qwen-vllm"]),
            ("card 1 = qwen-vllm-2", cards(s, "1") == ["qwen-vllm-2"]),
            ("vllm config", {k: running_of(s, "qwen-vllm")[0][k] for k in
                             ("cards", "engine", "model", "weights", "tp", "ctx", "parallel", "quant", "kv", "spec")}
             == {"cards": "0", "engine": "vllm", "model": "Qwen3.8-27B", "weights": "Qwen3.8-27B-INT4",
                 "tp": "1", "ctx": "131072", "parallel": "4", "quant": "INT4", "kv": "auto", "spec": "mtp:3"}),
            ("members", len(find(s, "local_serve_layout_members", layout="tp2")) == 3),
            ("api up", find(s, "local_serve_docker_api_up")[0][1] == 1),
        ])
        state("tp2, as configured", "tp2", {"qwen-vllm-tp2"}, checks=lambda s: [
            ("effective tp2/1", eff(s) == ("tp2", "tp2", 1)),
            ("tp2 on both cards", cards(s, "0") == cards(s, "1") == ["qwen-vllm-tp2"]),
            ("tp2 config", {k: running_of(s, "qwen-vllm-tp2")[0][k] for k in ("cards", "tp", "ctx", "parallel")}
             == {"cards": "0+1", "tp": "2", "ctx": "262144", "parallel": "8"}),
        ])
        text, _ = state("review, as configured", "review", {"qwen-vllm", "gemma"}, checks=lambda s: [
            ("effective review/1", eff(s) == ("review", "review", 1)),
            ("gemma config", {k: running_of(s, "gemma")[0][k] for k in
                              ("cards", "engine", "model", "tp", "ctx", "parallel", "quant", "kv", "spec")}
             == {"cards": "1", "engine": "llama.cpp", "model": "gemma-4-31B", "tp": "", "ctx": "131072",
                 "parallel": "2", "quant": "Q4_K_M", "kv": "q8_0", "spec": "draft-mtp:3"}),
            ("gemma running 1", running_of(s, "gemma")[1] == 1),
            ("tp2 stopped 0", running_of(s, "qwen-vllm-tp2")[1] == 0),
        ])
        sample = text
        state("review with an image loaded", "review", {"qwen-vllm", "qwen-image"}, checks=lambda s: [
            ("effective review+qwen-image/0", eff(s) == ("review+qwen-image", "review", 0)),
            ("card 1 = qwen-image", cards(s, "1") == ["qwen-image"]),
            ("image config", {k: running_of(s, "qwen-image")[0][k] for k in ("engine", "model")}
             == {"engine": "python app.py", "model": "Qwen/Qwen-Image-2.1"}),
        ])
        state("big, as configured", "big", {"gpt-oss"}, checks=lambda s: [
            ("effective big/1", eff(s) == ("big", "big", 1)),
            ("gpt-oss on both cards", cards(s, "0") == cards(s, "1") == ["gpt-oss"]),
            ("gpt-oss config", {k: running_of(s, "gpt-oss")[0][k] for k in
                                ("cards", "engine", "model", "ctx", "parallel", "quant", "kv", "spec")}
             == {"cards": "0+1", "engine": "llama.cpp", "model": "gpt-oss-120b", "ctx": "131072",
                 "parallel": "4", "quant": "MXFP4", "kv": "q8_0", "spec": ""}),
        ])
        state("big with an image loaded", "big", {"qwen-vllm", "qwen-image"}, checks=lambda s: [
            ("effective big+qwen-image/0", eff(s) == ("big+qwen-image", "big", 0)),
            ("card 0 = qwen-vllm", cards(s, "0") == ["qwen-vllm"]),
        ])
        state("tp2 with an image loaded", "tp2", {"qwen-vllm", "qwen-image"}, checks=lambda s: [
            ("effective tp2+qwen-image/0", eff(s) == ("tp2+qwen-image", "tp2", 0)),
        ])
        state("switching split -> tp2", "tp2", {"qwen-vllm", "qwen-vllm-2"}, checks=lambda s: [
            ("effective split/0", eff(s) == ("split", "tp2", 0)),
        ])
        state("half started", "split", {"qwen-vllm"}, checks=lambda s: [
            ("effective mixed/0", eff(s) == ("mixed", "split", 0)),
            ("card 1 free", find(s, "local_serve_card_container", card="1") == [
                ({"card": "1", "container": "free", "model": ""}, 0)]),
        ])
        state("nothing running, no mode file", None, set(), checks=lambda s: [
            ("configured defaults to split", find(s, "local_serve_layout_info", layout="split") != []),
            ("effective none/0", eff(s) == ("none", "split", 0)),
        ])
        state("gemma inspect fails", "review", {"qwen-vllm", "gemma"}, fail={"gemma"}, checks=lambda s: [
            ("gemma running -1", running_of(s, "gemma")[1] == -1),
            ("gemma keeps its last config", running_of(s, "gemma")[0]["model"] == "gemma-4-31B"),
            ("effective unknown/-1", eff(s) == ("unknown", "review", -1)),
            ("card 1 unknown", find(s, "local_serve_card_container", card="1")[0][0]["container"] == "unknown"),
            ("api down", find(s, "local_serve_docker_api_up")[0][1] == 0),
            ("errors counted", find(s, "local_serve_inspect_errors_total", container="gemma")[0][1] > 0),
            ("exporter up", find(s, "local_serve_layout_exporter_up")[0][1] == 1),
        ])
        api.shutdown()
        api.server_close()
        state("docker-api unreachable", "split", set(), checks=lambda s: [
            ("all -1", all(v == -1 for _, v in find(s, "local_serve_container_running"))),
            ("exporter still up", find(s, "local_serve_layout_exporter_up")[0][1] == 1),
        ])
    finally:
        proc.terminate()
        proc.wait(5)
        shutil.rmtree(layouts, ignore_errors=True)

    print("\n--- sample scrape (review layout) ---\n" + sample)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
