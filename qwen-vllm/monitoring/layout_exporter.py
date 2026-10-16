#!/usr/bin/env python3
"""Expose the GPU layout and each model container's serving config to Prometheus.

Stdlib only, like xpu_exporter.py. Every POLL_S seconds it reads
  - LAYOUTS_DIR/layouts.conf (`card` and `layout` lines) and LAYOUTS_DIR/mode (the
    layout `make mode` picked; missing = split, as in gpu-layout.sh), and
  - GET /containers/<name>/json for every container a `card` line names, through
    the docker-api socket proxy (DOCKER_HOST; never the socket itself),
and serves the last snapshot on :PORT/metrics. A failed read never crashes it: the
container's running value becomes -1 and an error counter goes up.

Metrics (labels in braces):
  local_serve_layout_info{layout}                      1, the configured layout
  local_serve_layout_members{layout,container,priority} 1 per layout member
  local_serve_container_running{container,cards,engine,model,weights,tp,ctx,parallel,
      quant,kv,spec}                                   1 running, 0 not, -1 unknown
  local_serve_container_state{container,state,health}  1 (docker's State.Status/Health)
  local_serve_card_container{card,container,model}     1 per running container on a
                                                       card; container="free" 0 if none
  local_serve_effective_layout{layout,configured}      1 when what runs is the
      configured layout, 0 when it differs (a switch, an on-demand model such as
      qwen-image, a crash), -1 when unknown. `layout` names what runs: a layout,
      "<layout>+qwen-image", "mixed" (no layout fits), "none" or "unknown".
  local_serve_docker_api_up, local_serve_inspect_errors_total{container},
  local_serve_layout_conf_errors_total, local_serve_layout_sample_age_seconds,
  local_serve_layout_exporter_up
"""
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LAYOUTS_DIR = os.environ.get("LAYOUTS_DIR", "/layouts")
DOCKER_HOST = os.environ.get("DOCKER_HOST", "tcp://docker-api:2375")
PORT = int(os.environ.get("PORT", "9102"))
POLL_S = float(os.environ.get("POLL_S", "5"))
TIMEOUT_S = float(os.environ.get("TIMEOUT_S", "3"))
DEFAULT_LAYOUT = "split"  # gpu-layout.sh's default when ./mode is missing

CONFIG_LABELS = ("engine", "model", "weights", "tp", "ctx", "parallel", "quant", "kv", "spec")

# A quantization named in the weights' file or directory name (Qwen3.8-27B-INT4,
# gemma-4-31B-it-Q4_K_M.gguf), used when no --quantization flag says it.
QUANT_IN_NAME = re.compile(
    r"(?<![A-Za-z0-9])(W\d+A\d+|INT\d+|NVFP4|MXFP4|FP8|AWQ|GPTQ|BF16|F16|F32|"
    r"I?Q\d(?:_[0-9A-Z]+)*)(?![A-Za-z0-9])", re.IGNORECASE)


# --- layouts.conf --------------------------------------------------------------------

def parse_layouts(text):
    """layouts.conf -> (cards: {container: [card, ...]}, layouts: {name: [container, ...]}).

    Both keep file order, which is the priority order gpu-layout.sh uses."""
    cards, layouts = {}, {}
    for line in text.splitlines():
        words = line.split("#", 1)[0].split()
        if len(words) >= 3 and words[0] == "card":
            cards[words[1]] = words[2:]
        elif len(words) >= 2 and words[0] == "layout":
            layouts[words[1]] = words[2:]
    return cards, layouts


def read_mode(layouts_dir):
    # `make mode` rewrites the file in place (`echo M > mode`), so a read can land
    # between the truncate and the write: read an empty file once more.
    for attempt in range(2):
        try:
            with open(os.path.join(layouts_dir, "mode")) as f:
                mode = f.read().strip()
        except FileNotFoundError:
            return DEFAULT_LAYOUT
        if mode:
            return mode
        time.sleep(0.1)
    return "unknown"


def simulate(cards, members, on_demand):
    """What gpu-layout.sh's reconcile keeps running for a layout: on-demand models
    first, then each member in order if all its cards are still free."""
    want, used = [], set()
    for c in list(on_demand) + list(members):
        if c in want or c not in cards:
            continue
        if not used.intersection(cards[c]):
            want.append(c)
            used.update(cards[c])
    return set(want)


def effective_layout(cards, layouts, configured, running):
    """Name what is running, from the set of running managed containers.

    Containers in no layout (qwen-image) are on-demand. The configured layout wins
    when several layouts would leave the same containers running."""
    in_layouts = {c for m in layouts.values() for c in m}
    on_demand = sorted(c for c in running if c not in in_layouts)
    order = ([configured] if configured in layouts else []) + [n for n in layouts if n != configured]
    base = next((n for n in order if simulate(cards, layouts[n], on_demand) == running), None)
    if base is None:
        base = "none" if not running else "mixed"
    name = "+".join([base] + on_demand)
    return name, 1 if name == configured else 0


# --- launch args ---------------------------------------------------------------------

def flag(argv, *names):
    """Value of the last `--name=value` / `--name value` / `-n value` in argv, else None."""
    found = None
    for i, a in enumerate(argv):
        for n in names:
            if a == n and i + 1 < len(argv):
                found = argv[i + 1]
            elif a.startswith(n + "="):
                found = a[len(n) + 1:]
    return found


def env_of(inspect):
    env = {}
    for e in (inspect.get("Config") or {}).get("Env") or []:
        k, _, v = e.partition("=")
        env[k] = v
    return env


def quant_from_name(weights):
    m = QUANT_IN_NAME.search(weights or "")
    return m.group(1).upper() if m else ""


def serving_config(inspect):
    """Engine and serving flags from a container's launch command. Unknown = ""."""
    argv = [inspect.get("Path") or ""] + list(inspect.get("Args") or [])
    env = env_of(inspect)
    exe = os.path.basename(argv[0])
    cfg = dict.fromkeys(CONFIG_LABELS, "")
    if exe == "vllm" or any(a.startswith("vllm.entrypoints") for a in argv):
        cfg["engine"] = "vllm"
        # `vllm serve MODEL ...`: the model is the first positional after `serve`.
        path = flag(argv, "--model")
        if path is None and "serve" in argv:
            rest = argv[argv.index("serve") + 1:]
            if rest and not rest[0].startswith("-"):
                path = rest[0]
        cfg["weights"] = os.path.basename((path or "").rstrip("/"))
        cfg["model"] = flag(argv, "--served-model-name") or cfg["weights"]
        cfg["tp"] = flag(argv, "--tensor-parallel-size", "-tp") or "1"
        cfg["ctx"] = flag(argv, "--max-model-len") or ""
        cfg["parallel"] = flag(argv, "--max-num-seqs") or ""
        cfg["quant"] = flag(argv, "--quantization", "-q") or quant_from_name(cfg["weights"])
        cfg["kv"] = flag(argv, "--kv-cache-dtype") or "auto"
        spec = flag(argv, "--speculative-config")
        if spec:
            try:
                s = json.loads(spec)
                cfg["spec"] = f'{s.get("method", "?")}:{s.get("num_speculative_tokens", "?")}'
            except (ValueError, AttributeError):
                cfg["spec"] = "on"
    elif exe in ("llama-server", "llama-cli") or exe.startswith("llama-"):
        cfg["engine"] = "llama.cpp"
        path = flag(argv, "--model", "-m") or ""
        cfg["weights"] = os.path.basename(path)
        cfg["model"] = flag(argv, "--alias", "-a") or cfg["weights"]
        cfg["ctx"] = flag(argv, "--ctx-size", "-c") or ""
        cfg["parallel"] = flag(argv, "--parallel", "-np") or ""
        cfg["quant"] = quant_from_name(cfg["weights"])
        k = flag(argv, "--cache-type-k", "-ctk") or "f16"
        v = flag(argv, "--cache-type-v", "-ctv") or "f16"
        cfg["kv"] = k if k == v else f"k:{k} v:{v}"
        spec = flag(argv, "--spec-type")
        if spec or flag(argv, "--model-draft", "-md"):
            cfg["spec"] = f'{spec or "draft"}:{flag(argv, "--spec-draft-n-max", "--draft-max") or "?"}'
    else:
        # Anything else (qwen-image's `python app.py`): the command and a *_MODEL env var.
        cfg["engine"] = " ".join([exe] + [os.path.basename(a) for a in argv[1:2]]).strip()
        model = next((v for k, v in env.items() if k.endswith("MODEL") and v), "")
        cfg["model"] = model
        cfg["weights"] = os.path.basename(model)
        cfg["quant"] = quant_from_name(cfg["weights"])
    return cfg


# --- polling -------------------------------------------------------------------------

def docker_base():
    if DOCKER_HOST.startswith("tcp://"):
        return "http://" + DOCKER_HOST[len("tcp://"):].rstrip("/")
    return DOCKER_HOST.rstrip("/")


def inspect(name):
    """GET /containers/NAME/json. Returns the JSON, or None when the container does not
    exist (404). Raises on anything else."""
    try:
        with urllib.request.urlopen(f"{docker_base()}/containers/{name}/json", timeout=TIMEOUT_S) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.snapshot = None       # last poll's text lines (without counters)
        self.updated = 0.0
        self.inspect_errors = {}   # container -> count
        self.conf_errors = 0
        self.conf = None           # last good (cards, layouts)
        self.configs = {}          # container -> last known serving config


STATE = State()


def esc(v):
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def series(name, labels, value):
    inner = ",".join(f'{k}="{esc(v)}"' for k, v in labels.items())
    return f"{name}{{{inner}}} {value}"


def poll_once(layouts_dir=None, fetch=None):
    """One poll: read the layouts and inspect every managed container. Returns the
    metric lines for the snapshot and updates STATE's counters."""
    layouts_dir = layouts_dir or LAYOUTS_DIR
    fetch = fetch or inspect
    try:
        with open(os.path.join(layouts_dir, "layouts.conf")) as f:
            conf = parse_layouts(f.read())
        if not conf[0]:
            raise ValueError("no card lines")
        STATE.conf = conf
    except (OSError, ValueError):
        STATE.conf_errors += 1
        conf = STATE.conf
    configured = read_mode(layouts_dir)
    out = []
    out += ["# HELP local_serve_layout_info The GPU layout `make mode` picked (layouts/mode).",
            "# TYPE local_serve_layout_info gauge",
            series("local_serve_layout_info", {"layout": configured}, 1)]
    if conf is None:
        return out + ["local_serve_docker_api_up 0"]
    cards, layouts = conf

    out += ["# HELP local_serve_layout_members Containers each layout keeps running, in priority order (1 = first).",
            "# TYPE local_serve_layout_members gauge"]
    for name, members in layouts.items():
        for i, c in enumerate(members, 1):
            out.append(series("local_serve_layout_members",
                              {"layout": name, "container": c, "priority": i}, 1))

    with ThreadPoolExecutor(max_workers=len(cards)) as pool:
        futures = {c: pool.submit(fetch, c) for c in cards}
    running, unknown, states, api_ok = set(), set(), {}, True
    for c, fut in futures.items():
        try:
            data = fut.result()
        except Exception:  # noqa: BLE001 - any failure is "unknown", never a crash
            STATE.inspect_errors[c] = STATE.inspect_errors.get(c, 0) + 1
            unknown.add(c)
            api_ok = False
            continue
        if data is None:
            states[c] = ("missing", "")
            continue
        st = data.get("State") or {}
        states[c] = (st.get("Status", ""), (st.get("Health") or {}).get("Status", ""))
        STATE.configs[c] = serving_config(data)
        if st.get("Running"):
            running.add(c)

    out += ["# HELP local_serve_container_running Model container running (1), stopped (0) or unknown (-1), with its serving config.",
            "# TYPE local_serve_container_running gauge"]
    for c, cs in cards.items():
        cfg = STATE.configs.get(c, dict.fromkeys(CONFIG_LABELS, ""))
        value = -1 if c in unknown else (1 if c in running else 0)
        out.append(series("local_serve_container_running",
                          {"container": c, "cards": "+".join(cs), **cfg}, value))
    out += ["# HELP local_serve_container_state Docker's State.Status and health check status per container.",
            "# TYPE local_serve_container_state gauge"]
    for c in cards:
        status, health = states.get(c, ("unknown", ""))
        out.append(series("local_serve_container_state",
                          {"container": c, "state": status, "health": health}, 1))

    out += ["# HELP local_serve_card_container Running container(s) on each card; container=\"free\" (0) when none.",
            "# TYPE local_serve_card_container gauge"]
    for card in sorted({g for cs in cards.values() for g in cs}, key=lambda g: (len(g), g)):
        on = [c for c, cs in cards.items() if card in cs and c in running]
        for c in on:
            out.append(series("local_serve_card_container",
                              {"card": card, "container": c,
                               "model": STATE.configs.get(c, {}).get("model", "")}, 1))
        if not on:
            held = "unknown" if any(card in cards[c] for c in unknown) else "free"
            out.append(series("local_serve_card_container",
                              {"card": card, "container": held, "model": ""}, 0 if held == "free" else -1))

    if unknown:
        eff, match = "unknown", -1
    else:
        eff, match = effective_layout(cards, layouts, configured, running)
    out += ["# HELP local_serve_effective_layout What is running, as a layout name; 1 = the configured one, 0 = differs, -1 = unknown.",
            "# TYPE local_serve_effective_layout gauge",
            series("local_serve_effective_layout", {"layout": eff, "configured": configured}, match),
            "# HELP local_serve_docker_api_up 1 when every inspect in the last poll got an answer.",
            "# TYPE local_serve_docker_api_up gauge",
            f"local_serve_docker_api_up {1 if api_ok else 0}"]
    return out


def poller():
    while True:
        try:
            lines = poll_once()
        except Exception:  # noqa: BLE001 - keep serving the last snapshot
            STATE.conf_errors += 1
            lines = None
        if lines is not None:
            with STATE.lock:
                STATE.snapshot = lines
                STATE.updated = time.time()
        time.sleep(POLL_S)


def render():
    with STATE.lock:
        lines = list(STATE.snapshot or [])
        age = time.time() - STATE.updated if STATE.updated else -1
        errors = dict(STATE.inspect_errors)
        conf_errors = STATE.conf_errors
    lines += ["# HELP local_serve_inspect_errors_total Failed container inspects through the docker-api proxy.",
              "# TYPE local_serve_inspect_errors_total counter"]
    for c in (STATE.conf[0] if STATE.conf else {}):
        lines.append(series("local_serve_inspect_errors_total", {"container": c}, errors.get(c, 0)))
    lines += ["# HELP local_serve_layout_conf_errors_total Polls that could not read layouts.conf (the last good one is kept).",
              "# TYPE local_serve_layout_conf_errors_total counter",
              f"local_serve_layout_conf_errors_total {conf_errors}",
              "# HELP local_serve_layout_sample_age_seconds Seconds since the last completed poll (-1 = none yet).",
              "# TYPE local_serve_layout_sample_age_seconds gauge",
              f"local_serve_layout_sample_age_seconds {age:.1f}",
              "# HELP local_serve_layout_exporter_up The layout exporter is serving.",
              "# TYPE local_serve_layout_exporter_up gauge",
              "local_serve_layout_exporter_up 1"]
    return ("\n".join(lines) + "\n").encode()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/metrics":
            self.send_error(404)
            return
        body = render()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    threading.Thread(target=poller, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
