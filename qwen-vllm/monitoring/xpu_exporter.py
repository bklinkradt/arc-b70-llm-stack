#!/usr/bin/env python3
"""Expose Intel GPU stats from `xpu-smi dump` as Prometheus metrics (stdlib only).

Every card in XPU_DEVICES (default "0,1"), labelled with its xpu-smi device id and
PCI address. The xe driver doesn't report a device-wide utilization %, so memory read
bandwidth is the best proxy for how hard a GPU is working.
"""
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEVICES = os.environ.get("XPU_DEVICES", "0,1").split(",")
PORT = 9101
INTERVAL = "2"

# xpu-smi metric id -> (prometheus name, help, scale factor to base units)
METRICS = [
    ("1", "xpu_power_watts", "GPU power draw in watts.", 1),
    ("2", "xpu_frequency_mhz", "GPU core frequency in MHz.", 1),
    ("3", "xpu_temperature_celsius", "GPU core temperature in Celsius.", 1),
    ("4", "xpu_memory_temperature_celsius", "GPU memory temperature in Celsius.", 1),
    ("5", "xpu_memory_utilization_ratio", "Fraction of GPU memory in use.", 0.01),
    ("18", "xpu_memory_used_bytes", "GPU memory used in bytes.", 1024 * 1024),
    ("6", "xpu_memory_read_bytes_per_second", "GPU memory read bandwidth.", 1000),
    ("7", "xpu_memory_write_bytes_per_second", "GPU memory write bandwidth.", 1000),
]

latest: dict[str, dict[str, float]] = {}  # device id -> metric -> value
last_update = 0.0
lock = threading.Lock()


def pci_addresses():
    """Device id -> PCI address, from `xpu-smi discovery` (empty on failure)."""
    try:
        out = subprocess.run(["xpu-smi", "discovery"], capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    found, dev = {}, None
    for line in out.splitlines():
        if m := re.match(r"\|?\s*(\d+)\s+\|?\s*Device Name", line):
            dev = m.group(1)
        elif dev is not None and (m := re.search(r"PCI BDF Address:\s*(\S+)", line)):
            found[dev] = m.group(1)
    return found


PCI = pci_addresses()


def collect():
    global last_update
    ids = ",".join(m[0] for m in METRICS)
    while True:
        proc = subprocess.Popen(
            # xpu-smi 2.x syntax (1.x used -d/-m/-i); metric ids are the legacy ones.
            ["xpu-smi", "dump", "--device", ",".join(DEVICES), "--metrics", ids,
             "--interval", INTERVAL],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        for line in proc.stdout:
            fields = [f.strip() for f in line.split(",")]
            # Data rows: timestamp, device id, then one column per metric.
            if len(fields) != len(METRICS) + 2 or fields[1] not in DEVICES:
                continue
            values = {}
            for (_, name, _, scale), raw in zip(METRICS, fields[2:]):
                try:
                    values[name] = float(raw) * scale
                except ValueError:  # "N/A"
                    pass
            with lock:
                latest[fields[1]] = values
                last_update = time.time()
        proc.wait()
        time.sleep(5)  # xpu-smi exited; restart it


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/metrics":
            self.send_error(404)
            return
        with lock:
            values = {dev: dict(v) for dev, v in latest.items()}
            age = time.time() - last_update if last_update else -1
        out = []
        for _, name, help_text, _ in METRICS:
            out += [f"# HELP {name} {help_text}", f"# TYPE {name} gauge"]
            for dev in DEVICES:
                if name in values.get(dev, {}):
                    out.append(f'{name}{{device="{dev}",pci="{PCI.get(dev, "")}"}} '
                               f'{values[dev][name]}')
        out += ["# HELP xpu_exporter_sample_age_seconds Seconds since the last xpu-smi sample.",
                "# TYPE xpu_exporter_sample_age_seconds gauge",
                f"xpu_exporter_sample_age_seconds {age}"]
        body = ("\n".join(out) + "\n").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    threading.Thread(target=collect, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
