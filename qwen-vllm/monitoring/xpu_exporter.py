#!/usr/bin/env python3
"""Expose Intel GPU stats from `xpu-smi dump` as Prometheus metrics (stdlib only).

The xe driver doesn't report a device-wide utilization %, so memory read
bandwidth is the best proxy for how hard the GPU is working during decode.
"""
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEVICE = "0"
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

latest: dict[str, float] = {}
last_update = 0.0
lock = threading.Lock()


def collect():
    global last_update
    ids = ",".join(m[0] for m in METRICS)
    while True:
        proc = subprocess.Popen(
            # xpu-smi 2.x syntax (1.x used -d/-m/-i); metric ids are the legacy ones.
            ["xpu-smi", "dump", "--device", DEVICE, "--metrics", ids, "--interval", INTERVAL],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        for line in proc.stdout:
            fields = [f.strip() for f in line.split(",")]
            # Data rows: timestamp, device id, then one column per metric.
            if len(fields) != len(METRICS) + 2 or fields[1] != DEVICE:
                continue
            values = {}
            for (_, name, _, scale), raw in zip(METRICS, fields[2:]):
                try:
                    values[name] = float(raw) * scale
                except ValueError:  # "N/A"
                    pass
            with lock:
                latest.clear()
                latest.update(values)
                last_update = time.time()
        proc.wait()
        time.sleep(5)  # xpu-smi exited; restart it


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/metrics":
            self.send_error(404)
            return
        with lock:
            values = dict(latest)
            age = time.time() - last_update if last_update else -1
        out = []
        for _, name, help_text, _ in METRICS:
            if name in values:
                out += [f"# HELP {name} {help_text}", f"# TYPE {name} gauge",
                        f'{name}{{device="{DEVICE}"}} {values[name]}']
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
