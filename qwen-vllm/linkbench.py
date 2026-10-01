#!/usr/bin/env python3
"""Measure the PCIe links and the two-card all-reduce that TP=2 runs on (torch xccl).

Prints host<->device copy bandwidth per card, then all-reduce time and bus bandwidth
from 160 KiB (one decode step) to 160 MiB. Run it in a vLLM image with both cards
free (stop llama-swap and the vLLM containers first):

  docker run --rm --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro \
    --group-add $(getent group video | cut -d: -f3) \
    --group-add $(getent group render | cut -d: -f3) \
    --shm-size 8g -e ZE_AFFINITY_MASK=0,1 \
    -e CCL_ENABLE_SYCL_KERNELS=1 -v $PWD:/w --entrypoint python3 \
    qwen-vllm-runtime:26.35 /w/linkbench.py

CCL_ENABLE_SYCL_KERNELS=0 measures oneCCL's fallback path. NO_AR=1 skips the
all-reduce; ITERS_MUL=50 repeats each size 50x as a stress test.
"""
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def copybw(dev):
    n = 1 << 30
    h = torch.empty(n, dtype=torch.uint8).pin_memory()
    d = torch.empty(n, dtype=torch.uint8, device=dev)
    for name, f in (("H2D", lambda: d.copy_(h, non_blocking=True)),
                    ("D2H", lambda: h.copy_(d, non_blocking=True))):
        f()
        torch.xpu.synchronize(dev)
        t = time.perf_counter()
        for _ in range(5):
            f()
        torch.xpu.synchronize(dev)
        print(f"card {dev} {name}: {5 * n / (time.perf_counter() - t) / 1e9:5.1f} GB/s", flush=True)


def worker(rank):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29511")
    torch.xpu.set_device(rank)
    dist.init_process_group("xccl", rank=rank, world_size=2)
    for kib in (160, 1024, 10 * 1024, 40 * 1024, 160 * 1024):
        x = torch.ones(kib * 512, dtype=torch.bfloat16, device=f"xpu:{rank}")
        iters = int(os.environ.get("ITERS_MUL", "1")) * (200 if kib <= 1024 else 20)
        for _ in range(5):
            dist.all_reduce(x)
        torch.xpu.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.xpu.synchronize()
        dt = (time.perf_counter() - t) / iters
        if rank == 0:
            # Bus bandwidth for n ranks is size * 2(n-1)/n, which is the size for two.
            print(f"all_reduce {kib:7d} KiB: {dt * 1e6:9.1f} us  busbw {kib * 1024 / dt / 1e9:6.2f} GB/s",
                  flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    for i in range(torch.xpu.device_count()):
        copybw(i)
    if not os.environ.get("NO_AR"):
        mp.spawn(worker, nprocs=2)
