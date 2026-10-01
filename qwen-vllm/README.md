# qwen-vllm

Self-hosted Qwen3.8-27B (INT4) on two Intel Arc Pro B70s, either one vLLM instance
per card or one instance across both (tensor parallel), served through Docker Compose. It includes an Open WebUI chat front end, a
Prometheus + Grafana monitoring stack.

## Services

| Service | Port | Purpose |
|---|---|---|
| `vllm` | 8000 (localhost) | OpenAI-compatible API (upstream vLLM v0.30.0 with Intel runtime 26.35, `runtime-26.35/Dockerfile`) on the first B70 (PCI 03:00.0) |
| `vllm-2` | 8001 (localhost) | The same server on the second B70 (PCI 08:00.0) |
| `vllm-tp2` | 8000 (localhost) | The same server across both cards (`--tensor-parallel-size=2`). Never runs alongside `vllm` |
| `open-webui` | 3000 (all interfaces) | Chat UI, talks to the running instances through `../gateway`'s `vllm-lb` |
| `grafana` | 3001 (all interfaces) | Dashboards (anonymous read-only; log in as admin to edit). Open WebUI and Grafana listen on the LAN; firewall them or bind them to 127.0.0.1 if that's not what you want |
| `prometheus` | — | Scrapes the vLLM instances and the GPU exporter every 5 s, keeps 30 days |
| `gpu-exporter` | — | Exports `xpu-smi` power, clocks, temperature, VRAM and bandwidth for the first B70 |

## Requirements

- An Intel Arc GPU with the `xe` driver, plus Docker with Compose
- The model at `~/models/Qwen3.8-27B-INT4`
  ([RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4), an
  AWQ INT4 quant with the MTP head). To use another directory, set `MODELS_DIR` in the
  environment or in `qwen-vllm/.env`.
- The host's `video` and `render` group IDs (`getent group video render`). The
  defaults are Fedora's 39 and 105; set `VIDEO_GID` / `RENDER_GID` if yours differ.

`ZE_AFFINITY_MASK` picks the card: `0` for `vllm`, `1` for `vllm-2`, `0,1` for
`vllm-tp2`. Level Zero only lists Intel devices, in PCI order, so a non-Intel iGPU is
ignored.

## Usage

`../gateway`'s llama-swap decides which of the three vLLM containers run, following the
GPU layout in `../gateway/llama-swap/layouts/layouts.conf`, and lends card 1 to
`../qwen-image` on demand. The containers are in the `managed` profile, which keeps
`docker compose up` from starting them behind llama-swap's back. From the repo root:

```bash
make up              # creates the vLLM containers, starts Open WebUI, monitoring and llama-swap
make mode            # which layout is active and what is running
make mode M=tp2      # switch: tp2 (both cards as one) or split (one instance per card)
make logs s=vllm     # first start compiles kernels; allow up to ~10 min
```

| Layout | Runs | While qwen-image holds card 1 |
|---|---|---|
| `split` (default) | `vllm` + `vllm-2` | `vllm` only; chat carries on |
| `tp2` | `vllm-tp2` | `vllm` only. Chat waits ~70 s at each swap (held in `vllm-lb`, not failed), and requests running on `vllm-tp2` are cut off |

`tp2` is ~1.5x faster per request and has ~1.5x the total throughput of `split` (see
[Tensor parallel (TP=2)](#tensor-parallel-tp2)). `split` never interrupts chat for
images and keeps two independent prefix caches. The choice survives restarts.

- API: `http://localhost:8000/v1` (`vllm` or `vllm-tp2`, whichever runs) and
  `http://localhost:8001/v1` (`vllm-2`, in `split` while card 1 isn't lent to
  qwen-image), model name `Qwen3.8-27B`, no API key. Remote access goes through
  `../gateway` with per-user API keys, which balances over the running instances.
- Chat: http://localhost:3000
- Dashboards: http://localhost:3001

The server supports tool calling (`qwen3_coder` parser), reasoning output
(`qwen3` parser), up to 4 images per prompt (capped at ~4 MP), a 128k context
and 4 concurrent sequences per card (8 for `vllm-tp2`).

## Documentation

[`docs/intel-xpu-stack.html`](docs/intel-xpu-stack.html) is a code-level guide to the
Intel stack under vLLM, from the engine down to the xe driver. It includes the
GDN conv-state bug behind the MTP corruption and a draft upstream issue. Open it
in a browser.

## Configuration notes

These settings were tuned with `bench.py`. The comments in `docker-compose.yml`
have the details.

- **Upstream vLLM v0.30.0 image.** It replaced Intel's
  `intel/llm-scaler-vllm:0.26.0-b2` on 2026-09-26. The llm-scaler image's GDN
  kernels lack an upstream fix (vllm-xpu-kernels #544/#545), so with MTP and
  prefix caching both on, long generations corrupted once they crossed a
  2048-token cache block: newline floods and gibberish in about 1 in 5 replies
  to opencode's ~20k-token prompts. v0.30.0 ships vllm-xpu-kernels 0.1.14.1,
  which includes the fix. [`docs/intel-xpu-stack.html`](docs/intel-xpu-stack.html)
  has the full analysis.
- **MTP speculative decoding is on, with 3 draft tokens.** Decode runs at
  ~68 tok/s on code and ~50 on prose, against ~31 without MTP. Validated with
  `repro.py` (0 of 40 corrupted), a 4000-token greedy run and 3 concurrent
  requests. Re-test after any image change (see [Re-testing MTP](#re-testing-mtp)).
  3 is the best depth: see [MTP draft depth](#mtp-draft-depth).
- **Prefix caching is on.** It skips re-processing the shared prefix of each
  opencode turn.
- **XPU graphs are off.** On llm-scaler they gave no decode gain and used
  2.5 GiB of VRAM that vLLM doesn't budget for. With MTP they also pad state
  slots in a way the SYCL GDN kernels don't handle.
- fp8 KV cache and fp16 dtype gave no meaningful gain.

### History: Intel llm-scaler image

Until 2026-09-26 this ran `intel/llm-scaler-vllm:0.26.0-b2` with MTP off. If you
use that image, never enable MTP while prefix caching is on. One experiment on it is
worth knowing about:

- **Intel runtime 26.31** on top of llm-scaler: compute-runtime 26.31, IGC 2.40.13
  and Level Zero 1.32. With MTP it cut corruption to 3 of 30 rounds but didn't fix
  it, because the bug was in the kernels. Building it taught two things: the PPA's
  `libigc2`/`libigdfcl2` must be force-removed before Intel's `intel-igc-*`
  packages install, and `libze-dev` is required, since oneCCL loads the
  unversioned `libze_loader.so` and fails with `oneCCL: ... ze_data was not
  initialized` without it.

## Benchmarking

`bench.py` measures streaming decode speed against the local server, and `prefill.py`
prompt-processing speed on unique 4k-63k-token prompts. Both only use the Python
standard library. `linkbench.py` measures the PCIe links and the two-card all-reduce;
its docstring has the `docker run` line.

```bash
python3 bench.py [runs] [prose|code] [concurrency] [temperature]
python3 bench.py 3 code 2    # 3 rounds of 2 concurrent code requests
python3 bench.py 5 code 1 0  # temperature 0: use this to compare configurations
```

The default temperature (0.7) changes how many MTP drafts are accepted from run
to run, which swings decode speed by 20+ tok/s. At temperature 0 runs repeat
within about ±1.5 tok/s. `bench.py` warns if other clients have requests
running, since those slow every run.

`repro.py` checks for output corruption with long, multi-turn, agent-style
prompts (~21k tokens), which `bench.py`'s short prompts don't catch. Run it
after changing the image or the MTP settings. It exits 1 if any response was
corrupted.

```bash
python3 repro.py 5 nocache   # force a full prefill every round
python3 repro.py 5 cache     # reuse the prefix, like opencode
```

Like `bench.py`, it reads `BENCH_URL` and `BENCH_MODEL`, so it can check other
servers, such as the ones in `../llamacpp-bench`.

### Re-testing MTP

Run this after any image or MTP change. On the broken llm-scaler image the
corruption showed up in about 1 in 5 responses, so a short run can pass by
chance. Test 40 rounds, a concurrent run, and check the kernel log for GPU
faults:

```bash
(cd .. && docker compose up -d --force-recreate vllm)
for m in nocache nocache nocache cache nocache nocache nocache cache; do python3 repro.py 5 $m | tail -1; done
python3 bench.py 2 code      # ~68 tok/s with MTP3, ~31 without
python3 bench.py 2 code 3    # 3 concurrent requests must not crash the engine
journalctl -k --since -30min | grep -iE 'xe .*(fault|reset|timed out)'
```

Keep MTP on only if every round is clean.

### MTP draft depth

`mtp-depth.sh` sweeps `num_speculative_tokens` on the second card
(`docker-compose.mtp.yml`: a copy of vllm-2 on `127.0.0.1:8002`, off `llm-net`, so
vllm-lb never routes users to it). Card 0 keeps serving chat; the script gives the
card back to llama-swap when it's done. `make mtp MTP_N=4` / `make mtp-down` in the
repo root start and stop one depth by hand.

Result on 2026-09-30 (v0.30.0, tok/s, raw output in `logs/2026-09-30-mtp-depth/`):

| | Depth 2 | Depth 3 | Depth 4 |
|---|---:|---:|---:|
| Code, temperature 0 | 63.7 | 72.2 | 75.1 |
| Prose, temperature 0 | 52.4 | 53.4 | 51.3 |
| Code, temperature 0.7 | 61.1 | 69.6 | 72.5 |
| Code, 4 concurrent (total) | 212 | 232 | 238 |
| Prose, 4 concurrent (total) | 175 | 175 | 162 |
| Draft acceptance by position | 88 / 73% | 87 / 73 / 60% | 85 / 71 / 57 / 47% |
| Corrupted `repro.py` rounds | 0/10 | 0/10 | 0/10 |

Depth 3 stays. Depth 4's fourth draft is accepted 47% of the time, which buys 2–4% on
code and costs 4–7% on prose; depth 2 loses on code everywhere. No depth crashed with
overlapping requests or logged `xe` faults. Adopting depth 4 would need the full
40-round check in [Re-testing MTP](#re-testing-mtp) first.

### Intel runtime 26.35

`runtime-26.35/Dockerfile` is the production image with only Intel's compute runtime
(26.27 → 26.35) and IGC, the compiler that turns SYCL kernels into GPU code (2.38.2 →
2.41.5), upgraded. It's the SYCL counterpart of the Mesa update that doubled llama.cpp's
Vulkan decode (`../llamacpp-bench`). Tested on 2026-09-30 through the MTP setup above
(`MTP_IMAGE=qwen-vllm-runtime:26.35 MTP_CACHE=vllm-cache-runtime DEPTHS=3` plus the
40-round `REPRO_PLAN`); raw output in `logs/2026-09-30-runtime-26.35/`.

| Depth 3 | Stock (26.27) | Runtime 26.35 |
|---|---:|---:|
| Code / prose, temperature 0 | 72.2 / 53.4 | 71.9 / 53.1 |
| Code, temperature 0.7 | 69.6 | 69.9 |
| Code / prose, 4 concurrent (total) | 232 / 175 | 230 / 176 |
| Draft acceptance by position | 87 / 73 / 60% | 87 / 72 / 60% |
| Corrupted `repro.py` rounds | | 0/40 |

No speed change on one card: vLLM's decode here is set by its own kernels, not by the
runtime or compiler version. It did fix TP=2, so since 2026-10-01 every vLLM container
runs this image (next section).

### Tensor parallel (TP=2)

`vllm-tp2` splits the model across both cards, which exchange partial results through
oneCCL all-reduces, about two per layer per step. On the stock image (runtime 26.27 /
IGC 2.38.2), oneCCL's SYCL all-reduce kernels hang both cards: an `xe` job timeout on
both at once, then GT resets and `UR_RESULT_ERROR_DEVICE_LOST`. The fallback path
(`CCL_ENABLE_SYCL_KERNELS=0`) works but goes through the host, which made TP=2 no
faster than one card. With runtime 26.35 / IGC 2.41.5 the SYCL kernels are stable, so
the bug was in Intel's compute runtime or compiler (the image upgrades both, so it
doesn't say which), not in oneCCL or the PCIe link. Measured 2026-10-01 with both cards
on PCIe Gen5 x8; raw output in `logs/2026-10-01-tp2-runtime-26.35/`.

`linkbench.py` (two-card all-reduce through torch's `xccl` backend):

| All-reduce size | Stock, SYCL kernels | 26.35, SYCL kernels | Fallback, either image |
|---|---:|---:|---:|
| 160 KiB (one decode step) | 15 us | 13 us | ~100 us |
| 10 MiB | 26.5 GB/s | 26.8 GB/s | 4.3 GB/s |
| 160 MiB | hangs both cards | 28.0 GB/s | 3.0 GB/s |

The SYCL kernels run at the link's limit: each card copies 28.7 GB/s to and from the
host. 26.35 passed three runs at 50x the iterations with no `xe` faults; the stock
image hung again straight after.

vLLM, temperature 0 (`bench.py`; prefill with `prefill.py`, which defeats the prefix
cache; load is total completion tokens over wall time with `--max-num-seqs=64`):

| | One card | TP=2, stock, fallback | TP=2, 26.35 |
|---|---:|---:|---:|
| Code / prose, 1 request | 71.3 / 54.3 | 69.5 / 52.0 | **109.0 / 82.9** |
| Code / prose, 4 at once (total) | 234 / 178 | 236 / 178 | **388 / 294** |
| Prefill, 4k / 17k / 63k-token prompt | 1506 / 1362 / 981 | 1449 / 1344 / 1142 | **2603 / 2369 / 1775** |
| Total at 16 / 32 concurrent | 273 / 278 | 410 / 509 | **669 / 843** |

Two single cards together give about twice the one-card column (~556 at 32, not
measured together), so TP=2 on 26.35 beats them at every load. `repro.py` found 0 of
40 rounds corrupted and 3 concurrent requests ran cleanly. Gen4 vs Gen5 made no
difference to one card (~49 tok/s at temperature 0.7 either way).

## Monitoring

The Grafana dashboard covers token throughput, per-request decode speed, time
to first token, prompt sizes, prefix-cache hit rate, MTP acceptance by draft
position, and GPU power, clocks, temperature, VRAM and memory bandwidth. The
`xe` driver doesn't report GPU utilization, so memory read bandwidth serves as
the load indicator.

The dashboard JSON is generated. Edit `monitoring/gen_dashboard.py`, then run:

```bash
python3 monitoring/gen_dashboard.py
```

Grafana reloads the dashboard within ~10 s.
