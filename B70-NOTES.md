# Arc Pro B70 pitfalls and fixes

Everything here cost time to find. Each entry gives the symptom, the cause, the fix, and
where the detail lives. Test machine: 2× Intel Arc Pro B70 (32 GB), each on PCIe Gen5 x8,
Fedora 44, kernel 7.2.7 (`xe` driver, GuC 70.72.1), vLLM v0.30.0 XPU. Findings are dated
because this stack moves fast: re-check anything older than a few months.

## Hardware and PCIe

### A card links at Gen4 even though the BIOS forces Gen5 (2026-10-01)
- **Symptom:** one card at 16 GT/s, the other at 32 GT/s. Forcing Gen5 in the BIOS
  doesn't help, and swapping slots moves the problem with the card.
- **Cause:** a setting stored on the card itself, "PCIe Gen4 Downgrade", visible in
  `sudo xpu-smi config -d <id>`. Reading it needs root; without root it shows N/A.
- **Fix:** `sudo xpu-smi config -d <id> --pciedowngrade 0`, then a full power-off. A
  reboot isn't enough.

### `lspci` says the GPU is at 2.5 GT/s x1
- **Cause:** the GPU sits behind a PCIe switch on the card (`8086:e2f0` / `8086:e2ff`),
  and its own link is virtual. The real link is on the card's upstream bridge (the
  `e2ff` device) and on the CPU root port above it.
- **Check:** `cat /sys/bus/pci/devices/<bridge>/current_link_{speed,width}`, or look for
  the kernel line "available PCIe bandwidth" in `dmesg`.
- **Does Gen4 vs Gen5 matter?** Not for single-card decode (~49 tok/s either way). It
  does for TP=2: the all-reduce runs at ~28 GB/s, the Gen5 x8 copy limit.

## vLLM on XPU

### TP=2 hangs both cards (fixed 2026-10-01)
- **Symptom:** `--tensor-parallel-size=2` freezes, the kernel logs `xe ... Check job
  timeout ... not started` on both cards at the same moment, then GT resets,
  devcoredumps and `UR_RESULT_ERROR_DEVICE_LOST`.
- **Cause:** oneCCL's SYCL all-reduce kernels on Intel compute runtime 26.27 / IGC 2.38.2,
  the versions in `vllm/vllm-openai-xpu:v0.30.0`. Not oneCCL itself, and not PCIe.
- **Workaround that loses the gain:** `CCL_ENABLE_SYCL_KERNELS=0` is stable but
  host-staged (~4 GB/s, ~100 µs per decode all-reduce), so TP=2 is no faster than one card.
- **Fix:** upgrade to compute runtime 26.35 + IGC 2.41.5
  ([`qwen-vllm/runtime-26.35/Dockerfile`](qwen-vllm/runtime-26.35/Dockerfile), a plain
  `dpkg -i` on top of the upstream image). TP=2 then decodes at 109 vs 71 tok/s and
  prefills ~1.75x faster. Details:
  [qwen-vllm/README.md](qwen-vllm/README.md#tensor-parallel-tp2). Measure your own cards
  with [`qwen-vllm/linkbench.py`](qwen-vllm/linkbench.py).

### Long MTP generations turn into gibberish (fixed 2026-09-26)
- **Symptom:** with MTP speculative decoding and prefix caching both on, about 1 in 5
  long replies degenerate into newline floods or gibberish once they cross a 2048-token
  cache block.
- **Cause:** the GDN conv-state kernels in Intel's `intel/llm-scaler-vllm:0.26.0-b2`
  lack vllm-xpu-kernels #544/#545.
- **Fix:** upstream `vllm/vllm-openai-xpu:v0.30.0` (vllm-xpu-kernels 0.1.14.1). Check any
  image with [`qwen-vllm/repro.py`](qwen-vllm/repro.py), whose ~21k-token agent-style
  prompts catch this and short benchmarks don't. The analysis is in
  [`qwen-vllm/docs/intel-xpu-stack.html`](qwen-vllm/docs/intel-xpu-stack.html).

### `oneCCL: ... ze_data was not initialized`
- **Cause:** oneCCL `dlopen`s the unversioned `libze_loader.so`, which only the
  `libze-dev` package provides. vLLM initialises oneCCL even on one GPU.
- **Fix:** keep `libze-dev` installed in custom images.

### oneCCL can't find the GPUs in Docker
- **Cause:** oneCCL scans `/dev/dri/by-path`, and device passthrough doesn't create
  those symlinks.
- **Fix:** pass `/dev/dri` and mount `/dev/dri/by-path:/dev/dri/by-path:ro`
  ([`qwen-vllm/docker-compose.yml`](qwen-vllm/docker-compose.yml)).

### Picking a card
- Level Zero lists only Intel GPUs, in PCI order, so an AMD or other iGPU never shifts
  the numbering. `ZE_AFFINITY_MASK=0`, `1` or `0,1` picks cards.
- Vulkan ignores `ZE_AFFINITY_MASK`; to hide a card from Vulkan, pass only its
  `/dev/dri/by-path/pci-…` nodes.

### Settings that didn't help
- **XPU graphs:** no decode gain, 2.5 GiB of VRAM vLLM doesn't budget for, and with MTP
  they pad state slots in a way the SYCL GDN kernels don't handle. Leave
  `VLLM_XPU_ENABLE_XPU_GRAPH=0`.
- **fp8 KV cache and fp16 dtype:** no meaningful gain.
- **Compute runtime 26.35 on one card:** same speed as 26.27. It only matters for TP=2.

### MTP draft depth
3 drafts is best for Qwen3.8-27B: ~68–72 tok/s on code and ~50–53 on prose, vs ~31
without MTP. Depth 4 gains 2–4% on code and loses 4–7% on prose.
[Sweep](qwen-vllm/README.md#mtp-draft-depth).

### Benchmark at temperature 0
With MTP, sampling at 0.7 changes how many drafts are accepted, which swings decode by
20+ tok/s between runs. At temperature 0, runs repeat within ±1.5 tok/s.
[`qwen-vllm/bench.py`](qwen-vllm/bench.py) takes the temperature as an argument.

## llama.cpp on the B70 (2026-09-29/30)

- **Mesa version decides Vulkan speed.** On Mesa 26.2 Vulkan decodes 1.58x faster than
  on Mesa 26.0.8 for the 27B, and beats SYCL at decode. SYCL still wins prompt
  processing. Mesa comes from the container image, not the host. Treat any Vulkan
  numbers from before Mesa 26.1 as stale.
- **`--split-mode row` is broken** on SYCL for every model tried (`pre-allocated tensor
  ... in a buffer (SYCL_Split) that cannot run the operation`).
- **`--split-mode tensor` faults the copy engine** under concurrent requests
  (`xe: Engine memory CAT error, class=bcs`).
- **vLLM is still ~1.5x the best llama.cpp figure** for serving this model.

Details: [llamacpp-bench/README.md](llamacpp-bench/README.md).
