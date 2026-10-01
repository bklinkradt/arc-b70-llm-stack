# llamacpp-bench

A throwaway experiment: llama.cpp's **Vulkan** backend (on two Mesa versions) against
its **SYCL** backend on one Arc Pro B70, and both against the production vLLM setup. Everything else in this
repo runs on SYCL / Level Zero (vLLM XPU, PyTorch XPU), and Vulkan had never been
tried. llama.cpp is the one engine that runs both backends on the same build.

This directory is not part of the stack. `../compose.yaml` doesn't include it, it joins
no shared network, it binds to 127.0.0.1 only, and nothing in the gateway points at it.

## Result (2026-09-30): Mesa decides it

**On Mesa 26.2, Vulkan beats SYCL at decode on Qwen3.8-27B, and vLLM still beats
both.** The 2026-09-29 run below found SYCL 1.4x faster, but its Vulkan image carried
Mesa 26.0.8. The same llama.cpp commit built on Fedora 44 with Mesa 26.2.3
(`vulkan-mesa26/Dockerfile`) decodes 1.58x faster on the 27B and 2.05x faster on the
dense 8B control, with nothing else changed. SYCL, re-run the same day, matched its
09-29 numbers within 0.1 tok/s. Mesa is Vulkan's userspace driver (ANV) and shader
compiler, and it comes from the container image, not the host.

- **Decode:** Vulkan leads by 10% without MTP and by 25% with it (45 / 33 tok/s code /
  prose against SYCL's 36 / 27).
- **Prompt processing:** SYCL still leads, and more so on long prompts (745 vs 597 tok/s
  at 4k tokens). On the 8B, SYCL leads on everything.
- **Serving:** vLLM + MTP (~68–72 / ~50–53) is still about 1.5x the best llama.cpp
  figure. Nothing here suggests changing the serving stack.

This matches a 2026-09-30 video (RepoChad, "AMD R9700 vs Intel Arc Pro B70 for Local
AI") that reported Mesa 26.1 roughly doubling Vulkan decode on the B70 for a MoE model.
Here the gain also shows on dense models. Anything benchmarked on Vulkan before Mesa
26.1 is stale.

### Qwen3.8-27B, unsloth `UD-Q4_K_M` GGUF (15.3 GiB), all runs on 2026-09-30

| | Vulkan, Mesa 26.0.8 | Vulkan, Mesa 26.2.3 | SYCL | vLLM (production, INT4) |
|---|---:|---:|---:|---:|
| llama-bench pp512 (tok/s) | 662 | 718 | **750** | |
| llama-bench pp4096 (tok/s) | 555 | 597 | **745** | |
| llama-bench tg128 (tok/s) | 14.5 | **22.9** | 20.9 | |
| bench.py code / prose, MTP off | 14.9 / 15.0 | **22.5 / 22.5** | 20.5 / 20.5 | ~31 |
| bench.py code / prose, MTP 3 drafts | 24.5 / 17.9 | **45.1 / 33.3** | 36.0 / 26.8 | ~68–72 / ~50–53 |
| MTP draft acceptance | 57% | 58% | 59% | |

The Mesa 26.0.8 `bench.py` rows are from 09-29; its llama-bench rows were re-run on
09-30 and landed within 2% of the 09-29 figures. `bench.py` figures are 3 runs at
temperature 0, single request. The vLLM column is from `../qwen-vllm/README.md`; it
wasn't re-run, because that card was serving users. Output on Mesa 26.2 at temperature
0 was coherent, and no `xe` faults were logged.

### Qwen3-8B `Q4_K_M` (control, dense)

| | Vulkan, Mesa 26.0.8 | Vulkan, Mesa 26.2.3 | SYCL |
|---|---:|---:|---:|
| llama-bench pp512 | 2053 | 2309 | **3216** |
| llama-bench pp4096 | 1139 | 1214 | **3078** |
| llama-bench tg128 | 40.1 | 82.2 | **89.6** |

### History: the 2026-09-29 result on Mesa 26.0.8

The first run concluded that SYCL wins, 1.4x at 27B decode and 2.2x on the 8B, and that
Vulkan's showing wasn't down to a missing driver feature, since Mesa 26.0.8 ANV already
reports fp16, bf16, integer dot product and `KHR_cooperative_matrix` on the B70. The
features were there; the older Mesa just generated slower code for them. Both backends
offload all 66 layers of the 27B model, GDN layers included. The only CPU-side buffer
is the 682 MiB token-embedding table, which is normal for llama.cpp.

### Versions

All three images are the same llama.cpp build: 0.5.0-dev b11243, commit `fc07d781e`.

- `full-vulkan`: Ubuntu 26.04, Mesa 26.0.8 ANV, Vulkan loader 1.4.341.
- `llamacpp-vulkan-mesa26` (built locally from `vulkan-mesa26/Dockerfile`): Fedora 44,
  Mesa 26.2.3 ANV. `docker compose build llama-vulkan-mesa26` rebuilds it; a rebuild
  picks up whatever Mesa Fedora 44 ships at the time.
- `full-intel` (SYCL): oneAPI 2026.1.1, compute-runtime 26.18.38308, IGC 2.34.4, Level Zero 1.32.

The ghcr digests are pinned in `docker-compose.yml`.

## Gemma garbled output on SYCL (2026-09-30)

The same video repeated a report of llama.cpp's SYCL backend garbling Gemma output on
Battlemage, even with the workaround flag set, and said it found no actual fix. The flag
at this commit is `GGML_SYCL_ENABLE_OPT` (default 1). Setting it to 0 turns off the
weight reorder, which covers Q4_0, Q8_0 and Q2_K through Q6_K and also runs for MTP draft
verification (batches of up to 8 tokens).

**Not reproduced.** Gemma 4 12B, two quants, three setups each, temperature 0: no
corrupted replies and no `xe` faults.

| Quant | Setup | Short prompts | Long, fresh prefix | Long, cached prefix |
|---|---|---:|---:|---:|
| ggml-org `Q4_0` | Vulkan, Mesa 26.2.3 (control) | 0/6 | 0/5 | 0/5 |
| | SYCL, `GGML_SYCL_ENABLE_OPT=1` | 0/6 | 0/5 | 0/5 |
| | SYCL, `GGML_SYCL_ENABLE_OPT=0` | 0/6 | 0/5 | 0/5 |
| unsloth `Q4_K_M` | Vulkan, Mesa 26.2.3 (control) | 0/6 | 0/5 | 0/5 |
| | SYCL, `GGML_SYCL_ENABLE_OPT=1` | 0/6 | 0/5 | 0/5 |
| | SYCL, `GGML_SYCL_ENABLE_OPT=0` | 0/6 | 0/5 | 0/5 |

- **Short prompts** cover code, bash, prose, arithmetic, SQL and German. The long rounds
  are `../qwen-vllm/repro.py`: ~23–25k-token multi-turn coding prompts, 700 tokens out.
- **Text differs between backends but stays coherent.** Compared with Vulkan's answers
  (difflib ratio), SYCL with the reorder off matched exactly on 4 of 6 Q4_0 prompts;
  with it on, answers drifted more (0.57–0.99). Every low-scoring pair was read: each
  took a different but sensible path from some token on. That is numerical difference,
  not corruption.
- **SYCL processes long Gemma prompts 5–6x faster than Vulkan.** A fresh ~23k-token
  round took 43–54 s on SYCL and 250–293 s on Vulkan, at the same ~22 tok/s decode.
  Vulkan's new decode speed doesn't help agent-style use on Gemma.
- **Limits:** the 12B only (the 31B shares its architecture but not its tensor shapes),
  one llama.cpp build, 16 replies per setup. It shows the bug isn't common on this build,
  not that it's fixed.

`garble-test.sh` reruns the whole matrix; outputs, server logs and the summary are in
`logs/2026-09-30-gemma-garble/`. It found no case to check, but the corruption check in
`repro.py` now also flags leaked special tokens and a word or phrase looping 15+ times
on one line.

## Two cards: llama.cpp split modes (2026-09-29)

vLLM's TP=2 needs oneCCL, whose SYCL all-reduce kernels hung both cards on the stock
image's Intel runtime. (Fixed by runtime 26.35 on 2026-10-01, after these runs: vLLM TP=2
now decodes ~1.5x faster than one card. See `../qwen-vllm/README.md`.) llama.cpp drives both cards from one process and
copies between them itself, so the question was whether its split modes do better.
These runs use the SYCL backend with both cards, and chat was stopped while they ran.

| `--split-mode` | pp512 | pp4096 | tg128 | Aggregate decode at 1 / 4 / 8 / 16 parallel |
|---|---:|---:|---:|---|
| `none` (card 0 only) | 743 | 738 | 20.9 | 9.7 / 39 / 45 / 33 |
| `layer` (pipeline) | 740 | 740 | 21.4 | 9.8 / 39 / 44 / 33 |
| `row` | crashes at load | | | |
| `tensor` (experimental) | 989 | 966 | **29.3** | 11.2 / 64 / **81** / 60 |

The first three columns come from `llama-bench`. Aggregate decode is `llama-batched-bench`
(S_TG, 512-token prompts, 128 generated per sequence). Its 1-sequence row sits well
below `llama-bench`'s tg128, so read those columns only against each other.

- **`tensor` works, and it's the only mode that helps.** It gives 1.4x single-stream
  decode, 1.3x prompt processing, and 1.8x peak aggregate decode. No GPU hangs or `xe`
  resets were logged during any run.
- **`layer` changes nothing.** Each token runs through both cards in turn, and the
  model already fits on one card.
- **`row` is broken in this build for every model.** It fails with `pre-allocated tensor
  (blk.0.attn_norm.weight) in a buffer (SYCL_Split) that cannot run the operation`,
  on the dense Qwen3-8B too, and with flash attention on or off.
- **It still doesn't beat vLLM.** Two-card llama.cpp decodes about as fast as one-card
  vLLM without MTP (29 vs ~31 tok/s), and one-card vLLM with MTP reaches ~68. At load,
  llama.cpp peaks at 81 tok/s aggregate at 8 sequences and then falls. vLLM measured
  ~370 tok/s on one card and 627 with TP=2 (with the oneCCL fallback, before the fix).
  The two-instance vLLM setup stays the right choice.

### `tensor` + MTP vs vLLM TP=2 (`bench.py`, temperature 0)

| | Code, 1 request | Prose, 1 request | Code, 4 at once (total) | Prose, 4 at once (total) |
|---|---:|---:|---:|---:|
| llama.cpp SYCL `tensor`, 2 cards | 28.6 | 28.5 | hung | not run |
| llama.cpp SYCL `tensor` + MTP, 2 cards | 56.6 | 42.0 | not run | not run |
| vLLM + MTP, 1 card (production) | 71.6 | 53.3 | 232 | 176 |
| vLLM TP=2 + MTP, 2 cards | 69.9 | 52.6 | 235 | 176 |

- **`tensor` hangs under concurrent requests.** Through `llama-server` with 4 parallel
  slots, card 0's copy engine faulted (`xe: Engine memory CAT error, class=bcs`, then
  an engine reset and a devcoredump; kernel log in
  `logs/2026-09-29-tensor-copy-engine-fault.log`). The server stopped making progress, so the
  llama.cpp runs with MTP were single-request only. `llama-batched-bench` hadn't
  triggered this. It's a different path from vLLM's oneCCL hang, but the same kind of
  failure: a cross-card copy faults on the copy engine.
- **With MTP, `tensor` reaches 79% of one-card vLLM** (56.6 vs 71.6 on code), using
  twice the hardware. Draft acceptance was 59%, the same as on one card.
- **vLLM TP=2 matched one-card vLLM within 3% at 1 and 4 requests**, with no GPU faults.
  That fits the 2026-09-28 finding that TP=2 only pays off at 16+ concurrent requests.

Chart of every run: `b70-decode-runs.html` (published as an artifact).

## Running it again

The experiment borrows the second card (PCI 08:00.0) from llama-swap, the same way
`make mtp` does. Card 0's vLLM keeps serving chat, whatever the GPU layout. Image generation is unavailable until
you give the card back. The containers get only that card's device nodes, so neither
backend can see card 0 or the AMD iGPU.

```bash
make -C llamacpp-bench take-card            # stops llama-swap and card 1's containers; runs qwen-vllm
make -C llamacpp-bench devices-vulkan devices-sycl
make -C llamacpp-bench llama-bench-vulkan   # raw numbers; no server running
make -C llamacpp-bench llama-bench-vulkan-mesa26
make -C llamacpp-bench llama-bench-sycl
make -C llamacpp-bench sycl                 # server on 127.0.0.1:8091 (vulkan: 8090, vulkan-mesa26: 8093)
make -C llamacpp-bench bench-sycl args="3 code 1 0"
make -C llamacpp-bench sycl SPEC=draft-mtp  # same, with MTP
./llamacpp-bench/garble-test.sh             # Gemma check, both quants x 3 setups; ~1.5 h
make -C llamacpp-bench release-card         # stops the experiment, restarts llama-swap

# Split modes need both cards, so all chat is down in between:
make -C llamacpp-bench take-both            # also stops qwen-vllm
make -C llamacpp-bench split-bench          # SPLITS="none layer tensor" by default
make -C llamacpp-bench sycl-2card SPEC=draft-mtp   # 2-card server on :8092; keep PARALLEL=1
make -C llamacpp-bench bench-sycl-2card args="3 code 1 0"
make -C llamacpp-bench release-both         # restarts llama-swap, which restores the layout
```

Only one backend runs at a time, because they share the card. `MODEL=<path under
$MODELS_DIR>` (default `~/models`) picks another GGUF; the default is
`Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf`, downloaded from
`unsloth/Qwen3.8-27B-GGUF`. `ALIAS=` sets the server's model name, and `SYCL_OPT=0` sets
`GGML_SYCL_ENABLE_OPT=0` on the one-card SYCL server. `bench-*` runs `../qwen-vllm/bench.py` with `BENCH_URL`
pointed at the experiment server.

Pass server flags as separate list items in `docker-compose.yml`. llama.cpp rewrites `_`
to `-` inside `--flag=value`, so `--model=…Q4_K_M.gguf` fails.
