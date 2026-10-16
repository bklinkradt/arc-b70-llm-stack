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

All three images were the same llama.cpp build for these runs: 0.5.0-dev b11243, commit
`fc07d781e`. Since 2026-10-16 `docker-compose.yml` pins b11669 (`620d2225c`) for SYCL
and Mesa 26.2 Vulkan (`VULKAN_COMMIT=fc07d781e` rebuilds the old one); `full-vulkan`
stays on b11243.

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

## ~100-120B MoE on both cards (2026-10-15)

Can two B70s (64 GB) run a model bigger than Qwen3.8-27B at a usable speed? Three MoE
models in the 106-122B class, each as a GGUF that fits the two cards' VRAM, on SYCL with
both cards in one process. The pinned llama.cpp build loads all three architectures.

**On b11243 (below) none of them was close to production; on b11669 gpt-oss-120b is.**
With the pinned build, Qwen3.5-122B-A10B decoded at 31 tok/s (23 at 41k tokens),
gpt-oss-120b at 36 (28), and GLM-4.5-Air at ~22 (10). vLLM TP=2 serves Qwen3.8-27B at
109 tok/s. On the newer b11669 (next section) gpt-oss-120b reaches 81 tok/s and 60 at
36k tokens, and Qwen3.5-122B 40 / 32, both with `tensor`.

| | GLM-4.5-Air | Qwen3.5-122B-A10B | gpt-oss-120b |
|---|---:|---:|---:|
| Total / active params | 106B / 12B | 122B / 10B | 117B / 5.1B |
| GGUF (all on GPU) | unsloth `UD-Q3_K_XL`, 51 GiB | unsloth `UD-Q3_K_XL`, 53 GiB | ggml-org `MXFP4`, 59 GiB |
| llama-bench pp512 / pp4096 | 260 / 256 | 409 / 399 | **516 / 506** |
| llama-bench tg128 | 22.2 | 32.3 | **37.5** |
| Aggregate decode at 1 / 4 / 8 / 16 parallel | 21 / 24 / 29 / 33 | 29 / 43 / 48 / 44 | 34 / 58 / **80** / 67 |
| bench.py code / prose | ~22 | 31.3 / 31.4 | **36.2 / 36.3** |
| `repro.py` ~36-42k tokens: prefill / decode | 192 / 10.0 | 275 / 23.0 | **280 / 27.7** |
| `repro.py` corrupted rounds | 0/3 | 0/3 | 0/3 |

All runs use `--split-mode layer`, flash attention, and for the server a 64k context with
a q8_0 KV cache, one slot. `bench.py` is 3 runs at temperature 0. The GLM bench.py figure is
`llama-bench`'s; its bench.py run was on `tensor` (27.9). Logs are in
`logs/2026-10-15-big-moe/`. Each model load takes ~6 minutes.

- **`tensor` split corrupts long prompts.** It decodes 10-30% faster on short prompts
  (GLM 29.6, Qwen 36.5 tok/s), but `repro.py` failed 2/3 rounds on GLM (`1, 1, 1, …`
  loops) and 3/3 on Qwen (mixed-script garbage), where `layer` passed 0/3 on the same
  server flags. On Qwen it also "prefilled" at 930-980 tok/s, 3.5x `layer`, so part of
  the work is wrong or skipped. With the concurrency hang above, `tensor` is unusable in
  this build; `SPLIT` now defaults to `layer`. b11669 fixes it for these MoEs (next
  section).
- **No MTP.** llama.cpp ignores GLM-4.5-Air's built-in draft layer (`blk.46.nextn.*` is
  logged as unused), and the Qwen3.5 GGUF has none, so decode can't be doubled the way
  MTP does for the 27B.
- **gpt-oss-120b fits with nothing on the CPU** under SYCL. MXFP4 was slow on SYCL for
  gpt-oss-20b (`../gemma/README.md`), but the 120b is the fastest of the three here.

### Newer build b11669 (620d2225c, 2026-10-16): faster, and `tensor` works on the MoEs

The `full-intel` image built 2026-10-15 (sha256:9ddf4bfc…, pinned in
`docker-compose.yml` since 2026-10-16; `SYCL_IMAGE=` tries another) has the 2026-09-30 tensor all-reduce sync rework (#29604), grouped MoE
XMX GEMM (#29245) and faster MXFP4 MoE (#29809). Same flags as above.

| | Qwen3.5-122B b11243 | Qwen3.5-122B b11669 | gpt-oss-120b b11243 | gpt-oss-120b b11669 |
|---|---:|---:|---:|---:|
| pp512 / pp4096, `layer` | 409 / 399 | 664 / 666 | 516 / 506 | 883 / 835 |
| pp512 / pp4096, `tensor` | 350 / 344 | **721 / 704** | | **943 / 901** |
| tg128, `layer` / `tensor` | 32.3 / 36.5 | 34.3 / **40.3** | 37.5 / - | **95.0** / 90.7 |
| Aggregate decode at 8 parallel, `layer` / `tensor` | 48 / 47 | 75 / 80 | 80 / - | **224** / 198 |
| `repro.py` ~36-42k, `layer`: prefill / decode, corrupted | 275 / 23.0, 0/3 | 534 / 24.0, 0/3 | 280 / 27.7, 0/3 | |
| `repro.py` ~36-42k, `tensor`: prefill / decode, corrupted | 3/3 corrupt | **589 / 31.6, 0/3** | | **480 / 60.1, 0/3** |

- **gpt-oss-120b is now the fastest large model by far:** 81 tok/s in `bench.py`
  (`tensor`), 60 tok/s decode at 36k tokens, 552 tok/s aggregate at 8 parallel (`layer`).
  That's 2.5x b11243, and close to production vLLM TP=2 on the 27B (109 / 83 with MTP).
- **`tensor` passed the long-prompt check on both MoEs** (0/3 each), and no `xe` faults
  were logged. On b11243 the same runs were 3/3 garbage (Qwen3.5) or a copy-engine fault
  and hang (Qwen3.8-27B, `Engine memory CAT error … class=bcs`, as on 2026-09-29).
- **GLM-4.5-Air gained little on b11669** (Q3_K experts miss the new MoE kernels):
  pp4096 296 / 286 (`layer` / `tensor`, from 256 / 261), tg128 22.1 / 30.3 (unchanged).
  `tensor` now passes `repro.py` too (0/3, from 2/3), at 209 / 16.3 tok/s prefill /
  decode at ~37k tokens; `layer` 211 / 10.0. `bench.py` 21.3 / 28.5.
- **Qwen3.8-27B GGUF is garbage at ~41k tokens on b11669 in both split modes**
  (`3333…`, 2/2 each, no faults), so it isn't tensor-specific. Not investigated: the 27B
  is served by vLLM, and this GGUF was never long-prompt tested on b11243 either.
- **Vulkan at the same commit still collapses on long prompts.** Qwen3.5-122B `layer`:
  pp4096 750, tg128 35.8 ± 4.4, `bench.py` 30.2 / 30.7, but 7.4-9.6 tok/s decode at 41k
  tokens (prefill 269-297), against SYCL's 24-32. PR #29476 (Intel GDN kernel tuning) is
  in this build; it helped short prompts, not long-context decode.

### vLLM TP=2: none of them fits

GLM-4.5-Air on vLLM was tried with `cyankiwi/GLM-4.5-Air-AWQ-4bit`: compressed-tensors
INT4, symmetric, group 32, which is the scheme vLLM v0.30's XPU MoE kernel takes
(`XPUExpertsWNA16`, `kInt4Static32`), with the `glm45` reasoning and tool parsers. It
failed at startup. vLLM sees 30.3 GiB per card, and its runtime leaves 28.4-28.6 GiB
free before loading, but the 59 GiB of weights need ~29.5 GiB per card at TP=2. The
same goes for the other two at 4-bit (gpt-oss-120b MXFP4 61 GiB, Qwen3.5-122B ~65 GiB).
llama.cpp gets these models in because a 3-bit K-quant is an option and its runtime
takes less VRAM.

- `--cpu-offload-gb` (UVA, supported on XPU) would load it, but offloaded weights cross
  PCIe on every decode step: 1-2 GiB per card per token at ~25 GB/s caps decode near
  10 tok/s, which measures the offload, not vLLM. Not run.
- The only GLM-4.5-Air that would fit, the pruned REAP-82B, has no vLLM-format INT4
  checkpoint (MLX only) as of 2026-10-15.

### Vulkan (Mesa 26.2) vs SYCL, both cards, `layer`

| | Qwen3.5-122B SYCL | Qwen3.5-122B Vulkan | gpt-oss-120b SYCL | gpt-oss-120b Vulkan |
|---|---:|---:|---:|---:|
| llama-bench pp512 / pp4096 | 409 / 399 | 516 / **650** | **516 / 506** | 298 / 323 |
| llama-bench tg128 | 32.3 | 31.0 ± 7.5 | 37.5 | 38.0 ± 11.6 |
| Aggregate decode at 1 / 4 / 8 / 16 parallel | 29 / 43 / 48 / 44 | 35 / **71** / 57 / 39 | 34 / 58 / 80 / 67 | 38 / 51 / 81 / 59 |
| bench.py code / prose | **31.3 / 31.4** | 29.2 / 30.1 | 36.2 / 36.3 | not run |
| `repro.py` ~41k tokens: prefill / decode | 275 / **23.0** | 268-300 / **5.3** | 280 / 27.7 | not run |

**SYCL stays the backend for long contexts.** Vulkan prefills short prompts up to 1.6x
faster on Qwen, but its decode at 41k tokens falls to 5.3 tok/s, a quarter of SYCL's,
and its decode swings by ±25% between runs. Vulkan reports only 29.3 GiB free per card
(SYCL ~31.9), so gpt-oss-120b needed 4 layers of experts in RAM (`CPU_MOE=4`), which
explains part of its lower prompt speed.

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
make -C llamacpp-bench split-bench MODEL=gpt-oss-120b-GGUF/gpt-oss-120b-MXFP4.gguf SPLITS=layer
make -C llamacpp-bench sycl-2card MODEL=... ALIAS=... CTX=65536 KV=q8_0   # SPLIT=layer default
make -C llamacpp-bench split-bench-vulkan / vulkan-2card / bench-vulkan-2card   # Vulkan, :8094
# CPU_MOE=N keeps the experts of the first N layers in system RAM, for MoEs over 64 GB.
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
