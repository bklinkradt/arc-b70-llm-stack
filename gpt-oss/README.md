# gpt-oss-120b (llama.cpp SYCL, both cards)

OpenAI's gpt-oss-120b (117B MoE, 5.1B active, MXFP4) on llama.cpp across both B70s.
It's the biggest model the stack serves, and only the `big` GPU layout runs it:

```bash
make mode M=big          # gpt-oss on both cards; Qwen3.8-27B stops
make mode M=review       # or split / tp2 to go back
```

- API: `http://localhost:8004/v1` (OpenAI) and `/v1/messages` (Anthropic), model
  `gpt-oss-120b`. Through the gateway, the same model name with an API key.
- Weights: `hf download ggml-org/gpt-oss-120b-GGUF --include "gpt-oss-120b-MXFP4*"
  --local-dir ~/models/gpt-oss-120b-GGUF` (59 GiB). `GPT_OSS_MODEL` (a path under
  `MODELS_DIR`) swaps in another GGUF; `make up` recreates the container with it.
- An image request takes card 1, so gpt-oss stops and `qwen-vllm` serves chat on card 0
  until the image model unloads (10 idle minutes); then gpt-oss loads again (~5 min).
  Requests for `gpt-oss-120b` fail in between.

## Settings

- **llama.cpp b11669** (`620d2225c`), newer than Gemma's build: its MXFP4 MoE kernels
  (#29809) took decode from ~37 to ~95 tok/s, and older builds corrupt long prompts
  with `--split-mode tensor`.
- **`--split-mode layer`**: each card holds half the layers. Faster at load than
  `tensor` here (224 vs 198 tok/s aggregate at 8 requests), with less cross-card traffic.
- **128k context (the model's maximum), q8_0 KV cache, 4 slots sharing it**
  (`--kv-unified`). One request can use the whole 128k; concurrent requests split it.
  The KV cache is only 2.4 GiB at 128k, because half of gpt-oss's layers attend to the
  last 128 tokens only. A slot adds ~15 MB, so fewer slots wouldn't buy more context.
- **`--fit` places the weights, not `--n-gpu-layers 999`.** Whole layers (~1.6 GiB each)
  put 31.3 GiB of weights on card 0 and 28.6 on card 1, so 128k ran out of memory on
  card 0 at warm-up (`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`); moving one layer
  (`--tensor-split 17,19`) also ran out: the gap is less than two layers. `--fit` splits a layer's
  tensors across the cards instead: 30.8 / 31.3 GiB used, all 37 layers on the GPUs.
  It sizes to the VRAM free at start (both whole cards in `big`); if it ever comes up
  short it moves weights to system RAM, which the log shows as a `CPU_Mapped model
  buffer` well over the usual ~590 MiB.
- No speculative decoding: the GGUF has no draft head. An EAGLE-3 draft exists
  (`eagle3-gpt-oss-120b-Q8_0.gguf`, 0.8 GiB) but is untested here.

## Speed (2026-10-16)

Measured with `../qwen-vllm/bench.py` and `repro.py` at temperature 0, layer split and
4 slots. The 36k row is from the earlier 64k configuration; at 128k one and four requests
decode the same (87.2 and 28.1 tok/s). The 60k and 121k rows ran with one slot at 128k,
which has the same memory layout.

| | gpt-oss-120b, both cards | Qwen3.8-27B, vLLM TP=2 + MTP |
|---|---:|---:|
| Decode, one request, code / prose (tok/s) | 87.7 / 87.9 | 109 / 83 |
| Decode at 4 / 8 requests, each (total) | 28 (113) / 28 (222) | |
| ~36k-token prompt: prefill / decode (tok/s) | 390 / 48.7 | |
| ~60k / ~121k-token prompt: prefill (tok/s) | 318 / 220 | |
| ~60k / ~121k-token prompt: decode (tok/s) | 37.8 / 24.2 | |

- `repro.py` (~36k-token agent prompts) came back clean in content. One round of three
  was flagged because gpt-oss reasons even with `enable_thinking: false`: when the
  700-token cap goes to reasoning, the answer is empty, which the check counts as
  corrupt. A replay of the same rounds gave full answers.
- At 60k and 121k tokens it found a code phrase planted halfway through the prompt. A
  121k-token prompt takes ~9 minutes to prefill; the gateway's timeout is 30 minutes.
- `tensor` split decodes faster on long prompts (60 tok/s at 36k, 480 prefill) but
  slower at load; see `../llamacpp-bench/README.md` for both modes and the other
  models tried (GLM-4.5-Air, Qwen3.5-122B-A10B).
- An image request in `big` took 60 s end to end, including the swap to `qwen-vllm` on
  card 0 and qwen-image on card 1.
