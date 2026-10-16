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
- **64k context, q8_0 KV cache, 4 slots sharing it** (`--kv-unified`). The weights
  leave ~4.5 GiB of the two cards' 64 GB; 128k with 4 slots ran out of VRAM at
  warm-up (`UR_RESULT_ERROR_OUT_OF_RESOURCES`).
- No speculative decoding: the GGUF has no draft head. An EAGLE-3 draft exists
  (`eagle3-gpt-oss-120b-Q8_0.gguf`, 0.8 GiB) but is untested here.

## Speed (2026-10-16)

Measured on this container as configured (layer split, 64k context, 4 slots), with
`../qwen-vllm/bench.py` and `repro.py` at temperature 0:

| | gpt-oss-120b, both cards | Qwen3.8-27B, vLLM TP=2 + MTP |
|---|---:|---:|
| Decode, one request, code / prose (tok/s) | 87.7 / 87.9 | 109 / 83 |
| Decode at 4 / 8 requests, each (total) | 28 (113) / 28 (222) | |
| ~36k-token prompt: prefill / decode (tok/s) | 390 / 48.7 | |

- `repro.py` (~36k-token agent prompts) came back clean in content. One round of three
  was flagged because gpt-oss reasons even with `enable_thinking: false`: when the
  700-token cap goes to reasoning, the answer is empty, which the check counts as
  corrupt. A replay of the same rounds gave full answers.
- `tensor` split decodes faster on long prompts (60 tok/s at 36k, 480 prefill) but
  slower at load; see `../llamacpp-bench/README.md` for both modes and the other
  models tried (GLM-4.5-Air, Qwen3.5-122B-A10B).
- An image request in `big` took 60 s end to end, including the swap to `qwen-vllm` on
  card 0 and qwen-image on card 1.
