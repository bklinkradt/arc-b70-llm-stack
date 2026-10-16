# Gemma 4 31B reviewer (llama.cpp SYCL)

Gemma 4 31B on the second B70, to review what Qwen3.8-27B (`../qwen-vllm`) writes. A
model from another family has other blind spots than the one that wrote the code.

```bash
hf download unsloth/gemma-4-31B-it-GGUF gemma-4-31B-it-Q4_K_M.gguf mtp-gemma-4-31B-it.gguf \
  --local-dir ~/models/gemma-4-31B-it-GGUF
make up                  # from the repo root: creates the container, stopped
make mode M=review       # Qwen on card 0, Gemma on card 1
```

Only the `review` layout runs it. Qwen drops to one card there (no TP=2: ~71 instead
of ~109 tok/s, 128k context), and an image request takes card 1 from Gemma until
qwen-image unloads.

- API: `http://localhost:8003/v1` (OpenAI) and `/v1/messages` (Anthropic), model
  `gemma-4-31B`, no key. Remote access goes through `../gateway` with an API key.
- `claude-review` (`../scripts`) runs Claude Code on it, like `claude-local` on Qwen.

## Settings

Measured 2026-10-02 (`../qwen-vllm/bench.py` / `prefill.py`, temperature 0): **47 tok/s**
decode on code, **~620 tok/s** prefill at 4k-16k tokens, so a 40k-token review prompt
takes about a minute before the answer starts. Card 1 holds 27.2 of 32 GB with the full
128k context allocated. `claude-review` on a planted path traversal and SQL injection
found both, with line numbers and correct fixes, in 32 s.

- **llama.cpp b11669** (`620d2225c`, since 2026-10-16), the build `../gpt-oss` and
  `../llamacpp-bench` use. Against b11243 on the same flags it measured the same:
  46.9 / 38.0 vs 46.2 / 37.7 tok/s decode (code / prose), 591 / 24.9 vs 585 / 25.0
  prefill / decode at ~43k tokens, 84-87% vs 83-90% MTP draft acceptance, 0/3
  `repro.py` rounds corrupted on both, and tool calls through `/v1/messages` on both.
  `GEMMA_IMAGE=` runs another build.
- **Q4_K_M** (unsloth, 18.3 GB): the quant type that tested clean on SYCL with the 12B
  (`../llamacpp-bench`). The 31B itself hasn't had the garble check.
- **MTP** with Gemma's separate draft head (`mtp-gemma-4-31B-it.gguf`), 3 drafts.
- **128k context, q8_0 KV cache**, two slots sharing it (`--kv-unified`): a review
  session's main agent and one subagent.
- llama.cpp rather than vLLM: SYCL is the backend Gemma was tested on here, and it
  processes long prompts 5-6x faster than Vulkan. It listens on port 8000 inside the
  container, as vLLM did.
- **It ran on vLLM 2026-10-11..12** (Google's QAT INT4, flags in git `cb0bbfd`, bake-off
  notes below). In the harness that saved ~15 s per security stage and nothing visible
  per review stage (the Qwen critic dominates), while the engine segfaulted once in ~6
  layout-switch starts (right after loading weights, so a run's review waited out its
  timeout) and the 49k context forced a 16k output cap. Back on llama.cpp for stability.

## Reviewer bake-off (2026-10-02, 2026-10-11)

One diff (a Flask notes service, +105 lines) with 10 planted bugs: inverted session
expiry, hardcoded secret fallback, pagination off-by-one, SQL injection, quota race,
delete without owner check, upload path traversal, stored XSS, empty-default admin token,
`debug=True` on 0.0.0.0. Each reviewer ran twice through Claude Code with the same
prompt (`claude-review`, or `claude-local` for Qwen). Score = planted bugs found.

| Reviewer | Found (run 1 / 2) | False alarms | Time per review | Decode / prefill (tok/s) |
|---|---:|---:|---:|---:|
| Gemma 4 31B, Q4_K_M (llama.cpp) | 8 / 6.5 | 0 | 119 / 80 s | 47 / ~620 |
| Gemma 4 26B-A4B (MoE), Google QAT Q4_0 | 4 / 7 | 0 | 219 / 72 s | 78 / ~1500 |
| GLM-4.7-Flash (Z.ai MoE), Q4_K_M | 6 / 4 | 1 (CSRF) | 193 / 92 s | 40 / 562 at 4k, 203 at 17k |
| gpt-oss-20b (OpenAI MoE), MXFP4 + EAGLE-3 draft | 8 / 6 | 1 (CSRF) | 166 / 191 s | 44 / ~1050 |
| Qwen3.8-27B INT4, one card (vLLM) | 9 / 8 | 0 | 484 / 133 s | ~71 / ~2400 |
| Nemotron 3.5 Lightning 30B-A3B (NVIDIA hybrid MoE), Q4_0 + MTP | 7 / 5 | 0 | 95 / 50 s | 83 / 1100 at 3k, 1900 at 12k |
| Devstral Small 2 24B (Mistral), Q4_K_M | - | - | - | 33 / ~1000 |
| Gemma 4 31B, Google QAT INT4 on vLLM, flash attention + MTP | 6 / 7.5 | 1 (IDOR on GET) | 68 / 63 s | 51 / ~1050-1250, ~690 at 40k |

- **The 31B stays, on llama.cpp** (vLLM was quicker but less stable, see Settings and the
  vLLM note below). The 26B-A4B reads 2.5x faster but swung from 4 to 7 between runs,
  and its second run's line numbers were off by up to 16. Switch with `GEMMA_MODEL`,
  `GEMMA_DRAFT`, `GEMMA_ALIAS` (see docker-compose.yml) to retry it.
- **GLM-4.7-Flash and gpt-oss-20b don't beat it** (2026-10-03, same llama.cpp build, 128k
  context, q8_0 KV, on card 1). GLM's prompt speed collapses with length on SYCL and it
  found the fewest. gpt-oss matched the 31B's first run but took longer per review and
  decodes slowly here (MXFP4 on SYCL). Both flagged CSRF, which doesn't apply to a
  bearer-token API.
- **Nemotron 3.5 Lightning is the fastest reviewer that works** (2026-10-11): 1.6-2x
  quicker per review than the 31B, but it found about one bug fewer per run, and its second
  run's line numbers were off by up to 22. It missed the XSS both times.
- **Devstral Small 2 doesn't work with Claude Code** (2026-10-11). Its template rejects
  the extra system messages Claude Code sends; a patched template fixes that, but on
  Claude Code's long prompt it drops the `[TOOL_CALLS]` token and writes tool calls as
  text, so no tool runs. Tool calls parse on short prompts.
- **Gemma on vLLM reviews ~1.5x faster than on llama.cpp, with ~40% of the context**
  (2026-10-11, `qwen-vllm-runtime:26.35`, `google/gemma-4-31B-it-qat-w4a16-ct`):
  - vLLM forces Triton attention for Gemma 4 (head dims 256 and 512 differ, no FA4 on
    XPU): ~150-170 tok/s prefill, 16.5 tok/s decode. `--attention-backend FLASH_ATTN`
    skips that and runs fine on the XPU kernels: ~1050-1250 tok/s prefill at 3-12k.
  - MTP with Google's draft (`google/gemma-4-31B-it-qat-q4_0-unquantized-assistant`,
    0.9 GB, `--speculative-config '{"model":...,"num_speculative_tokens":3}'`): 23 to
    51 tok/s decode, 61% of drafts accepted.
  - Only 49k context fits (fp8 KV, `--gpu-memory-utilization 0.96`, 59k tokens of KV):
    weights and the draft take 22.6 of 32 GB. The 31B Q4_K_M on llama.cpp holds 128k
    for two slots.
  - `repro.py`: 0/5 rounds corrupted at 40-43k tokens. Found as much as the llama.cpp
    31B within the noise; run 2 raised one false IDOR on a route that checks the owner.
- **Qwen found the most** and gave exact lines, but this code wasn't Qwen's: the test
  can't show whether it would miss its own mistakes, which is the case for a second
  family. It was also the slowest per review.
- **No reviewer caught the quota race** (count, then insert). Concurrency bugs need a
  stronger reviewer or a test.
- Two runs on one diff: a small sample. Treat the gaps between 6 and 9 as rough.

