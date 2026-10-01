# Qwen on two Intel Arc Pro B70s

Self-hosted Qwen models on two Intel Arc Pro B70s: Qwen3.8-27B for chat and
Qwen-Image-2.1 for images, served to the internet through an API gateway with
per-user keys.

Getting this to work well on Intel's XPU stack took a lot of debugging: a PCIe setting
stored on the card, a tensor-parallel hang fixed by a newer compute runtime, MTP output
corruption fixed by a newer image, and more. **[B70-NOTES.md](B70-NOTES.md) lists every
pitfall with its fix**, which may be the most useful part if you run Arc GPUs yourself.

**Tested on:** 2× Arc Pro B70 (32 GB) on PCIe Gen5 x8, Fedora 44, kernel 7.2.7 (`xe`),
vLLM v0.30.0 XPU with Intel compute runtime 26.35, model Qwen3.8-27B INT4. The configs
assume two B70s and this model. Card numbers, PCI addresses, group IDs and the model
directory are variables (`MODELS_DIR`, `VIDEO_GID`, `RENDER_GID`, `IMAGE_CARD_PCI`), but
other GPUs or models will need their flags reviewed.

| Qwen3.8-27B, MTP 3 drafts, temperature 0 | One B70 | Two B70s, TP=2 |
|---|---:|---:|
| Decode, code / prose (tok/s) | 71 / 54 | 109 / 83 |
| Prefill, 63k-token prompt (tok/s) | 981 | 1775 |
| Total throughput, 32 concurrent requests (tok/s) | 278 | 843 |

| Directory | What it is |
|---|---|
| [`qwen-vllm/`](qwen-vllm/README.md) | vLLM serving Qwen3.8-27B, one instance per card or one across both (tensor parallel), plus Open WebUI, Prometheus and Grafana. Also has the benchmark and corruption-check scripts and the Intel XPU stack guide |
| [`qwen-image/`](qwen-image/README.md) | Qwen-Image-2.1 with a Gradio UI and an OpenAI-compatible images API |
| [`gateway/`](gateway/README.md) | Caddy (TLS), LiteLLM (API keys, limits, usage), the chat load balancer, and llama-swap, which places the model containers on the cards (the GPU layout) and lends the second card to images on demand |
| [`llamacpp-bench/`](llamacpp-bench/README.md) | Standalone experiment, not part of the stack: llama.cpp Vulkan vs SYCL (Vulkan decodes faster on Mesa 26.2+), a Gemma garbled-output check, its two-card split modes, and tensor split + MTP vs vLLM TP=2. vLLM stays the choice |

```
internet ─► caddy ─► litellm ─┬─► vllm-lb ───┬─► qwen-vllm      B70 #1     split layout
            TLS      API keys  │   chat        ├─► qwen-vllm-2    B70 #2     split layout
                               │               └─► qwen-vllm-tp2  B70 #1+#2  tp2 layout
                               └─► llama-swap ───► qwen-image     B70 #2     on demand
                                   images
```

`make mode M=split|tp2` picks the GPU layout: one Qwen per card, or one across both cards
(~1.5x faster). Either way an image request takes card 1, and llama-swap puts the layout
back after 10 idle minutes. See `qwen-vllm/README.md` for the trade-offs.

`compose.yaml` pulls the three compose files into one project, `local-serve`. Each part
keeps its own compose file and README, and its relative paths resolve from its own
directory, so `gateway/.env` stays in `gateway/`.

## First-time setup

Needs Docker with Compose 2.20 or newer (for `include:`) and `make`.

1. Model weights (~20 GB), [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4):
   `hf download RedHatAI/Qwen3.8-27B-INT4 --local-dir ~/models/Qwen3.8-27B-INT4`.
   Elsewhere than `~/models`? Set `MODELS_DIR`. Qwen-Image downloads its own weights
   on first use.
2. Gateway secrets: `cp gateway/.env.example gateway/.env`, fill it in, `chmod 600`.
   DNS and router steps are in `gateway/README.md`.
3. Optional: `cp qwen-image/.env.example qwen-image/.env` for `HF_TOKEN` / `QWEN_OFFLOAD`.
4. `make build && make up`

`make up` creates the shared `llm-net` network if needed, and creates the model
containers (`qwen-vllm`, `qwen-vllm-2`, `qwen-vllm-tp2`, `qwen-image`) without starting
them. They are in the `managed` profile because llama-swap decides which of them hold
the cards. Then it starts everything else, and llama-swap starts the layout's
containers. The layout is `split` until `make mode M=tp2`.

## Day to day

```bash
make ps               # what's running
make logs s=litellm   # follow one service (omit s= for all)
make restart s=caddy  # after editing a config file
make down             # stop everything; volumes (keys, certs, dashboards) are kept
make bench args="5 code 1 0"
make mode / make mode M=tp2   # show / switch the GPU layout (split, tp2)
make mtp MTP_N=4 / make mtp-down   # MTP draft-depth experiment on the second card
```

Run `docker compose` commands from this directory. Running one from a subdirectory
starts a separate compose project, and it clashes with the fixed container names.

Local endpoints: vLLM `localhost:8000` (`qwen-vllm` or `qwen-vllm-tp2`) / `:8001`, LiteLLM admin `localhost:4000/ui`,
Open WebUI `:3000`, Grafana `:3001`, qwen-image Gradio `localhost:7860`.

## Data

State lives in named Docker volumes, so `make down` and rebuilds keep it:

| Volume | Holds |
|---|---|
| `gateway_postgres-data` | LiteLLM's database: every API key, its limits and usage. Back this up |
| `gateway_caddy-data` | TLS certificates and the ACME account |
| `qwen-vllm_open-webui-data` | Open WebUI accounts, chats and uploads |
| `qwen-vllm_grafana-data`, `qwen-vllm_prometheus-data` | Dashboards state and 30 days of metrics |
| `qwen-vllm_vllm-cache-runtime*` | vLLM's compiled-kernel caches; safe to delete (the next start recompiles, ~10 min) |

The names are fixed with `name:` in each compose file, so they don't depend on the
compose project name. Never run `docker compose down -v`: it deletes them.
