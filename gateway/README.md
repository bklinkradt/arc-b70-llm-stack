# gateway

Internet-facing API gateway for the models in `../qwen-vllm` (Qwen3.8-27B chat) and
`../qwen-image` (Qwen-Image-2.1 images) on two Arc Pro B70s. Each person gets their own
API key, with rate limits and usage tracking.

```
internet ─► router ─► caddy ─► litellm ─┬─► vllm-lb ───┬─► qwen-vllm     :8000  B70 #1     split
            :443/:80  TLS,     API keys, │   chat        ├─► qwen-vllm-2   :8000  B70 #2     split
                      WAN sees limits,   │               └─► qwen-vllm-tp2 :8000  B70 #1+#2  tp2
                      /v1/* only postgres└─► llama-swap ───► qwen-image    :7860  B70 #2     on demand
                                             images
```

llama-swap runs the model containers that the GPU layout asks for
(`llama-swap/layouts/layouts.conf`; `make mode M=split|tp2` from the repo root):

- **split** (default): one Qwen per card. An image request makes llama-swap stop
  `qwen-vllm-2` and load qwen-image on the second card, and chat carries on on the first.
- **tp2**: one Qwen across both cards, ~1.5x faster. An image request swaps
  `qwen-vllm-tp2` for `qwen-vllm` on the first card plus qwen-image on the second. Chat
  has no backend for ~70 s at each swap; `vllm-lb` holds new requests until one is up.

After 10 idle minutes llama-swap unloads qwen-image and puts the layout back.

The placement logic is `llama-swap/gpu-layout.sh`. `layouts.conf` lists the cards each
container uses and, per layout, the containers to run in priority order. On-demand
models (llama-swap's `config.yaml`) are placed first, then each layout entry starts if
its cards are free, and everything else is stopped. To add a model: create its
container (compose, `managed` profile), add a `card` line and, for a resident model, a
place in a layout, and allow it in `docker-api/Caddyfile`; for an on-demand model, add
an entry to `llama-swap/config.yaml` using `run-container` / `stop-container`.

| Service | Purpose |
|---|---|
| `caddy` | TLS (Let's Encrypt) on 80/443. The internet can reach only `/v1/*` and `/health/liveliness`; the admin API and UI answer only on the LAN (192.168.0.0/16, 10.0.0.0/8) and the tailnet (100.64.0.0/10). LAN and Tailscale clients can also use plain HTTP, `http://<this machine>/v1` |
| `litellm` | OpenAI-compatible proxy: virtual keys, per-key rate and parallel limits, spend logs (Postgres). Admin UI at http://localhost:4000/ui |
| `vllm-lb` | Spreads chat over whichever vLLM instances are up (`/health` checks), and holds requests for up to 4 minutes while none is (a tp2 swap). Sticky per API key: caddy tags each request with the key's last 8 characters (`X-Affinity`), LiteLLM forwards it, and each key stays on one card so its prefix cache keeps hitting. It moves to the other card only while its own is down |
| `llama-swap` | Starts the GPU layout's containers, loads qwen-image on the second card on demand (evicting the chat container there), and restores the layout after 10 idle minutes (`ttl`). On startup it applies the layout and stops qwen-image |
| `docker-api` | Docker socket proxy for llama-swap. It allows only start, stop and inspect of the model containers |
| `ddns` | Optional (`--profile ddns`). Keeps `DOMAIN` pointed at the home IP |

Models: `Qwen3.8-27B` (chat, vision, tools, reasoning) and `qwen-image`
(`/v1/images/generations`, `/v1/images/edits`, returns `b64_json`).

## Setup

The whole stack starts from the repo root with `make build && make up` (see
`../README.md`). Before the first start:

1. Secrets: `cp .env.example .env`, fill it in (`openssl rand -hex 32`), `chmod 600 .env`.
   Set `DOMAIN` to the hostname you'll use, e.g. `llm.example.com`.
2. DNS and dynamic IP: `cp ddns/config.example.json ddns/config.json`. Edit it for your
   DNS provider ([provider docs](https://github.com/qdm12/ddns-updater#configuration)),
   then `make ddns`. `make logs s=ddns` shows the updates.
3. Router: forward TCP 80 and 443 to this machine (give it a static LAN IP, e.g.
   in NetworkManager), and keep that address out of the router's DHCP pool or reserve it
   so nothing else gets it. Port 80 is needed for Let's Encrypt HTTP-01 and the HTTPS redirect.

Caddy gets the certificate on the first start (`make logs s=caddy`).

LAN clients: if `https://$DOMAIN` doesn't load from inside the house, the router doesn't
do hairpin NAT. Add a local DNS override (router or Pi-hole) pointing `DOMAIN` at
this machine's LAN IP, or use `http://<this machine's IP>/v1`.

Coding agents: give each person (or each agent) its own key. Routing is sticky per key,
so a key shared by many sessions puts them all on one card. Set the client's context
limit to 131072 so it compacts before vLLM rejects a request. An opencode provider:

```json
"local-qwen": {
  "npm": "@ai-sdk/openai-compatible",
  "options": { "baseURL": "http://<this machine>/v1", "apiKey": "{file:~/.config/opencode/litellm.key}" },
  "models": { "Qwen3.8-27B": { "tool_call": true, "reasoning": true, "attachment": true,
                               "limit": { "context": 131072, "output": 32768 } } }
}
```

## Managing keys

Run these on this machine (or anywhere on the LAN via `https://$DOMAIN`). The admin UI at
http://localhost:4000/ui (user `admin`, `LITELLM_UI_PASSWORD`) does the same.

```bash
set -a; . ./.env; set +a

# New key for a person
curl -s http://localhost:4000/key/generate -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' -d '{
    "key_alias": "alice",
    "models": ["Qwen3.8-27B", "qwen-image"],
    "rpm_limit": 30,
    "max_parallel_requests": 2
  }'

# List keys, then revoke one
curl -s "http://localhost:4000/key/list?return_full_object=true" -H "Authorization: Bearer $LITELLM_MASTER_KEY"
curl -s http://localhost:4000/key/delete -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' -d '{"key_aliases": ["alice"]}'

# Usage per key
curl -s http://localhost:4000/spend/logs -H "Authorization: Bearer $LITELLM_MASTER_KEY"
```

Never give out the master key. It can mint and revoke keys, and the internet can't use it for
admin calls anyway.

## What to send a user

```
Base URL: https://llm.example.com/v1
API key:  sk-...
Models:   Qwen3.8-27B (chat), qwen-image (images)
```

Any OpenAI-compatible client works. Images:

```bash
curl https://llm.example.com/v1/images/generations -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen-image", "prompt": "...", "size": "1024x1024"}'
# Extras beyond the OpenAI schema: steps (default 40), guidance, seed, negative_prompt.
# /v1/images/edits (multipart, field image) takes a picture; without size the output
# keeps its aspect ratio at ~resolution² pixels (resolution 1024-2048, default 2048).
```

What users notice: the first image after a quiet spell takes about 20-30 s longer while
qwen-image loads (over a minute in tp2, where the first card reloads at the same time).
In split, chat never waits for a swap, but while images are loaded it runs on one card,
so it slows down under heavy load. In tp2, chat stalls for ~70 s when an image request
arrives and again when images unload. A chat generation running on the evicted container
(`qwen-vllm-2`, or `qwen-vllm-tp2`) at that moment is cut off; `vllm-lb` only retries
requests that never reached a backend.

## Operations

- Which card is doing what: `make mode` from the repo root (layout, loaded on-demand
  models, and which model containers run). llama-swap's log shows each start and stop
  (`gpu-layout: ...`).
- Give the second card back to chat now:
  `docker run --rm --network llm-net curlimages/curl -s http://llama-swap:8080/unload`
- Local tools: Open WebUI uses `http://vllm-lb:8000/v1`. `make bench` (`../qwen-vllm/bench.py`) hits
  localhost:8000 directly (`qwen-vllm` or `qwen-vllm-tp2`; `qwen-vllm-2` is on localhost:8001).
- Don't `docker start` a model container by hand: llama-swap wouldn't know the card is
  taken. `docker exec llama-swap gpu-layout reconcile` puts things back to the layout;
  restarting llama-swap does the same and also unloads qwen-image.
- Config changes: `llama-swap/config.yaml` reloads automatically, and
  `layouts/layouts.conf` is read on every change (`make mode` applies it now); for
  `litellm/config.yaml`, the `Caddyfile`, `vllm-lb/Caddyfile` or `docker-api/Caddyfile`
  run `make restart s=litellm` (or `s=caddy`, `s=vllm-lb`, `s=docker-api`) from the repo
  root. After editing the llama-swap scripts, `docker compose up -d --build llama-swap`
  from the root.
