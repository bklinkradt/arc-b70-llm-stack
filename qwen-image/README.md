# Qwen-Image-2.1 on Intel Arc (Docker)

Runs [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) with a Gradio UI on the Intel Arc Pro B70. It uses PyTorch's XPU backend, not CUDA.

## Run

```bash
cp .env.example .env          # optional: set HF_TOKEN / QWEN_OFFLOAD
mkdir -p ~/.cache/huggingface outputs
```

Then `make build && make up` from the repo root builds the image and creates the container stopped; `../gateway`'s llama-swap starts it on demand. It is in the `managed` profile so a plain `docker compose up` never starts it.

The container runs on the second B70 (PCI 08:00.0), which it shares with `../qwen-vllm`'s chat containers, so `../gateway` (llama-swap) starts and stops it: it evicts the chat container on that card (`vllm-2`, or `vllm-tp2`, which `vllm` on the first card then replaces) and loads when an image is requested, and after 10 idle minutes it stops and the GPU layout gets the card back. To use it on its own without the gateway, `make -C llamacpp-bench take-card` from the repo root (stops llama-swap and frees the card), then `docker compose up qwen-image`.

Once it's running, open http://localhost:7860 for the Gradio UI. It is bound to localhost and has no authentication; remote access goes through the gateway's API keys. The first start downloads the weights (tens of GB) into `~/.cache/huggingface` on the host. Images are saved to `./outputs`.

## API

The same server has an OpenAI-compatible images API. It returns `b64_json` only, and API and UI requests share one queue:

- `POST /v1/images/generations`: JSON `{prompt, size, n}`. `size` maps to the recommended resolution nearest in aspect ratio (`1024x1024` is the draft size).
- `POST /v1/images/edits`: multipart with `image` (or `image[]`) and `prompt`. Without `size` the output keeps the input's aspect ratio at ~`resolution`² pixels (1024-2048, default 2048).
- `GET /v1/models`: lists `qwen-image` (the gateway's health check).

Both accept `steps` (default 40), `guidance`, `seed` and `negative_prompt` as extras.

GPU sanity check:

```bash
docker compose run --rm qwen-image python -c "import torch; print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"
```

## Render → photo (image input)

Upload an image, e.g. a screenshot of a rough 3D browser render, then click **Use render → photo prompt** (or write your own instruction) and generate. With an image uploaded, the size switches to **Match input image**. The output keeps the input's aspect ratio, and **Output resolution** sets its scale: 2048 on a 16:9 input gives 2720×1536. The model keeps the layout and camera angle but isn't pixel-exact. Hide grid lines, gizmos and UI overlays before taking the screenshot. Each result is saved next to a copy of its input (`*_input.png`).

At 2048 an image-to-image run takes about 4.5 minutes and peaks at about 26.5 GiB of VRAM. Lower the output resolution for faster drafts.

## Memory / offload

`QWEN_OFFLOAD` in `.env`:

- `auto` (default): keeps everything on the GPU if the weights fit with about 4 GiB of headroom. Otherwise it uses `model`.
- `none`: everything on the GPU (fastest).
- `model`: each component moves to the GPU only while it runs.
- `sequential`: layer-by-layer offload. Lowest VRAM use, and very slow.

If you get out-of-memory errors at 2048² or larger, switch to `model`. The startup log prints the weight size and the chosen mode. Each generation logs its peak VRAM.

`QWEN_VAE_TILE` (default `1024`): images larger than this decode in overlapping tiles, because a full 2048² decode needs more than 28 GiB. Keep tiles large. The diffusers default of 256 px leaves purple vertical streaks along the tile seams. Saved images are RGB, since the model's VAE outputs RGBA and the alpha is dropped.

## Troubleshooting

- `torch.xpu is not available`: check `docker compose run --rm qwen-image ls -l /dev/dri`. `docker-compose.yml` passes the card by PCI path (`/dev/dri/by-path/pci-0000:08:00.0-*`); if the cards move slots, check `ls -l /dev/dri/by-path`. The group IDs `105`/`39` are this host's `render`/`video` groups.
- Unsupported XPU op errors: add `PYTORCH_ENABLE_XPU_FALLBACK: "1"` to the service environment.
- The installed package versions (including the diffusers git commit) are recorded in `/opt/requirements.lock` inside the image.
