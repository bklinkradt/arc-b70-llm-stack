import base64
import gc
import inspect
import io
import math
import os
import random
import threading
import time
from pathlib import Path

import gradio as gr
import torch
import uvicorn
from diffusers import QwenImage21Pipeline
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

MODEL_ID = os.environ.get("QWEN_MODEL", "Qwen/Qwen-Image-2.1")
OFFLOAD = os.environ.get("QWEN_OFFLOAD", "auto").lower()
VAE_TILE = int(os.environ.get("QWEN_VAE_TILE", "1024"))  # 0 disables tiled decode
OUTPUT_DIR = Path("/app/outputs")
DEVICE = "xpu"
GIB = 1024**3

# Recommended resolutions from the model card.
ASPECT_RATIOS = {
    "1:1 (2048×2048)": (2048, 2048),
    "4:3 (2400×1792)": (2400, 1792),
    "3:4 (1792×2400)": (1792, 2400),
    "3:2 (2528×1696)": (2528, 1696),
    "2:3 (1696×2528)": (1696, 2528),
    "16:9 (2752×1536)": (2752, 1536),
    "9:16 (1536×2752)": (1536, 2752),
    "1:1 draft (1024×1024)": (1024, 1024),
}
MATCH_INPUT = "Match input image"
DEFAULT_ASPECT = "1:1 (2048×2048)"
DEFAULT_RESOLUTION = 2048
DEFAULT_STEPS = 40
DEFAULT_GUIDANCE = 4.0
# Model id served on the OpenAI-compatible API (/v1/images/*).
API_MODEL_NAME = os.environ.get("QWEN_API_MODEL_NAME", "qwen-image")

RENDER_TO_PHOTO_PROMPT = (
    "Transform this rough 3D render into a photorealistic photograph. Keep the exact composition, "
    "camera angle, layout, object positions and proportions. Replace the flat CG shading with realistic "
    "materials, textures, natural lighting, soft shadows and reflections, as if shot on a full-frame "
    "camera with a 35mm lens."
)


def check_device():
    if not torch.xpu.is_available():
        raise SystemExit(
            "torch.xpu is not available. Check that /dev/dri/renderD128 is passed into the "
            "container and that the container user is in the render group."
        )
    props = torch.xpu.get_device_properties(0)
    print(f"[qwen-image] torch {torch.__version__}, device: {props.name}, "
          f"{props.total_memory / GIB:.1f} GiB", flush=True)
    return props.total_memory


def weights_bytes(pipe):
    total = 0
    for component in pipe.components.values():
        if isinstance(component, torch.nn.Module):
            total += sum(p.numel() * p.element_size() for p in component.parameters())
    return total


def load_pipeline(vram_bytes):
    print(f"[qwen-image] loading {MODEL_ID} ...", flush=True)
    pipe = QwenImage21Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)

    size = weights_bytes(pipe)
    mode = OFFLOAD
    if mode == "auto":
        # Leave ~4 GiB headroom for activations and the VAE decode.
        mode = "none" if size < vram_bytes - 4 * GIB else "model"
    print(f"[qwen-image] weights {size / GIB:.1f} GiB, placement mode: {mode}", flush=True)

    if mode == "none":
        pipe.to(DEVICE)
    elif mode == "model":
        pipe.enable_model_cpu_offload(device=DEVICE)
    elif mode == "sequential":
        pipe.enable_sequential_cpu_offload(device=DEVICE)
    else:
        raise SystemExit(f"Unknown QWEN_OFFLOAD={OFFLOAD!r}; use auto|none|model|sequential")

    # A full 2048² decode needs >28 GiB, so the VAE decodes in tiles. The diffusers default
    # (256px tiles, 64px overlap = 4 latents at 16x compression) leaves purple seam streaks;
    # large tiles with a 1/4-tile blend zone are seamless.
    if VAE_TILE > 0:
        print(f"[qwen-image] VAE tiling: {VAE_TILE}px tiles, {VAE_TILE // 4}px overlap", flush=True)
        pipe.vae.enable_tiling(
            tile_sample_min_height=VAE_TILE,
            tile_sample_min_width=VAE_TILE,
            tile_sample_stride_height=VAE_TILE * 3 // 4,
            tile_sample_stride_width=VAE_TILE * 3 // 4,
        )
    return pipe


vram = check_device()
pipe = load_pipeline(vram)

call_params = inspect.signature(pipe.__call__).parameters
GUIDANCE_ARG = next((n for n in ("true_cfg_scale", "guidance_scale") if n in call_params), None)
HAS_NEGATIVE = "negative_prompt" in call_params


def run_pipe(kwargs):
    """Run the pipeline; on OOM return None with the failed attempt's memory released."""
    try:
        return pipe(**kwargs)
    except (torch.OutOfMemoryError, RuntimeError) as e:
        # Level Zero reports some allocation failures as a RuntimeError instead.
        if not isinstance(e, torch.OutOfMemoryError) and "OUT_OF_RESOURCES" not in str(e):
            raise
    # Free memory outside the except block (its traceback pins the failed run's tensors), and
    # offload whatever model the aborted run left on the GPU; the pipeline only does that on success.
    pipe.maybe_free_model_hooks()
    gc.collect()
    torch.xpu.empty_cache()
    return None


class OutOfGpuMemory(RuntimeError):
    pass


# Gradio's concurrency_limit only serialises its own queue; the HTTP API shares the same pipeline.
PIPE_LOCK = threading.Lock()


def render(prompt, negative_prompt, input_image, aspect, resolution, steps, guidance, seed):
    """Run one generation and save it. Raises ValueError on bad input, OutOfGpuMemory if the retry OOMs too."""
    if not prompt.strip():
        raise ValueError("Prompt is empty.")
    if aspect == MATCH_INPUT and input_image is None:
        raise ValueError("'Match input image' needs an input image.")
    seed = random.randint(0, 2**31 - 1) if seed is None or seed < 0 else int(seed)

    kwargs = dict(
        prompt=prompt,
        num_inference_steps=int(steps),
        generator=torch.Generator(DEVICE).manual_seed(seed),
    )
    if input_image is not None:
        # The pipeline resizes the condition image to ~resolution² and, without an explicit
        # width/height, keeps the input's aspect ratio for the output.
        kwargs["image"] = input_image
        kwargs["output_resolution"] = int(resolution)
    if aspect != MATCH_INPUT:
        kwargs["width"], kwargs["height"] = ASPECT_RATIOS[aspect]
    if HAS_NEGATIVE and negative_prompt and negative_prompt.strip():
        kwargs["negative_prompt"] = negative_prompt
    if GUIDANCE_ARG:
        kwargs[GUIDANCE_ARG] = float(guidance)

    with PIPE_LOCK:
        torch.xpu.reset_peak_memory_stats()
        start = time.time()
        result = run_pipe(kwargs)
        if result is None:
            # With an input image + negative prompt, the pipeline keeps a KV cache of the condition
            # tokens for both CFG branches, which overflows 32 GiB at 2048. Retry without the cache:
            # slower (condition tokens are recomputed every step) but lighter.
            print("[qwen-image] OOM, retrying with use_kv_cache=False", flush=True)
            result = run_pipe({**kwargs, "use_kv_cache": False})
        if result is None:
            raise OutOfGpuMemory(
                "Out of GPU memory. Lower the output resolution, clear the negative prompt, "
                "or set QWEN_OFFLOAD=sequential."
            )
        # The VAE decodes RGBA; drop the (near-opaque) alpha so viewers don't composite it.
        image = result.images[0].convert("RGB")
        elapsed = time.time() - start
        peak = torch.xpu.max_memory_allocated() / GIB
        del result
        alloc, reserved = torch.xpu.memory_allocated() / GIB, torch.xpu.memory_reserved() / GIB
        # Models are offloaded after each run; hand the allocator's cached blocks back to the driver
        # so an idle server doesn't hold VRAM.
        gc.collect()
        torch.xpu.empty_cache()
        print(f"[qwen-image] after run: allocated {alloc:.2f} GiB, reserved {reserved:.2f} GiB "
              f"-> {torch.xpu.memory_reserved() / GIB:.2f} GiB after empty_cache", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}_{seed}"
    path = OUTPUT_DIR / f"{stem}.png"
    image.save(path)
    if input_image is not None:
        input_image.save(OUTPUT_DIR / f"{stem}_input.png")
    mode = "img2img" if input_image is not None else "txt2img"
    info = f"{mode} · seed {seed} · {image.width}×{image.height} · {int(steps)} steps · {elapsed:.1f}s · peak VRAM {peak:.1f} GiB · {path.name}"
    print(f"[qwen-image] {info}", flush=True)
    return image, info


def generate(prompt, negative_prompt, input_image, aspect, resolution, steps, guidance, seed):
    try:
        return render(prompt, negative_prompt, input_image, aspect, resolution, steps, guidance, seed)
    except (ValueError, OutOfGpuMemory) as e:
        raise gr.Error(str(e)) from e


with gr.Blocks(title="Qwen-Image-2.1") as demo:
    gr.Markdown(f"## Qwen-Image-2.1 on {torch.xpu.get_device_name(0)}")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Input image (optional, e.g. a 3D render to make photorealistic)",
                type="pil", image_mode="RGB", sources=["upload", "clipboard"],
            )
            prompt = gr.Textbox(label="Prompt", lines=4)
            preset = gr.Button("Use render → photo prompt", size="sm")
            negative = gr.Textbox(label="Negative prompt", lines=2, visible=HAS_NEGATIVE)
            aspect = gr.Dropdown([MATCH_INPUT, *ASPECT_RATIOS], value=DEFAULT_ASPECT, label="Size")
            resolution = gr.Slider(
                1024, 2048, value=DEFAULT_RESOLUTION, step=128, visible=False,
                label="Output resolution (long-side scale for image input; output keeps the input's aspect ratio)",
            )
            steps = gr.Slider(1, 80, value=DEFAULT_STEPS, step=1, label="Steps")
            guidance = gr.Slider(1.0, 10.0, value=DEFAULT_GUIDANCE, step=0.1, label="CFG scale", visible=GUIDANCE_ARG is not None)
            seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            button = gr.Button("Generate", variant="primary")
        with gr.Column():
            output = gr.Image(label="Result", type="pil", format="png")
            info = gr.Markdown()
    preset.click(lambda: RENDER_TO_PHOTO_PROMPT, None, prompt)
    # Uploading an image switches the size to follow it; clearing it goes back to a fixed size.
    input_image.change(
        lambda img: (
            gr.update(value=MATCH_INPUT if img is not None else DEFAULT_ASPECT),
            gr.update(visible=img is not None),
        ),
        input_image, [aspect, resolution],
    )
    button.click(
        generate, [prompt, negative, input_image, aspect, resolution, steps, guidance, seed], [output, info],
        concurrency_limit=1,
    )



# OpenAI-compatible images API, mounted next to the Gradio UI. Always returns b64_json.
api = FastAPI()


def aspect_for_size(size):
    """Map an OpenAI "WxH" size to the recommended resolution closest in aspect ratio, then area."""
    try:
        width, height = (int(v) for v in size.lower().split("x"))
    except ValueError:
        raise ValueError(f"size must look like 1024x1024, got {size!r}") from None
    if width <= 0 or height <= 0:
        raise ValueError(f"size must be positive, got {size!r}")
    return min(
        ASPECT_RATIOS,
        key=lambda name: (
            round(abs(math.log((ASPECT_RATIOS[name][0] / ASPECT_RATIOS[name][1]) / (width / height))), 3),
            abs(ASPECT_RATIOS[name][0] * ASPECT_RATIOS[name][1] - width * height),
        ),
    )


def api_error(status, message):
    kind = "invalid_request_error" if status < 500 else "server_error"
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status)


def images_response(prompt, negative_prompt, input_image, aspect, resolution, n, steps, guidance, seed):
    if not 1 <= n <= 4:
        return api_error(400, "n must be between 1 and 4")
    data = []
    try:
        for i in range(n):
            image, _ = render(prompt, negative_prompt, input_image, aspect, resolution, steps, guidance,
                              seed if seed < 0 else seed + i)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            data.append({"b64_json": base64.b64encode(buf.getvalue()).decode()})
    except ValueError as e:
        return api_error(400, str(e))
    except OutOfGpuMemory as e:
        return api_error(503, str(e))
    return {"created": int(time.time()), "data": data}


class GenerationRequest(BaseModel):
    prompt: str
    model: str | None = None
    n: int = 1
    size: str | None = None
    response_format: str | None = None  # ignored: always b64_json
    # Extensions beyond the OpenAI schema.
    negative_prompt: str = ""
    steps: int = DEFAULT_STEPS
    guidance: float = DEFAULT_GUIDANCE
    seed: int = -1


@api.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": API_MODEL_NAME, "object": "model", "owned_by": "local"}]}


@api.post("/v1/images/generations")
def images_generations(req: GenerationRequest):
    try:
        aspect = aspect_for_size(req.size) if req.size and req.size != "auto" else DEFAULT_ASPECT
    except ValueError as e:
        return api_error(400, str(e))
    return images_response(req.prompt, req.negative_prompt, None, aspect, DEFAULT_RESOLUTION,
                           req.n, req.steps, req.guidance, req.seed)


@api.post("/v1/images/edits")
def images_edits(
    # OpenAI accepts the upload as "image" or "image[]" (LiteLLM sends the latter); uses the first.
    image: list[UploadFile] | None = File(None),
    image_list: list[UploadFile] | None = File(None, alias="image[]"),
    prompt: str = Form(...),
    model: str | None = Form(None),
    n: int = Form(1),
    size: str | None = Form(None),
    response_format: str | None = Form(None),
    negative_prompt: str = Form(""),
    resolution: int = Form(DEFAULT_RESOLUTION),
    steps: int = Form(DEFAULT_STEPS),
    guidance: float = Form(DEFAULT_GUIDANCE),
    seed: int = Form(-1),
):
    """img2img: without a size the output keeps the input's aspect ratio at ~resolution² pixels."""
    uploads = image or image_list
    if not uploads:
        return api_error(400, "image is required")
    try:
        input_image = Image.open(uploads[0].file).convert("RGB")
    except Exception:
        return api_error(400, "image is not a readable image file")
    if not 1024 <= resolution <= 2048:
        return api_error(400, "resolution must be between 1024 and 2048")
    try:
        aspect = aspect_for_size(size) if size and size != "auto" else MATCH_INPUT
    except ValueError as e:
        return api_error(400, str(e))
    return images_response(prompt, negative_prompt, input_image, aspect, resolution,
                           n, steps, guidance, seed)


demo.queue(max_size=16)
app = gr.mount_gradio_app(api, demo, path="/")
uvicorn.run(app, host="0.0.0.0", port=7860)
