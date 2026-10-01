#!/usr/bin/env python3
"""Generate or edit an image with the Gemini API and save it to a file.

Usage:
  gen_image.py "PROMPT" OUTPUT [--aspect 16:9] [--size 1K] [--width 1600]
                               [--ref IMAGE ...] [--model MODEL]

Needs GEMINI_API_KEY (or GOOGLE_API_KEY). Stdlib only; with Pillow installed
it can also write .webp/.jpg, convert formats and downscale with --width.
"""
import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")
ASPECTS = ["1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
SIZES = ["512px", "1K", "2K", "4K"]


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def find_image(obj):
    """Return (mime, base64 data) of the first image anywhere in the response."""
    if isinstance(obj, dict):
        inline = obj.get("inlineData") or obj.get("inline_data")
        if isinstance(inline, dict) and inline.get("data"):
            return inline.get("mimeType") or inline.get("mime_type") or "image/png", inline["data"]
        mime = obj.get("mimeType") or obj.get("mime_type") or ""
        if mime.startswith("image/") and isinstance(obj.get("data"), str):
            return mime, obj["data"]
        for v in obj.values():
            if found := find_image(v):
                return found
    elif isinstance(obj, list):
        for v in obj:
            if found := find_image(v):
                return found
    return None


def find_text(obj):
    """Collect any text the model returned (explains refusals / no image)."""
    if isinstance(obj, dict):
        out = [obj["text"]] if isinstance(obj.get("text"), str) else []
        return out + [t for v in obj.values() for t in find_text(v)]
    if isinstance(obj, list):
        return [t for v in obj for t in find_text(v)]
    return []


def call_api(model, body, key):
    req = urllib.request.Request(
        API.format(model=model), json.dumps(body).encode(),
        {"Content-Type": "application/json", "x-goog-api-key": key})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            try:
                detail = json.loads(detail)["error"]["message"]
            except (ValueError, KeyError, TypeError):
                pass
            # "limit: 0" means no quota at all (e.g. free tier without billing),
            # so waiting won't help.
            retryable = e.code in (429, 500, 502, 503, 504) and "limit: 0" not in str(detail)
            if retryable and attempt < 3:
                time.sleep(5 * 2 ** attempt)
                continue
            fail(f"Gemini API returned HTTP {e.code}: {detail}")
        except urllib.error.URLError as e:
            if attempt < 3:
                time.sleep(5 * 2 ** attempt)
                continue
            fail(f"could not reach Gemini API: {e.reason}")


def save(data, mime, out, width):
    """Write the image to `out`, converting/resizing with Pillow when needed."""
    want = mimetypes.guess_type(out.name)[0] or mime
    try:
        from io import BytesIO
        from PIL import Image
    except ImportError:
        if width:
            print("WARNING: Pillow not installed; --width ignored", file=sys.stderr)
        if want != mime:
            ext = mimetypes.guess_extension(mime) or ".png"
            print(f"WARNING: Pillow not installed; saving as {ext} instead of {out.suffix}",
                  file=sys.stderr)
            out = out.with_suffix(ext)
        out.write_bytes(data)
        return out, None

    img = Image.open(BytesIO(data))
    if width and img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    fmt = {"image/webp": "WEBP", "image/jpeg": "JPEG", "image/png": "PNG"}.get(want, "PNG")
    if fmt == "JPEG" and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(out, fmt, **({"quality": 85} if fmt in ("WEBP", "JPEG") else {}))
    return out, img.size


def main():
    p = argparse.ArgumentParser(description="Generate or edit an image with Gemini.")
    p.add_argument("prompt")
    p.add_argument("output", type=Path)
    p.add_argument("--aspect", default="1:1", choices=ASPECTS)
    p.add_argument("--size", default="1K", choices=SIZES, help="generated resolution")
    p.add_argument("--width", type=int, help="downscale to this width in px (needs Pillow)")
    p.add_argument("--ref", action="append", type=Path, default=[],
                   help="reference or source image to edit/match (repeatable)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        fail("set GEMINI_API_KEY (get one at https://aistudio.google.com/apikey)")

    parts = [{"text": args.prompt}]
    for ref in args.ref:
        if not ref.is_file():
            fail(f"reference image not found: {ref}")
        parts.append({"inline_data": {
            "mime_type": mimetypes.guess_type(ref.name)[0] or "image/png",
            "data": base64.b64encode(ref.read_bytes()).decode()}})
    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {"aspectRatio": args.aspect, "imageSize": args.size},
        },
    }

    resp = call_api(args.model, body, key)
    found = find_image(resp)
    if not found:
        text = " ".join(find_text(resp)).strip()
        reason = json.dumps(resp.get("promptFeedback") or (resp.get("candidates") or [{}])[0].get("finishReason"))
        fail(f"no image returned (reason: {reason}). {text[:500]}")

    mime, b64 = found
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out, size = save(base64.b64decode(b64), mime, args.output, args.width)
    dims = f" {size[0]}x{size[1]}" if size else ""
    print(f"saved {out}{dims} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
