#!/usr/bin/env python3
"""Chroma-key generated KIKI frames onto RGBA PNGs and build app icons."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(
    "/home/martin/.grok/sessions/"
    "%2Fhome%2Fmartin%2FDokumente%2FProjekte%2FProjectKIKI/"
    "01a0259e-33ae-7323-8a67-e8190882dbc8/images"
)
OUT = ROOT / "data" / "character" / "kiki"
ICON_ROOT = ROOT / "data" / "icons" / "hicolor"

# Sampled from canonical corners (muted lime, not pure #00FF00).
KEY = (148, 189, 109)

# source filename -> list of (subdir, filename)
MAPPING: list[tuple[str, str, str]] = [
    ("17.jpg", "idle", "00.png"),
    ("20.jpg", "idle", "01.png"),
    ("18.jpg", "idle", "02.png"),
    ("18.jpg", "idle_blink", "00.png"),
    ("23.jpg", "greet", "00.png"),
    ("24.jpg", "listening", "00.png"),
    ("25.jpg", "thinking", "00.png"),
    ("28.jpg", "thinking", "01.png"),
    ("21.jpg", "speaking", "00.png"),
    ("29.jpg", "speaking", "01.png"),
    ("22.jpg", "happy", "00.png"),
    ("19.jpg", "surprised", "00.png"),
    ("27.jpg", "sleeping", "00.png"),
    ("26.jpg", "error", "00.png"),
    ("31.jpg", "notification", "00.png"),
]


def _dist(r: int, g: int, b: int) -> float:
    return math.sqrt((r - KEY[0]) ** 2 + (g - KEY[1]) ** 2 + (b - KEY[2]) ** 2)


def chroma_key(im: Image.Image, hard: float = 38.0, soft: float = 88.0) -> Image.Image:
    """Remove the lime backdrop. The figure has no lime, so every similar pixel goes."""
    rgba = im.convert("RGBA")
    w, h = rgba.size
    src = rgba.load()
    assert src is not None
    out = Image.new("RGBA", (w, h))
    dst = out.load()
    assert dst is not None
    span = max(soft - hard, 1.0)
    for y in range(h):
        for x in range(w):
            r, g, b, _a = src[x, y]
            d = _dist(r, g, b)
            greenish = g > r + 16 and g > b + 28 and r < 205 and b < 155
            keyed = d < soft or (greenish and d < soft + 24)
            if not keyed:
                if g > r + 10 and g > b + 10:
                    g = min(g, max(r, b) + 4)
                dst[x, y] = (r, g, b, 255)
                continue
            if d <= hard or (greenish and d <= hard + 10):
                dst[x, y] = (0, 0, 0, 0)
            else:
                alpha = int(255 * min(1.0, (d - hard) / span))
                if g > r and g > b:
                    g = int((r + b) / 2)
                dst[x, y] = (r, g, b, alpha)
    alpha = out.getchannel("A").filter(ImageFilter.GaussianBlur(radius=0.45))
    r_ch, g_ch, b_ch, _ = out.split()
    # Fully transparent pixels must not keep keyed green RGB (premultiply-safe).
    merged = Image.merge("RGBA", (r_ch, g_ch, b_ch, alpha))
    px = merged.load()
    assert px is not None
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a == 0:
                px[x, y] = (0, 0, 0, 0)
            elif a < 255 and g > r + 6 and g > b + 6:
                px[x, y] = (r, int((r + b) / 2), b, a)
    return merged


def union_bbox(images: list[Image.Image], pad: int = 12) -> tuple[int, int, int, int]:
    minx, miny, maxx, maxy = 10**9, 10**9, 0, 0
    for im in images:
        box = im.getbbox()
        if not box:
            continue
        x0, y0, x1, y1 = box
        minx, miny = min(minx, x0), min(miny, y0)
        maxx, maxy = max(maxx, x1), max(maxy, y1)
    minx = max(0, minx - pad)
    miny = max(0, miny - pad)
    maxx = min(images[0].width, maxx + pad)
    maxy = min(images[0].height, maxy + pad)
    return minx, miny, maxx, maxy


def rounded_rect(size: int, radius: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for y in range(size):
        t = y / max(size - 1, 1)
        col = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3)) + (255,)
        draw.line([(0, y), (size, y)], fill=col)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    im.paste(overlay, (0, 0), mask)
    return im


def make_icons(bust: Image.Image) -> None:
    keyed = chroma_key(bust)
    box = keyed.getbbox()
    if box:
        keyed = keyed.crop(box)
    for size in (64, 128, 256, 512):
        canvas = rounded_rect(size, radius=int(size * 0.22), top=(32, 84, 168), bottom=(92, 52, 168))
        margin = int(size * 0.08)
        avail = size - 2 * margin
        ratio = min(avail / keyed.width, avail / keyed.height)
        nw, nh = max(1, int(keyed.width * ratio)), max(1, int(keyed.height * ratio))
        sprite = keyed.resize((nw, nh), Image.Resampling.LANCZOS)
        x = (size - nw) // 2
        y = size - nh - int(size * 0.04)
        canvas.paste(sprite, (x, y), sprite)
        dest_dir = ICON_ROOT / f"{size}x{size}" / "apps"
        dest_dir.mkdir(parents=True, exist_ok=True)
        canvas.save(dest_dir / "io.github.projectkiki.Kiki.png", "PNG")
        print(f"icon {size}x{size}")


def main() -> None:
    keyed_by_src: dict[str, Image.Image] = {}
    for src_name, _subdir, _fname in MAPPING:
        if src_name in keyed_by_src:
            continue
        path = SRC / src_name
        if not path.exists():
            raise SystemExit(f"missing source {path}")
        print(f"key {src_name}")
        keyed_by_src[src_name] = chroma_key(Image.open(path))

    bodies = [im for name, im in keyed_by_src.items() if name != "12.jpg"]
    crop = union_bbox(bodies)
    print("union crop", crop)

    for src_name, subdir, fname in MAPPING:
        dest = OUT / subdir / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        im = keyed_by_src[src_name].crop(crop)
        im.save(dest, "PNG")
        print(f"wrote {dest.relative_to(ROOT)} {im.size}")

    bust_path = SRC / "30.jpg"
    if bust_path.exists():
        make_icons(Image.open(bust_path))


if __name__ == "__main__":
    main()
