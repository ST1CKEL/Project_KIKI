"""Build a cairo input region from a pixbuf's alpha channel."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cairo
import gi

gi.require_version("GdkPixbuf", "2.0")
from gi.repository import GdkPixbuf  # noqa: E402

PixbufLoader = Callable[[Path], GdkPixbuf.Pixbuf]


class AlphaRegionCache:
    """Cache immutable alpha hit regions by asset path and render size."""

    def __init__(self, *, threshold: int = 24) -> None:
        self._threshold = threshold
        self._regions: dict[tuple[str, int, int], cairo.Region] = {}

    def get(
        self,
        path: Path,
        dest_width: int,
        dest_height: int,
        loader: PixbufLoader,
    ) -> cairo.Region:
        key = (str(path), dest_width, dest_height)
        region = self._regions.get(key)
        if region is None:
            region = alpha_region(
                loader(path),
                dest_width,
                dest_height,
                threshold=self._threshold,
            )
            self._regions[key] = region
        return region

    def clear(self) -> None:
        self._regions.clear()

    def __len__(self) -> int:
        return len(self._regions)


def alpha_region(
    pixbuf: GdkPixbuf.Pixbuf,
    dest_width: int,
    dest_height: int,
    *,
    threshold: int = 24,
) -> cairo.Region:
    src_w = pixbuf.get_width()
    src_h = pixbuf.get_height()
    if src_w <= 0 or src_h <= 0 or dest_width <= 0 or dest_height <= 0:
        return cairo.Region()
    if pixbuf.get_n_channels() < 4 or not pixbuf.get_has_alpha():
        return cairo.Region(cairo.RectangleInt(0, 0, dest_width, dest_height))
    pixels = pixbuf.get_pixels()
    rowstride = pixbuf.get_rowstride()
    n_ch = pixbuf.get_n_channels()
    sx = dest_width / src_w
    sy = dest_height / src_h
    region = cairo.Region()
    for y in range(src_h):
        x = 0
        row = y * rowstride
        while x < src_w:
            while x < src_w and pixels[row + x * n_ch + 3] < threshold:
                x += 1
            if x >= src_w:
                break
            start = x
            while x < src_w and pixels[row + x * n_ch + 3] >= threshold:
                x += 1
            rx = int(start * sx)
            ry = int(y * sy)
            rw = max(1, int((x - start) * sx))
            rh = max(1, int(sy) if sy >= 1 else 1)
            region.union(cairo.RectangleInt(rx, ry, rw, rh))
    return region
