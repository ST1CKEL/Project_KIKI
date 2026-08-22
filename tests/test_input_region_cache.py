from __future__ import annotations

from pathlib import Path

import kiki.ui.input_region as input_region


def test_alpha_region_cache_reuses_region(monkeypatch) -> None:
    built: list[tuple[object, int, int, int]] = []
    pixbuf = object()
    region = object()

    def fake_alpha_region(source, width, height, *, threshold):
        built.append((source, width, height, threshold))
        return region

    monkeypatch.setattr(input_region, "alpha_region", fake_alpha_region)
    cache = input_region.AlphaRegionCache(threshold=17)
    loader_calls: list[Path] = []

    def load(path: Path):
        loader_calls.append(path)
        return pixbuf

    path = Path("idle/00.png")
    assert cache.get(path, 128, 256, load) is region
    assert cache.get(path, 128, 256, load) is region
    assert len(cache) == 1
    assert loader_calls == [path]
    assert built == [(pixbuf, 128, 256, 17)]


def test_alpha_region_cache_separates_sizes_and_can_clear(monkeypatch) -> None:
    builds = 0

    def fake_alpha_region(_source, _width, _height, *, threshold):
        nonlocal builds
        assert threshold == 24
        builds += 1
        return object()

    monkeypatch.setattr(input_region, "alpha_region", fake_alpha_region)
    cache = input_region.AlphaRegionCache()

    def loader(_path: Path):
        return object()

    path = Path("idle/00.png")

    first = cache.get(path, 128, 256, loader)
    second = cache.get(path, 256, 512, loader)
    assert first is not second
    assert len(cache) == 2

    cache.clear()
    assert len(cache) == 0
    assert cache.get(path, 128, 256, loader) is not first
    assert builds == 3
