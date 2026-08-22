from __future__ import annotations

import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

from kiki.platform.autostart import set_enabled

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ID = "io.github.projectkiki.Kiki"


def test_release_versions_stay_in_sync() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]
    spec = (PROJECT_ROOT / "packaging/rpm/kiki.spec").read_text(encoding="utf-8")
    metainfo = ET.parse(PROJECT_ROOT / f"data/{APP_ID}.metainfo.xml").getroot()

    assert f"Version:        {version}" in spec
    assert 'scheme="rpm_prefix"' in spec
    assert "Requires:       python(abi) = %{kiki_python_version}" in spec
    assert "Requires:       espeak-ng" in spec
    assert "Requires:       python3-cffi" in spec
    assert "BuildRequires:  python3-cffi" in spec
    assert "License:        MIT" in spec
    assert "Requires:       vosk-api-devel >= 0.3.50" in spec
    assert "BuildArch:      x86_64" in spec
    assert "Requires:       gstreamer1-plugins-good" in spec
    assert "Requires:       pipewire-pulseaudio" in spec
    assert "Requires:       xdg-terminal-exec" in spec
    assert "Requires:       xdg-utils" in spec
    assert "Suggests:       ollama" in spec
    assert "Recommends:     ollama" not in spec
    assert metainfo.find(f"./releases/release[@version='{version}']") is not None
    assert "Pillow>=10" in pyproject["project"]["dependencies"]

    # The runtime version is a fourth source of truth and drifted unnoticed
    # once: the RPM said 0.6.0 while `kiki --version` still printed 0.5.0, which
    # only the packaging smoke test caught. Cheaper to catch here.
    from kiki import __version__

    assert __version__ == version
    # A bump without a changelog entry produces an RPM nobody can date.
    assert f"- {version}-1" in spec


def test_model_setup_keeps_compatible_default_and_documents_quality_profile() -> None:
    setup = (PROJECT_ROOT / "scripts/setup-local-model.sh").read_text(encoding="utf-8")

    assert 'MODEL="${1:-qwen3-vl:4b}"' in setup
    assert "ollama pull qwen3-vl:8b" in setup


def test_character_frame_normalizer_is_syntax_checked() -> None:
    script = (PROJECT_ROOT / "scripts/normalize-character-frame.sh").read_text(encoding="utf-8")
    rpm_builder = (PROJECT_ROOT / "scripts/build-rpm.sh").read_text(encoding="utf-8")
    spec = (PROJECT_ROOT / "packaging/rpm/kiki.spec").read_text(encoding="utf-8")

    assert "magick" in script
    assert "-extent 512x512" in script
    assert "scripts/normalize-character-frame.sh" in spec
    assert "--exclude='./vendor'" in rpm_builder


def test_desktop_and_service_are_safe_for_graphical_sessions() -> None:
    desktop = (PROJECT_ROOT / f"data/{APP_ID}.desktop").read_text(encoding="utf-8")
    service = (PROJECT_ROOT / "data/systemd/kiki.service").read_text(encoding="utf-8")

    assert "TryExec=kiki" in desktop
    assert "X-GNOME-UsesNotifications=true" in desktop
    assert "X-GNOME-Autostart-enabled=false" not in desktop
    assert "Wants=kiki-tts.service" not in service
    assert "ExecStart=/usr/bin/kiki" in service
    assert "WantedBy=graphical-session.target" in service


def test_autostart_normalizes_template(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "bundle"
    data_dir.mkdir()
    (data_dir / f"{APP_ID}.desktop").write_text(
        "\n".join(
            [
                "[Desktop Entry]",
                "Type=Application",
                "Name=KIKI",
                "Exec=kiki",
                "TryExec=kiki",
                "X-GNOME-Autostart-enabled=false",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("KIKI_DATA_DIR", str(data_dir))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    set_enabled(True, "/opt/kiki/bin/kiki")

    generated = (tmp_path / f"config/autostart/{APP_ID}.desktop").read_text(encoding="utf-8")
    assert "Exec=/opt/kiki/bin/kiki" in generated
    assert "TryExec=/opt/kiki/bin/kiki" in generated
    assert generated.count("X-GNOME-Autostart-enabled=true") == 1
    assert "X-GNOME-Autostart-enabled=false" not in generated


def test_every_harness_module_is_packaged() -> None:
    """A module added to services/kiki-llm/ but not installed ships a harness
    that cannot import itself.

    This nearly happened: batching.py and torch_batch.py were written after the
    spec's hand-kept file list and would have been left out of the RPM.
    """
    spec = (PROJECT_ROOT / "packaging/rpm/kiki.spec").read_text(encoding="utf-8")
    # The install step globs the directory instead of naming files one by one.
    assert "for part in services/kiki-llm/*.py; do" in spec
    assert "%{_libexecdir}/kiki/" in spec

    modules = sorted(
        p.name for p in (PROJECT_ROOT / "services" / "kiki-llm").glob("*.py")
    )
    assert "kiki_llm_server.py" in modules
    for required in ("batching.py", "torch_batch.py", "toolcalls.py", "vram.py"):
        assert required in modules, f"{required} fehlt im Harness-Verzeichnis"


def test_every_setup_entry_point_is_linked_and_listed() -> None:
    """A %files entry landing in %install by mistake broke the build once."""
    spec = (PROJECT_ROOT / "packaging/rpm/kiki.spec").read_text(encoding="utf-8")
    install, _, files = spec.partition("%files")
    for name in ("kiki-setup-model", "kiki-setup-tts", "kiki-setup-llm"):
        assert f"%{{_bindir}}/{name}\n" in files, f"{name} fehlt in %files"
        assert "ln -s ../libexec/kiki/" in install
        # A bare path in %install would be executed as a command.
        assert f"\n%{{_bindir}}/{name}\n" not in install
