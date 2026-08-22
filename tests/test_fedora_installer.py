from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = PROJECT_ROOT / "packaging/installer/kiki-fedora44.run.in"
BUILD = PROJECT_ROOT / "scripts/build-fedora-installer.sh"
SMOKE = PROJECT_ROOT / "scripts/smoke-test-installer.sh"


def test_installer_template_is_bounded_and_does_not_pipe_remote_shell() -> None:
    text = TEMPLATE.read_text(encoding="utf-8")

    assert 'os_id" != "fedora"' in text
    assert 'os_version" != "44"' in text
    assert 'arch" != "x86_64"' in text
    assert 'Dieser Installer enthält ein x86_64-RPM' in text
    assert "sha256sum" in text
    assert "rpm -qp --qf" in text
    assert "sudo dnf install" in text
    assert "--setopt=install_weak_deps=False" in text
    assert "Fedora-Paket 'ollama' jetzt installieren?" in text
    assert "PYTHONNOUSERSITE=1 kiki --prepare-voice-model" in text
    assert "qwen3-vl:2b|qwen3-vl:4b|qwen3-vl:8b" in text
    assert "https://opencode.ai/install --output" in text
    assert "curl |" not in text
    assert "eval " not in text
    assert "vosk-api-devel" in text
    assert "xdotool" not in text
    assert "ydotool" not in text


def test_installer_sources_are_valid_bash() -> None:
    for path in (TEMPLATE, BUILD, SMOKE):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_builder_appends_one_exact_rpm_payload_marker() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    builder = BUILD.read_text(encoding="utf-8")

    assert template.count("\n__KIKI_RPM_PAYLOAD_BELOW__\n") == 1
    assert "dd if=\"$RPM_PATH\"" in builder
    assert '"$OUTPUT" --verify-only' in builder


def test_installer_smoke_compares_payload_with_current_rpm() -> None:
    smoke = SMOKE.read_text(encoding="utf-8")

    assert "KIKI_RPM_SHA256" in smoke
    assert 'sha256sum "$RPM_PATH"' in smoke
    assert '"$EMBEDDED_SHA" == "$CURRENT_SHA"' in smoke
