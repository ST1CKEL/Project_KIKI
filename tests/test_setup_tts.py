from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP_TTS = ROOT / "scripts" / "setup-tts.sh"


def _executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _executable(fake_bin / "systemctl", "#!/usr/bin/env bash\nexit 0\n")
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(tmp_path / "home"),
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "XDG_DATA_HOME": str(tmp_path / "data"),
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )
    return env, fake_bin


def test_dummy_setup_uses_system_python_without_venv(tmp_path: Path) -> None:
    env, _fake_bin = _environment(tmp_path)
    system_python = shutil.which("python3", path=env["PATH"])
    assert system_python is not None

    result = subprocess.run(
        ["bash", str(SETUP_TTS), "--dummy"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "keine GPU- oder Pip-Pakete nötig" in result.stdout
    assert not (tmp_path / "data" / "kiki" / "tts-venv").exists()
    assert (tmp_path / "data" / "kiki" / "tts" / "kiki_tts_server.py").is_file()
    unit = (tmp_path / "config" / "systemd" / "user" / "kiki-tts.service").read_text(
        encoding="utf-8"
    )
    assert f'ExecStart="{system_python}"' in unit
    assert "--dummy" in unit


def test_gpu_setup_rejects_unusable_nvidia_driver_before_venv(tmp_path: Path) -> None:
    env, fake_bin = _environment(tmp_path)
    _executable(
        fake_bin / "python3.12",
        '#!/usr/bin/env bash\n[[ "${1:-}" == "--version" ]] && echo "Python 3.12.0" && exit 0\nexit 99\n',
    )
    _executable(fake_bin / "nvidia-smi", "#!/usr/bin/env bash\nexit 1\n")

    result = subprocess.run(
        ["bash", str(SETUP_TTS)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "NVIDIA-Treiber oder CUDA-GPU ist nicht erreichbar" in result.stdout
    assert not (tmp_path / "data" / "kiki" / "tts-venv").exists()


def test_setup_runs_through_the_packaged_symlink(tmp_path: Path) -> None:
    """The RPM links /usr/bin/kiki-setup-tts to /usr/libexec/kiki/setup-tts.

    Resolving $0 without following the symlink made the script look for
    kiki_tts_server.py in /usr/bin, so `kiki-setup-tts` never worked from an
    installed package — only from the source tree.
    """
    env, _fake_bin = _environment(tmp_path)
    libexec = tmp_path / "usr" / "libexec" / "kiki"
    usr_bin = tmp_path / "usr" / "bin"
    libexec.mkdir(parents=True)
    usr_bin.mkdir(parents=True)
    shutil.copy2(SETUP_TTS, libexec / "setup-tts")
    shutil.copy2(
        ROOT / "services" / "qwen3-tts" / "kiki_tts_server.py",
        libexec / "kiki_tts_server.py",
    )
    (usr_bin / "kiki-setup-tts").symlink_to("../libexec/kiki/setup-tts")

    result = subprocess.run(
        [str(usr_bin / "kiki-setup-tts"), "--dummy"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "nicht neben dem Setup-Skript gefunden" not in result.stdout
    assert (tmp_path / "data" / "kiki" / "tts" / "kiki_tts_server.py").is_file()
