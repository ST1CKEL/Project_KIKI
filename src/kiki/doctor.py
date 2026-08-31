"""Voice-stack doctor. Law 1: missing pieces are FAIL, never a quiet maybe."""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from kiki.config.runtime import load_runtime
from kiki.ipc.paths import runtime_dir
from kiki.paths import user_data_dir


def _ok(flag: bool) -> str:
    return "OK" if flag else "FAIL"


def _ping(path: Path, payload: dict | None = None, timeout: float = 0.6) -> dict:
    if not path.exists():
        return {"ok": False, "error": "socket missing"}
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        sock.connect(str(path))
        sock.sendall(json.dumps(payload or {"command": "healthz"}).encode("utf-8") + b"\n")
        line = sock.makefile("r", encoding="utf-8").readline()
        return json.loads(line) if line.strip() else {"ok": False, "error": "empty reply"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        sock.close()


def _user_unit(name: str) -> str:
    proc = subprocess.run(
        ["systemctl", "--user", "is-active", name],
        capture_output=True,
        text=True,
        check=False,
    )
    return (proc.stdout or proc.stderr or "").strip() or "unknown"


def run_diagnostics(*, as_json: bool = False) -> int:
    cfg = load_runtime()
    rt = cfg.socket_dir if cfg.socket_dir.exists() else runtime_dir()
    stt_py = user_data_dir() / "stt-venv" / "bin" / "python"
    kokoro_py = user_data_dir() / "kokoro-venv" / "bin" / "python"
    wake_model = user_data_dir() / "wake" / f"{cfg.wake.model_name}.onnx"
    vad_onnx = user_data_dir() / "vad" / "silero_vad.onnx"
    piper_hint = user_data_dir() / "piper"

    checks: list[dict[str, object]] = []

    def add(name: str, ok: bool, detail: str, required: bool = True) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail, "required": required})

    # GPU
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        try:
            out = subprocess.check_output(
                [nvidia, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
                encoding="utf-8",
                timeout=3,
            ).strip()
            add("gpu", True, out)
        except Exception as exc:
            add("gpu", False, str(exc))
    else:
        add("gpu", False, "nvidia-smi fehlt")

    add("pipewire_pwcat", shutil.which("pw-cat") is not None, "pw-cat")
    add("pipewire_wpctl", shutil.which("wpctl") is not None, "wpctl", required=False)
    add("stt_venv", stt_py.is_file(), str(stt_py))
    add("tts_venv", kokoro_py.is_file(), str(kokoro_py))
    add(
        "wake_model",
        wake_model.is_file(),
        f"{wake_model} — ohne diese Datei gibt es kein Weckwort, nur Hotkey/Klick",
        required=False,
    )
    add("silero_vad_onnx", vad_onnx.is_file(), str(vad_onnx))

    if stt_py.is_file():
        try:
            out = subprocess.check_output(
                [
                    str(stt_py),
                    "-c",
                    "import ctranslate2; print(ctranslate2.get_cuda_device_count())",
                ],
                encoding="utf-8",
                timeout=20,
            ).strip()
            add("stt_cuda", int(out) > 0, f"CUDA devices={out}")
        except Exception as exc:
            add("stt_cuda", False, str(exc))

    if kokoro_py.is_file():
        try:
            out = subprocess.check_output(
                [
                    str(kokoro_py),
                    "-c",
                    "import torch; print(torch.cuda.is_available())",
                ],
                encoding="utf-8",
                timeout=20,
            ).strip()
            add("tts_cuda", out.strip() in {"True", "true", "1"}, f"torch.cuda={out}")
        except Exception as exc:
            add("tts_cuda", False, str(exc))

    services = [
        "kiki-audio.service",
        "kiki-stt.service",
        "kiki-tts.service",
        "kiki-orchestrator.service",
        "kiki-pet.service",
    ]
    for svc in services:
        status = _user_unit(svc)
        add(svc, status == "active", status, required=False)

    for name in ("audio", "stt", "tts", "ui"):
        path = rt / f"{name}.sock"
        reply = _ping(path)
        ready = bool(
            reply.get("ready")
            or reply.get("status") in {"healthy", "degraded"}
            or (path.exists() and "error" not in reply)
        )
        add(f"socket_{name}", ready, str(reply)[:180], required=False)

    required_fail = [c for c in checks if c["required"] and not c["ok"]]
    code = 1 if required_fail else 0

    if as_json:
        json.dump({"ok": code == 0, "checks": checks, "runtime": str(rt)}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return code

    print("=" * 64)
    print("KIKI voice stack")
    print("=" * 64)
    print(f"runtime dir: {rt}")
    print(f"stt model:   {cfg.stt.model_name} device={cfg.stt.device}")
    print(f"tts primary: {cfg.tts.primary_engine}/{cfg.tts.primary_voice}")
    print(f"llm:         {cfg.llm.model} @ {cfg.llm.base_url}")
    print()
    for check in checks:
        mark = _ok(bool(check["ok"]))
        req = "required" if check["required"] else "optional"
        print(f"[{mark:4}] {check['name']:28} ({req}) {check['detail']}")
    print()
    if required_fail:
        print("Gesetz 1: fehlende Pflichtteile werden nicht still ersetzt.")
        print("Behebe die FAIL-Zeilen, bevor KIKI so tut, als würde sie hören oder sprechen.")
    else:
        print("Pflichtteile sichtbar. Weckwort braucht weiterhin das KIKI-ONNX.")
    print(f"piper voices dir (secondary): {piper_hint}")
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KIKI voice-stack doctor")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    return run_diagnostics(as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
