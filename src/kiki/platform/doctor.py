"""Passive, privacy-safe readiness report for a Fedora KIKI installation.

The doctor reads local facts and may ping configured *loopback* services.  It
never records audio, opens a portal, changes a setting or contacts an external
provider.  Those active checks belong to an explicit future smoke-test mode.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlparse

from kiki.ai.factory import active_model, create_provider
from kiki.config.settings import Settings
from kiki.platform.capabilities import detect_capabilities
from kiki.storage.secrets import MemorySecretStore
from kiki.voice.stt import vosk_model_ready, vosk_runtime_available
from kiki.voice.system_tts import system_tts_available
from kiki.voice.tts_client import tts_health

TARGET_FEDORA = "44"
MIN_RAM_KIB = 7_500_000
MIN_VRAM_MIB = 7_800


class DoctorStatus(StrEnum):
    READY = "ready"
    LIMITED = "limited"
    MISSING = "missing"


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: DoctorStatus
    detail: str
    required: bool = False


@dataclass(frozen=True)
class DoctorReport:
    status: DoctorStatus
    checks: tuple[DoctorCheck, ...]
    schema_version: int = 1

    @property
    def strict_ok(self) -> bool:
        return all(check.status is DoctorStatus.READY for check in self.checks if check.required)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    def to_text(self) -> str:
        marks = {
            DoctorStatus.READY: "✓",
            DoctorStatus.LIMITED: "⚠",
            DoctorStatus.MISSING: "✗",
        }
        lines = ["KIKI Doctor"]
        lines.extend(
            f"{marks[check.status]} {check.name}: {check.detail} [{check.status.value}]"
            for check in self.checks
        )
        lines.append(f"doctor={self.status.value}")
        return "\n".join(lines)


def build_doctor_report(settings: Settings) -> DoctorReport:
    checks = [
        _fedora_check(_read_text(Path("/etc/os-release"))),
        _display_check(),
        _ram_check(_read_text(Path("/proc/meminfo"))),
        _gpu_check(),
        _audio_check(),
        _voice_check(),
        _system_tts_check(),
        _opencode_check(settings.agents.opencode_binary),
    ]
    checks.extend(asyncio.run(_service_checks(settings)))
    if any(c.required and c.status is DoctorStatus.MISSING for c in checks):
        status = DoctorStatus.MISSING
    elif any(c.status is not DoctorStatus.READY for c in checks):
        status = DoctorStatus.LIMITED
    else:
        status = DoctorStatus.READY
    return DoctorReport(status=status, checks=tuple(checks))


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _key_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip()] = value.strip().strip('"')
    return values


def _fedora_check(text: str) -> DoctorCheck:
    values = _key_values(text)
    distro = values.get("ID", "").lower()
    version = values.get("VERSION_ID", "")
    if distro == "fedora" and version == TARGET_FEDORA:
        return DoctorCheck("fedora", DoctorStatus.READY, "Fedora 44", required=True)
    if distro == "fedora":
        return DoctorCheck(
            "fedora",
            DoctorStatus.LIMITED,
            f"Fedora {version or 'unbekannt'}; Ziel ist Fedora 44",
            required=True,
        )
    return DoctorCheck(
        "fedora",
        DoctorStatus.MISSING,
        "Fedora 44 nicht erkannt",
        required=True,
    )


def _display_check() -> DoctorCheck:
    backend = detect_capabilities().display_backend
    if backend in {"wayland", "x11"}:
        detail = "Wayland (primär)" if backend == "wayland" else "X11/XWayland"
        return DoctorCheck("display", DoctorStatus.READY, detail, required=True)
    return DoctorCheck("display", DoctorStatus.MISSING, "keine grafische Sitzung erkannt", required=True)


def _ram_check(text: str) -> DoctorCheck:
    values = _key_values(text.replace(":", "="))
    parts = values.get("MemTotal", "").split()
    raw = parts[0] if parts else ""
    try:
        kib = int(raw)
    except ValueError:
        return DoctorCheck("ram", DoctorStatus.MISSING, "Arbeitsspeicher unbekannt", required=True)
    gib = kib / (1024 * 1024)
    status = DoctorStatus.READY if kib >= MIN_RAM_KIB else DoctorStatus.LIMITED
    return DoctorCheck(
        "ram",
        status,
        f"{gib:.1f} GiB; lokales Ziel mindestens 8 GB",
        required=True,
    )


def _gpu_check() -> DoctorCheck:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return DoctorCheck("gpu", DoctorStatus.LIMITED, "keine NVIDIA-CUDA-GPU erkannt")
    try:
        result = subprocess.run(
            [
                binary,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return DoctorCheck("gpu", DoctorStatus.LIMITED, "GPU-Abfrage fehlgeschlagen")
    memories: list[int] = []
    names: list[str] = []
    for line in result.stdout.splitlines() if result.returncode == 0 else ():
        name, separator, raw_memory = line.rpartition(",")
        try:
            memory = int(raw_memory.strip()) if separator else 0
        except ValueError:
            memory = 0
        if memory:
            memories.append(memory)
            names.append(name.strip())
    if not memories:
        return DoctorCheck("gpu", DoctorStatus.LIMITED, "CUDA-VRAM unbekannt")
    best = max(memories)
    status = DoctorStatus.READY if best >= MIN_VRAM_MIB else DoctorStatus.LIMITED
    label = names[memories.index(best)] or "NVIDIA GPU"
    return DoctorCheck("gpu", status, f"{label}, {best / 1024:.1f} GiB VRAM")


def _audio_check() -> DoctorCheck:
    tools = [name for name in ("pw-play", "pactl") if shutil.which(name)]
    if tools:
        return DoctorCheck("audio", DoctorStatus.READY, f"Audio-Werkzeug: {tools[0]}")
    return DoctorCheck("audio", DoctorStatus.LIMITED, "PipeWire-Werkzeuge fehlen")


def _voice_check() -> DoctorCheck:
    runtime = vosk_runtime_available()
    model = vosk_model_ready()
    if runtime and model:
        return DoctorCheck("stt", DoctorStatus.READY, "Vosk und Sprachmodell bereit")
    if runtime:
        return DoctorCheck("stt", DoctorStatus.LIMITED, "Vosk bereit, Sprachmodell fehlt")
    return DoctorCheck("stt", DoctorStatus.LIMITED, "Vosk-Laufzeit fehlt")


def _system_tts_check() -> DoctorCheck:
    if system_tts_available():
        return DoctorCheck("tts_fallback", DoctorStatus.READY, "System-TTS verfügbar")
    return DoctorCheck("tts_fallback", DoctorStatus.LIMITED, "System-TTS fehlt")


def _opencode_check(binary: str) -> DoctorCheck:
    if shutil.which(binary):
        return DoctorCheck("opencode", DoctorStatus.READY, "OpenCode verfügbar")
    return DoctorCheck("opencode", DoctorStatus.LIMITED, "OpenCode nicht gefunden")


async def _service_checks(settings: Settings) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    provider_url = _provider_url(settings)
    if _is_loopback(provider_url):
        try:
            provider = create_provider(settings, MemorySecretStore())
            health = await asyncio.wait_for(provider.ping(active_model(settings)), timeout=2.0)
        except Exception:
            health = None
        if health is not None and health.ok and health.selected_model_present:
            checks.append(DoctorCheck("llm", DoctorStatus.READY, "lokales Modell bereit"))
        elif health is not None and health.ok:
            checks.append(DoctorCheck("llm", DoctorStatus.LIMITED, "Dienst bereit, Modell fehlt"))
        else:
            checks.append(DoctorCheck("llm", DoctorStatus.LIMITED, "lokaler LLM-Dienst nicht bereit"))
    else:
        checks.append(
            DoctorCheck(
                "llm",
                DoctorStatus.READY,
                "externer Provider konfiguriert; nicht kontaktiert",
            )
        )

    if not settings.tts.enabled:
        checks.append(DoctorCheck("tts_service", DoctorStatus.LIMITED, "TTS deaktiviert"))
    elif not _is_loopback(settings.tts.base_url):
        checks.append(
            DoctorCheck(
                "tts_service",
                DoctorStatus.LIMITED,
                "externer TTS-Dienst wird nicht kontaktiert",
            )
        )
    else:
        try:
            health = await tts_health(settings.tts.base_url, timeout=1.5)
        except Exception:
            health = None
        status = (
            DoctorStatus.READY if health is not None and health.ok and health.ready else DoctorStatus.LIMITED
        )
        detail = (
            "lokaler TTS-Dienst bereit" if status is DoctorStatus.READY else "lokaler TTS-Dienst nicht bereit"
        )
        checks.append(DoctorCheck("tts_service", status, detail))
    return checks


def _provider_url(settings: Settings) -> str:
    if settings.ai.provider == "kiki_harness":
        return settings.ai.kiki_harness.base_url
    if settings.ai.provider == "openai_compatible":
        return settings.ai.openai_compatible.base_url
    return settings.ai.ollama.base_url


def _is_loopback(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host in {"127.0.0.1", "::1", "localhost"}
