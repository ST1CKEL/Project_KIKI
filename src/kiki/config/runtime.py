"""Voice-first runtime configuration (`runtime.toml`). Separate from GTK settings."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kiki.ipc.paths import runtime_dir
from kiki.paths import config_dir

_PACKAGED = Path(__file__).with_name("runtime.toml")


class RuntimeConfigError(Exception):
    """runtime.toml is missing a required honest setting."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def _read(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def runtime_config_path() -> Path:
    env = os.environ.get("KIKI_RUNTIME_TOML", "").strip()
    if env:
        return Path(env)
    return config_dir() / "runtime.toml"


@dataclass(frozen=True)
class HardwareConfig:
    gpu_device_id: int = 0
    vram_safety_margin_mb: int = 2048
    llm_target_vram_mb: int = 5200
    vision_vram_mb: int = 4500


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    frame_ms: int = 32
    pre_roll_ms: int = 400
    max_turn_ms: int = 8000
    listen_cue: bool = True


@dataclass(frozen=True)
class VadConfig:
    enabled: bool = True
    model: str = "silero_vad"
    threshold: float = 0.55
    min_speech_ms: int = 220
    min_silence_ms: int = 280
    speech_pad_ms: int = 160


@dataclass(frozen=True)
class WakeConfig:
    enabled: bool = True
    engine: str = "openwakeword"
    model_name: str = "kiki"
    threshold: float = 0.58
    consecutive_frames: int = 2
    cooldown_ms: int = 1200
    barge_in_enabled: bool = True
    barge_in_ms: int = 120


@dataclass(frozen=True)
class SttConfig:
    backend: str = "faster-whisper"
    model_name: str = "Systran/faster-whisper-large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "de"
    beam_size: int = 2
    # Law 1: CPU is not an auto fallback. It is allowed only when set here.
    allow_cpu: bool = False


@dataclass(frozen=True)
class TtsConfig:
    primary_engine: str = "kokoro"
    primary_voice: str = "df_eva"
    secondary_engine: str = "piper"
    secondary_voice: str = "de_DE-eva_k-medium"
    language: str = "de"
    sample_rate: int = 24000
    cache_phrases: bool = True
    # Law 1: espeak-ng is never a hidden last resort.
    allow_espeak: bool = False


@dataclass(frozen=True)
class LlmConfig:
    provider: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:7b-instruct-q4_K_M"
    temperature: float = 0.6
    num_ctx: int = 8192
    vision_model: str = "qwen2.5-vl:7b-instruct-q4_K_M"


@dataclass(frozen=True)
class PolicyConfig:
    autonomy: str = "balanced"
    confirmation_timeout_s: int = 20


@dataclass(frozen=True)
class RuntimeConfig:
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    socket_dir: Path = field(default_factory=runtime_dir)
    raw: dict[str, Any] = field(default_factory=dict)

    def socket(self, name: str) -> Path:
        from kiki.ipc.paths import socket_path

        return socket_path(name, runtime=self.socket_dir)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name) or {}
    return value if isinstance(value, dict) else {}


def load_runtime(path: Path | None = None) -> RuntimeConfig:
    mapping = _read(_PACKAGED) if _PACKAGED.is_file() else {}
    user_path = path or runtime_config_path()
    if user_path.is_file() and user_path.resolve() != _PACKAGED.resolve():
        mapping = _deep_merge(mapping, _read(user_path))
    hw = _section(mapping, "hardware")
    audio = _section(mapping, "audio")
    vad = _section(mapping, "vad")
    vad_modes = vad.get("modes") if isinstance(vad.get("modes"), dict) else {}
    command_vad = vad_modes.get("command") if isinstance(vad_modes, dict) else {}
    wake = _section(mapping, "wakeword") or _section(mapping, "wake")
    stt = _section(mapping, "stt")
    tts = _section(mapping, "tts")
    llm = _section(mapping, "llm")
    vision = _section(mapping, "vision_agent")
    policy = _section(mapping, "policy")
    ipc = _section(mapping, "ipc")

    socket_dir_raw = str(ipc.get("socket_dir") or "").strip()
    if socket_dir_raw and "/1000/" in socket_dir_raw and os.getuid() != 1000:
        # A copied example still pointing at another user's runtime dir.
        socket_dir_raw = ""
    sock_dir = runtime_dir(socket_dir_raw or None)

    stt_device = str(stt.get("device") or "cuda").strip().lower()
    allow_cpu = bool(stt.get("allow_cpu", False)) or stt_device == "cpu"

    return RuntimeConfig(
        hardware=HardwareConfig(
            gpu_device_id=int(hw.get("gpu_device_id") or 0),
            vram_safety_margin_mb=int(hw.get("vram_safety_margin_mb") or 2048),
            llm_target_vram_mb=int(llm.get("target_vram_mb") or 5200),
            vision_vram_mb=int(vision.get("vram_requirement_mb") or 4500),
        ),
        audio=AudioConfig(
            sample_rate=int(audio.get("sample_rate_processing") or audio.get("sample_rate") or 16000),
            frame_ms=int(audio.get("frame_duration_ms") or 32),
            pre_roll_ms=int(audio.get("pre_roll_buffer_ms") or 400),
            max_turn_ms=int(audio.get("max_turn_ms") or 8000),
            listen_cue=bool(audio.get("listen_cue", True)),
        ),
        vad=VadConfig(
            enabled=bool(vad.get("enabled", True)),
            model=str(vad.get("model") or "silero_vad"),
            threshold=float(vad.get("threshold") or 0.55),
            min_speech_ms=int(vad.get("min_speech_ms") or 220),
            min_silence_ms=int(
                (command_vad or {}).get("min_silence_ms") or vad.get("min_silence_ms") or 280
            ),
            speech_pad_ms=int(vad.get("speech_pad_ms") or 160),
        ),
        wake=WakeConfig(
            enabled=bool(wake.get("enabled", True)),
            engine=str(wake.get("engine") or "openwakeword"),
            model_name=str(wake.get("model_name") or "kiki"),
            threshold=float(wake.get("threshold") or 0.58),
            consecutive_frames=int(wake.get("consecutive_frames") or 2),
            cooldown_ms=int(wake.get("cooldown_ms") or 1200),
            barge_in_enabled=bool(wake.get("barge_in_enabled", True)),
            barge_in_ms=int(wake.get("barge_in_fadeout_ms") or wake.get("barge_in_ms") or 120),
        ),
        stt=SttConfig(
            backend=str(stt.get("backend") or "faster-whisper"),
            model_name=str(stt.get("model_name") or "Systran/faster-whisper-large-v3-turbo"),
            device=stt_device,
            compute_type=str(stt.get("compute_type") or "float16"),
            language=str(stt.get("language") or "de"),
            beam_size=int(stt.get("beam_size") or 2),
            allow_cpu=allow_cpu,
        ),
        tts=TtsConfig(
            primary_engine=str(tts.get("primary_engine") or "kokoro"),
            primary_voice=str(tts.get("primary_voice") or "df_eva"),
            secondary_engine=str(tts.get("secondary_engine") or "piper"),
            secondary_voice=str(tts.get("secondary_voice") or "de_DE-eva_k-medium"),
            language=str(tts.get("language") or "de"),
            sample_rate=int(tts.get("sample_rate") or 24000),
            cache_phrases=bool(tts.get("cache_phrases", True)),
            allow_espeak=bool(tts.get("allow_espeak", False)),
        ),
        llm=LlmConfig(
            provider=str(llm.get("provider") or "ollama"),
            base_url=str(llm.get("base_url") or "http://127.0.0.1:11434"),
            model=str(llm.get("model") or "qwen2.5:7b-instruct-q4_K_M"),
            temperature=float(llm.get("temperature") or 0.6),
            num_ctx=int(llm.get("num_ctx") or 8192),
            vision_model=str(vision.get("model") or "qwen2.5-vl:7b-instruct-q4_K_M"),
        ),
        policy=PolicyConfig(
            autonomy=str(policy.get("autonomy") or "balanced"),
            confirmation_timeout_s=int(policy.get("confirmation_timeout_seconds") or 20),
        ),
        socket_dir=sock_dir,
        raw=mapping,
    )
