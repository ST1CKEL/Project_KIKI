"""TOML configuration with safe defaults and atomic writes."""

from __future__ import annotations

import logging
import tomllib
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kiki.paths import config_path

log = logging.getLogger(__name__)

_DEFAULTS_FILE = Path(__file__).with_name("defaults.toml")

VALID_PROVIDERS = ("ollama", "openai_compatible", "kiki_harness")
VALID_ANCHORS = ("bottom-right", "bottom-left", "top-right", "top-left")
VALID_TTS_SPEAKERS = (
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
)
DEFAULT_WORKSPACE_ROOTS: tuple[str, ...] = (
    "~/Projects",
    "~/Code",
    "~/Projekte",
    "~/Dokumente/Projekte",
)
VALID_TTS_LANGUAGES = (
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
)


class SettingsError(Exception):
    """Invalid configuration."""


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def default_mapping() -> dict[str, Any]:
    return _read_toml(_DEFAULTS_FILE)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _require_http_url(value: str, field_name: str) -> str:
    text = value.strip().rstrip("/")
    if not (text.startswith("http://") or text.startswith("https://")):
        raise SettingsError(f"{field_name} must be an http(s) URL")
    return text


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _bounded_int(value: Any, *, default: int, low: int, high: int) -> int:
    """Read an integer budget, falling back to the default on anything unusable."""
    if isinstance(value, bool) or value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return int(_clamp(number, low, high))


@dataclass
class AppSettings:
    autostart: bool = False
    privacy_panic: bool = False
    greet_on_start: bool = True


@dataclass
class PetSettings:
    scale: float = 1.0
    always_on_top: bool = True
    click_through_idle: bool = True
    anchor: str = "bottom-right"
    height_px: int = 260
    # Best-effort last position. -1 means unset. Wayland cannot restore this.
    last_x: int = -1
    last_y: int = -1


@dataclass
class CharacterSettings:
    id: str = "kiki-adult-v3"


@dataclass
class OllamaSettings:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3-vl:4b"
    # Ollama's own default is 4096, which system prompt + memories + history +
    # a thinking model's deliberation can exhaust before any answer appears.
    num_ctx: int = 8192
    # Ask a thinking-capable model to skip deliberation. Measured on
    # qwen3-vl:4b: "say only Hallo" took 10.5 s and 4841 characters of thinking
    # with it on, 4.9 s with it off. Harmless for models without the mode.
    think: bool = False
    # Prefill a closed, empty reasoning block so a thinking model cannot open
    # one. Measured 43.0 s -> 0.4 s to the first token; harmless otherwise.
    suppress_thinking: bool = True


@dataclass
class KikiHarnessSettings:
    """KIKI's own LLM service. Loopback only, like every other KIKI service."""

    base_url: str = "http://127.0.0.1:18770"
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    quantize: str = "int4"
    slots: int = 2


@dataclass
class OpenAICompatibleSettings:
    base_url: str = "https://api.x.ai/v1"
    model: str = "grok-4.5"


@dataclass
class AISettings:
    provider: str = "ollama"
    temperature: float = 0.7
    history_limit: int = 40
    # The personality half only. The invariant rules live in kiki.ai.persona and
    # are never stored here, so they cannot go stale in a user's config.
    system_prompt: str = ""
    ollama: OllamaSettings = field(default_factory=OllamaSettings)
    kiki_harness: KikiHarnessSettings = field(default_factory=KikiHarnessSettings)
    openai_compatible: OpenAICompatibleSettings = field(default_factory=OpenAICompatibleSettings)


@dataclass
class PersonaSettings:
    id: str = "begleiterin"
    # How KIKI addresses the user. Empty means she does not use a name.
    address: str = ""


@dataclass
class IntegrationToggle:
    enabled: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationsSettings:
    enabled: bool = True
    status_cards: bool = True
    datetime: IntegrationToggle = field(default_factory=IntegrationToggle)
    upower: IntegrationToggle = field(default_factory=IntegrationToggle)
    networkmanager: IntegrationToggle = field(default_factory=IntegrationToggle)
    disk: IntegrationToggle = field(default_factory=IntegrationToggle)


@dataclass
class ToolsSettings:
    model_tool_use: bool = False
    # "strict": the model may read unattended. "balanced": reads plus the
    # declared safety controls. "trusted": additionally opens folders, files,
    # terminal, editor and http(s) links inside registered workspaces.
    # "jarvis": acts unattended at every risk level — writes and external
    # actions included. Opt-in only; the hard deny list, the panic switch and
    # every spec that withheld auto_allow still bite at this level.
    autonomy: str = "balanced"
    max_steps: int = 6
    max_tool_calls: int = 12


@dataclass
class ScreenshotSettings:
    enabled: bool = True
    interactive: bool = True


@dataclass
class WakeSettings:
    # Off by default: an always-open microphone is the user's decision, not a
    # default that arrives with an update.
    enabled: bool = False
    phrases: tuple[str, ...] = ("kiki",)
    cooldown_ms: int = 2000
    command_timeout_s: int = 12
    # Once an explicitly woken conversation has answered, accept exactly one
    # more utterance without another wake word. The wake feature itself remains
    # off by default, so this never opens a microphone on its own.
    follow_up: bool = True


@dataclass
class ResponsePolicySettings:
    """What KIKI may read out loud.

    Every default is False: speech is a channel that can be overheard and
    recorded, so a category is spoken only once it has been switched on
    deliberately. Nothing here limits *how much* KIKI says -- she speaks the
    whole answer -- only what is removed from it first.
    """

    speak_code: bool = False
    speak_logs: bool = False
    speak_urls: bool = False
    speak_paths: bool = False
    speak_tables: bool = False
    speak_secrets: bool = False
    # Whole-answer policy for turns that started at the microphone. The full
    # answer stays in chat; only its spoken companion is shortened.
    concise_answers: bool = True
    open_chat_for_details: bool = True


@dataclass
class VoiceSettings:
    enabled: bool = True
    auto_send: bool = True
    wake: WakeSettings = field(default_factory=WakeSettings)
    response_policy: ResponsePolicySettings = field(default_factory=ResponsePolicySettings)


@dataclass
class TtsSettings:
    enabled: bool = True
    base_url: str = "http://127.0.0.1:18765"
    speaker: str = "Serena"
    language: str = "German"
    stream_sentences: bool = True
    fallback_to_system: bool = True
    # Opt-in: speak through VoicePlaybackController instead of the file-based
    # route. Off until a manual end-to-end test confirms the new path.
    use_controller_route: bool = False


@dataclass
class WatchSettings:
    """Proactive notices. KIKI may speak up, but never act, on her own."""

    enabled: bool = True
    speak: bool = True
    interval_s: int = 60
    quiet_start: str = "22:00"
    quiet_end: str = "08:00"
    cooldown_s: int = 1800
    max_per_hour: int = 6
    battery_enabled: bool = True
    battery_percent: int = 20
    disk_enabled: bool = True
    disk_percent: int = 90


@dataclass
class WorkspaceSettings:
    allowed_roots: tuple[str, ...] = DEFAULT_WORKSPACE_ROOTS


@dataclass
class AgentsSettings:
    opencode_binary: str = "opencode"
    default_model: str = ""
    plan_first: bool = True


@dataclass
class Settings:
    app: AppSettings = field(default_factory=AppSettings)
    pet: PetSettings = field(default_factory=PetSettings)
    character: CharacterSettings = field(default_factory=CharacterSettings)
    ai: AISettings = field(default_factory=AISettings)
    persona: PersonaSettings = field(default_factory=PersonaSettings)
    integrations: IntegrationsSettings = field(default_factory=IntegrationsSettings)
    tools: ToolsSettings = field(default_factory=ToolsSettings)
    screenshot: ScreenshotSettings = field(default_factory=ScreenshotSettings)
    voice: VoiceSettings = field(default_factory=VoiceSettings)
    tts: TtsSettings = field(default_factory=TtsSettings)
    watch: WatchSettings = field(default_factory=WatchSettings)
    workspaces: WorkspaceSettings = field(default_factory=WorkspaceSettings)
    agents: AgentsSettings = field(default_factory=AgentsSettings)
    extra: dict[str, Any] = field(default_factory=dict)

    def persona_prompt(self) -> str:
        """The personality half: the chosen preset, or the user's own text."""
        from kiki.ai.persona import CUSTOM_ID, get_persona

        if self.persona.id != CUSTOM_ID:
            preset = get_persona(self.persona.id)
            if preset is not None:
                return preset.prompt
        return self.ai.system_prompt

    def compose_prompt(self) -> str:
        """The full system prompt actually sent to a model."""
        from kiki.ai.persona import compose

        return compose(self.persona_prompt(), address=self.persona.address)

    def integrations_active(self) -> bool:
        return self.integrations.enabled and not self.app.privacy_panic

    def screenshot_allowed(self) -> bool:
        return self.screenshot.enabled and self.integrations_active()

    def voice_allowed(self) -> bool:
        return self.voice.enabled and not self.app.privacy_panic

    def tts_allowed(self) -> bool:
        return self.tts.enabled and not self.app.privacy_panic

    def to_mapping(self) -> dict[str, Any]:
        return {
            "app": {
                "autostart": self.app.autostart,
                "privacy_panic": self.app.privacy_panic,
                "greet_on_start": self.app.greet_on_start,
            },
            "pet": {
                "scale": self.pet.scale,
                "always_on_top": self.pet.always_on_top,
                "click_through_idle": self.pet.click_through_idle,
                "anchor": self.pet.anchor,
                "height_px": self.pet.height_px,
                "last_x": self.pet.last_x,
                "last_y": self.pet.last_y,
            },
            "character": {"id": self.character.id},
            "ai": {
                "provider": self.ai.provider,
                "temperature": self.ai.temperature,
                "history_limit": self.ai.history_limit,
                "system_prompt": self.ai.system_prompt,
                "ollama": {
                    "base_url": self.ai.ollama.base_url,
                    "model": self.ai.ollama.model,
                    "num_ctx": self.ai.ollama.num_ctx,
                    "think": self.ai.ollama.think,
                    "suppress_thinking": self.ai.ollama.suppress_thinking,
                },
                "kiki_harness": {
                    "base_url": self.ai.kiki_harness.base_url,
                    "model": self.ai.kiki_harness.model,
                    "quantize": self.ai.kiki_harness.quantize,
                    "slots": self.ai.kiki_harness.slots,
                },
                "openai_compatible": {
                    "base_url": self.ai.openai_compatible.base_url,
                    "model": self.ai.openai_compatible.model,
                },
            },
            "persona": {
                "id": self.persona.id,
                "address": self.persona.address,
            },
            "integrations": {
                "enabled": self.integrations.enabled,
                "status_cards": self.integrations.status_cards,
                "datetime": {"enabled": self.integrations.datetime.enabled},
                "upower": {"enabled": self.integrations.upower.enabled},
                "networkmanager": {"enabled": self.integrations.networkmanager.enabled},
                "disk": {
                    "enabled": self.integrations.disk.enabled,
                    **self.integrations.disk.extra,
                },
            },
            "tools": {
                "model_tool_use": self.tools.model_tool_use,
                "autonomy": self.tools.autonomy,
                "max_steps": self.tools.max_steps,
                "max_tool_calls": self.tools.max_tool_calls,
            },
            "screenshot": {
                "enabled": self.screenshot.enabled,
                "interactive": self.screenshot.interactive,
            },
            "voice": {
                "enabled": self.voice.enabled,
                "auto_send": self.voice.auto_send,
                "wake": {
                    "enabled": self.voice.wake.enabled,
                    "phrases": list(self.voice.wake.phrases),
                    "cooldown_ms": self.voice.wake.cooldown_ms,
                    "command_timeout_s": self.voice.wake.command_timeout_s,
                    "follow_up": self.voice.wake.follow_up,
                },
                "response_policy": {
                    "speak_code": self.voice.response_policy.speak_code,
                    "speak_logs": self.voice.response_policy.speak_logs,
                    "speak_urls": self.voice.response_policy.speak_urls,
                    "speak_paths": self.voice.response_policy.speak_paths,
                    "speak_tables": self.voice.response_policy.speak_tables,
                    "speak_secrets": self.voice.response_policy.speak_secrets,
                    "concise_answers": self.voice.response_policy.concise_answers,
                    "open_chat_for_details": self.voice.response_policy.open_chat_for_details,
                },
            },
            "tts": {
                "enabled": self.tts.enabled,
                "base_url": self.tts.base_url,
                "speaker": self.tts.speaker,
                "language": self.tts.language,
                "stream_sentences": self.tts.stream_sentences,
                "fallback_to_system": self.tts.fallback_to_system,
                "use_controller_route": self.tts.use_controller_route,
            },
            "watch": {
                "enabled": self.watch.enabled,
                "speak": self.watch.speak,
                "interval_s": self.watch.interval_s,
                "quiet_start": self.watch.quiet_start,
                "quiet_end": self.watch.quiet_end,
                "cooldown_s": self.watch.cooldown_s,
                "max_per_hour": self.watch.max_per_hour,
                "battery": {
                    "enabled": self.watch.battery_enabled,
                    "percent": self.watch.battery_percent,
                },
                "disk": {
                    "enabled": self.watch.disk_enabled,
                    "percent": self.watch.disk_percent,
                },
            },
            "workspaces": {
                "allowed_roots": list(self.workspaces.allowed_roots),
            },
            "agents": {
                "opencode_binary": self.agents.opencode_binary,
                "default_model": self.agents.default_model,
                "plan_first": self.agents.plan_first,
            },
        }

    def pet_height(self) -> int:
        return max(120, int(self.pet.height_px * self.pet.scale))


def settings_from_mapping(data: dict[str, Any]) -> Settings:
    app = data.get("app", {})
    pet = data.get("pet", {})
    character = data.get("character", {})
    ai = data.get("ai", {})
    ollama = ai.get("ollama", {})
    oai = ai.get("openai_compatible", {})
    integ = data.get("integrations", {})
    tools = data.get("tools", {})
    shot = data.get("screenshot", {})
    voice = data.get("voice", {})
    tts = data.get("tts", {})
    workspaces = data.get("workspaces", {})
    agents = data.get("agents", {})

    provider = str(ai.get("provider", "ollama"))
    if provider not in VALID_PROVIDERS:
        raise SettingsError(f"unknown AI provider: {provider}")
    anchor = str(pet.get("anchor", "bottom-right"))
    if anchor not in VALID_ANCHORS:
        raise SettingsError(f"unknown pet anchor: {anchor}")

    ollama_url = _require_http_url(str(ollama.get("base_url", "http://127.0.0.1:11434")), "ai.ollama.base_url")
    oai_url = _require_http_url(str(oai.get("base_url", "https://api.x.ai/v1")), "ai.openai_compatible.base_url")
    tts_url = _require_http_url(str(tts.get("base_url", "http://127.0.0.1:18765")), "tts.base_url")
    speaker = str(tts.get("speaker", "Serena")).strip() or "Serena"
    if speaker not in VALID_TTS_SPEAKERS:
        speaker = "Serena"
    language = str(tts.get("language", "German")).strip() or "German"
    if language not in VALID_TTS_LANGUAGES:
        language = "German"

    disk = integ.get("disk", {})
    return Settings(
        app=AppSettings(
            autostart=bool(app.get("autostart", False)),
            privacy_panic=bool(app.get("privacy_panic", False)),
            greet_on_start=bool(app.get("greet_on_start", True)),
        ),
        pet=PetSettings(
            scale=float(_clamp(float(pet.get("scale", 1.0)), 0.5, 2.5)),
            always_on_top=bool(pet.get("always_on_top", True)),
            click_through_idle=bool(pet.get("click_through_idle", True)),
            anchor=anchor,
            height_px=int(_clamp(int(pet.get("height_px", 260)), 120, 640)),
            last_x=int(pet.get("last_x", -1)),
            last_y=int(pet.get("last_y", -1)),
        ),
        character=CharacterSettings(id=str(character.get("id", "kiki-adult-v3"))),
        ai=AISettings(
            provider=provider,
            temperature=float(_clamp(float(ai.get("temperature", 0.7)), 0.0, 2.0)),
            history_limit=int(_clamp(int(ai.get("history_limit", 40)), 2, 200)),
            system_prompt=str(ai.get("system_prompt", "")),
            ollama=OllamaSettings(
                base_url=ollama_url,
                model=str(ollama.get("model", "qwen3-vl:4b")).strip() or "qwen3-vl:4b",
                num_ctx=_bounded_int(ollama.get("num_ctx"), default=8192, low=2048, high=131072),
                think=bool(ollama.get("think", False)),
                suppress_thinking=bool(ollama.get("suppress_thinking", True)),
            ),
            kiki_harness=KikiHarnessSettings(
                base_url=_require_http_url(
                    str((ai.get("kiki_harness") or {}).get("base_url", "http://127.0.0.1:18770")),
                    "ai.kiki_harness.base_url",
                ),
                model=str((ai.get("kiki_harness") or {}).get("model", "Qwen/Qwen3-4B-Instruct-2507")).strip()
                or "Qwen/Qwen3-4B-Instruct-2507",
                quantize=(
                    str((ai.get("kiki_harness") or {}).get("quantize", "int4")).strip().lower()
                    if str((ai.get("kiki_harness") or {}).get("quantize", "int4")).strip().lower()
                    in {"none", "int8", "int4"}
                    else "int4"
                ),
                slots=_bounded_int((ai.get("kiki_harness") or {}).get("slots"), default=2, low=1, high=8),
            ),
            openai_compatible=OpenAICompatibleSettings(
                base_url=oai_url,
                model=str(oai.get("model", "grok-4.5")).strip() or "grok-4.5",
            ),
        ),
        persona=_parse_persona(data.get("persona") or {}, ai),
        integrations=IntegrationsSettings(
            enabled=bool(integ.get("enabled", True)),
            status_cards=bool(integ.get("status_cards", True)),
            datetime=IntegrationToggle(enabled=bool(integ.get("datetime", {}).get("enabled", True))),
            upower=IntegrationToggle(enabled=bool(integ.get("upower", {}).get("enabled", True))),
            networkmanager=IntegrationToggle(
                enabled=bool(integ.get("networkmanager", {}).get("enabled", True))
            ),
            disk=IntegrationToggle(
                enabled=bool(disk.get("enabled", True)),
                extra={"path": str(disk.get("path", ""))} if "path" in disk else {},
            ),
        ),
        tools=ToolsSettings(
            model_tool_use=bool(tools.get("model_tool_use", False)),
            autonomy=str(tools.get("autonomy", "balanced")),
            max_steps=_bounded_int(tools.get("max_steps"), default=6, low=1, high=20),
            max_tool_calls=_bounded_int(tools.get("max_tool_calls"), default=12, low=0, high=64),
        ),
        screenshot=ScreenshotSettings(
            enabled=bool(shot.get("enabled", True)),
            interactive=bool(shot.get("interactive", True)),
        ),
        voice=VoiceSettings(
            enabled=bool(voice.get("enabled", True)),
            auto_send=bool(voice.get("auto_send", True)),
            wake=_parse_wake(voice.get("wake") or {}),
            response_policy=_parse_response_policy(voice.get("response_policy") or {}),
        ),
        tts=TtsSettings(
            enabled=bool(tts.get("enabled", True)),
            base_url=tts_url,
            speaker=speaker,
            language=language,
            stream_sentences=bool(tts.get("stream_sentences", True)),
            fallback_to_system=bool(tts.get("fallback_to_system", True)),
            # Second safety net: a mapping that predates the key must not
            # switch the route on by accident.
            use_controller_route=bool(tts.get("use_controller_route", False)),
        ),
        watch=_parse_watch(data.get("watch") or {}),
        workspaces=WorkspaceSettings(allowed_roots=_parse_workspace_roots(workspaces.get("allowed_roots"))),
        agents=AgentsSettings(
            opencode_binary=str(agents.get("opencode_binary", "opencode")).strip() or "opencode",
            default_model=str(agents.get("default_model", "")).strip(),
            plan_first=bool(agents.get("plan_first", True)),
        ),
    )


def _parse_workspace_roots(value: Any) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_WORKSPACE_ROOTS
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = [str(item) for item in value]
    else:
        raise SettingsError("workspaces.allowed_roots must be a list of strings")
    cleaned: list[str] = []
    for item in items:
        text = item.strip()
        if not text:
            continue
        if text in {"/", "~", "$HOME"}:
            raise SettingsError("workspaces.allowed_roots must not be / or $HOME")
        cleaned.append(text)
    if not cleaned:
        raise SettingsError("workspaces.allowed_roots must not be empty")
    return tuple(cleaned)


def _parse_watch(data: dict[str, Any]) -> WatchSettings:
    battery = data.get("battery") or {}
    disk = data.get("disk") or {}
    return WatchSettings(
        enabled=bool(data.get("enabled", True)),
        speak=bool(data.get("speak", True)),
        interval_s=_bounded_int(data.get("interval_s"), default=60, low=15, high=3600),
        quiet_start=str(data.get("quiet_start", "22:00")).strip() or "22:00",
        quiet_end=str(data.get("quiet_end", "08:00")).strip() or "08:00",
        cooldown_s=_bounded_int(data.get("cooldown_s"), default=1800, low=0, high=86400),
        max_per_hour=_bounded_int(data.get("max_per_hour"), default=6, low=0, high=60),
        battery_enabled=bool(battery.get("enabled", True)),
        battery_percent=_bounded_int(battery.get("percent"), default=20, low=1, high=95),
        disk_enabled=bool(disk.get("enabled", True)),
        disk_percent=_bounded_int(disk.get("percent"), default=90, low=50, high=99),
    )


def _parse_persona(data: dict[str, Any], ai: dict[str, Any]) -> PersonaSettings:
    """Read the persona, migrating configs written before presets existed.

    Such a config has a hand-written or older built-in `ai.system_prompt` and no
    `[persona]` table. Treating it as the custom persona keeps the user's own
    wording; the invariant rules are supplied by the package either way.
    """
    from kiki.ai.persona import CUSTOM_ID, DEFAULT_PERSONA_ID, valid_persona_ids

    raw = str(data.get("id", "")).strip().lower()
    if raw in valid_persona_ids():
        chosen = raw
    elif str(ai.get("system_prompt", "")).strip():
        chosen = CUSTOM_ID
    else:
        chosen = DEFAULT_PERSONA_ID
    return PersonaSettings(
        id=chosen,
        address=" ".join(str(data.get("address", "")).split())[:60],
    )


def _parse_response_policy(data: dict[str, Any]) -> ResponsePolicySettings:
    """Anything unreadable falls back to not speaking that category.

    Fail closed: a damaged config must not be the reason a secret is read out.
    """

    def allowed(name: str) -> bool:
        return data.get(name) is True

    def enabled_unless_explicitly_disabled(name: str) -> bool:
        # Broken hand-written config must stay concise and keep omitted detail
        # visible. Only a real TOML false relaxes either behaviour.
        return data.get(name, True) is not False

    return ResponsePolicySettings(
        speak_code=allowed("speak_code"),
        speak_logs=allowed("speak_logs"),
        speak_urls=allowed("speak_urls"),
        speak_paths=allowed("speak_paths"),
        speak_tables=allowed("speak_tables"),
        speak_secrets=allowed("speak_secrets"),
        concise_answers=enabled_unless_explicitly_disabled("concise_answers"),
        open_chat_for_details=enabled_unless_explicitly_disabled(
            "open_chat_for_details"
        ),
    )


def _parse_wake(data: dict[str, Any]) -> WakeSettings:
    raw = data.get("phrases")
    phrases: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            text = " ".join(str(item).lower().split())
            if text and text not in phrases:
                phrases.append(text)
    if not phrases:
        phrases = ["kiki"]
    return WakeSettings(
        # An unreadable or empty phrase list must never leave the microphone
        # armed with nothing to match, so it falls back to the default word.
        enabled=bool(data.get("enabled", False)),
        phrases=tuple(phrases),
        cooldown_ms=_bounded_int(data.get("cooldown_ms"), default=2000, low=0, high=30000),
        command_timeout_s=_bounded_int(data.get("command_timeout_s"), default=12, low=2, high=120),
        follow_up=data.get("follow_up", True) is True,
    )


def load_settings(path: Path | None = None) -> Settings:
    mapping = default_mapping()
    target = path or config_path()
    if target.is_file():
        try:
            user = _read_toml(target)
            mapping = _deep_merge(mapping, user)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            log.warning("Could not read %s: %s — using defaults", target, exc)
    return settings_from_mapping(mapping)


def save_settings(settings: Settings, path: Path | None = None) -> None:
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _mapping_to_toml(settings.to_mapping())
    tmp = target.with_suffix(".toml.tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(target)


def _escape(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        if "\n" in value:
            # Triple-quoted multiline.
            cleaned = value.replace('"""', '\\"\\"\\"')
            return f'"""{cleaned}"""'
        return f'"{_escape(value)}"'
    if isinstance(value, list):
        inner = ", ".join(_format_value(item) for item in value)
        return f"[{inner}]"
    raise TypeError(f"unsupported TOML value {type(value)}")


def _mapping_to_toml(data: dict[str, Any], prefix: str = "") -> str:
    """Minimal TOML writer for our nested dicts of scalars."""
    chunks: list[str] = []
    scalars = {k: v for k, v in data.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in data.items() if isinstance(v, dict)}
    if prefix and scalars:
        chunks.append(f"[{prefix}]")
    elif not prefix:
        pass
    for key, value in scalars.items():
        chunks.append(f"{key} = {_format_value(value)}")
    if scalars:
        chunks.append("")
    for key, value in tables.items():
        child = f"{prefix}.{key}" if prefix else key
        chunks.append(_mapping_to_toml(value, child))
    return "\n".join(chunks).rstrip() + "\n"
