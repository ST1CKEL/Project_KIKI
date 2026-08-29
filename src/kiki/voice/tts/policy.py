"""How much of an answer KIKI actually says out loud.

The full answer always stays in the chat. Speech is a shortened, safe companion:
it never carries code, logs, URLs, paths, tables or anything that looks like a
secret, and it is capped in both sentences and characters.

Redaction happens before length limiting on purpose. Dropping a code block first
means the sentence budget is spent on prose the user can actually follow, not on
a fence that would have been skipped anyway.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class VoiceMode(StrEnum):
    SILENT = "silent"
    CONCISE = "concise"
    NORMAL = "normal"
    DETAILED = "detailed"


# --- what must never be spoken ---------------------------------------------

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_UNCLOSED_FENCE = re.compile(r"```.*\Z", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`]*`")
_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]*)\)")
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
# The lookbehind excludes ":" and "/" so "https://example.com" is not read
# as a path — otherwise enabling speak_urls would still mangle every URL.
_PATH = re.compile(r"(?<![\w.:/])(?:~|\.{1,2})?/[\w.\-/]{2,}")
_WINPATH = re.compile(r"\b[A-Za-z]:\\[\w\\.\- ]+")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
# A leading "-" is a markdown bullet far more often than a diff line, so
# single-sign lines only count as diff when a real diff header is present.
_DIFF_HEADER = re.compile(r"^\s*(?:---\s|\+\+\+\s|@@)", re.MULTILINE)
_DIFF_LINE = re.compile(r"^\s*(?:[+-]{3}|@@|[+-])\s?\S.*$", re.MULTILINE)
_LOG_LINE = re.compile(
    r"^\s*(?:\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}|\[(?:DEBUG|INFO|WARN|WARNING|ERROR|TRACE)\]"
    r"|(?:DEBUG|INFO|WARNING|ERROR|TRACE|CRITICAL)\s+[\w.]+:).*$",
    re.MULTILINE | re.IGNORECASE,
)
# Anything shaped like a credential. Deliberately broad: a false positive costs
# a spoken word, a false negative reads a token out loud.
_SECRET = re.compile(
    r"\b(?:sk-|ghp_|gho_|github_pat_|xox[baprs]-|AKIA|ASIA|eyJ[\w-]{8,})[\w\-./+=]{6,}"
    r"|\b(?:api[_-]?key|token|secret|passwor[dt]|passphrase|bearer|credential)s?\b\s*[:=]\s*\S+",
    re.IGNORECASE,
)
_MD_MARKUP = re.compile(r"[*_~#>]+")
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.MULTILINE)
_SPACE = re.compile(r"[ \t]+")
# A terminator only ends a sentence when whitespace or the end follows.
# Without the lookahead "https://example.com" splits after "example.".
_SENTENCE = re.compile(r".+?(?:[.!?…]+(?=\s|$)|$)", re.DOTALL)


@dataclass(frozen=True)
class VoicePolicyConfig:
    default_mode: VoiceMode = VoiceMode.CONCISE
    concise_max_sentences: int = 2
    concise_max_characters: int = 300
    normal_max_sentences: int = 3
    normal_max_characters: int = 500
    detailed_speech: bool = False
    speak_code: bool = False
    speak_logs: bool = False
    speak_urls: bool = False
    speak_paths: bool = False
    speak_tables: bool = False
    speak_secrets: bool = False

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> VoicePolicyConfig:
        raw = dict(data or {})

        def flag(name: str, default: bool) -> bool:
            return bool(raw.get(name, default))

        def count(name: str, default: int, high: int) -> int:
            try:
                value = int(raw.get(name, default))
            except (TypeError, ValueError):
                return default
            return max(0, min(high, value))

        mode_raw = str(raw.get("default_mode", "concise")).strip().lower()
        try:
            mode = VoiceMode(mode_raw)
        except ValueError:
            # An unreadable mode must not make KIKI talk more than intended.
            mode = VoiceMode.CONCISE
        return cls(
            default_mode=mode,
            concise_max_sentences=count("concise_max_sentences", 2, 20),
            concise_max_characters=count("concise_max_characters", 300, 4000),
            normal_max_sentences=count("normal_max_sentences", 3, 40),
            normal_max_characters=count("normal_max_characters", 500, 8000),
            detailed_speech=flag("detailed_speech", False),
            speak_code=flag("speak_code", False),
            speak_logs=flag("speak_logs", False),
            speak_urls=flag("speak_urls", False),
            speak_paths=flag("speak_paths", False),
            speak_tables=flag("speak_tables", False),
            speak_secrets=flag("speak_secrets", False),
        )

    def limits_for(self, mode: VoiceMode) -> tuple[int, int]:
        """(max sentences, max characters). Zero means unlimited."""
        if mode is VoiceMode.SILENT:
            return 0, 0
        if mode is VoiceMode.CONCISE:
            return self.concise_max_sentences, self.concise_max_characters
        if mode is VoiceMode.NORMAL:
            return self.normal_max_sentences, self.normal_max_characters
        # detailed: unlimited only when explicitly enabled, else like normal.
        if self.detailed_speech:
            return 0, 0
        return self.normal_max_sentences, self.normal_max_characters


@dataclass
class SpeechPlan:
    """What will be spoken, and what was left out.

    `removed` names categories, never the removed content — a secret must not
    reach a log by way of a diagnostic field.
    """

    mode: VoiceMode
    text: str = ""
    removed: list[str] = field(default_factory=list)
    truncated: bool = False

    @property
    def speaks(self) -> bool:
        return bool(self.text.strip())


class VoiceResponsePolicy:
    def __init__(self, config: VoicePolicyConfig | None = None) -> None:
        self._config = config or VoicePolicyConfig()

    @property
    def config(self) -> VoicePolicyConfig:
        return self._config

    def plan(self, answer: str, *, mode: VoiceMode | None = None) -> SpeechPlan:
        chosen = mode or self._config.default_mode
        plan = SpeechPlan(mode=chosen)
        if chosen is VoiceMode.SILENT or not (answer or "").strip():
            return plan

        text, removed = self._redact(answer)
        plan.removed = removed
        text = self._tidy(text)
        if not text:
            return plan

        max_sentences, max_chars = self._config.limits_for(chosen)
        text, truncated = _limit(text, max_sentences, max_chars)
        plan.text = text
        plan.truncated = truncated
        return plan

    def redact_chunk(self, raw: str) -> str:
        """Everything the policy forbids, removed, then tidied for speech.

        For one chunk of a longer answer, so the length cap is deliberately not
        applied: the cap is about a whole answer, and a chunk the cap emptied
        would silently swallow speech that was allowed.

        Both voice routes go through here. Redaction that only one route
        applies is redaction that depends on a config flag to protect anything.

        `plan()` is deliberately not used: it also caps length, and a cap
        applied per chunk truncates every chunk of a streamed answer rather
        than the answer as a whole.
        """
        cleaned, _removed = self._redact(raw)
        return self._tidy(cleaned)

    def _redact(self, text: str) -> tuple[str, list[str]]:
        removed: list[str] = []
        out = text

        def drop(pattern: re.Pattern[str], label: str, repl: str = " ") -> None:
            nonlocal out
            replaced, hits = pattern.subn(repl, out)
            if hits:
                out = replaced
                if label not in removed:
                    removed.append(label)

        # Secrets first: a token inside a code fence must not survive because
        # some later rule happened to keep the fence.
        if not self._config.speak_secrets:
            drop(_SECRET, "secrets")
        if not self._config.speak_code:
            drop(_FENCE, "code")
            drop(_UNCLOSED_FENCE, "code")
            drop(_INLINE_CODE, "code")
        if not self._config.speak_logs:
            drop(_LOG_LINE, "logs")
            if _DIFF_HEADER.search(out):
                drop(_DIFF_LINE, "diff")
        if not self._config.speak_tables:
            drop(_TABLE_ROW, "tables")
        # Markdown links become their label before URLs are stripped, so
        # "[Doku](https://…)" still speaks the word "Doku".
        if self._config.speak_urls:
            out = _MD_LINK.sub(r"\1 \2", out)
        else:
            replaced, hits = _MD_LINK.subn(r"\1", out)
            if hits:
                out = replaced
            drop(_URL, "urls")
        if not self._config.speak_paths:
            drop(_PATH, "paths")
            drop(_WINPATH, "paths")
        return out, removed

    @staticmethod
    def _tidy(text: str) -> str:
        out = _BULLET.sub("", text)
        out = _MD_MARKUP.sub("", out)
        out = _SPACE.sub(" ", out)
        lines = [line.strip() for line in out.splitlines()]
        return " ".join(line for line in lines if line).strip()


def _limit(text: str, max_sentences: int, max_chars: int) -> tuple[str, bool]:
    """Cut to whole sentences first, characters only as a backstop."""
    truncated = False
    if max_sentences > 0:
        sentences = [s.strip() for s in _SENTENCE.findall(text) if s.strip()]
        if len(sentences) > max_sentences:
            sentences = sentences[:max_sentences]
            truncated = True
        if sentences:
            text = " ".join(sentences)
    if max_chars > 0 and len(text) > max_chars:
        cut = text[:max_chars]
        # Prefer the last sentence end, then the last word boundary.
        for marker in (". ", "! ", "? "):
            index = cut.rfind(marker)
            if index > max_chars // 2:
                return cut[: index + 1].strip(), True
        space = cut.rfind(" ")
        return (cut[:space] if space > 0 else cut).strip(), True
    return text.strip(), truncated
