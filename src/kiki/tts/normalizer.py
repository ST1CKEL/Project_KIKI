#!/usr/bin/env python3
"""Project KIKI - German Spoken Text Normalizer & Tech Lexicon

Responsibilities:
1. Markdown, code, URL, and symbol sanitization for speech synthesis.
2. Expands German numbers, ordinals, decimals, times, dates, currencies, and IP addresses.
3. Technical pronunciation lexicon for Linux, Dev, and AI terminology (PipeWire, Proxmox, etc.).
"""

from __future__ import annotations

import html
import re

# Domain-specific German phonetic substitutions for tech & system terms
TECH_LEXICON: dict[str, str] = {
    "pipewire": "Peipweier",
    "proxmox": "Proxmox",
    "kubernetes": "Kubernetis",
    "k8s": "Kubernetis",
    "qwen": "Kwen",
    "ollama": "Ollahma",
    "vlan": "Vau-Lahn",
    "wlan": "W-Lahn",
    "ssh": "S-S-H",
    "dns": "D-N-S",
    "nvidia": "Enwiddia",
    "wayland": "Weiland",
    "fedora": "Fedora",
    "gnome": "Gnohm",
    "docker": "Docker",
    "podman": "Podmän",
    "systemctl": "System-Control",
    "systemd": "System-D",
    "github": "Git-Habb",
    "gitlab": "Git-Labb",
    "gui": "G-U-I",
    "cli": "C-L-I",
    "vram": "V-Ramm",
    "ram": "Ramm",
    "cpu": "C-P-U",
    "gpu": "G-P-U",
    "api": "A-P-I",
    "sqlite": "S-Q-L-eit",
    "json": "Dschäison",
    "yaml": "Jammel",
    "toml": "Tommel",
    "localhost": "Lokel-Host",
    "ip": "I-P",
}

_ONES = {
    0: "null", 1: "eins", 2: "zwei", 3: "drei", 4: "vier", 5: "fünf",
    6: "sechs", 7: "sieben", 8: "acht", 9: "neun", 10: "zehn",
    11: "elf", 12: "zwölf", 13: "dreizehn", 14: "vierzehn", 15: "fünfzehn",
    16: "sechzehn", 17: "siebzehn", 18: "achtzehn", 19: "neunzehn",
}

_TENS = {
    2: "zwanzig", 3: "dreißig", 4: "vierzig", 5: "fünfzig",
    6: "sechzig", 7: "siebzig", 8: "achtzig", 9: "neunzig",
}

_ORDINAL_STEMS = {
    1: "erst", 2: "zweit", 3: "dritt", 7: "siebt", 8: "acht",
}


def _number_to_german_words(n: int) -> str:
    """Converts an integer 0..999,999 to German cardinal words."""
    if n < 0:
        return f"minus {_number_to_german_words(-n)}"
    if n in _ONES:
        return _ONES[n]
    if n < 100:
        ten, unit = divmod(n, 10)
        return _TENS[ten] if unit == 0 else f"{'ein' if unit == 1 else _ONES[unit]}und{_TENS[ten]}"
    if n < 1000:
        hundred, rest = divmod(n, 100)
        prefix = "einhundert" if hundred == 1 else f"{_ONES[hundred]}hundert"
        return prefix if rest == 0 else f"{prefix}{_number_to_german_words(rest)}"
    if n < 1000000:
        thousand, rest = divmod(n, 1000)
        prefix = "eintausend" if thousand == 1 else f"{_number_to_german_words(thousand)}tausend"
        return prefix if rest == 0 else f"{prefix}{_number_to_german_words(rest)}"
    return str(n)


class GermanTextNormalizer:
    def __init__(self, custom_lexicon: dict[str, str] | None = None):
        self.lexicon = {**TECH_LEXICON, **(custom_lexicon or {})}

    def normalize(self, text: str) -> str:
        """Sanitizes and normalizes markdown and technical text for spoken audio."""
        if not text:
            return ""

        # 1. Clean markdown & code blocks
        t = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
        t = re.sub(r"`[^`]*`", " ", t)
        t = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", t)
        t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
        t = re.sub(r"\b(?:https?://|www\.)\S+", " Link ", t, flags=re.IGNORECASE)
        t = re.sub(r"^#{1,6}\s*", "", t, flags=re.MULTILINE)
        t = re.sub(r"[*_~]{1,3}", "", t)
        t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.MULTILINE)
        t = re.sub(r"^\s*\d+\.\s+", "", t, flags=re.MULTILINE)
        t = html.unescape(t)

        # 2. IPv4 address expansion (e.g. 192.168.1.1 -> "192 Punkt 168 Punkt 1 Punkt 1")
        def _expand_ip(match: re.Match) -> str:
            parts = match.group(0).split(".")
            return " Punkt ".join(_number_to_german_words(int(p)) for p in parts)
        t = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", _expand_ip, t)

        # 3. Currency expansion
        t = re.sub(r"(\d+)\s*€", lambda m: f"{_number_to_german_words(int(m.group(1)))} Euro", t)
        t = re.sub(r"€\s*(\d+)", lambda m: f"{_number_to_german_words(int(m.group(1)))} Euro", t)
        t = re.sub(r"(\d+)\s*\$", lambda m: f"{_number_to_german_words(int(m.group(1)))} Dollar", t)
        t = re.sub(r"(\d+)\s*%", lambda m: f"{_number_to_german_words(int(m.group(1)))} Prozent", t)

        # 4. Units & Times
        t = re.sub(r"(\d+)\s*°C", lambda m: f"{_number_to_german_words(int(m.group(1)))} Grad Celsius", t)
        t = re.sub(r"(\d+)\s*GB", lambda m: f"{_number_to_german_words(int(m.group(1)))} Gigabyte", t, flags=re.IGNORECASE)
        t = re.sub(r"(\d+)\s*MB", lambda m: f"{_number_to_german_words(int(m.group(1)))} Megabyte", t, flags=re.IGNORECASE)
        t = re.sub(r"(\d+)\s*ms\b", lambda m: f"{_number_to_german_words(int(m.group(1)))} Millisekunden", t, flags=re.IGNORECASE)
        t = re.sub(r"(\d+)\s*s\b", lambda m: f"{_number_to_german_words(int(m.group(1)))} Sekunden", t)
        t = re.sub(r"(\d{1,2}):(\d{2})\s*Uhr", lambda m: f"{_number_to_german_words(int(m.group(1)))} Uhr {_number_to_german_words(int(m.group(2)))}", t)

        # 5. Standalone numbers
        def _expand_num(match: re.Match) -> str:
            n = int(match.group(0))
            return _number_to_german_words(n) if n <= 999999 else match.group(0)
        t = re.sub(r"\b\d+\b", _expand_num, t)

        # 6. Apply Tech Lexicon substitutions on word boundaries
        for term, spoken in self.lexicon.items():
            pattern = rf"\b{re.escape(term)}\b"
            t = re.sub(pattern, spoken, t, flags=re.IGNORECASE)

        # 7. Collapse whitespace and fix punctuation spacing
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"\s+([,.;:!?])", r"\1", t)
        return t.strip()
