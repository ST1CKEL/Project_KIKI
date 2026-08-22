"""Safe, HTML-free Markdown subset for chat rendering."""

from __future__ import annotations

import re
from dataclasses import dataclass

_INLINE = re.compile(
    r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[[^\]]+\]\([^)]+\))"
)


@dataclass(frozen=True)
class Span:
    text: str
    kind: str = "text"  # text, bold, italic, code, link
    href: str | None = None


@dataclass(frozen=True)
class Paragraph:
    spans: tuple[Span, ...]


@dataclass(frozen=True)
class Heading:
    level: int
    spans: tuple[Span, ...]


@dataclass(frozen=True)
class ListBlock:
    ordered: bool
    items: tuple[tuple[Span, ...], ...]


@dataclass(frozen=True)
class CodeBlock:
    language: str
    code: str


Block = Paragraph | Heading | ListBlock | CodeBlock


def parse_markdown(source: str) -> list[Block]:
    lines = source.replace("\r\n", "\n").split("\n")
    blocks: list[Block] = []
    i = 0
    para: list[str] = []
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            _flush_para(para, blocks)
            lang = line[3:].strip()
            i += 1
            body: list[str] = []
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            blocks.append(CodeBlock(language=lang, code="\n".join(body)))
            i += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            _flush_para(para, blocks)
            blocks.append(Heading(level=len(heading.group(1)), spans=_parse_inline(heading.group(2))))
            i += 1
            continue
        bullet = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if bullet:
            _flush_para(para, blocks)
            ordered = bullet.group(2).endswith(".")
            items: list[tuple[Span, ...]] = [_parse_inline(bullet.group(3))]
            i += 1
            while i < len(lines):
                nxt = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[i])
                if not nxt:
                    break
                items.append(_parse_inline(nxt.group(3)))
                i += 1
            blocks.append(ListBlock(ordered=ordered, items=tuple(items)))
            continue
        if not line.strip():
            _flush_para(para, blocks)
            i += 1
            continue
        para.append(line)
        i += 1
    _flush_para(para, blocks)
    return blocks


def _flush_para(para: list[str], blocks: list[Block]) -> None:
    if not para:
        return
    text = " ".join(part.strip() for part in para if part.strip())
    if text:
        blocks.append(Paragraph(spans=_parse_inline(text)))
    para.clear()


def _parse_inline(text: str) -> tuple[Span, ...]:
    spans: list[Span] = []
    pos = 0
    for match in _INLINE.finditer(text):
        if match.start() > pos:
            spans.append(Span(text[pos : match.start()]))
        token = match.group(0)
        if token.startswith("**"):
            spans.append(Span(token[2:-2], kind="bold"))
        elif token.startswith("*"):
            spans.append(Span(token[1:-1], kind="italic"))
        elif token.startswith("`"):
            spans.append(Span(token[1:-1], kind="code"))
        elif token.startswith("["):
            label, href = re.match(r"\[([^\]]+)\]\(([^)]+)\)", token).groups()  # type: ignore[union-attr]
            spans.append(Span(label, kind="link", href=href))
        pos = match.end()
    if pos < len(text):
        spans.append(Span(text[pos:]))
    return tuple(spans) if spans else (Span(text),)


def escape_pango(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def spans_to_pango(spans: tuple[Span, ...]) -> str:
    parts: list[str] = []
    for span in spans:
        body = escape_pango(span.text)
        if span.kind == "bold":
            parts.append(f"<b>{body}</b>")
        elif span.kind == "italic":
            parts.append(f"<i>{body}</i>")
        elif span.kind == "code":
            parts.append(f"<tt>{body}</tt>")
        elif span.kind == "link":
            href = escape_pango(span.href or "")
            parts.append(f"{body} (<tt>{href}</tt>)")
        else:
            parts.append(body)
    return "".join(parts)
