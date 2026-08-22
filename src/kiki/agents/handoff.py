"""Turn chat draft/history into a coding-session task. No execution."""

from __future__ import annotations

from typing import Any

_MAX_TASK = 4000
_MAX_SUMMARY = 1500


def coding_task_from_chat(*, draft: str = "", last_user: str = "") -> str:
    text = (draft or "").strip() or (last_user or "").strip()
    if not text:
        raise ValueError("Keine Aufgabe im Chat.")
    return text[:_MAX_TASK]


def format_coding_briefing(task: str) -> str:
    """Advisor scaffold. Does not start an agent."""
    body = (task or "").strip()
    if not body:
        raise ValueError("Keine Aufgabe.")
    if body.startswith("Aufgabe:"):
        return body[:_MAX_TASK]
    return (
        f"Aufgabe: {body}\n\n"
        "Bitte nur planen, nicht umsetzen:\n"
        "- Teilaufgaben\n"
        "- Unklare Punkte / Annahmen\n"
        "- Akzeptanzkriterien\n"
        "- Risiko (niedrig/mittel/hoch) und voraussichtlich betroffene Dateien\n"
        "- Welche Tests danach laufen sollen\n"
    )[:_MAX_TASK]


def session_summary_for_chat(session: Any, *, plan_excerpt: str = "", dirty: bool = False) -> str:
    kind = getattr(session, "kind", None)
    kind_v = getattr(kind, "value", kind) or "?"
    status = getattr(session, "status", None)
    status_v = getattr(status, "value", status) or "?"
    lines = [
        "Zusammenfassung der lokalen Coding-Session (nicht vom Chat-Modell erzeugt):",
        f"- Art: {kind_v}",
        f"- Status: {status_v}",
        f"- Branch vorher: {getattr(session, 'git_branch_before', None) or '?'}",
        f"- Exit: {getattr(session, 'exit_code', None)}",
    ]
    if dirty:
        lines.append("- Arbeitsbaum hat uncommitted Änderungen.")
    summary = (getattr(session, "summary", None) or "").strip()
    if summary:
        lines.append(f"- Kurz: {summary.splitlines()[0][:220]}")
    excerpt = (plan_excerpt or "").strip()
    if excerpt:
        lines.append(f"- Plan: {excerpt.splitlines()[0][:220]}")
    return "\n".join(lines)[:_MAX_SUMMARY]
