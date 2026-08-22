from __future__ import annotations

from kiki.ui.widgets.agent_output_view import MonoText


class DiffView(MonoText):
    def show(self, stat: str, patch: str, *, truncated: bool = False) -> None:
        notice = "\n\n[Diff gekürzt]\n" if truncated else ""
        body = (stat or "").strip()
        if body:
            body += "\n\n"
        self.set_text(body + (patch or "(kein Diff)") + notice)
