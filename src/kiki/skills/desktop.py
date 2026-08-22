"""Desktop perception tools. Capture is started from the UI after confirmation."""

from __future__ import annotations

from typing import Any

from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ToolSpec


def _screen_stub(_params: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": False,
        "error": "Bildschirmfoto nur nach Klick in der KIKI-Oberfläche und Nutzerfreigabe.",
    }


class DesktopPerceptionSkill:
    id = "desktop_perception"
    name = "Desktop wahrnehmen"
    description = "Bildschirmfoto — nur mit Bestätigung, nie im Hintergrund."

    def tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="capture_screen",
                title="Bildschirmfoto",
                description="Nimmt ein Bild des Desktops auf und darf es nur nach Freigabe an das lokale Modell senden.",
                risk=RiskLevel.READ,
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                handler=_screen_stub,
                effect="Erzeugt ein Bildschirmfoto und sendet es an das lokale Modell. Der Inhalt des Desktops verlässt den Rechner nicht, wenn Ollama lokal läuft.",
                target="Bildschirm",
                auto_allow=False,
                requires_integration=True,
            )
        ]
