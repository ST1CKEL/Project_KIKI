"""Podman/Docker status and bounded lifecycle. No free `exec`, no shell."""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any

from kiki.tools.policy import RiskLevel
from kiki.tools.registry import ToolSpec

log = logging.getLogger(__name__)

_TIMEOUT = 15
_SAFE_NAME = r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$"
_ACTIONS = frozenset({"start", "stop", "restart", "status"})


def _runtime() -> str | None:
    return shutil.which("podman") or shutil.which("docker")


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    binary = _runtime()
    if binary is None:
        raise RuntimeError("Weder podman noch docker ist installiert.")
    return subprocess.run(
        [binary, *args],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
        check=False,
    )


def _list(_params: dict[str, Any]) -> dict[str, Any]:
    proc = _run(["ps", "-a", "--format", "{{.Names}}\t{{.Status}}\t{{.Image}}"])
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or proc.stdout).strip()[:400]}
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return {"ok": True, "containers": lines[:40], "runtime": _runtime()}


def _action(params: dict[str, Any]) -> dict[str, Any]:
    import re

    name = str(params.get("name") or "").strip()
    action = str(params.get("action") or "").strip().lower()
    if not re.fullmatch(_SAFE_NAME, name):
        return {"ok": False, "error": "Ungültiger Container-Name."}
    if action not in _ACTIONS:
        return {"ok": False, "error": f"Aktion {action!r} ist nicht erlaubt."}
    # SECURITY: argv is a fixed verb plus a validated name. Model text never
    # becomes a shell string, and `exec`/`run`/`rm` are not in the allowlist.
    proc = _run([action, name] if action != "status" else ["inspect", "-f", "{{.State.Status}}", name])
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or proc.stdout).strip()[:400]}
    return {"ok": True, "action": action, "name": name, "output": proc.stdout.strip()[:400]}


class ContainerSkill:
    id = "containers"
    name = "Container"
    description = "Podman/Docker auflisten und begrenzt steuern."

    def tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="container_list",
                title="Container auflisten",
                description="Listet lokale Podman- oder Docker-Container mit Status.",
                risk=RiskLevel.READ,
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                handler=_list,
                effect="Liest die Containerliste, ändert nichts.",
                target="podman/docker",
                auto_allow=True,
                model_callable=True,
                audit_parameters=(),
            ),
            ToolSpec(
                name="container_action",
                title="Container starten oder stoppen",
                description="Startet, stoppt oder startet einen benannten Container neu. Kein exec, kein rm.",
                risk=RiskLevel.WRITE,
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Container-Name"},
                        "action": {
                            "type": "string",
                            "enum": ["start", "stop", "restart", "status"],
                        },
                    },
                    "required": ["name", "action"],
                    "additionalProperties": False,
                },
                handler=_action,
                effect="Ändert den Lebenszyklus eines bestehenden Containers.",
                target="podman/docker",
                auto_allow=False,
                model_callable=True,
                audit_parameters=("name", "action"),
            ),
        ]
