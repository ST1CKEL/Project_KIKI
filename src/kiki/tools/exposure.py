"""Which tools the model gets to see, and in what shape.

The model can only request what is declared here, and this list is rebuilt from
the live policy on every turn. Flipping the panic switch or disabling
integrations therefore removes tools from the model's view immediately, instead
of leaving it to guess why a call came back denied.
"""

from __future__ import annotations

from typing import Any

from kiki.tools.policy import Origin, ToolPolicy
from kiki.tools.registry import ToolRegistry, ToolSpec


def exposed_specs(
    registry: ToolRegistry,
    policy: ToolPolicy,
    *,
    panic: bool,
    integrations_enabled: bool,
    profile: str = "observe",
) -> list[ToolSpec]:
    """Return the specs a model-initiated call could currently get past `gate`."""
    out: list[ToolSpec] = []
    for spec in registry.all():
        blocked = policy.gate(
            name=spec.name,
            spec=spec,
            panic=panic,
            integrations_enabled=integrations_enabled,
            profile=profile,
            origin=Origin.MODEL,
        )
        if blocked is None:
            out.append(spec)
    return sorted(out, key=lambda s: s.name)


def function_schema(spec: ToolSpec) -> dict[str, Any]:
    """One OpenAI-style function declaration. Ollama accepts the same shape."""
    description = spec.description.strip()
    if not spec.auto_allow:
        description = f"{description} Diese Aktion fragt den Nutzer vor der Ausführung."
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": description,
            "parameters": spec.parameters,
        },
    }


def declarations(
    registry: ToolRegistry,
    policy: ToolPolicy,
    *,
    panic: bool,
    integrations_enabled: bool,
    profile: str = "observe",
) -> list[dict[str, Any]]:
    return [
        function_schema(spec)
        for spec in exposed_specs(
            registry,
            policy,
            panic=panic,
            integrations_enabled=integrations_enabled,
            profile=profile,
        )
    ]
