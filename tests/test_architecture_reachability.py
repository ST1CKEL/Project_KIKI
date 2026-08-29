"""What the running application may reach, and what it may not.

The whole consolidation was about removing a second way to do things. A second
way does not come back as a decision -- it comes back as one convenient import
that nobody notices. This is the guard against that.

The measurement is the import graph from `kiki.application`, following package
`__init__` modules, because importing `kiki.harness.confirmation` really does
execute `kiki/harness/__init__.py`. That detail is the whole reason the
superseded runner stayed reachable long after nothing used it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
ENTRY = "kiki.application"

# Superseded by `kiki.assistant`: one runner for the chat path and the agent
# path, every tool call through `ToolGateway`. Kept on disk for now; reaching
# them from the application would mean two runners again.
RETIRED = {
    "kiki.harness.runner",
    "kiki.harness.session",
    "kiki.harness.tools",
    "kiki.harness.gateway_source",
    "kiki.ai.agent_loop",
}

# Still shipped: the run vocabulary, the provider adapter, the confirmation
# display record, the trace, and the two tools that became ToolSpecs.
STILL_LIVE = {
    "kiki.harness.models",
    "kiki.harness.adapter",
    "kiki.harness.confirmation",
    "kiki.harness.trace",
    "kiki.harness.notes",
    "kiki.harness.system_status",
    "kiki.assistant.runner",
    "kiki.assistant.run_service",
    "kiki.tools.gateway",
    "kiki.tools.confirmation",
    "kiki.tools.routine_gateway",
}


def _module_name(path: Path) -> str:
    name = str(path.relative_to(SRC).with_suffix("")).replace("/", ".")
    return name.removesuffix(".__init__")


def _graph() -> dict[str, set[str]]:
    edges: dict[str, set[str]] = {}
    for path in SRC.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        found: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("kiki"):
                    found.add(node.module)
            elif isinstance(node, ast.Import):
                found.update(a.name for a in node.names if a.name.startswith("kiki"))
        edges[_module_name(path)] = found
    # Importing a submodule executes its package __init__, so add that edge.
    for name, targets in list(edges.items()):
        edges[name] = targets | {
            t.rsplit(".", 1)[0]
            for t in targets
            if t.count(".") >= 2 and t.rsplit(".", 1)[0] in edges
        }
    return edges


def _reachable() -> set[str]:
    edges = _graph()
    seen: set[str] = set()
    stack = [ENTRY]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(edges.get(current, ()))
    return seen


@pytest.mark.parametrize("module", sorted(RETIRED))
def test_the_application_does_not_reach_a_retired_module(module) -> None:
    """One import would put a second runner back in the shipped application."""
    assert module not in _reachable(), (
        f"{module} is reachable from {ENTRY} again. It was replaced by "
        "kiki.assistant; if it is genuinely needed, remove it from RETIRED "
        "and say why in the commit."
    )


@pytest.mark.parametrize("module", sorted(STILL_LIVE))
def test_a_module_that_should_ship_is_still_reachable(module) -> None:
    """The other direction: cutting too much is also a regression."""
    assert module in _reachable()


def test_the_package_root_re_exports_nothing_heavy() -> None:
    """`kiki/harness/__init__.py` used to import the whole surface eagerly.

    Nothing imported it that way, and it was the last edge holding the retired
    runner in the graph.
    """
    source = (SRC / "kiki" / "harness" / "__init__.py").read_text(encoding="utf-8")
    for retired in ("runner", "session", "tools", "gateway_source"):
        assert f"from kiki.harness.{retired} import" not in source


def test_no_test_imports_a_retired_module() -> None:
    """The gap this guard originally had.

    Reachability was measured from `kiki.application` through `src/` only, so
    four test files kept importing `RunBusyError` from the superseded runner.
    Nothing caught it until the module was actually deleted and collection
    broke. A retired module is retired for the tests too.
    """
    offenders: list[str] = []
    for path in (Path(__file__).parent).glob("*.py"):
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8")
        for module in RETIRED:
            if f"from {module} import" in text or f"import {module}\n" in text:
                offenders.append(f"{path.name} -> {module}")
    assert not offenders, offenders


def test_the_run_vocabulary_is_shared_not_duplicated() -> None:
    """`RunBusyError` lives with the run models.

    It used to live beside the legacy runner, so `kiki.assistant` importing one
    exception dragged a thousand lines of superseded code into the graph.
    """
    models = (SRC / "kiki" / "harness" / "models.py").read_text(encoding="utf-8")
    assert "class RunBusyError" in models
    for module in ("__init__", "runner", "run_service"):
        text = (SRC / "kiki" / "assistant" / f"{module}.py").read_text(encoding="utf-8")
        assert "from kiki.harness.runner import" not in text


def test_both_agent_paths_use_one_runner() -> None:
    """Chat and agent on the same runner -- the point of the whole migration."""
    chat = (SRC / "kiki" / "ai" / "chat_service.py").read_text(encoding="utf-8")
    app = (SRC / "kiki" / "application.py").read_text(encoding="utf-8")
    assert "AssistantRunner" in chat
    assert "AssistantRunner" in app
    for gone in ("HarnessSession(", "AgentRunner("):
        assert gone not in app, gone
