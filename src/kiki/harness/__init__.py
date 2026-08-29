"""The original agent harness. Superseded, and kept deliberately.

**Not to be confused with `kiki.agents`** (plural), which adapts external coding
agents like opencode.

What is still shipped from here:

* `models` -- the run vocabulary (`AgentRun`, `RunStatus`, `ToolCall`,
  `RunBusyError`, the closed `ERROR_CODES` set). `kiki.assistant` builds on it.
* `adapter` -- the provider-to-`ModelAction` translation.
* `confirmation` -- the display record a dialog is handed. It mints nothing;
  `kiki.tools.confirmation` is the only thing that issues a grant.
* `notes`, `system_status` -- two tools, as `ToolSpec`s in the production
  registry.
* `trace` -- the structured local run trace.

What is no longer reached from the application: `runner` (`AgentRunner`),
`session` (`HarnessSession`), `tools` (the harness-owned `ToolRegistry`) and
`gateway_source`. `kiki.assistant.runner` replaced them -- one runner for the
chat path and the agent path, every tool call through `ToolGateway`.

This module used to re-export the whole surface eagerly. Nothing imported it
that way, and the re-export was the last edge keeping the superseded runner in
the application's import graph -- so importing anything from this package
dragged a thousand lines of dead code along with it. Import the submodule you
mean instead.
"""
