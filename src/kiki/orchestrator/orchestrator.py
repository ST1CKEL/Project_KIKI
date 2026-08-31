"""Entry shim: `python -m kiki.orchestrator.orchestrator` still works."""

from kiki.orchestrator.service import main

if __name__ == "__main__":
    raise SystemExit(main())
