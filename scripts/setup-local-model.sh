#!/usr/bin/env bash
# Pull the default local KIKI model: small, German-capable, vision.
set -euo pipefail

MODEL="${1:-qwen3-vl:4b}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama fehlt. Fedora: https://ollama.com/download oder das offizielle Install-Skript."
  echo "Danach: ollama serve"
  exit 1
fi

echo "Lade ${MODEL} (Deutsch + Vision) …"
ollama pull "${MODEL}"
echo
echo "Fertig. In KIKI unter Einstellungen → Ollama-Modell: ${MODEL}"
echo "Kleiner/schneller:  ollama pull qwen3-vl:2b"
echo "Mehr Qualität:     ollama pull qwen3-vl:8b"
echo "Alternative:        ollama pull gemma3:4b"
