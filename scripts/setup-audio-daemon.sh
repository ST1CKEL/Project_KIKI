#!/usr/bin/env bash
# System packages + audio venv for kiki-audio (capture, Silero, openWakeWord).
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"

if command -v dnf >/dev/null 2>&1; then
  sudo dnf install -y \
    python3-gobject gtk4 libadwaita \
    gstreamer1 gstreamer1-plugins-good \
    pipewire pipewire-utils pipewire-pulseaudio \
    python3-numpy
fi

set +e
"$ROOT/scripts/setup-wakeword.sh"
rc=$?
set -e
if [[ "$rc" -eq 1 ]]; then
  echo "Audio-Python-Pakete fehlgeschlagen (nicht das fehlende KIKI-ONNX)." >&2
  exit 1
fi
if [[ "$rc" -eq 2 ]]; then
  echo "Pakete ok. Weckwort-ONNX fehlt noch — Hotkey/Klick funktionieren trotzdem."
fi
echo "Audio-Daemon-Abhängigkeiten vorbereitet."
