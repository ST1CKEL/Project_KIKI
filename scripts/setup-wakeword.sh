#!/usr/bin/env bash
# Audio-venv for kiki-audio: ONNX Silero-VAD + openWakeWord (ONNX only).
#
# openwakeword 0.6.0 declares tflite-runtime on Linux. That wheel does not
# exist for Python 3.12+. We only use inference_framework="onnx", so tflite
# is installed with --no-deps and never imported.
set -euo pipefail

DATA="${XDG_DATA_HOME:-$HOME/.local/share}/kiki"
VENV="$DATA/audio-venv"
WAKE="$DATA/wake"
VAD_DIR="$DATA/vad"
# Pinned Silero v5 ONNX (16 kHz, 512-sample window). No PyTorch in the audio process.
SILERO_ONNX_URL="${KIKI_SILERO_ONNX_URL:-https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx}"
SILERO_ONNX="$VAD_DIR/silero_vad.onnx"

mkdir -p "$WAKE" "$VAD_DIR"

pick_python() {
  # GStreamer/GI live on the distro interpreter. ML packages must too.
  if command -v python3 >/dev/null 2>&1 && python3 -c "import gi" 2>/dev/null; then
    command -v python3
    return 0
  fi
  for cand in python3.14 python3.13 python3.12; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import gi" 2>/dev/null; then
      command -v "$cand"
      return 0
    fi
  done
  echo "Kein Python mit PyGObject (gi) gefunden. Installiere python3-gobject." >&2
  return 1
}

PY="$(pick_python)"
SYS_VER="$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
echo "Audio-Python: $PY ($SYS_VER) — muss GI/GStreamer laden können."

recreate=0
if [[ ! -x "$VENV/bin/python" ]]; then
  recreate=1
else
  VENV_VER="$("$VENV/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
  if [[ "$VENV_VER" != "$SYS_VER" ]]; then
    echo "Bestehendes audio-venv ist Python $VENV_VER, System/GI ist $SYS_VER — lege neu an."
    recreate=1
  fi
fi
if [[ "$recreate" -eq 1 ]]; then
  rm -rf "$VENV"
  echo "Lege audio-venv mit system-site-packages an (GI/GStreamer vom System)."
  "$PY" -m venv --system-site-packages "$VENV"
fi

"$VENV/bin/python" -m pip install -U pip
# ONNX stack only. No tflite-runtime, no torch in this venv.
"$VENV/bin/python" -m pip install -U \
  "onnxruntime>=1.17,<2" \
  "numpy>=1.24" \
  "scipy>=1.10,<2" \
  "scikit-learn>=1.3,<2" \
  "tqdm>=4.0,<5" \
  "requests>=2.0,<3"

echo "Installiere openwakeword 0.6.0 ohne tflite-runtime (--no-deps, ONNX-Pfad)."
"$VENV/bin/python" -m pip install --no-deps "openwakeword==0.6.0"

if [[ ! -f "$SILERO_ONNX" ]]; then
  echo "Lade Silero-VAD ONNX nach $SILERO_ONNX"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$SILERO_ONNX_URL" -o "$SILERO_ONNX.partial"
  else
    wget -q "$SILERO_ONNX_URL" -O "$SILERO_ONNX.partial"
  fi
  mv -f "$SILERO_ONNX.partial" "$SILERO_ONNX"
fi
echo "Silero-VAD ONNX: $SILERO_ONNX ($(wc -c < "$SILERO_ONNX") Bytes)"

echo "Prüfe Imports …"
"$VENV/bin/python" - <<'PY'
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: F401

import numpy  # noqa: F401
import onnxruntime  # noqa: F401
import openwakeword  # noqa: F401
from openwakeword.model import Model  # noqa: F401

print("gi/Gst:", "ok")
print("onnxruntime:", onnxruntime.__version__)
print("openwakeword:", getattr(openwakeword, "__version__", "0.6.0"))
print("numpy:", numpy.__version__)
PY

# Shared ONNX feature extractors (melspec + embedding). Not English wake words.
"$VENV/bin/python" - <<'PY' || true
from openwakeword.utils import download_models
try:
    download_models(model_names=["melspectrogram", "embedding"])
    print("openWakeWord Feature-Modelle (melspec/embedding) geladen.")
except TypeError:
    print("WARN: download_models() kennt keine model_names — überspringe Auto-Download.")
except Exception as exc:
    print(f"WARN: Feature-Modelle nicht geladen ({exc}). Erstes Weckwort-Load versucht es erneut.")
PY

echo
echo "openWakeWord ist installiert (ONNX, ohne tflite-runtime)."
echo "Ein englisches Standardwort wird NICHT als KIKI verwendet."
echo "Lege ein für „KIKI“ trainiertes ONNX hier ab:"
echo "  $WAKE/kiki.onnx"
echo
echo "Training: https://github.com/dscripka/openWakeWord#training-new-models"
echo "Ohne diese Datei: Hotkey (Super+K) und Klick auf die Figur funktionieren,"
echo "das Weckwort nicht. Der Doctor meldet das als FAIL."
if [[ -f "$WAKE/kiki.onnx" ]]; then
  echo "Gefunden: $WAKE/kiki.onnx"
  exit 0
fi
echo "TODO: kiki.onnx fehlt — das ist absichtlich sichtbar, kein Fallback."
exit 2
