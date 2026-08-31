#!/usr/bin/env bash
# German Piper voice for KIKI. Official Kokoro has no German weights on this
# install — Piper de_DE-eva_k-medium is the verified female voice.
set -euo pipefail

DATA="${XDG_DATA_HOME:-$HOME/.local/share}/kiki"
VENV="$DATA/piper-venv"
VOICE_DIR="$DATA/piper"
VOICE="de_DE-kerstin-low"
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/kerstin/low"

mkdir -p "$VOICE_DIR"

pick_python() {
  for cand in python3.12 python3.13 python3.14 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
      echo "$cand"
      return 0
    fi
  done
  return 1
}

PY="$(pick_python)"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Lege piper-venv an mit $PY"
  "$PY" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install -U pip
"$VENV/bin/python" -m pip install -U "piper-tts>=1.2"

fetch() {
  local url="$1" dest="$2"
  if [[ -f "$dest" ]]; then
    return 0
  fi
  echo "Lade $dest"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL -L "$url" -o "$dest.partial"
  else
    wget -q "$url" -O "$dest.partial"
  fi
  mv -f "$dest.partial" "$dest"
}

fetch "$BASE/${VOICE}.onnx" "$VOICE_DIR/${VOICE}.onnx"
fetch "$BASE/${VOICE}.onnx.json" "$VOICE_DIR/${VOICE}.onnx.json"

echo "Probe …"
printf '%s\n' "Guten Tag, ich bin Kiki." | "$VENV/bin/piper" \
  --model "$VOICE_DIR/${VOICE}.onnx" \
  --output_file /tmp/kiki-piper-probe.wav
ls -l /tmp/kiki-piper-probe.wav
echo "Piper bereit: $VOICE_DIR/${VOICE}.onnx"
echo "Binary: $VENV/bin/piper"
