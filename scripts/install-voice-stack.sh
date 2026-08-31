#!/usr/bin/env bash
# Install systemd user units for the voice-first KIKI stack.
# Does not silently enable a broken ear — doctor runs at the end.
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
UNIT_SRC="$ROOT/systemd/user"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
CONF_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/kiki"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}/kiki"
ENV_FILE="$CONF_DIR/voice.env"
RUNTIME_DST="$CONF_DIR/runtime.toml"

mkdir -p "$UNIT_DIR" "$CONF_DIR" "$DATA/wake" "$DATA/piper" "$DATA/tts"

STT_PY="$DATA/stt-venv/bin/python"
TTS_PY="$DATA/kokoro-venv/bin/python"
AUDIO_PY="$DATA/audio-venv/bin/python"
if [[ ! -x "$AUDIO_PY" ]]; then
  AUDIO_PY="$(command -v python3)"
fi
if [[ ! -x "$STT_PY" ]]; then
  echo "WARN: STT-Venv fehlt ($STT_PY). Führe scripts/setup-stt.sh aus." >&2
  STT_PY="$(command -v python3)"
fi
if [[ ! -x "$TTS_PY" ]]; then
  echo "WARN: Kokoro-Venv fehlt ($TTS_PY). Führe scripts/setup-kokoro-tts.sh aus." >&2
  TTS_PY="$(command -v python3)"
fi

CUDNN=""
if [[ -d "$DATA/stt-venv/lib" ]]; then
  CUDNN="$(find "$DATA/stt-venv/lib" -path '*nvidia/cudnn/lib' -type d 2>/dev/null | head -n1)"
  CUBLAS="$(find "$DATA/stt-venv/lib" -path '*nvidia/cublas/lib' -type d 2>/dev/null | head -n1)"
  if [[ -n "${CUDNN:-}" && -n "${CUBLAS:-}" ]]; then
    CUDNN="$CUDNN:$CUBLAS"
  fi
fi

cat > "$ENV_FILE" <<EOF
PYTHONPATH=$ROOT/src
KIKI_DATA_DIR=$ROOT/data
KIKI_STT_PYTHON=$STT_PY
KIKI_TTS_PYTHON=$TTS_PY
KIKI_AUDIO_PYTHON=$AUDIO_PY
HF_HUB_DISABLE_TELEMETRY=1
PATH=$DATA/piper-venv/bin:$DATA/audio-venv/bin:$HOME/.local/bin:/usr/bin
EOF
if [[ -n "${CUDNN:-}" ]]; then
  printf 'LD_LIBRARY_PATH=%s\n' "$CUDNN" >> "$ENV_FILE"
fi

if [[ ! -f "$RUNTIME_DST" ]]; then
  cp -f "$ROOT/src/kiki/config/runtime.toml" "$RUNTIME_DST"
  echo "runtime.toml nach $RUNTIME_DST kopiert"
else
  echo "bestehende $RUNTIME_DST belassen"
fi

for unit in kiki-audio kiki-stt kiki-tts kiki-orchestrator kiki-pet; do
  cp -f "$UNIT_SRC/${unit}.service" "$UNIT_DIR/${unit}.service"
done

# Audio daemon must run with a Python that sees GI + silero/openwakeword.
# If an audio-venv exists, point ExecStart at it via a drop-in.
# systemd user units do not expand ${VAR} in ExecStart on Fedora 44.
# Concrete paths via drop-ins.
write_python_dropin() {
  local unit="$1" python="$2" module="$3"
  mkdir -p "$UNIT_DIR/${unit}.service.d"
  cat > "$UNIT_DIR/${unit}.service.d/python.conf" <<EOF
[Service]
ExecStart=
ExecStart=$python -m $module
EOF
}
write_python_dropin kiki-audio "$AUDIO_PY" kiki.audio.daemon
write_python_dropin kiki-stt "$STT_PY" kiki.stt.service
write_python_dropin kiki-tts "$TTS_PY" kiki.tts.service

systemctl --user daemon-reload
echo "Units installiert unter $UNIT_DIR"
echo "Start: systemctl --user enable --now kiki-audio kiki-stt kiki-tts kiki-orchestrator kiki-pet"
echo "Hotkey: $AUDIO_PY -m kiki.trigger  (GNOME: Super+K darauf binden)"
echo
PYTHONPATH="$ROOT/src" python3 -m kiki.doctor || true
