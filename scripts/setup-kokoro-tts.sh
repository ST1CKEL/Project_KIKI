#!/usr/bin/env bash
# Local Kokoro-TTS (82M) service for KIKI.
# Natural, ultra-fast German & English speech synthesis (CUDA/CPU).
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}/kiki/tts"
VENV="${XDG_DATA_HOME:-$HOME/.local/share}/kiki/kokoro-venv"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
MODEL="${KIKI_TTS_MODEL:-hexgrad/Kokoro-82M}"
DUMMY=0

if [[ -f "$SCRIPT_DIR/../services/kokoro-tts/kiki_kokoro_server.py" ]]; then
  SERVER_SOURCE="$SCRIPT_DIR/../services/kokoro-tts/kiki_kokoro_server.py"
elif [[ -f "$SCRIPT_DIR/kiki_kokoro_server.py" ]]; then
  SERVER_SOURCE="$SCRIPT_DIR/kiki_kokoro_server.py"
else
  echo "kiki_kokoro_server.py wurde nicht gefunden."
  exit 1
fi

if [[ "${1:-}" == "--dummy" ]]; then
  DUMMY=1
fi

pick_python() {
  for cand in python3.12 python3.13 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
      echo "$cand"
      return 0
    fi
  done
  return 1
}

GPU=0
if [[ "$DUMMY" -eq 1 ]]; then
  if ! PYTHON_EXEC="$(command -v python3)"; then
    echo "python3 fehlt."
    exit 1
  fi
  echo "Python: $("$PYTHON_EXEC" --version)"
else
  if ! PY="$(pick_python)"; then
    echo "Kein geeignetes Python gefunden (empfohlen: Python 3.12)."
    exit 1
  fi
  echo "Python: $("$PY" --version)"
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    GPU=1
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  else
    echo "Keine CUDA-GPU sichtbar — Kokoro läuft auf CPU (durch 82M Parameter ebenfalls extrem schnell)."
  fi
fi

mkdir -p "$DATA" "$UNIT_DIR"
cp -f "$SERVER_SOURCE" "$DATA/kiki_kokoro_server.py"
for part in "$(dirname "$SERVER_SOURCE")"/*.py; do
  [[ -f "$part" ]] || continue
  cp -f "$part" "$DATA/$(basename "$part")"
done

if [[ "$DUMMY" -eq 0 ]]; then
  if [[ ! -x "$VENV/bin/python" ]]; then
    echo "Lege venv an: $VENV"
    "$PY" -m venv "$VENV"
  fi
  PYTHON_EXEC="$VENV/bin/python"
  "$PYTHON_EXEC" -m pip install -U pip wheel setuptools
  echo "Installiere Kokoro-TTS & PyTorch …"
  if [[ "$GPU" -eq 1 ]]; then
    "$PYTHON_EXEC" -m pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu128 || \
    "$PYTHON_EXEC" -m pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu124 || \
    "$PYTHON_EXEC" -m pip install -U torch torchaudio
  else
    "$PYTHON_EXEC" -m pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  fi
  "$PYTHON_EXEC" -m pip install -U "kokoro>=0.8.4" soundfile misaki[de] spacy
  "$PYTHON_EXEC" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
fi

UNIT="$UNIT_DIR/kiki-tts.service"
if [[ "$DUMMY" -eq 1 ]]; then
  EXTRA="--dummy"
else
  EXTRA="--device auto --model ${MODEL}"
fi

cat > "$UNIT" <<EOF
[Unit]
Description=KIKI Kokoro-TTS (82M Natural Speech)
PartOf=graphical-session.target
After=graphical-session.target

[Service]
Type=simple
ExecStart="${PYTHON_EXEC}" "${DATA}/kiki_kokoro_server.py" --host 127.0.0.1 --port 18765 ${EXTRA}
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=HF_HUB_DISABLE_TELEMETRY=1
TimeoutStartSec=0

[Install]
WantedBy=graphical-session.target
EOF

if command -v systemctl >/dev/null 2>&1; then
  systemctl --user daemon-reload || true
  systemctl --user enable --now kiki-tts.service || true
fi

echo
echo "Kokoro-TTS Dienst bereit: http://127.0.0.1:18765/health"
echo "  systemctl --user status kiki-tts.service"
if [[ "$DUMMY" -eq 0 ]]; then
  echo "Beim ersten Start lädt Hugging Face ${MODEL} (nur ca. 300 MB)."
fi
echo "In KIKI: Stimme df_sarah (weiblich, natürlich) oder df_eva / df_nicole."
