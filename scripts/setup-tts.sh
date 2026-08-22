#!/usr/bin/env bash
# Local Qwen3-TTS 0.6B CustomVoice service (CUDA) for KIKI.
# GTK stays on system Python. This venv is Python 3.12 + PyTorch only.
set -euo pipefail

# Resolve symlinks: the RPM installs this as /usr/libexec/kiki/setup-tts and
# links /usr/bin/kiki-setup-tts to it. Using $0 unresolved would look for the
# server file in /usr/bin, where it is not, so the packaged setup always failed.
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}/kiki/tts"
VENV="${XDG_DATA_HOME:-$HOME/.local/share}/kiki/tts-venv"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
MODEL="${KIKI_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice}"
DUMMY=0

if [[ -f "$SCRIPT_DIR/../services/qwen3-tts/kiki_tts_server.py" ]]; then
  SERVER_SOURCE="$SCRIPT_DIR/../services/qwen3-tts/kiki_tts_server.py"
elif [[ -f "$SCRIPT_DIR/kiki_tts_server.py" ]]; then
  SERVER_SOURCE="$SCRIPT_DIR/kiki_tts_server.py"
else
  echo "kiki_tts_server.py wurde nicht neben dem Setup-Skript gefunden."
  exit 1
fi

if [[ "${1:-}" == "--dummy" ]]; then
  DUMMY=1
fi

pick_python() {
  for cand in python3.12 python3.13; do
    if command -v "$cand" >/dev/null 2>&1; then
      echo "$cand"
      return 0
    fi
  done
  return 1
}

if [[ "$DUMMY" -eq 1 ]]; then
  if ! PYTHON_EXEC="$(command -v python3)"; then
    echo "python3 fehlt. Der Dummy-Test benötigt nur das System-Python."
    exit 1
  fi
  echo "Python: $("$PYTHON_EXEC" --version)"
else
  if ! PY="$(pick_python)"; then
    echo "Python 3.12 fehlt (qwen-tts läuft nicht auf Fedora-Python 3.14)."
    echo "  sudo dnf install -y python3.12 python3.12-devel"
    exit 1
  fi
  echo "Python: $("$PY" --version)"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi fehlt. Ohne CUDA: $0 --dummy  (nur PipeWire-Testton)"
    exit 1
  fi
  if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "NVIDIA-Treiber oder CUDA-GPU ist nicht erreichbar."
    echo "Treiber prüfen, oder ohne CUDA testen: $0 --dummy"
    exit 1
  fi
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

mkdir -p "$DATA" "$UNIT_DIR"
cp -f "$SERVER_SOURCE" "$DATA/kiki_tts_server.py"

if [[ "$DUMMY" -eq 0 ]]; then
  if [[ ! -x "$VENV/bin/python" ]]; then
    echo "Lege venv an: $VENV"
    "$PY" -m venv "$VENV"
  fi
  PYTHON_EXEC="$VENV/bin/python"
  "$PYTHON_EXEC" -m pip install -U pip wheel
  echo "Installiere PyTorch (CUDA) + qwen-tts …"
  "$PYTHON_EXEC" -m pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu128
  "$PYTHON_EXEC" -m pip install -U "qwen-tts>=0.1.1" soundfile
  "$PYTHON_EXEC" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA nicht sichtbar in dieser venv. Treiber/Wheel prüfen.")
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
Description=KIKI Qwen3-TTS (0.6B CustomVoice)
PartOf=graphical-session.target
After=graphical-session.target

[Service]
Type=simple
ExecStart="${PYTHON_EXEC}" "${DATA}/kiki_tts_server.py" --host 127.0.0.1 --port 18765 ${EXTRA}
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=HF_HUB_DISABLE_TELEMETRY=1
TimeoutStartSec=0

[Install]
WantedBy=graphical-session.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now kiki-tts.service

echo
echo "TTS-Dienst: http://127.0.0.1:18765/health"
echo "  systemctl --user status kiki-tts.service"
if [[ "$DUMMY" -eq 0 ]]; then
  echo "Beim ersten CUDA-Start lädt Hugging Face ${MODEL} (ca. 1–2 GB)."
else
  echo "Dummy-Modus aktiv: kurzer Testton, keine GPU- oder Pip-Pakete nötig."
fi
echo "In KIKI: Einstellungen → Sprachausgabe → Stimme Serena / Deutsch."
echo "Dummy-Test ohne GPU: $0 --dummy"
