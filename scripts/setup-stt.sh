#!/usr/bin/env bash
# Local faster-whisper STT service for KIKI.
# Own venv on purpose: ctranslate2 ships a different CUDA stack than the
# torch-based tts/llm services, so the venvs must not be shared.
set -euo pipefail

# Resolve symlinks: the RPM installs this as /usr/libexec/kiki/setup-stt and
# links /usr/bin/kiki-setup-stt to it. Using $0 unresolved would look for the
# server file in /usr/bin, where it is not.
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}/kiki/stt"
VENV="${XDG_DATA_HOME:-$HOME/.local/share}/kiki/stt-venv"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
MODEL="${KIKI_STT_MODEL:-Systran/faster-whisper-large-v3-turbo}"
LANGUAGE="${KIKI_STT_LANGUAGE:-de}"
DUMMY=0

if [[ -f "$SCRIPT_DIR/../services/kiki-stt/kiki_stt_server.py" ]]; then
  SERVER_SOURCE="$SCRIPT_DIR/../services/kiki-stt/kiki_stt_server.py"
elif [[ -f "$SCRIPT_DIR/kiki_stt_server.py" ]]; then
  SERVER_SOURCE="$SCRIPT_DIR/kiki_stt_server.py"
else
  echo "kiki_stt_server.py wurde nicht neben dem Setup-Skript gefunden."
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
    echo "python3 fehlt. Der Dummy-Test benötigt nur das System-Python."
    exit 1
  fi
  echo "Python: $("$PYTHON_EXEC" --version)"
else
  if ! PY="$(pick_python)"; then
    echo "Kein geeignetes Python gefunden."
    exit 1
  fi
  echo "Python: $("$PY" --version)"
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    GPU=1
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  else
    echo "Keine CUDA-GPU sichtbar — der Dienst läuft auf CPU (faster-whisper-small ist auch dort schnell genug)."
  fi
fi

mkdir -p "$DATA" "$UNIT_DIR"
cp -f "$SERVER_SOURCE" "$DATA/kiki_stt_server.py"
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
  "$PYTHON_EXEC" -m pip install -U pip wheel
  echo "Installiere faster-whisper (ctranslate2) und qwen-asr …"
  "$PYTHON_EXEC" -m pip install -U faster-whisper qwen-asr
  if [[ "$GPU" -eq 1 ]]; then
    "$PYTHON_EXEC" -m pip install -U nvidia-cublas-cu12 nvidia-cudnn-cu12
  fi
  "$PYTHON_EXEC" - <<PY
import ctranslate2
print("ctranslate2", ctranslate2.__version__, "cuda_devices", ctranslate2.get_cuda_device_count())
PY
fi

UNIT="$UNIT_DIR/kiki-stt.service"
if [[ "$DUMMY" -eq 1 ]]; then
  EXTRA="--dummy"
  LIBLINE=""
elif [[ "$GPU" -eq 1 ]]; then
  EXTRA="--device auto --model ${MODEL} --language ${LANGUAGE}"
  SITE_PACKAGES="$("$VENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")"
  LIBLINE="Environment=LD_LIBRARY_PATH=${SITE_PACKAGES}/nvidia/cudnn/lib:${SITE_PACKAGES}/nvidia/cublas/lib"
else
  EXTRA="--device cpu --model ${MODEL} --language ${LANGUAGE}"
  LIBLINE=""
fi
cat > "$UNIT" <<EOF
[Unit]
Description=KIKI faster-whisper STT
PartOf=graphical-session.target
After=graphical-session.target

[Service]
Type=simple
ExecStart="${PYTHON_EXEC}" "${DATA}/kiki_stt_server.py" --host 127.0.0.1 --port 18775 ${EXTRA}
${LIBLINE}
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=HF_HUB_DISABLE_TELEMETRY=1
TimeoutStartSec=0

[Install]
WantedBy=graphical-session.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now kiki-stt.service

echo
echo "STT-Dienst: http://127.0.0.1:18775/health"
echo "  systemctl --user status kiki-stt.service"
if [[ "$DUMMY" -eq 0 ]]; then
  echo "Beim ersten Start lädt Hugging Face ${MODEL} (ca. 0,5 GB)."
fi
echo "Dummy-Test ohne Modell: $0 --dummy"
