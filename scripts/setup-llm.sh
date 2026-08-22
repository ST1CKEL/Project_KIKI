#!/usr/bin/env bash
# KIKI's own LLM harness (PyTorch/transformers, CUDA). Neither Ollama nor
# llama.cpp: KIKI drives generation itself so it can ban reasoning tokens,
# render tool declarations through the model's template and budget VRAM.
set -euo pipefail

# Resolve symlinks: the RPM installs this as /usr/libexec/kiki/setup-llm and
# links /usr/bin/kiki-setup-llm to it. Using $0 unresolved would look for the
# server files in /usr/bin, where they are not.
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"

DATA="${XDG_DATA_HOME:-$HOME/.local/share}/kiki/llm"
# Shared with the TTS service on purpose: both need the same torch + CUDA +
# transformers stack, and transformers 4.57 already supports Qwen3, so nothing
# has to be upgraded underneath the running TTS model.
VENV="${XDG_DATA_HOME:-$HOME/.local/share}/kiki/tts-venv"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
MODEL="${KIKI_LLM_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
QUANTIZE="${KIKI_LLM_QUANTIZE:-int4}"
SLOTS="${KIKI_LLM_SLOTS:-4}"
# Continuous batching: one forward pass serves every active sequence.
# Measured on an RTX 5060 Ti with Qwen3-4B int4, same binary, only this
# flag differing: 1 request 1.00x, 2 requests 1.74x, 4 requests 3.28x.
BATCH="${KIKI_LLM_BATCH:-4}"
PORT="${KIKI_LLM_PORT:-18770}"
ECHO_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --echo) ECHO_ONLY=1 ;;
    --model=*) MODEL="${arg#*=}" ;;
    --quantize=*) QUANTIZE="${arg#*=}" ;;
    --slots=*) SLOTS="${arg#*=}" ;;
    --batch=*) BATCH="${arg#*=}" ;;
    -h|--help)
      echo "Aufruf: $0 [--echo] [--model=ID] [--quantize=int4|int8|none] [--slots=N] [--batch=N]"
      exit 0 ;;
    *) echo "Unbekannte Option: $arg" >&2; exit 1 ;;
  esac
done

for name in kiki_llm_server.py toolcalls.py vram.py; do
  if [[ -f "$SCRIPT_DIR/../services/kiki-llm/$name" ]]; then
    SRC_DIR="$SCRIPT_DIR/../services/kiki-llm"
  elif [[ -f "$SCRIPT_DIR/$name" ]]; then
    SRC_DIR="$SCRIPT_DIR"
  else
    echo "$name wurde nicht neben dem Setup-Skript gefunden." >&2
    exit 1
  fi
done

mkdir -p "$DATA" "$UNIT_DIR"
for name in kiki_llm_server.py toolcalls.py vram.py; do
  cp -f "$SRC_DIR/$name" "$DATA/$name"
done

PYTHON_EXEC="$(command -v python3)"
if [[ "$ECHO_ONLY" -eq 0 ]]; then
  if [[ ! -x "$VENV/bin/python" ]]; then
    echo "Die gemeinsame venv fehlt: $VENV" >&2
    echo "Zuerst die Sprachausgabe einrichten: ./scripts/setup-tts.sh" >&2
    exit 1
  fi
  PYTHON_EXEC="$VENV/bin/python"
  "$PYTHON_EXEC" - <<'PY'
import sys
try:
    import torch, transformers  # noqa: F401
except ImportError as exc:
    sys.exit(f"venv unvollständig: {exc}")
if not torch.cuda.is_available():
    sys.exit("CUDA ist in dieser venv nicht sichtbar. Treiber/Wheel prüfen.")
print(f"torch {torch.__version__} cuda ok, transformers {transformers.__version__}")
PY
  if [[ "$QUANTIZE" == "int4" || "$QUANTIZE" == "int8" ]]; then
    echo "Installiere bitsandbytes für $QUANTIZE …"
    "$PYTHON_EXEC" -m pip install -q -U bitsandbytes
  fi
fi

UNIT="$UNIT_DIR/kiki-llm.service"
if [[ "$ECHO_ONLY" -eq 1 ]]; then
  EXTRA="--echo"
else
  EXTRA="--model ${MODEL} --quantize ${QUANTIZE} --device auto --batch ${BATCH}"
fi
cat > "$UNIT" <<EOF
[Unit]
Description=KIKI LLM harness (eigene Modell-Laufzeit)
PartOf=graphical-session.target
After=graphical-session.target

[Service]
Type=simple
ExecStart="${PYTHON_EXEC}" "${DATA}/kiki_llm_server.py" --host 127.0.0.1 --port ${PORT} --slots ${SLOTS} ${EXTRA}
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=HF_HUB_DISABLE_TELEMETRY=1
TimeoutStartSec=0

[Install]
WantedBy=graphical-session.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now kiki-llm.service

echo
echo "LLM-Harness: http://127.0.0.1:${PORT}/health"
echo "  systemctl --user status kiki-llm.service"
if [[ "$ECHO_ONLY" -eq 0 ]]; then
  echo "Beim ersten Start lädt Hugging Face ${MODEL} (mehrere GB)."
  echo "In KIKI: Einstellungen → Anbieter → KIKI-Harness."
else
  echo "Echo-Modus: kein Modell, nur das Protokoll."
fi
