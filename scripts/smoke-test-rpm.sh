#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RPM_PATH="${1:-}"

if [[ -z "$RPM_PATH" ]]; then
  RPM_PATH="$(find "$PROJECT_ROOT/dist" -maxdepth 1 -type f -name 'kiki-*.x86_64.rpm' | sort -V | tail -n 1)"
fi
if [[ -z "$RPM_PATH" || ! -f "$RPM_PATH" ]]; then
  echo "Kein KIKI-RPM gefunden. Zuerst ./scripts/build-rpm.sh ausführen." >&2
  exit 1
fi
RPM_PATH="$(realpath "$RPM_PATH")"

SMOKE_ROOT="$(mktemp -d /tmp/kiki-rpm-smoke.XXXXXX)"
cleanup() {
  case "$SMOKE_ROOT" in
    /tmp/kiki-rpm-smoke.*) rm -rf -- "$SMOKE_ROOT" ;;
  esac
}
trap cleanup EXIT

(
  cd "$SMOKE_ROOT"
  rpm2cpio "$RPM_PATH" | cpio -id --quiet
)

PYTHON_SITE="$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib", scheme="rpm_prefix"))')"
export PYTHONPATH="$SMOKE_ROOT$PYTHON_SITE"
export PYTHONNOUSERSITE=1
export KIKI_DATA_DIR="$SMOKE_ROOT/usr/share/kiki"
export XDG_CONFIG_HOME="$SMOKE_ROOT/home/.config"
export XDG_DATA_HOME="$SMOKE_ROOT/home/.local/share"
export XDG_CACHE_HOME="$SMOKE_ROOT/home/.cache"
export XDG_STATE_HOME="$SMOKE_ROOT/home/.local/state"

test -x "$SMOKE_ROOT/usr/bin/kiki"
test -f "$KIKI_DATA_DIR/character/kiki/manifest.toml"
test -f "$KIKI_DATA_DIR/character/kiki/idle/00.png"
test -f "$KIKI_DATA_DIR/character/kiki-adult-v3/manifest.toml"
test -f "$KIKI_DATA_DIR/character/kiki-adult-v3/idle/00.png"
test -f "$SMOKE_ROOT/usr/share/applications/io.github.projectkiki.Kiki.desktop"
test -f "$SMOKE_ROOT/usr/share/metainfo/io.github.projectkiki.Kiki.metainfo.xml"
test -f "$SMOKE_ROOT/usr/lib/systemd/user/kiki.service"
rpm -qp --requires "$RPM_PATH" | grep -q '^vosk-api-devel >= 0.3.50'
rpm -qp --requires "$RPM_PATH" | grep -q '^espeak-ng'
rpm -qp --requires "$RPM_PATH" | grep -q '^gstreamer1-plugins-good'
rpm -qp --requires "$RPM_PATH" | grep -q '^xdg-terminal-exec'
rpm -qp --requires "$RPM_PATH" | grep -q '^xdg-utils'

desktop-file-validate \
  "$SMOKE_ROOT/usr/share/applications/io.github.projectkiki.Kiki.desktop"
appstreamcli validate --no-net --override=url-homepage-missing=info \
  "$SMOKE_ROOT/usr/share/metainfo/io.github.projectkiki.Kiki.metainfo.xml"

EXPECTED_VERSION="$(rpm -qp --qf '%{VERSION}' "$RPM_PATH")"
ACTUAL_VERSION="$("$SMOKE_ROOT/usr/bin/kiki" --version)"
test "$ACTUAL_VERSION" = "kiki $EXPECTED_VERSION"
echo "$ACTUAL_VERSION"
"$SMOKE_ROOT/usr/bin/kiki" --check
python3 -c 'from kiki.character.assets import load_character_pack; p = load_character_pack(); assert p.id == "kiki-adult-v3" and p.aspect > 0 and len(p.clips) == 12; print(f"character={p.id}, clips={len(p.clips)}")'

echo "RPM-Smoke-Test bestanden: $RPM_PATH"
