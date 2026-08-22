#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALLER="${1:-}"
if [[ -z "$INSTALLER" ]]; then
  INSTALLER="$(find "$PROJECT_ROOT/dist" -maxdepth 1 -type f -name 'KIKI-*-Fedora-44-x86_64.run' -print | sort -V | tail -n 1)"
fi
[[ -n "$INSTALLER" && -x "$INSTALLER" ]] || {
  echo "Kein ausführbarer Fedora-Installer gefunden." >&2
  exit 1
}
RPM_PATH="${2:-$(find "$PROJECT_ROOT/dist" -maxdepth 1 -type f -name 'kiki-*.x86_64.rpm' -print | sort -V | tail -n 1)}"
[[ -n "$RPM_PATH" && -f "$RPM_PATH" ]] || {
  echo "Kein passendes RPM zum Vergleich mit dem Installer gefunden." >&2
  exit 1
}

"$INSTALLER" --help >/dev/null
"$INSTALLER" --verify-only
EMBEDDED_SHA="$(awk -F'"' '/^KIKI_RPM_SHA256=/{print $2; exit}' "$INSTALLER")"
CURRENT_SHA="$(sha256sum "$RPM_PATH" | awk '{print $1}')"
[[ "$EMBEDDED_SHA" == "$CURRENT_SHA" ]] || {
  echo "Installer enthält nicht das aktuelle RPM: $EMBEDDED_SHA != $CURRENT_SHA" >&2
  exit 1
}
echo "Installer-Smoke-Test bestanden: $INSTALLER"
