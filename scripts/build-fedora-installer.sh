#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE="$PROJECT_ROOT/packaging/installer/kiki-fedora44.run.in"
DIST_DIR="$PROJECT_ROOT/dist"
VERSION="$({
  python3 -c 'import pathlib, sys, tomllib; print(tomllib.loads(pathlib.Path(sys.argv[1]).read_text())["project"]["version"])' \
    "$PROJECT_ROOT/pyproject.toml"
} 2>/dev/null)"
RPM_PATH="${1:-}"
OUTPUT="${2:-$DIST_DIR/KIKI-${VERSION}-Fedora-44-x86_64.run}"

[[ -f "$TEMPLATE" ]] || { echo "Installer-Template fehlt: $TEMPLATE" >&2; exit 1; }
if [[ -z "$RPM_PATH" ]]; then
  # Newest binary RPM for this version; the release may have been bumped.
  RPM_PATH="$(
    ls -t "$DIST_DIR"/kiki-"$VERSION"-*.fc44.x86_64.rpm 2>/dev/null | head -n 1 || true
  )"
fi
[[ -n "$RPM_PATH" && -f "$RPM_PATH" ]] || {
  echo "RPM fehlt: $RPM_PATH" >&2
  echo "Zuerst ./scripts/build-rpm.sh ausführen." >&2
  exit 1
}
[[ "$(rpm -qp --qf '%{NAME}-%{VERSION}-%{RELEASE}.%{ARCH}' "$RPM_PATH")" == \
  "kiki-${VERSION}-"*.fc44.x86_64 ]] || {
  echo "RPM-Header passt nicht zum Installerziel." >&2
  exit 1
}

mkdir -p "$DIST_DIR"
BUILD_TMP="$(mktemp /tmp/kiki-installer-build.XXXXXX)"
cleanup() {
  case "$BUILD_TMP" in
    /tmp/kiki-installer-build.*) rm -f -- "$BUILD_TMP" ;;
  esac
}
trap cleanup EXIT

RPM_SHA256="$(sha256sum "$RPM_PATH" | awk '{print $1}')"
RPM_SIZE="$(wc -c < "$RPM_PATH")"
RPM_NEVRA="$(rpm -qp --qf '%{NAME}-%{VERSION}-%{RELEASE}.%{ARCH}' "$RPM_PATH")"
sed \
  -e "s/@KIKI_VERSION@/$VERSION/g" \
  -e "s/@KIKI_RPM_SHA256@/$RPM_SHA256/g" \
  -e "s/@KIKI_RPM_SIZE@/$RPM_SIZE/g" \
  -e "s/@KIKI_RPM_NEVRA@/$RPM_NEVRA/g" \
  "$TEMPLATE" > "$BUILD_TMP"
bash -n "$BUILD_TMP"
install -m0755 "$BUILD_TMP" "$OUTPUT"
dd if="$RPM_PATH" of="$OUTPUT" oflag=append conv=notrunc status=none

echo "Installer gebaut: $OUTPUT"
echo "SHA-256: $(sha256sum "$OUTPUT" | awk '{print $1}')"
"$OUTPUT" --verify-only
