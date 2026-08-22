#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPEC="$PROJECT_ROOT/packaging/rpm/kiki.spec"
DIST_DIR="$PROJECT_ROOT/dist"
SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1787270400}"

VERSION="$({
  python3 -c 'import pathlib, sys, tomllib; print(tomllib.loads(pathlib.Path(sys.argv[1]).read_text())["project"]["version"])' \
    "$PROJECT_ROOT/pyproject.toml"
} 2>/dev/null)"
SPEC_VERSION="$(sed -n 's/^Version:[[:space:]]*//p' "$SPEC" | head -n 1)"

if [[ -z "$VERSION" || "$VERSION" != "$SPEC_VERSION" ]]; then
  echo "Versionskonflikt: pyproject.toml=$VERSION, Spec=$SPEC_VERSION" >&2
  exit 1
fi
if ! command -v rpmbuild >/dev/null 2>&1; then
  echo "rpmbuild fehlt. Installiere auf Fedora: sudo dnf install rpm-build" >&2
  exit 1
fi

RPM_TOPDIR="$(mktemp -d /tmp/kiki-rpmbuild.XXXXXX)"
cleanup() {
  case "$RPM_TOPDIR" in
    /tmp/kiki-rpmbuild.*) rm -rf -- "$RPM_TOPDIR" ;;
  esac
}
trap cleanup EXIT

mkdir -p "$RPM_TOPDIR"/{BUILD,BUILDROOT,RPMS,SOURCES,SPECS,SRPMS,tmp} "$DIST_DIR"

tar \
  --sort=name \
  --mtime="@$SOURCE_DATE_EPOCH" \
  --owner=0 --group=0 --numeric-owner \
  --exclude='./.git' \
  --exclude='./.venv' \
  --exclude='./build' \
  --exclude='./dist' \
  --exclude='./vendor' \
  --exclude='./.pytest_cache' \
  --exclude='./.ruff_cache' \
  --exclude='./.kiki-dev' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --transform="s#^\.#kiki-$VERSION#" \
  -C "$PROJECT_ROOT" -cf - . \
  | gzip -n > "$RPM_TOPDIR/SOURCES/kiki-$VERSION.tar.gz"

rpmbuild -ba "$SPEC" \
  --define "_topdir $RPM_TOPDIR" \
  --define "_tmppath $RPM_TOPDIR/tmp" \
  --define "_smp_build_ncpus 1" \
  --define "_smp_mflags -j1"

while IFS= read -r -d '' artifact; do
  install -m0644 "$artifact" "$DIST_DIR/$(basename "$artifact")"
done < <(find "$RPM_TOPDIR/RPMS" "$RPM_TOPDIR/SRPMS" -type f -name '*.rpm' -print0)

echo
echo "RPM-Build abgeschlossen:"
find "$DIST_DIR" -maxdepth 1 -type f -name 'kiki-*.rpm' -print | sort
