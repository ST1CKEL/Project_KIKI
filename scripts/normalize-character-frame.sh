#!/usr/bin/env bash
# Normalize an image-generated character pose into KIKI's production canvas.
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Aufruf: $0 QUELLE.png ZIEL.png" >&2
  exit 2
fi
if ! command -v magick >/dev/null 2>&1; then
  echo "ImageMagick fehlt (`sudo dnf install ImageMagick`)." >&2
  exit 1
fi

SOURCE="$1"
DEST="$2"
if [[ ! -f "$SOURCE" ]]; then
  echo "Quelldatei fehlt: $SOURCE" >&2
  exit 1
fi

WORK_DIR="$(mktemp -d /tmp/kiki-frame-normalize.XXXXXX)"
cleanup() {
  case "$WORK_DIR" in
    /tmp/kiki-frame-normalize.*) rm -rf -- "$WORK_DIR" ;;
  esac
}
trap cleanup EXIT

CUTOUT="$WORK_DIR/cutout.png"
PART="$WORK_DIR/final.png"
CHANNELS="$(magick identify -format '%[channels]' "$SOURCE")"

case "$CHANNELS" in
  *a*)
    magick "$SOURCE" "$CUTOUT"
    ;;
  *)
    # Built-in image generation occasionally paints a light checkerboard
    # instead of returning alpha. It is connected to the canvas edge, while
    # KIKI's dark outline protects light clothing and eye highlights.
    magick "$SOURCE" \
      -alpha on \
      -bordercolor white -border 1 \
      -fuzz 9% -fill none -draw 'alpha 0,0 floodfill' \
      -shave 1x1 \
      "$CUTOUT"
    ;;
esac

# Equal visible height and a fixed square canvas prevent jumps between poses.
magick "$CUTOUT" \
  -trim +repage \
  -resize '460x460>' \
  -gravity center -background none -extent 512x512 \
  -strip -define png:color-type=6 \
  "$PART"

CORNER_ALPHA="$(magick "$PART" -format '%[fx:p{0,0}.a]' info:)"
BOUNDING_BOX="$(magick identify -format '%@' "$PART")"
if [[ "$CORNER_ALPHA" != "0" || "$BOUNDING_BOX" == "0x0+0+0" ]]; then
  echo "Freistellung fehlgeschlagen: alpha=$CORNER_ALPHA, bbox=$BOUNDING_BOX" >&2
  exit 1
fi

mkdir -p "$(dirname "$DEST")"
install -m0644 "$PART" "$DEST"
echo "$DEST: 512x512 RGBA, bbox=$BOUNDING_BOX"
