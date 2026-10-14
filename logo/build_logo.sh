#!/usr/bin/env bash
# Build the NiSpace logo set. All inputs are downloaded from pinned remote sources (see remote.py).
# Usage: ./build_logo.sh [python]        (default: python3 on PATH; needs numpy, scipy, nibabel,
#                                         scikit-image, matplotlib, fonttools; rsvg-convert for PNGs)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${1:-python3}"
W="$HERE/work"; OUT="$HERE/final"
mkdir -p "$W" "$OUT"

# 1 · brain scene: right lateral view of the MNI152NLin6Asym FreeSurfer pial surface (GIN),
#     Sylvian line from the DK temporal border, cortical thickness (5 mm smoothed)
"$PY" "$HERE/brain_scene.py" "$W"
SCENE="$W/brain_scene.npz"
FONT="$("$PY" "$HERE/remote.py" "$W/data" JetBrainsMono-Bold.ttf)"
FONT_NAME="$("$PY" "$HERE/remote.py" "$W/data" JetBrainsMono-ExtraBold.ttf)"

# 2 · hexagon cells (coverage-scaled, tinted rims); 18 columns for the logo, 12 for the small icons
#     args: SCENE ncol mode out ring gap bg mincov mincov_gap fullcov tint
for N in 18 12; do
  "$PY" "$HERE/hex_logo.py" "$SCENE" $N scale "$W/hex_n$N.svg" 0.08 0.8 "" 0.12 0.12 2 0.35 > /dev/null
done

# 3 · graphic in font units: "ni" + brain, cells blowing into a hexbin band along a regression line
#     args: cells out layout bg frame seed land brain_h base1 line_h gap_to_line plot_aspect rim_moved rim_brain
#     gap_to_line "0.06+1p": one lattice step right of the c->e letter gap (moves the plot without re-drawing it);
#     rims: 2.2 on moved cells (visible on dark backgrounds), 1.0 in the brain
"$PY" "$HERE/hexbin_logo.py" "$W/hex_n18.svg.cells.json" "$W/lockup.svg" lockup "" 0 3 0.55 1.25 1.0 0.9 "0.06+1p" 0.75 2.2 1.0
# favicon: the same partly dissolved brain (12 columns), without flying cells, plot and line
NSP_BRAIN_ONLY=1 "$PY" "$HERE/hexbin_logo.py" "$W/hex_n12.svg.cells.json" "$W/favicon.svg" lockup "" 0 3 0.55 1.25 1.0 0.9 0.06 0.75

# 4 · logos with the text as outlines (one version for all backgrounds)
"$PY" "$HERE/compose_logo.py" "$W/lockup.svg" "$FONT" "$OUT/nispace_logo.svg" 0 ""
#   full name without a box, ExtraBold, 0.18 F below the "p"; n·i·s·p·a·c·e 1.2x in light red (35% white) /
#   light navy (40% white), the rest grey #808893: all ~3.5:1 on white and on dark grey
NSP_FONT_NAME="$FONT_NAME" NSP_HALO_TXT=0 NSP_NAME_GAP=0.18 NSP_HL_SCALE=1.2 NSP_REST="#808893" \
  "$PY" "$HERE/compose_logo.py" "$W/lockup.svg" "$FONT" "$OUT/nispace_logo_fullname.svg" 1 "" halo 0.35 white mid "" 0.35 0.40
"$PY" "$HERE/square_svg.py" "$W/lockup.svg" "$OUT/nispace_icon.svg" "$W/lockup.svg.layout.json"      # the logo graphic without text
"$PY" "$HERE/square_svg.py" "$W/favicon.svg" "$OUT/nispace_favicon.svg" "$W/favicon.svg.layout.json"

# 5 · PNG exports
command -v rsvg-convert > /dev/null || { echo "rsvg-convert not found: SVGs done, PNGs skipped"; exit 0; }
for w in 600 1200 2400; do
  rsvg-convert -w $w "$OUT/nispace_logo.svg" -o "$OUT/nispace_logo_${w}.png"
  rsvg-convert -w $w "$OUT/nispace_logo_fullname.svg" -o "$OUT/nispace_logo_fullname_${w}.png"
done
for s in 128 256 512; do rsvg-convert -w $s -h $s "$OUT/nispace_icon.svg" -o "$OUT/nispace_icon_${s}.png"; done
for s in 16 32 48 180; do rsvg-convert -w $s -h $s "$OUT/nispace_favicon.svg" -o "$OUT/nispace_favicon_${s}.png"; done
echo "done: $OUT"
