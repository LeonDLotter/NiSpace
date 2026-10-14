"""Pad an SVG's viewBox to a square (content centred).
With a layout JSON from hexbin_logo.py, the graphic's own bounding box is used (i.e. without the text block).
Usage: square_svg.py IN.svg OUT.svg [LAYOUT.json]"""
import json
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
m = re.search(r'viewBox="([^"]+)"', s)
x, y, w, h = map(float, m.group(1).split())
if len(sys.argv) > 3:
    x, y, w, h = json.load(open(sys.argv[3]))["graphic_bbox"]
side = max(w, h)
# explicit width/height so the SVG has an intrinsic size when used as <img>
box = (f'width="{side:.2f}" height="{side:.2f}" '
       f'viewBox="{x - (side - w) / 2:.2f} {y - (side - h) / 2:.2f} {side:.2f} {side:.2f}"')
open(dst, "w").write(s.replace(m.group(0), box, 1))
