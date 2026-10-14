"""Centre a logo SVG on a solid background of a fixed pixel size (e.g. the GitHub social preview, 1280 x 640).
The logo is scaled to FILL of the card height, or less if it would not fit the width with the same margin.
Usage: card_svg.py IN.svg OUT.svg WIDTH HEIGHT [fill=0.72] [bg=#FFFFFF]"""
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
CW, CH = float(sys.argv[3]), float(sys.argv[4])
FILL = float(sys.argv[5]) if len(sys.argv) > 5 else 0.72
BG = sys.argv[6] if len(sys.argv) > 6 else "#FFFFFF"

s = open(src).read()
x, y, w, h = map(float, re.search(r'viewBox="([^"]+)"', s).group(1).split())
inner = re.sub(r"^<svg[^>]*>\n?|</svg>\s*$", "", s.strip())
sc = min(FILL * CH / h, (CW - (1 - FILL) * CH) / w)       # same margin on all sides at most
lw, lh = w * sc, h * sc
open(dst, "w").write(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{CW:.0f}" height="{CH:.0f}" viewBox="0 0 {CW:.0f} {CH:.0f}">\n'
    f'<rect width="{CW:.0f}" height="{CH:.0f}" fill="{BG}"/>\n'
    f'<svg x="{(CW - lw) / 2:.2f}" y="{(CH - lh) / 2:.2f}" width="{lw:.2f}" height="{lh:.2f}" viewBox="{x} {y} {w} {h}">\n'
    f'{inner}\n</svg>\n</svg>\n')
