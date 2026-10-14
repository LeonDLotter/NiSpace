"""Hex-tiled NiSpace brain with a smooth outside.
mode 'clip' : full hex lattice clipped by the smooth brain outline; Sylvian gap + outline drawn as white strokes
mode 'scale': whole hexagons, each scaled by the fraction of its area inside the (smoothed) brain minus the gap
Usage: hex_logo.py SCENE.npz NCOL MODE OUT.svg [ring=0.05] [gap=0.8] [bg]"""
import sys, math
import numpy as np
from scipy import ndimage
from scipy.stats import rankdata
from matplotlib.path import Path

sc = np.load(sys.argv[1])
ncol, mode, out = int(sys.argv[2]), sys.argv[3], sys.argv[4]
ring = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
gapw = float(sys.argv[6]) if len(sys.argv) > 6 else 0.8
bg = sys.argv[7] if len(sys.argv) > 7 else None
hit, saimg, s = sc["hit"], sc["saimg"], float(sc["s"])
brain, syl = sc["brain"], sc["syl_ext"]
H, W = hit.shape
RED, NAVY = np.array((178, 32, 31)), np.array((2, 61, 107))     # thin cortex -> red, thick -> navy

p = 60 / ncol; rp = p * math.sqrt(3) / 2; R = p / math.sqrt(3)
VBW, VBH = 64, 4 + H * s
gw = gapw * p; sw = ring * p

def cr_path(P, closed):
    P = [tuple(x) for x in P]
    if closed and P[0] == P[-1]: P = P[:-1]
    n = len(P); d = f"M{P[0][0]:.2f} {P[0][1]:.2f}"
    for i in range(n if closed else n - 1):
        p0 = P[i - 1] if closed else P[max(i - 1, 0)]
        p1, p2 = P[i], P[(i + 1) % n]
        p3 = P[(i + 2) % n] if closed else P[min(i + 2, n - 1)]
        d += (f" C{p1[0] + (p2[0] - p0[0]) / 6:.2f} {p1[1] + (p2[1] - p0[1]) / 6:.2f}"
              f" {p2[0] - (p3[0] - p1[0]) / 6:.2f} {p2[1] - (p3[1] - p1[1]) / 6:.2f} {p2[0]:.2f} {p2[1]:.2f}")
    return d + (" Z" if closed else "")

# inside test in viewBox coords: smooth outline polygon (densified) minus the gap band
def dense(P, closed, n=400):
    P = np.array(P, float)
    if closed: P = np.vstack([P, P[:1]])
    t = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    tt = np.linspace(0, t[-1], n)
    return np.c_[np.interp(tt, t, P[:, 0]), np.interp(tt, t, P[:, 1])]
outline = Path(dense(brain, True))
sylD = dense(syl, False, 300)
def inside(pts):
    ok = outline.contains_points(pts)
    d = np.min(np.hypot(pts[:, None, 0] - sylD[None, :, 0], pts[:, None, 1] - sylD[None, :, 1]), axis=1)
    return ok & (d > gw / 2)
def value(pts):
    ix = np.clip(((pts[:, 0] - 2) / s).astype(int), 0, W - 1); iy = np.clip(((pts[:, 1] - 2) / s).astype(int), 0, H - 1)
    return saimg[iy, ix]

corners = np.array([(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in range(30, 390, 60)])
hexpath = Path(corners)
g = np.linspace(-1, 1, 15); sub = np.array([(a, b) for a in g for b in g]); sub = sub[hexpath.contains_points(sub)]

# lattice phase: centre the lattice on the brain's bounding box
x0, x1 = brain[:, 0].min(), brain[:, 0].max(); y0, y1 = brain[:, 1].min(), brain[:, 1].max()
cells = []
for r in range(-2, int((VBH) / rp) + 3):
    for c in range(-2, ncol + 3):
        cx = x0 - p + (c + 0.5 * (r % 2)) * p; cy = y0 - rp + r * rp
        pts = np.c_[cx + sub[:, 0] * R, cy + sub[:, 1] * R]
        ins = inside(pts); cov = ins.mean()
        if cov == 0: continue
        if mode == "clip" and not outline.contains_points(pts).any(): continue
        v = np.nanmean(value(pts[ins] if ins.any() else pts))
        cells.append((cx, cy, cov, v))
MINCOV = float(sys.argv[8]) if len(sys.argv) > 8 else 0.2          # outer edge
MINCOV_GAP = float(sys.argv[9]) if len(sys.argv) > 9 else MINCOV    # cells next to the Sylvian gap
def near_gap(cx, cy):
    return np.min(np.hypot(sylD[:, 0] - cx, sylD[:, 1] - cy)) < gw / 2 + R and outline.contains_points([(cx, cy)])[0]
for cx, cy, cov, v in cells:
    if 0.02 < cov < MINCOV and near_gap(cx, cy): print(f"  gap cell ({cx:.1f},{cy:.1f}) cov={cov:.2f}")
cells = [c for c in cells if (mode == "clip") or c[2] >= (MINCOV_GAP if near_gap(c[0], c[1]) else MINCOV)]
rk = (rankdata([c[3] for c in cells]) - 1) / (len(cells) - 1)
col = lambda t: "#%02X%02X%02X" % tuple(np.round(RED + (NAVY - RED) * t).astype(int))

FULLCOV = float(sys.argv[10]) if len(sys.argv) > 10 else 0.999   # cells at/above this coverage: full size + outline
print("  coverage >=", [ (th, sum(c[2] >= th for c in cells)) for th in (0.999, 0.95, 0.9, 0.85, 0.8, 0.75)])
els, els_plain = [], []          # full-size cells get the white outline, scaled-down cells none
TINT = float(sys.argv[11]) if len(sys.argv) > 11 else None       # tinted-rim mode: rim = cell colour mixed with white
import json as _json
_json.dump({'R': R, 'p': p, 'sw': sw, 'VBW': VBW, 'VBH': VBH, 'cells': [[c[0], c[1], c[2], float(t)] for c, t in zip(cells, rk)], 'brain': brain.tolist(), 'syl': syl.tolist()}, open(out + '.cells.json', 'w'))
if TINT is not None:
    # every cell: gap of sw to its neighbours (background shows through) + a rim of width sw in a lighter tint;
    # gap and rim scale with the cell, no snapping
    tint = lambda t: "#%02X%02X%02X" % tuple(np.round((RED + (NAVY - RED) * t) * (1 - TINT) + 255 * TINT).astype(int))
    for (cx, cy, cov, v), t in zip(cells, rk):
        k = math.sqrt(cov)
        rr = k * (R - sw / math.sqrt(3) - sw / math.sqrt(3))
        if rr <= 0: continue
        pts = " ".join(f"{cx + rr * dx:.2f},{cy + rr * dy:.2f}" for dx, dy in corners)
        els_plain.append(f'<polygon points="{pts}" fill="{col(t)}" stroke="{tint(t)}" stroke-width="{k * sw:.3f}"/>')
    cells = []
for (cx, cy, cov, v), t in zip(cells, rk):
    full = mode == "clip" or cov >= FULLCOV
    k = 1.0 if full else math.sqrt(cov)
    # stroked cells show colour out to R - 2*sw/sqrt(3); unstroked ones are drawn to that same visible extent
    rr = k * R - (1 if full else 2) * sw / math.sqrt(3)
    if rr <= 0: continue
    pts = " ".join(f"{cx + rr * dx:.2f},{cy + rr * dy:.2f}" for dx, dy in corners)
    (els if full else els_plain).append(f'<polygon points="{pts}" fill="{col(t)}"/>')

bpath, spath = cr_path(brain, True), cr_path(syl, False)
parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VBW} {VBH:.1f}">']
if bg: parts.append(f'<rect width="100%" height="100%" fill="{bg}"/>')
if mode == "clip":
    parts.append(f'<defs><clipPath id="nsp-brain"><path d="{bpath}"/></clipPath></defs>')
    parts.append('<g clip-path="url(#nsp-brain)">')
    parts.append(f'<g stroke="#FFFFFF" stroke-width="{sw:.3f}" stroke-linejoin="round">' + "".join(els) + "</g>")
    parts.append(f'<path d="{spath}" fill="none" stroke="#FFFFFF" stroke-width="{gw:.3f}" stroke-linecap="round"/>')
    parts.append("</g>")
    parts.append(f'<path d="{bpath}" fill="none" stroke="#FFFFFF" stroke-width="{2 * sw:.3f}" stroke-linejoin="round"/>')
else:
    parts.append(f'<g stroke="#FFFFFF" stroke-width="{sw:.3f}" stroke-linejoin="round">' + "".join(els) + "</g>")
    parts.append("<g>" + "".join(els_plain) + "</g>")
parts.append("</svg>")
open(out, "w").write("\n".join(parts) + "\n")
print(out, mode, len(els), "outlined +", len(els_plain), "plain cells")
