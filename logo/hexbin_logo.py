"""NiSpace brain dissolving into a hexbin plot.
Landed cells snap onto the brain's own hex lattice along a diagonal band; in-flight cells stay off-grid;
a regression line replaces the axes. Layered rendering: lighter base stays at the grid position,
true-colour cell on top (moved cells carry a matching rim).
Usage: hexbin_logo.py CELLS.json OUT.svg LAYOUT[side|inside] [bg] [frame=0|1] [seed]"""
import os, sys, json, math
import numpy as np

D = json.load(open(sys.argv[1]))
out, layout = sys.argv[2], sys.argv[3]
bg = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else None
frame = len(sys.argv) > 5 and sys.argv[5] == "1"
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 3
rng = np.random.default_rng(seed)
R, sw = D["R"], D["sw"]
p = R * math.sqrt(3); rp = 1.5 * R                         # lattice pitch (pointy-top hexes)
RED, NAVY = np.array((178, 32, 31)), np.array((2, 61, 107))
TINT, INSET = 0.35, 1.0
col = lambda t: RED + (NAVY - RED) * t
hexc = lambda c: "#%02X%02X%02X" % tuple(np.round(c).astype(int))

cx, cy, cov, tv = np.array(D["cells"]).T
n = len(cx); k = np.sqrt(cov); rad = k * R
bx0, bx1 = float((cx - rad).min()), float((cx + rad).max())
by0, by1 = float((cy - rad).min()), float((cy + rad).max())

# lockup: everything in font units (font size F = 100, JetBrains Mono advance 0.6 F)
#   line 1:  "ni" + brain (brain's right edge = end of "space")      line 2: "space"
#   scatter: right of both lines, from the brain's top to the line-2 baseline
F = 100.0; ADV = 0.6 * F
LK = {}; SC = 1.0
if layout == "lockup":
    BH = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0      # brain height in F
    BASE1 = float(sys.argv[9]) if len(sys.argv) > 9 else 0.85  # line-1 baseline below the brain top, in F
    LH = float(sys.argv[10]) if len(sys.argv) > 10 else 1.0    # line height, in F
    sb = BH * F / (by1 - by0)
    XB1 = 5 * ADV
    cx = XB1 - (bx1 - bx0) * sb + (cx - bx0) * sb
    cy = (cy - by0) * sb
    R, sw = R * sb, sw * sb
    p = R * math.sqrt(3); rp = 1.5 * R
    rad = k * R
    bx0, bx1 = float((cx - rad).min()), float((cx + rad).max())
    by0, by1 = float((cy - rad).min()), float((cy + rad).max())
    SC = sb
    LK = {"F": F, "base1": BASE1 * F, "base2": (BASE1 + LH) * F, "brain_x0": bx0, "brain_x1": bx1}

# lattice through the brain's cells: row index from y, parity from a reference cell
ref = int(np.argmax(cov))
def site(r, c):
    return cx[ref] + (c + 0.5 * ((r % 2) - (0 if True else 0))) * p, cy[ref] + r * rp
r_ref_par = 0  # rows are counted relative to the reference cell's row
def lattice_xy(r, c):
    shift = 0.5 * (((r % 2) + 2) % 2)
    return cx[ref] + (c + shift) * p, cy[ref] + r * rp

# which cells leave: frontal part, ragged front
U0, U1 = 0.60, 0.90
u = (cx - bx0) / (bx1 - bx0)
f = np.clip((u - U0) / (U1 - U0) + rng.normal(0, 0.06, n), 0, 1)
f = f * f * (3 - 2 * f)
mv = f > 0
g = f ** 1.3
LAND = float(sys.argv[7]) if len(sys.argv) > 7 else 0.45                                               # share of the leaving cells that land in the plot
fr = np.full(n, -1.0); fr[mv] = np.argsort(np.argsort(f[mv])) / max(mv.sum() - 1, 1)
landed = mv & (fr >= 1 - LAND)
flying = mv & ~landed

# ---- target band (hexbin) --------------------------------------------------------------------
if layout == "side":
    H = (by1 - by0) * 0.85; mid = (by0 + by1) / 2
    PX0 = bx1 + 16; PX1 = PX0 + 1.3 * H
    PY0, PY1 = mid - H / 2, mid + H / 2
    A = np.array([PX0 + 2, PY1 - 3]); B = np.array([PX1 - 2, PY0 + 3])   # regression line ends
    allow = lambda x, y: (x > PX0 + R) & (x < PX1 - R) & (y > PY0 + R) & (y < PY1 - R)
elif layout == "lockup":
    GAPF = float(sys.argv[11]) if len(sys.argv) > 11 else 0.7   # flight gap brain -> plot, in F
    ASP = float(sys.argv[12]) if len(sys.argv) > 12 else 1.3    # plot width / height
    PY0, PY1 = by0, LK["base2"]
    PX0 = 5 * ADV + GAPF * F                 # GAPF = 0: the line starts where a 6th letter would start
    PX1 = PX0 + ASP * (PY1 - PY0)
    A = np.array([PX0, PY1]); B = np.array([PX1, PY0])
    allow = lambda x, y: (x > PX0 + R) & (x < PX1 - R) & (y > PY0 + R) & (y < PY1 - R)
    LK.update({"plot": [PX0, PY0, PX1, PY1]})
else:  # inside: the brain sits at the lower left of one big plot, cells stream to the upper right
    S = (bx1 - bx0) * 1.35
    PX0, PY1 = bx0 - 4, by1 + 4
    PX1, PY0 = PX0 + S * 1.1, PY1 - S
    A = np.array([(bx0 + bx1) / 2 - 6, (by0 + by1) / 2 + 6]); B = np.array([PX1 - 3, PY0 + 3])
    far = lambda x, y: (x > bx1 + 1.5 * p) | (y < by0 - 1.5 * rp)
    allow = lambda x, y: far(x, y) & (x < PX1 - R) & (y > PY0 + R)
dvec = (B - A) / np.linalg.norm(B - A); nvec = np.array([-dvec[1], dvec[0]])
rows = range(int((PY0 - cy[ref]) / rp) - 2, int((PY1 - cy[ref]) / rp) + 3)
cols = range(int((PX0 - cx[ref]) / p) - 2, int((PX1 - cx[ref]) / p) + 3)
sites = np.array([lattice_xy(r, c) for r in rows for c in cols])
sites = sites[allow(sites[:, 0], sites[:, 1])]
along = (sites - A) @ dvec; across = (sites - A) @ nvec
L = np.linalg.norm(B - A)
sites = sites[(along > 0) & (along < L)]; along = (sites - A) @ dvec; across = (sites - A) @ nvec
# band: closeness to the line + jitter, with the band slightly wider in the middle (looks like data)
score = np.abs(across) / (1 + 0.6 * np.sin(np.pi * along / L)) + rng.normal(0, 0.22 * p, len(sites))
NL = int(landed.sum())
pick = np.argsort(score)[:NL]
band = sites[pick][np.argsort(along[pick])]                 # ordered along the trend
q = np.zeros(n)
li = np.nonzero(landed)[0]; li = li[np.argsort(tv[li])]     # lowest value -> start of the line
px, py = cx.copy(), cy.copy()
px[li], py[li] = band[:, 0], band[:, 1]
q[li] = np.linspace(0, 1, len(li))

# in-flight cells: off-grid, on the way from their origin towards a point on the band
fi = np.nonzero(flying)[0]
if len(fi):
    qf = (np.argsort(np.argsort(tv[fi])) / max(len(fi) - 1, 1))
    q[fi] = qf
    tgt = A + np.outer(qf * L, dvec) + np.outer(rng.normal(0, 1.2 * p, len(fi)), nvec)
    gg = np.clip(0.15 + 0.75 * fr[fi] / (1 - LAND), 0, 0.9)    # further along the more frontal
    d = tgt - np.c_[cx[fi], cy[fi]]; Ld = np.linalg.norm(d, axis=1) + 1e-9
    swirl = rng.uniform(-1, 1, len(fi)) * 4 * SC * np.sin(np.pi * gg)
    px[fi] = cx[fi] + d[:, 0] * gg - d[:, 1] / Ld * swirl
    py[fi] = cy[fi] + d[:, 1] * gg + d[:, 0] / Ld * swirl
    # relax only the flying cells against everything
    P = np.c_[px, py]
    need = (rad[fi][:, None] + rad[None, :]) * 0.95
    selfm = fi[:, None] == np.arange(n)[None, :]
    for it in range(300):
        dd = P[fi][:, None, :] - P[None, :, :]
        dist = np.hypot(dd[..., 0], dd[..., 1]) + selfm * 1e9
        ov = np.clip(need - dist, 0, None)
        if ov.max() < 1e-3: break
        P[fi] += (dd / dist[..., None] * ov[..., None]).sum(1) * 0.5
    px, py = P.T
tdisp = np.where(mv, tv + (q - tv) * np.where(landed, 1.0, g), tv)

# ---- draw --------------------------------------------------------------------------------------
corners = np.array([(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in range(30, 390, 60)])
hp = lambda x, y, r: " ".join(f"{x + r * a:.2f},{y + r * b:.2f}" for a, b in corners)
BRAIN_ONLY = os.environ.get("NSP_BRAIN_ONLY") == "1"         # only the partly dissolved brain, no flight/plot/line
base, top = [], []
for i in np.argsort(f):
    c0 = col(tv[i]); c = col(tdisp[i])
    l0 = c0 * (1 - TINT) + 255 * TINT; l1 = c * (1 - TINT) + 255 * TINT
    rb = k[i] * (R - sw / math.sqrt(3)); rt = rb - INSET * k[i] * sw * 2 / math.sqrt(3)
    if rt <= 0: continue
    base.append(f'<polygon points="{hp(cx[i], cy[i], rb)}" fill="{hexc(l0)}"/>')
    if mv[i]:
        if BRAIN_ONLY: continue                              # favicon: the imprint stays, the blown cells are gone
        top.append(f'<polygon points="{hp(px[i], py[i], (rb + rt) / 2)}" fill="{hexc(c)}" stroke="{hexc(l1)}" stroke-width="{INSET * k[i] * sw:.3f}"/>')
    else:
        top.append(f'<polygon points="{hp(px[i], py[i], rt)}" fill="{hexc(c)}"/>')

GREY, HALO = "#8C95A1", "#023D6B"
lw = (0.8 if layout == "lockup" else 0.5) * SC
ext = (0.0 if layout == "lockup" else 0.06) * L
la, lb = (A, B + dvec * ext) if layout == "lockup" else (A - dvec * ext, B + dvec * ext)
line = f"M{la[0]:.2f} {la[1]:.2f} L{lb[0]:.2f} {lb[1]:.2f}"
deco_under, deco_over = [], []
if frame:
    fpath = f"M{PX0:.2f} {PY0:.2f} V{PY1:.2f} H{PX1:.2f}"
    deco_under += [f'<path d="{fpath}" fill="none" stroke="{HALO}" stroke-width="{lw + 0.3:.2f}" stroke-linecap="round" stroke-linejoin="round"/>',
                   f'<path d="{fpath}" fill="none" stroke="{GREY}" stroke-width="{lw:.2f}" stroke-linecap="round" stroke-linejoin="round"/>']
# regression line: behind the cells in the brain (inside layout), on top of the band
deco_line = [f'<path d="{line}" fill="none" stroke="{HALO}" stroke-width="{lw * 1.5:.2f}" stroke-linecap="round"/>',
             f'<path d="{line}" fill="none" stroke="{GREY}" stroke-width="{lw:.2f}" stroke-linecap="round"/>']
if BRAIN_ONLY:
    deco_under, deco_line = [], []
    allx, ally = np.r_[cx - rad, cx + rad], np.r_[cy - rad, cy + rad]
else:
    allx = np.r_[px - rad, px + rad, cx - rad, cx + rad, [la[0], lb[0]] + ([PX0, PX1] if frame else [])]
    ally = np.r_[py - rad, py + rad, cy - rad, cy + rad, [la[1], lb[1]] + ([PY0, PY1] if frame else [])]
x0, y0 = allx.min() - 1.5 * SC, ally.min() - 1.5 * SC
W, Hh = allx.max() + 1.5 * SC - x0, ally.max() + 1.5 * SC - y0
LK["graphic_bbox"] = [float(x0), float(y0), float(W), float(Hh)]   # the graphic alone, without the text block
if layout == "lockup":   # include the text block: x from 0, down to the descender of the "p" in "space"
    x0 = min(x0, -0.04 * F); y0 = min(y0, -0.02 * F)
    W = max(allx.max() + 1.5 * SC, 5 * ADV) - x0; Hh = max(ally.max() + 1.5 * SC, LK["base2"] + 0.32 * F) - y0
    LK["viewBox"] = [x0, y0, W, Hh]
    LK["ink_right"] = float(max(allx.max(), lb[0] + lw))
    LK["pitch"] = p; LK["R"] = R
    json.dump(LK, open(out + ".layout.json", "w"), indent=1)
parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.1f} {y0:.1f} {W:.1f} {Hh:.1f}">']
if bg: parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{W:.1f}" height="{Hh:.1f}" fill="{bg}"/>')
if layout == "inside":
    parts += deco_under + deco_line + base + top
else:
    parts += deco_under + deco_line + base + top
parts.append("</svg>")
open(out, "w").write("\n".join(parts) + "\n")
print(out, layout, "landed", NL, "flying", len(fi), "static", int((~mv).sum()))
