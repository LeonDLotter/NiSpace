"""Compose the final NiSpace logo: graphic SVG (font units) + "ni" / "space" (+ optional full name) as outlines.
All alignment uses the letters' ink extents, so "ni", "space" and the full name share one left edge and the
full name ends exactly at the graphic's right edge.
Usage: compose_logo.py GRAPHIC.svg FONT.ttf OUT.svg [fullname=0|1] [bg] [namestyle=plain|box|hex]"""
import os, sys, json, re, math
import numpy as np
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.boundsPen import BoundsPen

gsvg, fontf, out = sys.argv[1], sys.argv[2], sys.argv[3]
fullname = len(sys.argv) > 4 and sys.argv[4] == "1"
bg = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None
style = sys.argv[6] if len(sys.argv) > 6 else "plain"
BOXMIX = float(sys.argv[7]) if len(sys.argv) > 7 else 0.35      # box = brain base tint (cell colour + 35% white)
BOXTXT = sys.argv[8] if len(sys.argv) > 8 else "white"
HL = sys.argv[9] if len(sys.argv) > 9 else "none"
BOXCOL = sys.argv[10] if len(sys.argv) > 10 else ""
TINT_RED = float(sys.argv[11]) if len(sys.argv) > 11 else 0.35
TINT_BLUE = float(sys.argv[12]) if len(sys.argv) > 12 else 0.35   # "" -> brain tint; "navy" | "navytint" | "grad" | "grey" | #hex     # none | dim | hex | dimhex : mark the letters that form "nispace"
LK = json.load(open(gsvg + ".layout.json"))
F = LK["F"]

font = TTFont(fontf)
gs = font.getGlyphSet(); cmap = font.getBestCmap(); upm = font["head"].unitsPerEm

def draw(txt, x, base, size, ls=0.0, pen=None, g=None):
    g = g or gs; s = size / upm
    for ch in txt:
        gn = cmap[ord(ch)]
        g[gn].draw(TransformPen(pen, (s, 0, 0, -s, x, base)))
        x += g[gn].width * s + ls
    return pen
def path(txt, x, base, size, ls=0.0):
    return draw(txt, x, base, size, ls, SVGPathPen(gs)).getCommands()
def ink(txt, size, ls=0.0, g=None):   # (xmin, xmax) of the ink with the origin at x = 0
    b = draw(txt, 0, 0, size, ls, BoundsPen(g or gs), g).bounds
    return b[0], b[2]

RED, NAVY = np.array((178, 32, 31)), np.array((2, 61, 107))
TINT = 0.35
col = lambda t: RED + (NAVY - RED) * t
hexc = lambda c: "#%02X%02X%02X" % tuple(np.round(c).astype(int))
mix = lambda c, w: c * (1 - w) + 255 * w
RIM = 0.07 * F         # drawn under the fill, so half of it shows
HALO_TXT = float(os.environ.get("NSP_HALO_TXT", 0.16))   # halo of the highlighted name letters without a box, in name font sizes
_rest = os.environ.get("NSP_REST", "#808893")             # HL "mid": colour of the other letters ("blend": half way to their word's colour)
REST_BLEND = _rest == "blend"
REST = np.array((128.0, 136, 147)) if REST_BLEND else np.array([int(_rest[i:i + 2], 16) for i in (1, 3, 5)], float)
NAME_GAP = os.environ.get("NSP_NAME_GAP")                 # style "halo": ink gap "space" (descender of p) -> name, in F

def glyphs(d, c, rim=RIM):
    return (f'<path d="{d}" fill="{hexc(mix(c, TINT))}" stroke="{hexc(mix(c, TINT))}" stroke-width="{rim:.2f}" stroke-linejoin="round"/>'
            f'<path d="{d}" fill="{hexc(c)}"/>')

# left edge = ink start of the big letters; both lines start there
l_ni, _ = ink("ni", F); l_sp, _ = ink("space", F)
L = min(l_ni, l_sp)
x_ni, x_sp = L - l_ni, L - l_sp
els = [glyphs(path("ni", x_ni, LK["base1"], F), RED), glyphs(path("space", x_sp, LK["base2"], F), NAVY)]

x0, y0, W, H = LK["viewBox"]
x0 = L - 0.04 * F; W = LK["ink_right"] + 0.04 * F - x0
Rgt = LK["ink_right"]
under = []
if fullname:
    name = "neuroimaging spatial colocalization environment"
    nf = os.environ.get("NSP_FONT_NAME")                       # optional weight for the whole name (same advances)
    fontN = TTFont(nf) if nf else font
    gsN = fontN.getGlyphSet()
    pad = 0.0 if style in ("plain", "halo") else 0.09 * F      # inner padding for box / hex strip
    target = (Rgt - L) - 2 * pad
    fs = target / (len(name) * 0.6)
    a0, a1 = ink(name, fs, 0.0, gsN)
    ls = (target - (a1 - a0)) / (len(name) - 1)             # stretch so the ink spans exactly the target width
    a0, a1 = ink(name, fs, ls, gsN)
    # equal ink gaps: bottom of "n" -> top of "s" == bottom of "s" -> top of the box
    bn = draw("n", 0, 0, F, 0, BoundsPen(gs)).bounds; bs = draw("s", 0, 0, F, 0, BoundsPen(gs)).bounds
    gap_ns = (LK["base2"] + bs[1]) - (LK["base1"] + bn[3])
    gap = bs[3] + gap_ns                                      # from the line-2 baseline to the box top
    print(f"  gap n->s {gap_ns:.2f}, s->box {gap - bs[3]:.2f}")
    base3 = LK["base2"] + gap + pad + 0.75 * fs
    xn = L + pad - a0
    MAUVE = mix(col(0.5), TINT)                               # lighter brain tone, readable on white and black
    if style == "plain":
        els.append(f'<path d="{path(name, xn, base3, fs, ls)}" fill="{hexc(MAUVE)}"/>')
        H = base3 + 0.25 * fs - y0
    else:
        top, bot = base3 - 0.75 * fs - pad, base3 + 0.25 * fs + pad
        BOXC = NAVY
        if style == "halo":   # no box: placed by its ink top further below (NAME_GAP under the "p", else one n->s gap)
            p_low = LK["base2"] + draw("space", 0, 0, F, 0, BoundsPen(gs)).bounds[3]   # lowest ink of "space" (y down)
        elif style == "outline":   # the box as a rim only, nothing filled
            h = bot - top
            base3 = (top + bot) / 2 + 0.30 * fs
            rect = f'x="{L:.2f}" y="{top:.2f}" width="{Rgt - L:.2f}" height="{h:.2f}" rx="{0.28 * h:.2f}"'
            under.append(f'<rect {rect} fill="none" stroke="{hexc(mix(NAVY, TINT))}" stroke-width="{RIM / 2:.2f}"/>')
        elif style == "box":
            h = bot - top
            base3 = (top + bot) / 2 + 0.30 * fs                    # optical centre: centre the x-height, not the em box
            boxc = mix(col(0.5), BOXMIX)
            fill = hexc(boxc); rimc = None
            if BOXCOL == "navy": boxc = NAVY; fill = hexc(NAVY); rimc = hexc(mix(NAVY, TINT))
            elif BOXCOL == "navytint": boxc = mix(NAVY, TINT); fill = hexc(boxc)
            elif BOXCOL == "grey": boxc = np.array((140, 149, 161)); fill = hexc(boxc)
            elif BOXCOL == "grad":
                boxc = col(0.5)
                under.append(f'<defs><linearGradient id="nsp-grad" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="{hexc(RED)}"/><stop offset="1" stop-color="{hexc(NAVY)}"/></linearGradient></defs>')
                fill = "url(#nsp-grad)"; rimc = hexc(mix(col(0.5), TINT))
            elif BOXCOL in ("pale", "white"):   # light box; the rim must be darker than the fill to show on white
                boxc = mix(col(0.5), 0.90) if BOXCOL == "pale" else np.array((255.0, 255, 255))
                fill = hexc(boxc); rimc = hexc(mix(NAVY, TINT))    # the halo of "space"
            elif BOXCOL.startswith("#"):
                boxc = np.array([int(BOXCOL[i:i + 2], 16) for i in (1, 3, 5)]); fill = BOXCOL
            rect = f'x="{L:.2f}" y="{top:.2f}" width="{Rgt - L:.2f}" height="{h:.2f}" rx="{0.28 * h:.2f}"'
            if rimc:   # rim behind the box, like the letters' rim: only the outer half shows
                under.append(f'<rect {rect} fill="{rimc}" stroke="{rimc}" stroke-width="{RIM:.2f}" stroke-linejoin="round"/>')
            under.append(f'<rect {rect} fill="{fill}"/>')
            BOXC = boxc
        elif style == "hex":   # hex strip on the brain's lattice pitch, light tints running red -> navy
            R = LK["R"] * 0.9; p = R * math.sqrt(3); rp = 1.5 * R
            corners = [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in range(30, 390, 60)]
            r_ = 0
            y = top + R * 0.5
            while y - R * 0.5 < bot:
                x = L + R * 0.87 + (0.5 * p if r_ % 2 else 0)
                while x + R * 0.87 <= Rgt + 1e-6:
                    t = (x - L) / (Rgt - L)
                    rr = R - 0.06 * F * 0.25
                    pts = " ".join(f"{x + rr * a:.2f},{y + rr * b:.2f}" for a, b in corners)
                    under.append(f'<polygon points="{pts}" fill="{hexc(mix(col(t), 0.80))}"/>')
                    x += p
                y += rp; r_ += 1
            bot = y - rp + R
        tcol = "#FFFFFF" if BOXTXT == "white" else hexc(NAVY)
        boxc = BOXC
        NI, SPACE = {0, 5}, {13, 14, 15, 21, 36}            # n·i (neuroImaging), s·p·a (SPAtial), c, e
        dimc = hexc(boxc + (np.array((255, 255, 255)) - boxc) * 0.45) if BOXTXT == "white" else hexc(mix(NAVY, 0.55))
        s_ = fs / upm; x = xn
        rest = os.environ.get("NSP_FONT_REST")     # optional lighter weight for the non-highlighted letters
        gs_rest = TTFont(rest).getGlyphSet() if rest else gsN
        # highlighted letters can be bigger: centred in their monospace cell (the first one keeps its ink left edge),
        # on the common baseline ("base") or with their x-height top on the others' ("top")
        HLK = float(os.environ.get("NSP_HL_SCALE", 1.0)); ALIGN = os.environ.get("NSP_HL_ALIGN", "base")
        # spacing of the bigger letters: "cell" = squeezed into the normal monospace cell (tighter next to them);
        # "equal" = the cell grows by the extra ink, so all ink gaps equal the normal ones;
        # "scaled" = the whole cell grows with the letter (gaps around and within "spa" get wider)
        SPACING = os.environ.get("NSP_HL_SPACING", "cell")
        def inkw(gn):
            b = BoundsPen(gsN); gsN[gn].draw(b); return b.bounds[2] - b.bounds[0]
        def adv(gn, hl):
            a = gsN[gn].width * s_
            if not hl or HLK == 1 or SPACING == "cell": return a
            return a + inkw(gn) * s_ * (HLK - 1) if SPACING == "equal" else a * HLK
        def place(i_, gn, x, hl):
            if not hl or HLK == 1: return x, 0.0, s_
            w = gsN[gn].width * s_
            lsb = fontN["hmtx"][gn][1] * s_
            if SPACING == "cell":
                dx = -lsb * (HLK - 1) if i_ == 0 else w * (1 - HLK) / 2
            elif SPACING == "equal" or i_ == 0:   # the ink starts where a normal letter's would (first letter: left edge)
                dx = -lsb * (HLK - 1)
            else:
                dx = 0.0
            return x + dx, (xh * (HLK - 1) if ALIGN == "top" else 0.0), s_ * HLK
        def ink_x(x):   # ink extent of the whole name starting at x
            xs = []
            for i_, ch in enumerate(name):
                gn = cmap[ord(ch)]; hl = i_ in NI or i_ in SPACE
                if ch != " ":
                    gx, dy, sc = place(i_, gn, x, hl)
                    b1 = BoundsPen(gsN); gsN[gn].draw(TransformPen(b1, (sc, 0, 0, -sc, gx, 0))); xs += [b1.bounds[0], b1.bounds[2]]
                x += adv(gn, hl) + ls
            return min(xs), max(xs)
        if HLK != 1 and SPACING != "cell":   # wider cells: shrink size and spacing together so the name still spans the target
            i0, i1 = ink_x(0.0); f_ = (target) / (i1 - i0)
            fs, ls = fs * f_, ls * f_; s_ = fs / upm
            xn = L + pad - i0 * f_
            print(f"  name refit x{f_:.3f} for {SPACING} spacing")
        xh = fontN["OS/2"].sxHeight * s_
        x = xn
        if style == "halo":   # place the line by its real ink top (bigger letters included)
            bp = BoundsPen(gsN); xx = xn; ys = []
            for i_, ch in enumerate(name):
                gn = cmap[ord(ch)]; hl = i_ in NI or i_ in SPACE
                if ch != " ":
                    gx, dy, sc = place(i_, gn, xx, hl)
                    b1 = BoundsPen(gsN); gsN[gn].draw(TransformPen(b1, (sc, 0, 0, -sc, gx, dy))); ys += [b1.bounds[1], b1.bounds[3]]
                xx += adv(gn, hl) + ls
            ink_top = (p_low + float(NAME_GAP) * F) if NAME_GAP else LK["base2"] + gap
            base3 = ink_top - min(ys); bot = base3 + max(ys)
            print(f"  name ink top {ink_top:.2f}, gap below p {(ink_top - p_low) / F:.3f} F")
        for i_, ch in enumerate(name):
            gn = cmap[ord(ch)]
            hl = i_ in NI or i_ in SPACE
            if HL == "tint":   # non-highlighted letters: light red in "neuroimaging", light blue after it
                fill = tcol if hl else hexc(mix(RED, TINT_RED) if i_ < 12 else mix(NAVY, TINT_BLUE))
            elif HL == "color":  # n·i red, s·p·a·c·e navy (as "ni" / "space" above); the rest as in "tint"
                fill = hexc(RED if i_ in NI else NAVY) if hl else hexc(mix(RED, TINT_RED) if i_ < 12 else mix(NAVY, TINT_BLUE))
            elif HL == "colorgrey":  # as "color", the rest in one neutral grey
                fill = hexc(RED if i_ in NI else NAVY) if hl else hexc(mix(np.array((140.0, 149, 161)), TINT_BLUE))
            elif HL == "mid":  # without a box: mid tones (~3.5:1 on white and on dark grey) — n·i light red, s·p·a·c·e light navy
                c_hl = mix(RED, TINT_RED) if i_ in NI or (not hl and i_ < 12) else mix(NAVY, TINT_BLUE)
                fill = hexc(c_hl) if hl else (hexc((c_hl + REST) / 2) if REST_BLEND else hexc(REST))
            else:
                fill = tcol if (HL in ("none", "hex") or hl) else dimc
            if ch != " ":
                g_ = gsN if hl else gs_rest            # the pen needs the same glyph set (composites, e.g. "i")
                gx, dy, sc = place(i_, gn, x, hl)
                pen = SVGPathPen(g_); g_[gn].draw(TransformPen(pen, (sc, 0, 0, -sc, gx, base3 + dy)))
                if hl and style in ("halo", "outline") and HALO_TXT > 0:   # no box behind: highlighted letters get the light halo of "ni" / "space"
                    halo = hexc(mix(RED if i_ in NI else NAVY, TINT))
                    els.append(f'<path d="{pen.getCommands()}" fill="{halo}" stroke="{halo}" stroke-width="{HALO_TXT * fs:.2f}" stroke-linejoin="round"/>')
                els.append(f'<path d="{pen.getCommands()}" fill="{fill}"/>')
                if hl and HL in ("hex", "dimhex"):
                    bp = BoundsPen(gsN); gsN[gn].draw(TransformPen(bp, (s_, 0, 0, -s_, x, base3)))
                    cxm = (bp.bounds[0] + bp.bounds[2]) / 2; r = 0.075 * fs
                    yy = base3 + 0.22 * fs + r * 0.3
                    pts = " ".join(f"{cxm + r * math.cos(math.radians(a)):.2f},{yy + r * math.sin(math.radians(a)):.2f}" for a in range(30, 390, 60))
                    els.append(f'<polygon points="{pts}" fill="{hexc(RED if i_ in NI else NAVY)}"/>')
            x += adv(gn, hl) + ls
        H = bot + RIM / 2 + 0.02 * F - y0                      # the box rim reaches RIM / 2 below the box

g = open(gsvg).read()
inner = re.sub(r"^<svg[^>]*>\n?|</svg>\s*$", "", g.strip())
inner = re.sub(r'<rect [^>]*/>\n?', "", inner)
# explicit width/height: without an intrinsic size, <img width:auto> in a shrink-to-fit box collapses to 0 (rtd sidebar)
head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.1f}" height="{H:.1f}" '
        f'viewBox="{x0:.1f} {y0:.1f} {W:.1f} {H:.1f}">')
bgrect = f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{W:.1f}" height="{H:.1f}" fill="{bg}"/>' if bg else ""
open(out, "w").write("\n".join([head, bgrect, inner, *under, *els, "</svg>"]) + "\n")
print(out, style if fullname else "short", f"left ink {L:.1f}, right {Rgt:.1f}")
