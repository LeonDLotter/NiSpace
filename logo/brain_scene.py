"""Brain scene for the logo: right lateral view of the FreeSurfer pial surface of the MNI152NLin6Asym template.
Writes WORK/brain_scene.npz with
  hit      silhouette mask (image px, 0.5 mm)
  saimg    cortical thickness on the visible surface (gap-filled, 5 mm smoothed)
  s        image px -> viewBox units (brain spans x = 2..62)
  brain    outline polygon (viewBox units)
  syl_ext  Sylvian line (DK temporal border vs. operculum/insula), extended through the front edge
Usage: brain_scene.py WORK_DIR"""
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import measure

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import fetch

WORK = Path(sys.argv[1]); CACHE = WORK / "data"
PX, SMOOTH = 0.5, 5.0                                     # pixel size and thickness smoothing, mm

coords, faces = nib.freesurfer.read_geometry(fetch("rh.pial", CACHE))
lab, _, names = nib.freesurfer.read_annot(fetch("rh.aparc.annot", CACHE))
names = [n.decode() for n in names]
vname = np.array([names[l] if l >= 0 else "unknown" for l in lab])
thick = nib.freesurfer.read_morph_data(fetch("rh.thickness", CACHE)).astype(float)

# ---- orthographic z-buffer render from +x: u = y (posterior -> anterior), v = -z (top -> bottom)
u = (coords[:, 1] - coords[:, 1].min()) / PX
v = (coords[:, 2].max() - coords[:, 2]) / PX
dep = coords[:, 0]
W, H = int(np.ceil(u.max())) + 2, int(np.ceil(v.max())) + 2
zbuf = np.full((H, W), -np.inf); vid = np.full((H, W), -1)
for f in faces:
    xs, ys, zs = u[f], v[f], dep[f]
    x0, x1 = int(np.floor(xs.min())), int(np.ceil(xs.max()))
    y0, y1 = int(np.floor(ys.min())), int(np.ceil(ys.max()))
    gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
    den = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
    if abs(den) < 1e-12: continue
    w0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / den
    w1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / den
    w2 = 1 - w0 - w1
    m = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
    if not m.any(): continue
    z = w0 * zs[0] + w1 * zs[1] + w2 * zs[2]
    near = f[np.argmax(np.stack([w0, w1, w2]), 0)]
    yy, xx = np.nonzero(m); yy_, xx_ = yy + y0, xx + x0
    upd = z[m] > zbuf[yy_, xx_]
    zbuf[yy_[upd], xx_[upd]] = z[m][upd]; vid[yy_[upd], xx_[upd]] = near[m][upd]
hit = ndimage.binary_fill_holes(vid >= 0)

# ---- visible-surface classes and thickness
TEMP = {"superiortemporal", "middletemporal", "inferiortemporal", "temporalpole", "transversetemporal", "bankssts"}
NBR = {"parsopercularis", "parstriangularis", "parsorbitalis", "precentral", "postcentral", "supramarginal",
       "insula", "lateralorbitofrontal"}
ok = vid >= 0
timg = np.zeros((H, W), bool); nimg = np.zeros((H, W), bool); img = np.full((H, W), np.nan)
timg[ok] = np.isin(vname, list(TEMP))[vid[ok]]
nimg[ok] = np.isin(vname, list(NBR))[vid[ok]]
t_ = thick[vid[ok]]; t_[t_ == 0] = np.nan; img[ok] = t_
_, (iy, ix) = ndimage.distance_transform_edt(~np.isfinite(img), return_indices=True)
img = ndimage.gaussian_filter(img[iy, ix], SMOOTH / PX)

def region(mask, s=1.0):
    mask = (ndimage.gaussian_filter(mask.astype(float), s / PX) > 0.5) & hit
    labs, n = ndimage.label(ndimage.binary_fill_holes(mask))
    return labs == (np.argmax(np.bincount(labs.ravel())[1:]) + 1) if n else mask
temporal = region(timg); nbr = region(nimg) & ~temporal

# ---- outline and Sylvian line in viewBox units
PAD = 6
s = 60 / W
inside = ndimage.distance_transform_edt(np.pad(hit, PAD))
xy = lambda cc: [(round(2 + (x - PAD) * s, 2), round(2 + (y - PAD) * s, 2)) for y, x in cc]

def closed(mask, sm=0.8, tol=0.5):
    a = np.pad(ndimage.gaussian_filter(mask.astype(float), sm / PX), PAD)
    cc = max(measure.find_contours(a, 0.5), key=len)
    return xy(measure.approximate_polygon(cc, tolerance=tol / PX))

def border(mask, other, sm=0.6, tol=0.5, edge=1.5, near=2.0):
    """Longest run of the mask's contour that lies inside the brain and next to `other`."""
    a = np.pad(ndimage.gaussian_filter(mask.astype(float), sm / PX), PAD)
    cc = max(measure.find_contours(a, 0.5), key=len)
    od = ndimage.distance_transform_edt(~np.pad(other, PAD))
    keep = np.array([inside[int(y), int(x)] > edge / PX and od[int(y), int(x)] < near / PX for y, x in cc])
    if not keep.all():
        k0 = np.argmin(keep); cc, keep = np.roll(cc, -k0, 0), np.roll(keep, -k0)
    segs, cur = [], []
    for p, k in zip(cc, keep):
        if k: cur.append(p)
        elif cur: segs.append(np.array(cur)); cur = []
    if cur: segs.append(np.array(cur))
    segs = [sg for sg in segs if len(sg) * PX >= 4]
    return xy(measure.approximate_polygon(max(segs, key=len), tolerance=tol / PX))

brain = closed(hit)
syl = np.array(border(temporal, nbr), float)
if syl[0, 0] > syl[-1, 0]: syl = syl[::-1]                # posterior -> anterior
dvec = syl[-1] - syl[-3]; dvec /= np.hypot(*dvec)
syl_ext = np.vstack([syl, syl[-1] + dvec * 4, syl[-1] + dvec * 10])   # open the gap through the front edge

np.savez(WORK / "brain_scene.npz", hit=hit, saimg=img, s=s, syl_ext=syl_ext, brain=np.array(brain), syl=syl)
print(f"brain scene: {len(coords)} vertices, image {H}x{W}, Sylvian line {len(syl)} points")
