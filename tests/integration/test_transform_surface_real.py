"""Surface transforms in nispace.transform against Connectome Workbench on the real NiSpace templates.

surf_to_surf re-implements ``wb_command -metric-resample`` / ``-label-resample`` with ADAP_BARY_AREA,
``-current-roi`` (source medial wall) and ``-metric-mask`` (target medial wall), i.e. the neuromaps
defaults. Validated 2026-10-26 on all density pairs below: continuous max |diff| <= 5e-5 (float32
inputs), labels identical wherever a valid source vertex exists. Where none exists, Workbench writes a
new "unused" label key (in the right hemisphere this can be label 1, a left-hemisphere parcel);
NiSpace writes 0.

mni_to_surf from MNI152NLin6Asym must equal sampling the volume at the registration-fusion vertex
positions (what neuromaps' mni152_to_fslr / mni152_to_fsaverage do).
"""

import shutil
import subprocess

import numpy as np
import nibabel as nib
import pytest
from scipy.interpolate import interpn

from nispace.datasets import fetch_template
from nispace.transform import SpaceTransformer, surf_to_surf, mni_to_surf, _surface_weights

WB = shutil.which("wb_command")
needs_wb = pytest.mark.skipif(WB is None, reason="Connectome Workbench (wb_command) not installed")

PAIRS = [("fsLR", "32k", "fsLR", "4k"), ("fsLR", "4k", "fsLR", "32k"), ("fsLR", "32k", "fsaverage", "41k"),
         ("fsaverage", "164k", "fsLR", "32k"), ("fsaverage", "10k", "fsaverage", "41k"),
         ("fsLR", "8k", "fsaverage", "3k")]


def _tpl(space, den, desc, hemi):
    return fetch_template(space, res=den, desc=desc, hemi=hemi, verbose=False)


def _arr(path):
    return np.asarray(nib.load(str(path)).darrays[0].data)


def _wb_resample(tmp_path, data, labels, sf, df, st, dt, hemi):
    cross = sf != st
    sph = lambda sp, den: _tpl(sp, den, "sphere_space-fsaverage" if (cross and sp == "fsLR") else "sphere", hemi)
    args = [sph(sf, df), sph(st, dt), "ADAP_BARY_AREA"]
    roi = ["-area-metrics", _tpl(sf, df, "vaavg", hemi), _tpl(st, dt, "vaavg", hemi),
           "-current-roi", _tpl(sf, df, "medial", hemi)]
    din, dout = tmp_path / "in.func.gii", tmp_path / "out.func.gii"
    nib.save(nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(data)]), din)
    lt = nib.gifti.GiftiLabelTable()
    for k in range(int(labels.max()) + 1):
        lab = nib.gifti.GiftiLabel(key=k, red=0.5, green=0.5, blue=0.5, alpha=1)
        lab.label = f"p{k}"
        lt.labels.append(lab)
    lin, lout = tmp_path / "in.label.gii", tmp_path / "out.label.gii"
    nib.save(nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(labels, intent="NIFTI_INTENT_LABEL")],
                            labeltable=lt), lin)
    run = lambda *a: subprocess.run([WB, *map(str, a)], check=True, capture_output=True)
    run("-metric-resample", din, *args, dout, *roi)
    run("-metric-mask", dout, _tpl(st, dt, "medial", hemi), dout)
    run("-label-resample", lin, *args, lout, *roi)
    return _arr(dout), _arr(lout)


@needs_wb
@pytest.mark.parametrize("sf,df,st,dt", PAIRS)
@pytest.mark.parametrize("hemi", ["L", "R"])
def test_surf_to_surf_matches_workbench(tmp_path, sf, df, st, dt, hemi):
    xyz = _arr(_tpl(sf, df, "sphere", hemi)) / 100
    rng = np.random.default_rng(0)
    data = (np.sin(3 * xyz[:, 0]) + np.cos(2 * xyz[:, 1]) * xyz[:, 2]
            + 0.1 * rng.standard_normal(len(xyz))).astype(np.float32)
    labels = (1 + (xyz[:, 0] > 0) + 2 * (xyz[:, 1] > 0.3) + 4 * (np.sin(5 * xyz[:, 2]) > 0)).astype(np.int32)
    wb_data, wb_labels = _wb_resample(tmp_path, data, labels, sf, df, st, dt, hemi)

    ours = surf_to_surf(data, sf, st, den_to=dt, hemi=hemi, verbose=False).ravel()
    np.testing.assert_allclose(ours, wb_data, atol=1e-4)

    ours_lab = surf_to_surf(labels, sf, st, den_to=dt, hemi=hemi, method="nearest", verbose=False).ravel()
    weights, trg_mask = _surface_weights(sf, df, st, dt, hemi)
    valid = trg_mask & (np.diff(weights.indptr) > 0)
    np.testing.assert_array_equal(ours_lab[valid], wb_labels[valid])
    assert (ours_lab[~valid] == 0).all()


def test_surf_to_surf_gifti_label_parcellation_roundtrip_structure():
    from nispace.datasets import fetch_parcellation
    parc = fetch_parcellation("Schaefer200Parcels7Networks", return_parcellation_only=True)
    parc.set_active_space("fsLR")
    imgs = parc._image_obj
    out = surf_to_surf(imgs, "fsLR", "fsaverage", den_to="41k", method="nearest", verbose=False)
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].darrays[0].data.shape == (40962,)
    assert out[0].labeltable is not None and len(out[0].labeltable.labels) > 0
    ids_in = set(np.unique(np.concatenate([np.asarray(i.darrays[0].data) for i in imgs]))) - {0}
    ids_out = set(np.unique(np.concatenate([o.darrays[0].data for o in out]))) - {0}
    assert ids_out <= ids_in and len(ids_in - ids_out) == 0


@pytest.mark.parametrize("space,den", [("fsLR", "32k"), ("fsLR", "4k"), ("fsaverage", "10k")])
def test_mni_to_surf_nlin6_equals_regfusion_sampling(space, den):
    img = nib.load(str(fetch_template("MNI152NLin6Asym", desc="T1w", res="2mm", verbose=False)))
    data = img.get_fdata()
    grid = [np.arange(s) for s in data.shape]
    st = SpaceTransformer("MNI152NLin6Asym", space, den_to=den, verbose=False).fit()
    out = st.transform(img)
    for h, o in zip("LR", out):
        coords = _arr(fetch_template(space, res=den, desc="regfusion_space-MNI152NLin6Asym", hemi=h,
                                     verbose=False)).astype(float)
        vox = nib.affines.apply_affine(np.linalg.inv(img.affine), coords)
        np.testing.assert_allclose(o.darrays[0].data, interpn(grid, data, vox), rtol=1e-5, atol=1e-3)
    # the sampling matrix is cached per grid: a second map on the same grid reuses it
    assert len(st._sampling) == 2
    st.transform(img)
    assert len(st._sampling) == 2


def test_mni_to_surf_2009c_composed_close_to_nlin6():
    """The same template in 2009c and NLin6 should give similar surface values once the EasyReg field is
    composed in, and clearly more similar than treating 2009c as if it were NLin6."""
    t6 = fetch_template("MNI152NLin6Asym", desc="brainmask", res="1mm", verbose=False)
    t9 = fetch_template("MNI152NLin2009cAsym", desc="brainmask", res="1mm", verbose=False)
    ref = np.concatenate([g.darrays[0].data for g in mni_to_surf(t6, "MNI152NLin6Asym", "fsLR", verbose=False)])
    comp = np.concatenate([g.darrays[0].data for g in mni_to_surf(t9, "MNI152NLin2009cAsym", "fsLR", verbose=False)])
    naive = np.concatenate([g.darrays[0].data for g in mni_to_surf(t9, "MNI152NLin6Asym", "fsLR", verbose=False)])
    err_comp, err_naive = np.abs(comp - ref).mean(), np.abs(naive - ref).mean()
    assert err_comp < err_naive
