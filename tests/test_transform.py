"""Unit tests for nispace.transform / nispace.utils.utils_transforms on synthetic meshes and volumes
(no data downloads). Real-template checks against Workbench live in
tests/integration/test_transform_surface_real.py."""

import logging

import numpy as np
import nibabel as nib
import pytest
from scipy.interpolate import interpn
from scipy.spatial import ConvexHull

from nispace.transform import SpaceTransformer, surf_to_mni, _split_surface, _merge_surface
from nispace.utils.utils_transforms import (_closest_point_barycentric, barycentric_weights,
                                            adap_bary_area_weights, resample_popular, sample_field,
                                            volume_sampling_matrix)


def _sphere_mesh(n, seed=0, jitter=0.0):
    """Triangulated sphere (radius 100): Fibonacci points, convex hull faces, optional jitter."""
    i = np.arange(n) + 0.5
    phi, theta = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    xyz = np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    if jitter:
        xyz += np.random.default_rng(seed).normal(scale=jitter, size=xyz.shape)
    xyz = 100 * xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    return xyz, ConvexHull(xyz).simplices


# KERNELS ==========================================================================================

def test_closest_point_barycentric_regions():
    a, b, c = np.array([0., 0, 0]), np.array([1., 0, 0]), np.array([0., 1, 0])
    pts = np.array([[0.2, 0.2, 1.0],     # above interior -> planar barycentric
                    [-1., -1, 0],        # vertex a region
                    [0.5, -1, 0],        # edge ab region
                    [2., 2, 0]])         # edge bc region
    w, d2 = _closest_point_barycentric(pts, a, b, c)
    np.testing.assert_allclose(w[0], [0.6, 0.2, 0.2])
    np.testing.assert_allclose(d2[0], 1.0)
    np.testing.assert_allclose(w[1], [1, 0, 0])
    np.testing.assert_allclose(w[2], [0.5, 0.5, 0])
    np.testing.assert_allclose(w[3], [0, 0.5, 0.5])
    np.testing.assert_allclose(w.sum(axis=1), 1)


def test_barycentric_weights_identity_and_row_sums():
    xyz, faces = _sphere_mesh(500)
    W = barycentric_weights(xyz, faces, xyz)
    np.testing.assert_allclose(W.toarray(), np.eye(len(xyz)), atol=1e-9)
    other, _ = _sphere_mesh(300, jitter=0.01)
    W2 = barycentric_weights(xyz, faces, other)
    np.testing.assert_allclose(np.asarray(W2.sum(axis=1)).ravel(), 1, atol=1e-12)
    assert (np.diff(W2.indptr) <= 3).all()


def test_adap_bary_area_identity_constant_and_roi():
    cur, cur_f = _sphere_mesh(800)
    new, new_f = _sphere_mesh(200, jitter=0.02)
    area_c, area_n = np.full(len(cur), 1.0), np.full(len(new), 4.0)
    # same mesh -> identity
    W_id = adap_bary_area_weights((cur, cur_f), (cur, cur_f), area_c, area_c)
    np.testing.assert_allclose(W_id.toarray(), np.eye(len(cur)), atol=1e-9)
    # downsampling: every source vertex contributes (adaptive scatter), constants are preserved
    W = adap_bary_area_weights((cur, cur_f), (new, new_f), area_c, area_n)
    assert np.asarray((W > 0).sum(axis=0)).ravel().min() > 0
    np.testing.assert_allclose(W @ np.ones(len(cur)), 1, atol=1e-12)
    # ROI: excluded source vertices get no weight, remaining rows renormalised (or empty)
    roi = (cur[:, 2] > 0).astype(float)
    W_roi = adap_bary_area_weights((cur, cur_f), (new, new_f), area_c, area_n, cur_roi=roi)
    assert W_roi[:, roi == 0].nnz == 0
    rs = np.asarray(W_roi.sum(axis=1)).ravel()
    np.testing.assert_allclose(rs[rs > 0], 1, atol=1e-12)
    assert (rs[new[:, 2] < -40] == 0).all()   # > 1 coarse mesh spacing (~25) away from the ROI


@pytest.mark.parametrize("src,trg", [("a", "b"), ("b", "a")])
def test_adap_bary_area_matches_stored_workbench_output(src, trg):
    """CI check against Workbench without Workbench: outputs stored by tests/data/make_wb_resample_fixture.py."""
    from pathlib import Path
    fx = np.load(Path(__file__).parent / "data" / "wb_resample_fixture.npz")
    W = adap_bary_area_weights((fx[f"{src}_xyz"], fx[f"{src}_faces"]), (fx[f"{trg}_xyz"], fx[f"{trg}_faces"]),
                               fx[f"{src}_area"], fx[f"{trg}_area"], cur_roi=fx[f"{src}_roi"])
    np.testing.assert_allclose(W @ fx[f"{src}_data"].astype(np.float64), fx[f"wb_{src}{trg}_data"], atol=1e-5)
    has = np.diff(W.indptr) > 0       # Workbench writes a new "unused" label key where no source is valid
    np.testing.assert_array_equal(resample_popular(W, fx[f"{src}_labels"])[has], fx[f"wb_{src}{trg}_labels"][has])
    assert (~has).sum() > 0           # the ROI cap makes sure that case is exercised


def test_resample_popular_majority_and_fill():
    import scipy.sparse as sp
    W = sp.csr_matrix(np.array([[0.2, 0.3, 0.5, 0.0],
                                [0.4, 0.4, 0.0, 0.2],
                                [0.0, 0.0, 0.0, 0.0]]))
    labels = np.array([7, 7, 3, 9])
    # row 0 is a tie (7: 0.2 + 0.3, 3: 0.5) -> Workbench keeps the label reaching the maximum first (7)
    np.testing.assert_array_equal(resample_popular(W, labels, fill=0), [7, 7, 0])
    np.testing.assert_array_equal(resample_popular(W, np.array([3, 3, 7, 9])), [3, 3, 0])
    np.testing.assert_array_equal(resample_popular(W, np.vstack([labels, labels + 1])),
                                  [[7, 7, 0], [8, 8, 0]])


def test_volume_sampling_matrix_matches_interpn():
    rng = np.random.default_rng(1)
    vol = rng.normal(size=(9, 11, 7))
    affine = np.array([[2., 0, 0, -8], [0, 2, 0, -10], [0, 0, 2, -6], [0, 0, 0, 1]])
    vox = rng.uniform([0, 0, 0], [8, 10, 6], size=(200, 3))
    world = vox @ affine[:3, :3].T + affine[:3, 3]
    grid = [np.arange(s) for s in vol.shape]
    M, inside = volume_sampling_matrix(world, affine, vol.shape, method="linear")
    assert inside.all()
    np.testing.assert_allclose(M @ vol.ravel(), interpn(grid, vol, vox), atol=1e-10)
    Mn, _ = volume_sampling_matrix(world, affine, vol.shape, method="nearest")
    np.testing.assert_allclose(Mn @ vol.ravel(), interpn(grid, vol, vox, method="nearest"))
    # outside the grid -> empty row
    M_out, inside_out = volume_sampling_matrix(np.array([[1000., 0, 0]]), affine, vol.shape)
    assert not inside_out[0] and M_out.nnz == 0


def test_sample_field_identity():
    shape = (6, 7, 8)
    affine = np.diag([2., 2, 2, 1]); affine[:3, 3] = [-5, -6, -7]
    ijk = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), axis=-1)
    field = nib.Nifti1Image((ijk @ affine[:3, :3].T + affine[:3, 3]).astype(np.float32), affine)
    pts = np.array([[-3.3, -2.0, 0.5], [1.0, 2.2, 3.9]])
    np.testing.assert_allclose(sample_field(field, pts), pts, atol=1e-5)


# API ==============================================================================================

def test_split_merge_surface_roundtrip():
    arr = np.arange(2 * 4002, dtype=float)
    parts, fmt, lts = _split_surface(arr, ("L", "R"))
    assert parts["L"].shape == (1, 4002)
    np.testing.assert_array_equal(_merge_surface(parts, ("L", "R"), fmt), arr)
    multi = np.vstack([arr, arr + 1])
    parts, fmt, _ = _split_surface(multi, ("L", "R"))
    np.testing.assert_array_equal(_merge_surface(parts, ("L", "R"), fmt), multi)
    imgs = _merge_surface(parts, ("L", "R"), ("gifti", 2))
    assert isinstance(imgs, tuple) and len(imgs) == 2 and len(imgs[0].darrays) == 2


def test_surface_to_mni_not_implemented():
    with pytest.raises(NotImplementedError, match="Wu et al"):
        SpaceTransformer("fsLR", "MNI152NLin6Asym")
    with pytest.raises(NotImplementedError):
        surf_to_mni(np.zeros(2 * 32492), "fsLR", "MNI152NLin6Asym")


def test_space_transformer_validation():
    with pytest.raises(ValueError, match="Density"):
        SpaceTransformer("fsLR", "fsaverage", den_to="32k")
    with pytest.raises(ValueError, match="method"):
        SpaceTransformer("fsLR", "fsaverage", method="cubic")
    with pytest.raises(ValueError, match="den_from"):
        SpaceTransformer("fsLR", "fsaverage").fit()
    with pytest.raises(RuntimeError, match="not fitted"):
        SpaceTransformer("fsLR", "fsLR").transform(np.zeros(2 * 32492))
    st = SpaceTransformer("FSLR", "fsaverage", den_to="10k")
    assert (st.space_from, st.space_to, st.den_to, st.method) == ("fsLR", "fsaverage", "10k", "linear")


def test_surface_identity_returns_input(tmp_path):
    x = np.random.default_rng(0).normal(size=(3, 2 * 32492))
    np.testing.assert_array_equal(SpaceTransformer("fsLR", "fsLR", den_to="32k").fit(x).transform(x), x)
    # paths in -> GIFTI images out; a spurious label table on continuous data must not cast to int
    paths = []
    for h, row in zip("LR", x[0].reshape(2, -1)):
        img = nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(row.astype(np.float32))])
        label = nib.gifti.GiftiLabel(key=0)
        label.label = "???"
        img.labeltable.labels.append(label)
        paths.append(tmp_path / f"hemi-{h}.func.gii")
        img.to_filename(paths[-1])
    out = SpaceTransformer("fsLR", "fsLR", den_to="32k").fit_transform(tuple(paths))
    assert isinstance(out[0], nib.GiftiImage)
    np.testing.assert_allclose(np.concatenate([o.agg_data() for o in out]), x[0], rtol=1e-6)


@pytest.mark.parametrize("data_space,expected", [("MNI152NLin2009cAsym", "MNI152NLin2009cAsym"),
                                                 ("mni152nlin2009casym", "MNI152NLin2009cAsym"),
                                                 ("MNIColin27", "MNIColin27"),
                                                 ("MNI152", "MNI152NLin6Asym")])
def test_parcellater_projects_volume_from_its_mni_space(monkeypatch, data_space, expected):
    """Parcellater must pass the data's actual MNI space to the volume-to-surface projection (the generic
    "MNI152" falls back to MNI152NLin6Asym, as in neuromaps)."""
    import nispace.transform as T
    from nispace.parcellate import Parcellater
    seen = {}

    def fake_mni_to_surf(img, mni_from, surf_to, den_to=None, hemi=None, method="linear", **kw):
        seen.update(mni_from=mni_from, surf_to=surf_to, den_to=den_to, method=method)
        return tuple(nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(np.ones(32492, np.float32))])
                     for _ in range(2))

    monkeypatch.setattr(T, "mni_to_surf", fake_mni_to_surf)
    labels = np.r_[np.ones(16000), 2 * np.ones(16492)].astype(np.int32)
    parc = tuple(nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(labels + 2 * h, intent="NIFTI_INTENT_LABEL")])
                 for h in range(2))
    vol = nib.Nifti1Image(np.ones((10, 10, 10), np.float32), np.eye(4))
    out = Parcellater(parc, "fsLR", resampling_target="parcellation").fit().transform(
        vol, data_space, background_value=False)
    assert seen == dict(mni_from=expected, surf_to="fsLR", den_to="32k", method="linear")
    np.testing.assert_allclose(out, 1)


@pytest.mark.parametrize("parc_space,data_space,target,expected", [
    ("MNI152NLin6Asym", "MNI152NLin2009cAsym", "data", ("parc", "MNI152NLin6Asym", "MNI152NLin2009cAsym", "nearest")),
    ("MNI152NLin6Asym", "MNI152NLin2009cAsym", "parcellation", ("data", "MNI152NLin2009cAsym", "MNI152NLin6Asym", "linear")),
    ("MNI152NLin6Asym", "MNI152", "data", None),          # unspecified data space: never warp
    ("MNI152", "MNI152NLin2009cAsym", "data", None),      # unspecified parcellation space: never warp
    ("MNI152NLin6Asym", "mni152nlin6asym", "data", None), # same space (case-insensitive)
])
def test_parcellater_mni_warp_rule(monkeypatch, parc_space, data_space, target, expected):
    from nispace.parcellate import Parcellater
    calls = []

    def fake_warp(self, img, mni_from, mni_to, reference, method, cache_result=False):
        calls.append(("parc" if cache_result else "data", mni_from, mni_to, method))
        return img

    monkeypatch.setattr(Parcellater, "_mni_warp", fake_warp)
    parc = nib.Nifti1Image(np.repeat([1, 2], 500).reshape(10, 10, 10).astype(np.int16), np.eye(4))
    data = nib.Nifti1Image(np.ones((10, 10, 10), np.float32), np.eye(4))
    p = Parcellater(parc, parc_space, resampling_target=target).fit()
    for _ in range(2):
        out = p.transform(data, data_space, background_value=False)
    np.testing.assert_allclose(out, 1)
    assert calls == ([] if expected is None else [expected, expected])


def test_space_aliases_all_contain_mni():
    from nispace.transform import _SPACE_ALIASES, _resolve_space, _is_mni_space_name, _KNOWN_SPACES
    assert all("mni" in a.lower() for a in _SPACE_ALIASES)
    assert _resolve_space("mni-fsl") == "MNI152NLin6Asym"
    assert _resolve_space("MNI-SPM-TPM") == "CAT12IXI555"
    assert _resolve_space("MNI-CAT12") == "MNI152NLin2009cAsym"
    for old in ["FSL", "SPM-TPM", "CAT12"]:
        assert _resolve_space(old, strict=False) is None
    # templateflow / neuromaps names of the ICBM 2009 a/b templates map to the 2009c fields
    assert _resolve_space("MNI152NLin2009aAsym") == "MNI152NLin2009cAsym"
    assert _resolve_space("mni152nlin2009bsym") == "MNI152NLin2009cSym"
    assert _resolve_space("MNI152NLin2009Asym") == "MNI152NLin2009cAsym"   # differs from ...aSym only in case
    assert _resolve_space("MNI152NLin2009aSym") == "MNI152NLin2009cSym"
    for amb in ["MNI-SPM", "SPM12", "mni152nlin2009asym"]:
        with pytest.raises(ValueError, match="ambiguous"):
            _resolve_space(amb)
    assert all(_is_mni_space_name(s) for s in list(_SPACE_ALIASES) + _KNOWN_SPACES)
    assert not any(_is_mni_space_name(s) for s in ["fsLR", "fsaverage"])
    # partial names must contain "MNI" too, so every accepted name is detected as volumetric
    assert _resolve_space("MNIColin") == "MNIColin27"
    assert _resolve_space("CVSavg35inMNI") == "CVSavg35inMNI152"
    assert _resolve_space("cat12ixi555") == "CAT12IXI555"
    for partial in ["Colin", "CVS", "ixi", "305", "2009cAsym"]:
        assert _resolve_space(partial, strict=False) is None
        assert not _is_mni_space_name(partial)
    with pytest.raises(ValueError, match="must contain 'MNI'"):
        _resolve_space("Colin")


def test_nispace_unspecified_spaces():
    from nispace.api import NiSpace, _unspecified_parc_space
    nsp = NiSpace(x=np.ones((1, 5)), y=np.ones((1, 5)), verbose=False)
    assert nsp._data_space == ["MNI152"] * 3
    vol = nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4))
    assert _unspecified_parc_space(vol, "MNI152NLin2009cAsym") == "MNI152NLin2009cAsym"
    assert _unspecified_parc_space(vol, "MNI152") == "MNI152NLin6Asym"
    assert _unspecified_parc_space("parc.nii.gz", "fsLR") == "MNI152NLin6Asym"
    assert _unspecified_parc_space(("L.label.gii", "R.label.gii"), "MNI152NLin2009cAsym") is None


def test_parcellater_warns_once_for_unresolved_mni_name(monkeypatch, caplog):
    """A typo in a specific MNI name silently disables warps, so it is warned about (once); the generic
    "MNI152" is not."""
    import nispace.parcellate as P
    monkeypatch.setattr(P, "_INFO_SEEN", set())
    parc = nib.Nifti1Image(np.repeat([1, 2], 500).reshape(10, 10, 10).astype(np.int16), np.eye(4))
    data = nib.Nifti1Image(np.ones((10, 10, 10), np.float32), np.eye(4))
    p = P.Parcellater(parc, "MNI152NLin6Asym").fit()
    with caplog.at_level(logging.WARNING, logger="nispace"):
        for space in ["MNI152", "MNI152NLin2009cAsymm", "MNI152NLin2009cAsymm"]:
            p.transform(data, space, background_value=False)
    assert sum("does not identify" in r.message for r in caplog.records) == 1


def test_civet_not_supported():
    from nispace.parcellate import Parcellater
    with pytest.raises(ValueError, match="CIVET"):
        SpaceTransformer("civet", "fsLR")
    with pytest.raises(ValueError, match="CIVET"):
        Parcellater(None, "civet")


def test_surface_hemi_pair_list_is_one_map():
    """[L, R] (as neuromaps accepts it) is one map; a list of (L, R) tuples is several maps."""
    x = np.random.default_rng(0).normal(size=2 * 4002).astype(np.float32)
    L, R = (nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(h)]) for h in x.reshape(2, -1))
    st = SpaceTransformer("fsLR", "fsLR", den_to="4k")
    out = st.fit_transform([L, R])
    assert isinstance(out, tuple) and len(out) == 2
    np.testing.assert_array_equal(np.concatenate([o.agg_data() for o in out]), x)
    many = st.transform([(L, R), (L, R)])
    assert isinstance(many, list) and len(many) == 2 and len(many[0]) == 2


def test_transform_dispatcher_accepts_order(monkeypatch):
    import nispace.transform as T
    seen = {}

    class FakeTransformer:
        def __init__(self, space_from, space_to, **kwargs):
            seen.update(kwargs)

        def fit_transform(self, data):
            return data

    monkeypatch.setattr(T, "SpaceTransformer", FakeTransformer)
    T.transform("img", "MNI152NLin6Asym", "MNI152NLin2009cAsym", order=0)
    assert seen["method"] == 0
    with pytest.raises(ValueError, match="either"):
        T.transform("img", "MNI152NLin6Asym", "MNI152NLin2009cAsym", order=0, method="nearest")


def test_compute_transform_displacement_constant_shift(monkeypatch, tmp_path):
    """A coordinate-map field storing (own RAS position + 2 mm in x) has a displacement of 2 mm everywhere."""
    import nispace.transform as T
    shape, affine = (4, 5, 6), np.diag([2., 2, 2, 1])
    ijk = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), axis=-1)
    field = ijk @ affine[:3, :3].T + affine[:3, 3] + [2, 0, 0]
    nib.Nifti1Image(field.astype(np.float32), affine).to_filename(tmp_path / "field.nii.gz")
    monkeypatch.setattr(T, "_resolve_field", lambda *a, **k: tmp_path / "field.nii.gz")
    out = T.compute_transform_displacement("MNI152NLin2009cAsym", "MNI152NLin6Asym", verbose=False)
    assert out.shape == shape
    np.testing.assert_allclose(out.get_fdata(), 2, rtol=1e-6)


def test_mni_to_surf_synthetic(monkeypatch, tmp_path):
    """Volume-to-surface without downloads (synthetic vertex positions): values = trilinear / nearest
    sampling, NaN / 0 outside the grid, 4D and single-hemisphere output, label GIFTIs for integer images,
    sampling matrix reused across instances for the same grid."""
    import nispace.transform as T
    rng = np.random.default_rng(2)
    affine = np.array([[2., 0, 0, -8], [0, 2, 0, -10], [0, 0, 2, -6], [0, 0, 0, 1]])
    shape = (9, 11, 7)
    grid = [np.arange(s) for s in shape]
    vox = {h: rng.uniform(0, [8, 10, 6], size=(50, 3)) for h in "LR"}
    coords = {h: v @ affine[:3, :3].T + affine[:3, 3] for h, v in vox.items()}
    coords["R"][-1] = [500, 0, 0]                                            # outside the grid
    monkeypatch.setattr(T, "_OPERATOR_CACHE", {})
    monkeypatch.setattr(T, "_surface_coords_in_mni", lambda mni_from, space_to, den_to, hemi, hash_check: coords[hemi])
    n_built, build = [], T.volume_sampling_matrix
    monkeypatch.setattr(T, "volume_sampling_matrix", lambda *a, **k: n_built.append(1) or build(*a, **k))

    vol = rng.normal(size=shape)
    L, R = T.mni_to_surf(nib.Nifti1Image(vol.astype(np.float32), affine), "MNI152NLin6Asym", "fsLR",
                         verbose=False)
    np.testing.assert_allclose(L.darrays[0].data, interpn(grid, vol, vox["L"]), rtol=1e-5, atol=1e-5)
    assert np.isnan(R.darrays[0].data[-1]) and np.isfinite(R.darrays[0].data[:-1]).all()

    vol4 = nib.Nifti1Image(np.stack([vol, 2 * vol], axis=-1).astype(np.float32), affine)
    L4 = T.mni_to_surf(vol4, "MNI152NLin6Asym", "fsLR", hemi="L", verbose=False)
    assert isinstance(L4, nib.GiftiImage) and len(L4.darrays) == 2
    np.testing.assert_allclose(L4.darrays[1].data, 2 * L4.darrays[0].data, rtol=1e-6)
    assert len(n_built) == 2                                                 # new instance, same grid: cached

    labels = rng.integers(1, 4, size=shape).astype(np.int16)
    Ll, Rl = T.mni_to_surf(nib.Nifti1Image(labels, affine), "MNI152NLin6Asym", "fsLR", method="nearest",
                           verbose=False)
    np.testing.assert_array_equal(Ll.darrays[0].data, interpn(grid, labels, vox["L"], method="nearest"))
    assert Rl.darrays[0].data.dtype == np.int32 and Rl.darrays[0].data[-1] == 0
    assert {lab.key for lab in Rl.labeltable.labels} == set(np.unique(Rl.darrays[0].data))
    Rl.to_filename(tmp_path / "R.label.gii")                                 # label table serialises
    assert len(nib.load(tmp_path / "R.label.gii").labeltable.labels) == len(Rl.labeltable.labels)


@pytest.mark.parametrize("drop", [False, True])
def test_distance_matrix_surface_downsampling(monkeypatch, caplog, drop):
    """get_distance_matrix resamples surface parcellations with surf_to_surf (nearest) and keeps the original
    density if that would drop parcels."""
    import nispace.transform as T
    import nispace.nulls as N
    gii = lambda lab: nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(lab.astype(np.int32))])
    parc = tuple(gii(np.arange(32492) % 4 + 1 + 4 * h) for h in range(2))
    seen = {}

    def fake_surf_to_surf(p, surf_from, surf_to, den_to=None, hemi=None, method=None, verbose=True):
        seen.update(spaces=(surf_from, surf_to), den_to=den_to, hemi=hemi, method=method)
        return tuple(gii(np.arange(4002) % (3 if drop else 4) + 1 + 4 * h) for h in range(2))

    def fake_centroids(parc, parc_space, parc_hemi, parc_density):
        seen["density"] = parc_density
        return np.random.default_rng(0).normal(size=(8, 3))

    monkeypatch.setattr(T, "surf_to_surf", fake_surf_to_surf)
    monkeypatch.setattr(N, "find_surf_parc_centroids", fake_centroids)
    with caplog.at_level(logging.WARNING, logger="nispace"):
        dist = N.get_distance_matrix(parc, "fsLR", parc_resample="4k", surf_euclidean=True, verbose=False)
    assert seen["spaces"] == ("fsLR", "fsLR") and seen["den_to"] == "4k"
    assert seen["method"] == "nearest" and seen["hemi"] is None
    assert seen["density"] == ("32k" if drop else "4k")
    assert dist.shape == (8, 8)
    assert any("would drop" in r.message for r in caplog.records) == drop


def test_transforms_module_is_deprecated_alias(caplog):
    import importlib, sys
    sys.modules.pop("nispace.transforms", None)
    with caplog.at_level(logging.WARNING, logger="nispace"):
        old = importlib.import_module("nispace.transforms")
    from nispace.transform import mni_to_mni
    assert old.mni_to_mni is mni_to_mni
    assert any("nispace.transforms" in r.message and "deprecated" in r.message for r in caplog.records)
