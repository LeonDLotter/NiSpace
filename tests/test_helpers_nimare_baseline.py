"""Unit tests for the reporting-baseline null sampling in helpers/nimare.py.

No NiMARE or network access needed: the coordinate draw (_sample_null_idx) and
the baseline resolution (_resolve_reporting_baseline) are tested in isolation
on small synthetic masks; fetching from the 'mabaseline' reference dataset is
monkeypatched. End-to-end runs live in tests/integration/test_helpers_nimare.py.
"""

import nibabel as nib
import numpy as np
import pytest

import nispace.datasets as nsp_datasets
from nispace.helpers.nimare import _sample_null_idx, _resolve_reporting_baseline


def _mask_img(shape=(6, 7, 5), affine=None):
    mask = np.zeros(shape, dtype=np.float32)
    mask[1:-1, 1:-1, 1:-1] = 1
    return nib.Nifti1Image(mask, np.eye(4) if affine is None else affine)


def _mask_ijk(mask_img):
    return np.vstack(np.where(mask_img.get_fdata())).T


# ---------------------------------------------------------------------------
# _sample_null_idx
# ---------------------------------------------------------------------------

def test_sample_uniform_is_unchanged_rng_call():
    # weights=None must reproduce the pre-baseline draw exactly (seeded results unchanged)
    a = _sample_null_idx(1000, 50, 20, np.random.default_rng(7))
    b = np.random.default_rng(7).integers(1000, size=(50, 20))
    np.testing.assert_array_equal(a, b)


def test_sample_concentrated_weights_stay_in_support():
    w = np.zeros(1000)
    support = [3, 500, 999]
    w[support] = [0.2, 0.5, 0.3]
    idx = _sample_null_idx(1000, 200, 50, np.random.default_rng(0), weights=w)
    assert idx.shape == (200, 50)
    assert set(np.unique(idx).tolist()) <= set(support)


def test_sample_flat_weights_are_uniform():
    n = 20
    idx = _sample_null_idx(n, 1000, 200, np.random.default_rng(0), weights=np.full(n, 1 / n))
    freq = np.bincount(idx.ravel(), minlength=n) / idx.size
    assert np.abs(freq - 1 / n).max() < 0.005  # 2e5 draws: SE per cell ~5e-4


def test_sample_weight_length_mismatch_raises():
    with pytest.raises(ValueError, match="weights"):
        _sample_null_idx(10, 5, 5, np.random.default_rng(0), weights=np.full(9, 1 / 9))


# ---------------------------------------------------------------------------
# _resolve_reporting_baseline
# ---------------------------------------------------------------------------

def test_resolve_none_returns_none():
    assert _resolve_reporting_baseline(None, _mask_img()) is None


def test_resolve_array_aligned_with_where_order():
    mask_img = _mask_img()
    ijk = _mask_ijk(mask_img)
    target = 17
    arr = np.zeros(mask_img.shape)
    arr[tuple(ijk[target])] = 5.0
    w = _resolve_reporting_baseline(arr, mask_img)
    assert w.shape == (len(ijk),)
    assert w[target] == pytest.approx(1.0)
    assert np.count_nonzero(w) == 1


def test_resolve_normalizes_within_mask_only():
    mask_img = _mask_img()
    arr = np.ones(mask_img.shape)  # values outside the mask must not count
    w = _resolve_reporting_baseline(arr, mask_img)
    np.testing.assert_allclose(w, 1 / len(_mask_ijk(mask_img)))


def test_resolve_nifti_on_other_grid_is_resampled_nonnegative():
    # 2 mm target grid, shifted by 0.5 mm so values are truly interpolated
    affine = np.array([[2, 0, 0, .5], [0, 2, 0, .5], [0, 0, 2, .5], [0, 0, 0, 1.]])
    mask_img = _mask_img(shape=(10, 10, 10), affine=affine)
    base = np.ones((20, 20, 20))
    base[:10] = 100.0  # sharp step: a cubic spline would overshoot here
    w = _resolve_reporting_baseline(nib.Nifti1Image(base, np.eye(4)), mask_img)
    assert np.isfinite(w).all() and (w >= 0).all()
    assert w.sum() == pytest.approx(1.0)
    x_mm = _mask_ijk(mask_img)[:, 0] * 2 + .5
    assert w[x_mm < 9].mean() > 10 * w[x_mm > 11].mean()


def test_resolve_4d_single_volume_image():
    mask_img = _mask_img()
    arr = np.ones(mask_img.shape)
    img4d = nib.Nifti1Image(arr[..., None], mask_img.affine)
    np.testing.assert_allclose(
        _resolve_reporting_baseline(img4d, mask_img),
        _resolve_reporting_baseline(arr, mask_img),
    )


def test_resolve_negative_raises():
    mask_img = _mask_img()
    arr = np.ones(mask_img.shape)
    arr[2, 2, 2] = -1
    with pytest.raises(ValueError, match="negative"):
        _resolve_reporting_baseline(arr, mask_img)


def test_resolve_all_zero_raises():
    mask_img = _mask_img()
    with pytest.raises(ValueError, match="positive, finite sum"):
        _resolve_reporting_baseline(np.zeros(mask_img.shape), mask_img)


def test_resolve_nan_treated_as_zero_with_warning(caplog):
    mask_img = _mask_img()
    arr = np.ones(mask_img.shape)
    arr[2, 2, 2] = np.nan
    with caplog.at_level("WARNING"):
        w = _resolve_reporting_baseline(arr, mask_img)
    assert any("NaN" in r.message for r in caplog.records)
    assert (w == 0).sum() == 1
    assert w.sum() == pytest.approx(1.0)


def test_resolve_wrong_shape_array_raises():
    with pytest.raises(ValueError, match="shape"):
        _resolve_reporting_baseline(np.ones((3, 3, 3)), _mask_img())


def test_resolve_unsupported_type_raises():
    with pytest.raises(TypeError, match="reporting_baseline"):
        _resolve_reporting_baseline(1.0, _mask_img())


def test_resolve_file_path(tmp_path):
    mask_img = _mask_img()
    arr = np.arange(np.prod(mask_img.shape), dtype=float).reshape(mask_img.shape)
    fp = tmp_path / "baseline.nii.gz"
    nib.Nifti1Image(arr, mask_img.affine).to_filename(fp)
    np.testing.assert_allclose(
        _resolve_reporting_baseline(str(fp), mask_img),
        _resolve_reporting_baseline(arr, mask_img),
    )


def test_resolve_unknown_query_raises_without_download(monkeypatch):
    def _no_fetch(*args, **kwargs):
        raise AssertionError("fetch_reference must not be called for an unmatched query")
    monkeypatch.setattr(nsp_datasets, "fetch_reference", _no_fetch)
    with pytest.raises(ValueError, match="must match exactly one"):
        _resolve_reporting_baseline("no-such-baseline", _mask_img())


def test_resolve_query_fetches_single_mabaseline_map_in_original_space(monkeypatch, tmp_path):
    mask_img = _mask_img()
    fp = tmp_path / "baseline.nii.gz"
    nib.Nifti1Image(np.ones(mask_img.shape), mask_img.affine).to_filename(fp)
    calls = {}

    def _fake_fetch(dataset, maps=None, space=None, **kwargs):
        calls.update(dataset=dataset, maps=maps, space=space)
        return [fp]

    monkeypatch.setattr(nsp_datasets, "fetch_reference", _fake_fetch)
    expected = [m for m in nsp_datasets.reference_lib["mabaseline"]["map"] if "neurosynth" in m]
    assert len(expected) == 1
    w = _resolve_reporting_baseline("neurosynth", mask_img, verbose=False)
    assert calls == {"dataset": "mabaseline", "maps": expected[0], "space": "MNIOriginal"}
    assert w.sum() == pytest.approx(1.0)


def test_resolve_uniform_keyword_returns_none():
    # the only way to force uniform sampling for a SCALE result
    assert _resolve_reporting_baseline("uniform", _mask_img()) is None
    assert _resolve_reporting_baseline("UNIFORM", _mask_img()) is None


# ---------------------------------------------------------------------------
# Coordinate pools (deterministic SCALE/SALE null)
#
# These need NiMARE: pool coordinates must be snapped to voxels with the exact
# convention NiMARE uses when it places kernels (mm2vox truncates, not rounds),
# otherwise a focus could be counted on a different voxel than it is modelled.
# ---------------------------------------------------------------------------

def _vox2mm(mask_img, ijk):
    from nimare.utils import vox2mm
    return vox2mm(np.asarray(ijk, dtype=float), mask_img.affine)


def test_coord_pool_weights_are_hit_counts():
    pytest.importorskip("nimare")
    from nispace.helpers.nimare import _coord_pool_to_weights
    mask_img = _mask_img()
    ijk = _mask_ijk(mask_img)
    pool = _vox2mm(mask_img, [ijk[0]] * 3 + [ijk[5]])
    w = _coord_pool_to_weights(pool, mask_img)
    assert w.shape == (len(ijk),)
    assert w.sum() == pytest.approx(1.0)
    expected = np.zeros(len(ijk))
    expected[[0, 5]] = [0.75, 0.25]
    np.testing.assert_allclose(w, expected)


def test_coord_pool_via_ndarray_and_table_and_dataset():
    pytest.importorskip("nimare")
    pd = pytest.importorskip("pandas")
    from nispace.helpers.nimare import _coord_pool_to_weights
    mask_img = _mask_img()
    ijk = _mask_ijk(mask_img)
    pool = _vox2mm(mask_img, [ijk[0], ijk[0], ijk[9]])
    expected = _coord_pool_to_weights(pool, mask_img)

    # (n, 3) array
    np.testing.assert_allclose(_resolve_reporting_baseline(pool, mask_img), expected)
    # table with x/y/z columns
    table = pd.DataFrame(pool, columns=["x", "y", "z"])
    np.testing.assert_allclose(_resolve_reporting_baseline(table, mask_img), expected)
    # anything exposing .coordinates (NiMARE dataset duck type)
    class _Dset:
        coordinates = table
    np.testing.assert_allclose(_resolve_reporting_baseline(_Dset(), mask_img), expected)


def test_coord_pool_table_without_xyz_columns_raises():
    pd = pytest.importorskip("pandas")
    table = pd.DataFrame({"i": [1], "j": [1], "k": [1]})
    with pytest.raises(ValueError, match="missing"):
        _resolve_reporting_baseline(table, _mask_img())


def test_coord_pool_drops_coordinates_outside_mask(caplog):
    pytest.importorskip("nimare")
    from nispace.helpers.nimare import _coord_pool_to_weights
    mask_img = _mask_img()
    ijk = _mask_ijk(mask_img)
    pool = _vox2mm(mask_img, [ijk[0], ijk[1], [0, 0, 0], [99, 99, 99]])  # bg + out of bounds
    with caplog.at_level("WARNING"):
        w = _coord_pool_to_weights(pool, mask_img)
    assert "outside the estimator's mask" in caplog.text
    expected = np.zeros(len(ijk))
    expected[[0, 1]] = 0.5
    np.testing.assert_allclose(w, expected)


def test_coord_pool_on_single_voxel_raises():
    pytest.importorskip("nimare")
    from nispace.helpers.nimare import _coord_pool_to_weights
    mask_img = _mask_img()
    pool = _vox2mm(mask_img, [_mask_ijk(mask_img)[0]] * 5)
    with pytest.raises(ValueError, match="cannot define a null distribution"):
        _coord_pool_to_weights(pool, mask_img)


# ---------------------------------------------------------------------------
# _scale_estimator_weights: reuse of a SCALE estimator's own null
# ---------------------------------------------------------------------------

class _FakeEstimator:
    def __init__(self, mask_img, sampling_prob=None, xyz=None):
        self.masker = type("M", (), {"mask_img": mask_img})()
        self.sampling_prob = sampling_prob
        self.xyz = xyz


def test_scale_weights_from_sampling_prob_are_normalized():
    from nispace.helpers.nimare import _scale_estimator_weights
    mask_img = _mask_img()
    n_vox = int((mask_img.get_fdata() != 0).sum())
    prob = np.arange(1, n_vox + 1, dtype=float)
    w = _scale_estimator_weights(_FakeEstimator(mask_img, sampling_prob=prob))
    np.testing.assert_allclose(w, prob / prob.sum())


def test_scale_weights_prefer_sampling_prob_over_xyz():
    pytest.importorskip("nimare")
    from nispace.helpers.nimare import _scale_estimator_weights
    mask_img = _mask_img()
    n_vox = int((mask_img.get_fdata() != 0).sum())
    prob = np.full(n_vox, 1 / n_vox)
    pool = _vox2mm(mask_img, _mask_ijk(mask_img)[:2])
    w = _scale_estimator_weights(_FakeEstimator(mask_img, sampling_prob=prob, xyz=pool))
    np.testing.assert_allclose(w, prob)


def test_scale_weights_fall_back_to_xyz_pool(caplog):
    pytest.importorskip("nimare")
    from nispace.helpers.nimare import _scale_estimator_weights, _coord_pool_to_weights
    mask_img = _mask_img()
    pool = _vox2mm(mask_img, _mask_ijk(mask_img)[:4])
    # no sampling_prob at all
    w = _scale_estimator_weights(_FakeEstimator(mask_img, xyz=pool))
    np.testing.assert_allclose(w, _coord_pool_to_weights(pool, mask_img))
    # unusable sampling_prob -> warn, then still use the pool
    with caplog.at_level("WARNING"):
        w2 = _scale_estimator_weights(
            _FakeEstimator(mask_img, sampling_prob=np.ones(3), xyz=pool)
        )
    assert "sampling_prob" in caplog.text
    np.testing.assert_allclose(w2, w)


def test_scale_weights_none_when_unusable(caplog):
    pytest.importorskip("nimare")
    from nispace.helpers.nimare import _scale_estimator_weights
    mask_img = _mask_img()
    assert _scale_estimator_weights(_FakeEstimator(mask_img)) is None
    # degenerate pool (nimare-gpu sets xyz=zeros((1, 3)) when none is given)
    with caplog.at_level("WARNING"):
        assert _scale_estimator_weights(
            _FakeEstimator(mask_img, xyz=np.zeros((1, 3)))
        ) is None
    assert "own null" in caplog.text


# ---------------------------------------------------------------------------
# Continuous-mode validation against NiMARE's own p-values
# ---------------------------------------------------------------------------

class _FakeResult:
    """Minimal MetaResult stand-in: .maps dict plus get_map() raising like NiMARE's."""

    def __init__(self, **maps):
        self.maps = maps

    def get_map(self, name):
        arr = self.maps.get(name)
        if arr is None:
            raise ValueError(f"No map with name '{name}' found.")
        return nib.Nifti1Image(arr, np.eye(4))


def _validate_continuous(result, caplog):
    from nispace.helpers.nimare import _validate_null_maps
    rng = np.random.default_rng(0)
    parc_arr = np.zeros((4, 4, 4), dtype=np.float32)
    parc_arr[:2] = 1
    parc_arr[2:] = 2
    parc_idc = np.array([1, 2], dtype=np.float32)
    # set the level on the module logger, not just the root one: other tests can
    # leave it above INFO, and then nothing would propagate to caplog
    with caplog.at_level("INFO", logger="nispace.helpers.nimare"):
        _validate_null_maps(
            rng.random((20, 2)), None, result, "stat",
            parc_arr, parc_idc, None, None, "size", 0.05, None,
        )


def test_validation_uses_logp_when_no_p_map(caplog):
    # SCALE results carry 'logp' but no 'p' map -- this used to raise ValueError
    rng = np.random.default_rng(1)
    shape = (4, 4, 4)
    _validate_continuous(
        _FakeResult(stat=rng.random(shape), logp=rng.random(shape)), caplog
    )
    assert "Spearman" in caplog.text


def test_validation_skips_without_p_or_logp(caplog):
    rng = np.random.default_rng(2)
    _validate_continuous(_FakeResult(stat=rng.random((4, 4, 4))), caplog)
    assert "neither a 'p' nor a 'logp' map" in caplog.text
