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
