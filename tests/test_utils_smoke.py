"""Smoke tests (well-formedness, not deep correctness) for the image-utility
functions in utils/utils.py that operate on plain nibabel Nifti1Image/
GiftiImage objects and don't need real brain data -- synthetic small volumes
suffice. Per the living testing-batches plan, this batch's job is proving
each live function runs cleanly on valid input and returns the documented
type/shape, not exhaustively verifying its numerics.
"""

import numpy as np
import nibabel as nib
import pytest

from nispace.utils.utils import (
    get_background_value, vect_to_vol_arr, vol_to_vect_arr, parc_vect_to_vol,
    relabel_gifti_parc, relabel_nifti_parc, merge_parcellations,
    correlate_hemispheres, mirror_nifti, mirror_gifti, correlated_vector,
)


@pytest.fixture
def synthetic_parc_nifti():
    """6x6x6 volume with 3 equal-sized parcels along the x-axis."""
    arr = np.zeros((6, 6, 6), dtype=np.int32)
    arr[0:2, :, :] = 1
    arr[2:4, :, :] = 2
    arr[4:6, :, :] = 3
    return nib.Nifti1Image(arr, affine=np.eye(4))


# ── vect_to_vol_arr / vol_to_vect_arr / parc_vect_to_vol ─────────────────

def test_parc_vect_to_vol_broadcasts_values(synthetic_parc_nifti):
    vect = np.array([10.0, 20.0, 30.0])
    vol = parc_vect_to_vol(vect, synthetic_parc_nifti)
    assert isinstance(vol, nib.Nifti1Image)
    assert set(np.unique(vol.get_fdata())) == {10.0, 20.0, 30.0}


def test_vect_to_vol_arr_and_vol_to_vect_arr_roundtrip(synthetic_parc_nifti):
    parc_arr = synthetic_parc_nifti.get_fdata().astype(np.float64)
    parc_idc = np.array([1.0, 2.0, 3.0])
    vect = np.array([10.0, 20.0, 30.0])

    vol_arr = vect_to_vol_arr(vect, parc_arr, parc_idc)
    assert set(np.unique(vol_arr)) == {10.0, 20.0, 30.0}

    back = vol_to_vect_arr(vol_arr, parc_arr, parc_idc, np.array([], dtype=np.float64))
    np.testing.assert_allclose(back, vect)


def test_vect_to_vol_arr_length_mismatch_raises(synthetic_parc_nifti):
    parc_arr = synthetic_parc_nifti.get_fdata().astype(np.float64)
    parc_idc = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        vect_to_vol_arr(np.array([1.0, 2.0]), parc_arr, parc_idc)


# ── get_background_value ─────────────────────────────────────────────────

def test_get_background_value_detects_border_median():
    arr = np.full((10, 10, 10), 100.0)
    arr[4:6, 4:6, 4:6] = 5.0  # interior "signal" region, border stays 100
    img = nib.Nifti1Image(arr, affine=np.eye(4))
    bg = get_background_value(img)
    assert bg == pytest.approx(100.0)


def test_get_background_value_nan_border_returns_nan():
    arr = np.full((10, 10, 10), 1.0)
    arr[0, 0, 0] = np.nan
    img = nib.Nifti1Image(arr, affine=np.eye(4))
    assert np.isnan(get_background_value(img))


# ── relabel_nifti_parc / relabel_gifti_parc ──────────────────────────────

def test_relabel_nifti_parc_reassigns_labels(synthetic_parc_nifti):
    relabeled = relabel_nifti_parc(synthetic_parc_nifti, new_labels=[10, 20, 30])
    assert set(np.trim_zeros(np.unique(relabeled.get_fdata()))) == {10.0, 20.0, 30.0}


def test_relabel_nifti_parc_default_labels_sequential(synthetic_parc_nifti):
    relabeled = relabel_nifti_parc(synthetic_parc_nifti)
    assert set(np.trim_zeros(np.unique(relabeled.get_fdata()))) == {1.0, 2.0, 3.0}


def test_relabel_gifti_parc_reassigns_labels():
    data = np.array([1, 1, 2, 2, 3, 3], dtype=np.float32)
    darray = nib.gifti.GiftiDataArray(data)
    gii = nib.GiftiImage(darrays=[darray])
    relabeled = relabel_gifti_parc(gii, new_labels=[10, 20, 30])
    assert set(np.trim_zeros(np.unique(relabeled.agg_data()))) == {10.0, 20.0, 30.0}


def test_relabel_gifti_parc_requires_giftiimage():
    with pytest.raises(ValueError):
        relabel_gifti_parc(np.array([1, 2, 3]))


# ── merge_parcellations ───────────────────────────────────────────────────

def test_merge_parcellations_quick_offsets_second_parc_labels(synthetic_parc_nifti):
    arr2 = np.zeros((6, 6, 6), dtype=np.int32)
    arr2[0:3, :, :] = 1
    parc2 = nib.Nifti1Image(arr2, affine=np.eye(4))
    merged = merge_parcellations([synthetic_parc_nifti, parc2], quick=True)
    assert isinstance(merged, nib.Nifti1Image)
    # second parc's label 1 offset by first parc's max label (3) -> 4
    assert 4.0 in np.unique(merged.get_fdata())


def test_merge_parcellations_slow_returns_labels_series(synthetic_parc_nifti):
    arr2 = np.zeros((6, 6, 6), dtype=np.int32)
    arr2[0:3, :, :] = 1
    parc2 = nib.Nifti1Image(arr2, affine=np.eye(4))
    merged, labels_merged = merge_parcellations([synthetic_parc_nifti, parc2], quick=False)
    assert isinstance(merged, nib.Nifti1Image)
    # 3 parcels from first + 1 from second = 4 sequential new labels
    assert set(labels_merged.index) == {1, 2, 3, 4}


def test_merge_parcellations_rejects_non_list():
    with pytest.raises(ValueError):
        merge_parcellations("not_a_list")


# ── correlate_hemispheres / mirror_nifti / mirror_gifti ──────────────────

def test_correlate_hemispheres_perfectly_symmetric_data_gives_corr_one(rng):
    arr = rng.normal(size=(8, 6, 6)) + 5.0  # offset from 0 -- default mask excludes exact zeros
    arr[4:] = arr[:4][::-1]  # mirror-symmetric across x
    assert correlate_hemispheres(arr) == pytest.approx(1.0, abs=1e-6)


def test_mirror_nifti_returns_same_type(synthetic_parc_nifti):
    mirrored = mirror_nifti(synthetic_parc_nifti)
    assert isinstance(mirrored, nib.Nifti1Image)
    assert mirrored.shape == synthetic_parc_nifti.shape


def test_mirror_gifti_left_to_right_copies_left_to_right():
    lh = np.array([1.0, 2.0, 3.0])
    rh = np.array([9.0, 9.0, 9.0])
    out_lh, out_rh = mirror_gifti((lh, rh), direction="left_to_right")
    np.testing.assert_allclose(out_lh, lh)
    np.testing.assert_allclose(out_rh, lh)


def test_mirror_gifti_rejects_non_tuple():
    with pytest.raises(ValueError):
        mirror_gifti(np.array([1, 2, 3]))


# ── correlated_vector ─────────────────────────────────────────────────────
# Regression tests: _corr_vector (numba-jitted) called np.nanstd(x, ddof=0)
# and np.std(x, ddof=0) -- this numba version's nanstd/std overloads reject
# the ddof kwarg entirely (TypingError at every call, not a corner case),
# making correlated_vector() 100% broken before the fix. Confirmed via a
# minimal njit repro before patching; fixed by hand-rolling the population
# std (ddof=0) via nanmean/mean of squared deviations instead.

def test_correlated_vector_correlation_one_returns_input_unchanged():
    data = np.arange(10.0)
    out = correlated_vector(data, correlation=1.0)
    np.testing.assert_array_equal(out, data)


def test_correlated_vector_achieves_approximately_target_correlation(rng):
    data = rng.normal(size=2000)
    out = correlated_vector(data, correlation=0.6, seed=42)
    assert np.corrcoef(out, data)[0, 1] == pytest.approx(0.6, abs=0.05)


def test_correlated_vector_preserves_nan_positions(rng):
    data = rng.normal(size=20)
    data[3] = np.nan
    out = correlated_vector(data, correlation=0.5, seed=1)
    assert np.isnan(out[3])
    assert np.isfinite(np.delete(out, 3)).all()
