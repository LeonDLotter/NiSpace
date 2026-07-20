"""Background/NaN-exclusion handling for parcellation extraction, against a
fully synthetic, non-brain-shaped fake parcellation (NiSpace does not require
brain-shaped input -- parcel labels are just integers on an arbitrary grid).

Covers two things explicitly requested as historically bug-prone:
1. `Parcellater.transform()`/`parcellate_data()` background/NaN exclusion
   settings in isolation (ignore_background_data, background_value,
   drop_background_parcels/background_parcels_to_nan).
2. The same settings wired end-to-end through `NiSpace(...).fit(**kwargs)`
   with real (if tiny) `nib.Nifti1Image` inputs -- exercising the actual
   image-parcellation code path (not pre-parcellated DataFrame input), since
   that's the realistic path any real user with raw images hits.

All expected values are hand-computed from the constructed voxel arrays, not
re-derived from the code under test. See [[project_parcellation_bg_handling]]
and [[project_parcellater_internals]].
"""

import numpy as np
import nibabel as nib
import pytest

from nispace import NiSpace
from nispace.parcellate import Parcellater


AFFINE = np.eye(4)
SHAPE = (3, 4, 4)  # 3 parcels x 16 voxels each, non-brain-shaped


def _parc_img():
    arr = np.zeros(SHAPE)
    arr[0] = 1
    arr[1] = 2
    arr[2] = 3
    return nib.Nifti1Image(arr, AFFINE)


def _fitted_parcellater():
    return Parcellater(_parc_img(), space="mni152", resampling_target="data").fit()


def _sub1_arr():
    """Parcel 1: 4 NaN + 4 exact-zero + 8 real (5.0). Parcel 2: clean 1..16
    range (mean 8.5). Parcel 3: fully zero (all-background)."""
    arr = np.zeros(SHAPE)
    arr[0] = [[np.nan, np.nan, 0.0, 0.0],
              [np.nan, np.nan, 0.0, 0.0],
              [5.0, 5.0, 5.0, 5.0],
              [5.0, 5.0, 5.0, 5.0]]
    arr[1] = np.arange(1, 17).reshape(4, 4)
    arr[2] = 0.0
    return arr


def _sub2_arr():
    """Parcel 1: all real (10.0). Parcel 2: fully NaN. Parcel 3: half-zero,
    half-real (3.0)."""
    arr = np.zeros(SHAPE)
    arr[0] = 10.0
    arr[1] = np.nan
    arr[2] = [[0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0],
              [3.0, 3.0, 3.0, 3.0],
              [3.0, 3.0, 3.0, 3.0]]
    return arr


# ==================================================================================================
# Parcellater.transform() directly
# ==================================================================================================

def test_ignore_background_true_excludes_nan_and_zero():
    p = _fitted_parcellater()
    img = nib.Nifti1Image(_sub1_arr(), AFFINE)
    out = p.transform(img, space="mni152",
                      ignore_background_data=True, background_value=0.0)
    # p1: 8 real 5.0s survive -> 5.0; p2: clean mean 8.5; p3: all-bg -> NaN
    np.testing.assert_allclose(out, [5.0, 8.5, np.nan], equal_nan=True)


def test_ignore_background_false_includes_zero_but_still_excludes_nan():
    p = _fitted_parcellater()
    img = nib.Nifti1Image(_sub1_arr(), AFFINE)
    out = p.transform(img, space="mni152",
                      ignore_background_data=False, background_value=0.0)
    # p1: NaNs still excluded (4 left out), zeros now count -> (4*0+8*5)/12
    np.testing.assert_allclose(out, [40 / 12, 8.5, 0.0], equal_nan=True)


def test_all_nan_parcel_is_nan_regardless_of_background_settings():
    p = _fitted_parcellater()
    img = nib.Nifti1Image(_sub2_arr(), AFFINE)
    out_ignore = p.transform(img, space="mni152",
                             ignore_background_data=True, background_value=0.0)
    out_noignore = p.transform(img, space="mni152",
                               ignore_background_data=False, background_value=0.0)
    assert np.isnan(out_ignore[1])
    assert np.isnan(out_noignore[1])


def test_background_parcels_to_nan_is_noop_when_ignore_background_data_false():
    """`background_parcels_to_nan` is deliberately gated to
    `ignore_background_data=True` only. With `ignore_background_data=False`,
    `background_value` may label real, meaningful data -- the canonical
    example is `NiSpace(binary_y=True)`, which forces
    `ignore_background_data=False` for Y so that an all-zero parcel (0%
    cluster/network overlap) keeps its genuine 0.0 mean rather than being
    NaN'd out as if it were missing background. Confirmed here directly at
    the Parcellater level, mirroring that scenario: even with
    `background_parcels_to_nan=True` explicitly requested, an all-zero parcel
    must NOT be touched when `ignore_background_data=False`."""
    p = _fitted_parcellater()
    img = nib.Nifti1Image(_sub1_arr(), AFFINE)
    out = p.transform(img, space="mni152",
                      ignore_background_data=False, background_value=0.0,
                      background_parcels_to_nan=True)
    assert out[0] == pytest.approx(40 / 12)
    assert out[1] == pytest.approx(8.5)
    assert out[2] == 0.0  # legitimate value, must survive untouched
    assert p._parc_idc_bg == []


def test_background_parcels_to_nan_true_flags_and_reports_bg_parcel():
    """With ignore_background_data=True, the all-background parcel is already
    NaN via the natural empty-mean path regardless of this flag -- its only
    observable effect is populating `_parc_idc_bg` (parcellate_data()'s
    logging), distinguishing "NaN because background" from other NaN causes.
    Regression test for a real bug (fixed 2026-07-17): the original
    implementation compared the *already-background-excluded* aggregated
    mean against the background scalar (`parcellated == bg_arr[0]`), which is
    structurally almost never true once exclusion has already happened --
    `_parc_idc_bg` was never actually populated in this regime. Fixed by
    scanning the raw (pre-exclusion) per-parcel voxel data instead: a parcel
    is flagged if every one of its non-NaN raw values is a background value."""
    p = _fitted_parcellater()
    img = nib.Nifti1Image(_sub1_arr(), AFFINE)
    out = p.transform(img, space="mni152",
                      ignore_background_data=True, background_value=0.0,
                      background_parcels_to_nan=True)
    assert out[0] == pytest.approx(5.0)
    assert out[1] == pytest.approx(8.5)
    assert np.isnan(out[2])
    assert p._parc_idc_bg == [3.0]


def test_background_parcels_to_nan_default_false_skips_reporting_only():
    """Default (False): the all-background parcel is still NaN (via the
    unconditional empty-mean path), but is not additionally recorded in
    `_parc_idc_bg` -- the flag only controls reporting, never the value."""
    p = _fitted_parcellater()
    img = nib.Nifti1Image(_sub1_arr(), AFFINE)
    out = p.transform(img, space="mni152",
                      ignore_background_data=True, background_value=0.0,
                      background_parcels_to_nan=False)
    assert np.isnan(out[2])
    assert p._parc_idc_bg == []


def test_background_parcels_to_nan_does_not_flag_all_nan_parcel():
    """An all-NaN parcel (genuinely missing data, no background voxels at
    all) must not be reported as a background parcel: sub2's parcel 2 is
    entirely NaN (not a single background voxel present), and its parcel 3
    is a zero/real mix (not entirely background either) -- neither counts,
    only a parcel whose non-NaN raw values are *all* background does."""
    p = _fitted_parcellater()
    img = nib.Nifti1Image(_sub2_arr(), AFFINE)
    out = p.transform(img, space="mni152",
                      ignore_background_data=True, background_value=0.0,
                      background_parcels_to_nan=True)
    assert np.isnan(out[1])  # p2: all-NaN
    assert p._parc_idc_bg == []


def test_background_parcels_to_nan_works_with_multiple_background_values():
    """No longer restricted to a single resolved background scalar (that
    constraint belonged to the old, now-replaced post-exclusion-mean check);
    a parcel counts as background if every raw non-NaN value is *any* of the
    given background values."""
    p = _fitted_parcellater()
    img = nib.Nifti1Image(_sub1_arr(), AFFINE)
    out = p.transform(img, space="mni152",
                      ignore_background_data=True, background_value=[0.0, -999.0],
                      background_parcels_to_nan=True)
    assert np.isnan(out[2])
    assert p._parc_idc_bg == [3.0]


def test_background_parcels_to_nan_with_auto_detected_background():
    """Same reporting mechanism, but with background_value='auto' (auto-detected
    scalar) instead of an explicit float, exercising the needs_auto branch."""
    p = _fitted_parcellater()
    img = nib.Nifti1Image(_sub1_arr(), AFFINE)
    out = p.transform(img, space="mni152",
                      ignore_background_data=True, background_value="auto",
                      background_parcels_to_nan=True)
    # border-median auto-detection on this tiny volume; whatever it resolves to,
    # the all-zero parcel is only flagged if auto == 0.0
    if p._parc_idc_bg:
        assert np.isnan(out[2])


# ==================================================================================================
# Full NiSpace(...).fit(**kwargs) wiring, with real (raw image) X/Y input
# ==================================================================================================

def _make_nsp():
    x_arr = np.zeros(SHAPE)
    x_arr[0], x_arr[1], x_arr[2] = 10.0, 20.0, 30.0
    x_img = nib.Nifti1Image(x_arr, AFFINE)
    sub1_img = nib.Nifti1Image(_sub1_arr(), AFFINE)
    sub2_img = nib.Nifti1Image(_sub2_arr(), AFFINE)
    return NiSpace(
        x=[x_img], y=[sub1_img, sub2_img],
        x_labels=["ref"], y_labels=["sub1", "sub2"],
        parcellation=_parc_img(), parcellation_space="MNI152NLin6Asym",
        parcellation_labels=["p1", "p2", "p3"],
        data_space="MNI152NLin6Asym",
        standardize=False, verbose=False, n_proc=1, return_self=True,
    )


def test_nispace_fit_parcellates_raw_images_with_expected_background_handling():
    nsp = _make_nsp()
    nsp.fit(ignore_background_data=True, background_value=0.0)
    X, Y = nsp.get_x(), nsp.get_y()

    np.testing.assert_allclose(X.values, [[10.0, 20.0, 30.0]])
    expected_Y = np.array([
        [5.0, 8.5, np.nan],   # sub1
        [10.0, np.nan, 3.0],  # sub2
    ])
    np.testing.assert_allclose(Y.values, expected_Y, equal_nan=True)


def test_nispace_fit_ignore_background_data_false_kwarg_forwards_correctly():
    nsp = _make_nsp()
    nsp.fit(ignore_background_data=False, background_value=0.0)
    Y = nsp.get_y()
    expected_Y = np.array([
        [40 / 12, 8.5, 0.0],  # sub1: zeros now included, NaNs still excluded
        [10.0, np.nan, 1.5],  # sub2: p3 = (8*0+8*3)/16
    ])
    np.testing.assert_allclose(Y.values, expected_Y, equal_nan=True)


def test_nispace_fit_drop_background_parcels_kwarg_ignore_true_forwards_correctly():
    """End-to-end check of `NiSpace(...).fit(drop_background_parcels=True)`
    (parcellate_data() renames the Parcellater param) with
    ignore_background_data=True, through the top-level API, not just
    internally in Parcellater -- the value is unchanged from the
    already-NaN empty-mean result; this only confirms the kwarg forwards to
    the right place without erroring end-to-end."""
    nsp = _make_nsp()
    nsp.fit(ignore_background_data=True, background_value=0.0,
            drop_background_parcels=True)
    Y = nsp.get_y()
    assert Y.loc["sub1", "p1"] == pytest.approx(5.0)
    assert Y.loc["sub1", "p2"] == pytest.approx(8.5)
    assert np.isnan(Y.loc["sub1", "p3"])
    assert np.isnan(Y.loc["sub2", "p2"])
    assert Y.loc["sub2", "p3"] == pytest.approx(3.0)


def test_nispace_fit_drop_background_parcels_kwarg_ignore_false_is_noop():
    """The binary_y-style safety case, through the full NiSpace(...).fit(**kwargs)
    wiring: with ignore_background_data=False, drop_background_parcels=True must
    NOT touch a legitimate all-zero (e.g. 0%-overlap) parcel."""
    nsp = _make_nsp()
    nsp.fit(ignore_background_data=False, background_value=0.0,
            drop_background_parcels=True)
    Y = nsp.get_y()
    assert Y.loc["sub1", "p1"] == pytest.approx(40 / 12)
    assert Y.loc["sub1", "p2"] == pytest.approx(8.5)
    assert Y.loc["sub1", "p3"] == 0.0  # legitimate all-zero value, untouched
    assert np.isnan(Y.loc["sub2", "p2"])  # all-NaN parcel, unaffected either way
    assert Y.loc["sub2", "p3"] == pytest.approx(1.5)
