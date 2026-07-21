"""Background/NaN-exclusion handling for parcellation extraction, against a
fully synthetic, non-brain-shaped fake parcellation (NiSpace does not require
brain-shaped input -- parcel labels are just integers on an arbitrary grid).

Covers things explicitly requested as historically bug-prone:
1. `Parcellater.transform()`/`parcellate_data()` background/NaN exclusion
   settings in isolation (the unified `background_value` parameter --
   scalar/list/'auto'/False -- and `report_background_parcels`).
2. The same settings wired end-to-end through `NiSpace(...).fit(**kwargs)`
   with real (if tiny) `nib.Nifti1Image` inputs -- exercising the actual
   image-parcellation code path (not pre-parcellated DataFrame input), since
   that's the realistic path any real user with raw images hits. Includes the
   per-role dict form (`background_value={"y": ...}`) and its interaction
   with `binary_y=True`'s automatic Y-only safety default.
3. The deprecated `ignore_background_data`/`drop_background_parcels` kwargs
   still work and emit the project's `_DEPR_...` + `lgr.warning(...)`
   deprecation signal (not `DeprecationWarning`/`warnings.warn` -- this
   codebase's own convention, see `_DEPR_*` constants in `parcellate.py`,
   `io.py`, `api.py`).
4. `min_num_valid_datapoints`/`min_fraction_valid_datapoints` coverage
   thresholds (previously untested in this file), including a regression
   test for a real guard-condition bug (fixed alongside the background_value
   redesign): the check used to silently no-op whenever `background_value`
   was explicitly `None` (a documented valid `'auto'` alias, not "disabled").

All expected values are hand-computed from the constructed voxel arrays, not
re-derived from the code under test. See [[project_bg_coverage_deep_dive]],
[[project_parcellation_bg_handling]], and [[project_parcellater_internals]].
"""

import numpy as np
import nibabel as nib
import pytest

import nispace.parcellate as parcellate_mod
import nispace.io as io_mod
import nispace.api as api_mod
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
# Parcellater.transform() directly -- background_value semantics
# ==================================================================================================

def test_background_value_scalar_excludes_nan_and_zero():
    p = _fitted_parcellater()
    img = nib.Nifti1Image(_sub1_arr(), AFFINE)
    out = p.transform(img, space="mni152", background_value=0.0)
    # p1: 8 real 5.0s survive -> 5.0; p2: clean mean 8.5; p3: all-bg -> NaN
    np.testing.assert_allclose(out, [5.0, 8.5, np.nan], equal_nan=True)


def test_background_value_false_includes_zero_but_still_excludes_nan():
    p = _fitted_parcellater()
    img = nib.Nifti1Image(_sub1_arr(), AFFINE)
    out = p.transform(img, space="mni152", background_value=False)
    # p1: NaNs still excluded (4 left out), zeros now count -> (4*0+8*5)/12
    np.testing.assert_allclose(out, [40 / 12, 8.5, 0.0], equal_nan=True)


def test_all_nan_parcel_is_nan_regardless_of_background_settings():
    p = _fitted_parcellater()
    img = nib.Nifti1Image(_sub2_arr(), AFFINE)
    out_excl = p.transform(img, space="mni152", background_value=0.0)
    out_incl = p.transform(img, space="mni152", background_value=False)
    assert np.isnan(out_excl[1])
    assert np.isnan(out_incl[1])


def test_report_background_parcels_is_noop_when_background_value_false():
    """`report_background_parcels` is deliberately gated to background
    exclusion being enabled. With `background_value=False`, the value may
    label real, meaningful data -- the canonical example is
    `NiSpace(binary_y=True)`, which resolves Y's `background_value` to
    `False` so that an all-zero parcel (0% cluster/network overlap) keeps its
    genuine 0.0 mean rather than being flagged as if it were missing
    background. Confirmed here directly at the Parcellater level, mirroring
    that scenario: even with `report_background_parcels=True` explicitly
    requested, an all-zero parcel must NOT be touched when
    `background_value=False`."""
    p = _fitted_parcellater()
    img = nib.Nifti1Image(_sub1_arr(), AFFINE)
    out = p.transform(img, space="mni152",
                      background_value=False, report_background_parcels=True)
    assert out[0] == pytest.approx(40 / 12)
    assert out[1] == pytest.approx(8.5)
    assert out[2] == 0.0  # legitimate value, must survive untouched
    assert p._parc_idc_bg == []


def test_report_background_parcels_true_flags_and_reports_bg_parcel():
    """With background exclusion enabled, the all-background parcel is
    already NaN via the natural empty-mean path regardless of this flag --
    its only observable effect is populating `_parc_idc_bg`
    (`parcellate_data()`'s logging), distinguishing "NaN because background"
    from other NaN causes."""
    p = _fitted_parcellater()
    img = nib.Nifti1Image(_sub1_arr(), AFFINE)
    out = p.transform(img, space="mni152",
                      background_value=0.0, report_background_parcels=True)
    assert out[0] == pytest.approx(5.0)
    assert out[1] == pytest.approx(8.5)
    assert np.isnan(out[2])
    assert p._parc_idc_bg == [3.0]


def test_report_background_parcels_default_false_skips_reporting_only():
    """Default (False): the all-background parcel is still NaN (via the
    unconditional empty-mean path), but is not additionally recorded in
    `_parc_idc_bg` -- the flag only controls reporting, never the value."""
    p = _fitted_parcellater()
    img = nib.Nifti1Image(_sub1_arr(), AFFINE)
    out = p.transform(img, space="mni152",
                      background_value=0.0, report_background_parcels=False)
    assert np.isnan(out[2])
    assert p._parc_idc_bg == []


def test_report_background_parcels_does_not_flag_all_nan_parcel():
    """An all-NaN parcel (genuinely missing data, no background voxels at
    all) must not be reported as a background parcel: sub2's parcel 2 is
    entirely NaN (not a single background voxel present), and its parcel 3
    is a zero/real mix (not entirely background either) -- neither counts,
    only a parcel whose non-NaN raw values are *all* background does. Also
    verifies the derivation is correctly distinct from a resampling-dropped
    parcel (n_total==0), which likewise must never be reported as background
    (see test_report_background_parcels_never_flags_dropped_parcel)."""
    p = _fitted_parcellater()
    img = nib.Nifti1Image(_sub2_arr(), AFFINE)
    out = p.transform(img, space="mni152",
                      background_value=0.0, report_background_parcels=True)
    assert np.isnan(out[1])  # p2: all-NaN
    assert p._parc_idc_bg == []


def test_report_background_parcels_works_with_multiple_background_values():
    """Not restricted to a single resolved background scalar -- a parcel
    counts as background if every raw non-NaN value is *any* of the given
    background values."""
    p = _fitted_parcellater()
    img = nib.Nifti1Image(_sub1_arr(), AFFINE)
    out = p.transform(img, space="mni152",
                      background_value=[0.0, -999.0], report_background_parcels=True)
    assert np.isnan(out[2])
    assert p._parc_idc_bg == [3.0]


def test_report_background_parcels_with_auto_detected_background():
    """Same reporting mechanism, with the default `background_value='auto'`.
    Bare 'auto' always resolves to `['auto', 0.0]` (auto-detected value AND
    exact zero, both excluded) -- so the all-zero parcel is always flagged,
    unconditionally, regardless of whatever the auto-detected scalar itself
    resolves to on this tiny synthetic volume."""
    p = _fitted_parcellater()
    img = nib.Nifti1Image(_sub1_arr(), AFFINE)
    out = p.transform(img, space="mni152",
                      background_value="auto", report_background_parcels=True)
    assert np.isnan(out[2])
    assert p._parc_idc_bg == [3.0]


def _dropped_parcel_img():
    """Data covering only slices 0-1 of SHAPE=(3,4,4) -- parcel 3 (slice 2 of
    the parcellation) has zero corresponding voxels after resampling onto
    this smaller grid, i.e. it is dropped, not merely background/NaN."""
    arr = np.zeros((2, 4, 4))
    arr[0] = 10.0
    arr[1] = 20.0
    return nib.Nifti1Image(arr, AFFINE)


def test_report_background_parcels_never_flags_dropped_parcel():
    """A parcel dropped during resampling (n_total==0: zero voxels carry that
    label in the resampled parcellation at all) is a structurally different
    cause of NaN than "all raw values were background" (n_total>0 but
    n_valid==0) -- `all_background` must stay False for it, and
    `report_background_parcels` must record it in `_parc_idc_dropped`, never
    in `_parc_idc_bg`, even though both end up NaN in the output. Regression
    coverage for the integration-level (not just the raw
    `vol_to_vect_arr_stats` unit-level, see test_utils_smoke.py) version of
    this invariant, through the actual `Parcellater.transform()` resampling
    path."""
    p = _fitted_parcellater()
    img = _dropped_parcel_img()
    out = p.transform(img, space="mni152",
                      background_value=0.0, report_background_parcels=True)
    np.testing.assert_allclose(out, [10.0, 20.0, np.nan], equal_nan=True)
    assert p._parc_idc_dropped == [3.0]
    assert p._parc_idc_bg == []


def test_fill_dropped_false_excludes_dropped_parcel_from_output():
    """With `fill_dropped=False`, the returned array only covers parcels
    present in the resampled parcellation -- shorter than
    `self.parcellation_idc`, with no NaN placeholder for the dropped parcel.
    `_parc_idc_dropped` still records it regardless."""
    p = _fitted_parcellater()
    img = _dropped_parcel_img()
    out = p.transform(img, space="mni152",
                      background_value=0.0, fill_dropped=False)
    np.testing.assert_allclose(out, [10.0, 20.0])
    assert len(out) == 2
    assert p._parc_idc_dropped == [3.0]


# ==================================================================================================
# Parcellater.transform() directly -- min_num/min_fraction_valid_datapoints
# ==================================================================================================

def test_min_num_valid_datapoints_excludes_low_coverage_parcels():
    p = _fitted_parcellater()
    img = nib.Nifti1Image(_sub1_arr(), AFFINE)
    out = p.transform(img, space="mni152",
                      background_value=0.0, min_num_valid_datapoints=10)
    # p1: 8 valid (< 10) -> excluded; p2: 16 valid -> kept; p3: 0 valid -> excluded (already NaN)
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(8.5)
    assert np.isnan(out[2])
    assert set(p._parc_idc_excl) == {1.0, 3.0}


def test_min_fraction_valid_datapoints_excludes_low_coverage_parcels():
    p = _fitted_parcellater()
    img = nib.Nifti1Image(_sub1_arr(), AFFINE)
    out = p.transform(img, space="mni152",
                      background_value=0.0, min_fraction_valid_datapoints=0.6)
    # p1: 8/16=0.5 (< 0.6) -> excluded; p2: 16/16=1.0 -> kept; p3: 0/16 -> excluded (already NaN)
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(8.5)
    assert np.isnan(out[2])
    assert set(p._parc_idc_excl) == {1.0, 3.0}


def test_min_valid_datapoints_regression_guard_bug_with_background_value_none():
    """Regression test for a real bug: the old guard condition checked the
    raw `background_value is not None`, so an explicit `background_value=None`
    (a documented valid 'auto' alias, not 'disabled') silently skipped the
    entire coverage-threshold block. Use a threshold that exceeds every
    parcel's total voxel count (16) so the assertion doesn't depend on what
    the auto-detected background value happens to resolve to -- if the old
    bug were still present, nothing would be excluded and this would fail."""
    p = _fitted_parcellater()
    img = nib.Nifti1Image(_sub1_arr(), AFFINE)
    out = p.transform(img, space="mni152",
                      background_value=None, min_num_valid_datapoints=17)
    assert np.isnan(out).all()
    assert set(p._parc_idc_excl) == {1.0, 2.0, 3.0}


# ==================================================================================================
# Parcellater.transform() directly -- deprecated ignore_background_data kwarg
# ==================================================================================================

def test_transform_ignore_background_data_legacy_kwarg_still_works_and_warns(monkeypatch):
    calls = []
    monkeypatch.setattr(parcellate_mod.lgr, "warning", lambda msg: calls.append(msg))
    p = _fitted_parcellater()
    img = nib.Nifti1Image(_sub1_arr(), AFFINE)
    out = p.transform(img, space="mni152",
                      ignore_background_data=False, background_value=0.0)
    np.testing.assert_allclose(out, [40 / 12, 8.5, 0.0], equal_nan=True)
    assert len(calls) == 1
    assert "ignore_background_data" in calls[0]


def test_transform_ignore_background_data_legacy_true_still_works(monkeypatch):
    monkeypatch.setattr(parcellate_mod.lgr, "warning", lambda msg: None)
    p = _fitted_parcellater()
    img = nib.Nifti1Image(_sub1_arr(), AFFINE)
    out = p.transform(img, space="mni152",
                      ignore_background_data=True, background_value=0.0)
    np.testing.assert_allclose(out, [5.0, 8.5, np.nan], equal_nan=True)


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
    nsp.fit(background_value=0.0)
    X, Y = nsp.get_x(), nsp.get_y()

    np.testing.assert_allclose(X.values, [[10.0, 20.0, 30.0]])
    expected_Y = np.array([
        [5.0, 8.5, np.nan],   # sub1
        [10.0, np.nan, 3.0],  # sub2
    ])
    np.testing.assert_allclose(Y.values, expected_Y, equal_nan=True)


def test_nispace_fit_background_value_false_kwarg_forwards_correctly():
    nsp = _make_nsp()
    nsp.fit(background_value=False)
    Y = nsp.get_y()
    expected_Y = np.array([
        [40 / 12, 8.5, 0.0],  # sub1: zeros now included, NaNs still excluded
        [10.0, np.nan, 1.5],  # sub2: p3 = (8*0+8*3)/16
    ])
    np.testing.assert_allclose(Y.values, expected_Y, equal_nan=True)


def test_nispace_fit_report_background_parcels_kwarg_forwards_correctly():
    """End-to-end check of `NiSpace(...).fit(report_background_parcels=True)`
    through the top-level API, not just internally in Parcellater -- the
    value is unchanged from the already-NaN empty-mean result; this only
    confirms the kwarg forwards to the right place without erroring
    end-to-end."""
    nsp = _make_nsp()
    nsp.fit(background_value=0.0, report_background_parcels=True)
    Y = nsp.get_y()
    assert Y.loc["sub1", "p1"] == pytest.approx(5.0)
    assert Y.loc["sub1", "p2"] == pytest.approx(8.5)
    assert np.isnan(Y.loc["sub1", "p3"])
    assert np.isnan(Y.loc["sub2", "p2"])
    assert Y.loc["sub2", "p3"] == pytest.approx(3.0)


def test_nispace_fit_report_background_parcels_is_noop_with_background_value_false():
    """The binary_y-style safety case, through the full NiSpace(...).fit(**kwargs)
    wiring: with background_value=False, report_background_parcels=True must
    NOT touch a legitimate all-zero (e.g. 0%-overlap) parcel."""
    nsp = _make_nsp()
    nsp.fit(background_value=False, report_background_parcels=True)
    Y = nsp.get_y()
    assert Y.loc["sub1", "p1"] == pytest.approx(40 / 12)
    assert Y.loc["sub1", "p2"] == pytest.approx(8.5)
    assert Y.loc["sub1", "p3"] == 0.0  # legitimate all-zero value, untouched
    assert np.isnan(Y.loc["sub2", "p2"])  # all-NaN parcel, unaffected either way
    assert Y.loc["sub2", "p3"] == pytest.approx(1.5)


def test_nispace_fit_legacy_ignore_background_data_kwarg_still_works_and_warns(monkeypatch):
    calls = []
    monkeypatch.setattr(api_mod.lgr, "warning", lambda msg: calls.append(msg))
    nsp = _make_nsp()
    nsp.fit(ignore_background_data=False, background_value=0.0)
    Y = nsp.get_y()
    expected_Y = np.array([
        [40 / 12, 8.5, 0.0],
        [10.0, np.nan, 1.5],
    ])
    np.testing.assert_allclose(Y.values, expected_Y, equal_nan=True)
    assert any("ignore_background_data" in c for c in calls)


def test_parcellate_data_legacy_drop_background_parcels_kwarg_warns(monkeypatch):
    calls = []
    monkeypatch.setattr(io_mod.lgr, "warning", lambda msg: calls.append(msg))
    from nispace.io import parcellate_data
    out = parcellate_data(
        [nib.Nifti1Image(_sub1_arr(), AFFINE)],
        data_labels=["sub1"],
        data_space="MNI152NLin6Asym",
        parcellation=_parc_img(), parc_space="MNI152NLin6Asym",
        parc_labels=["p1", "p2", "p3"],
        background_value=0.0, drop_background_parcels=True,
        verbose=False, n_proc=1,
    )
    assert np.isnan(out.loc["sub1", "p3"])
    assert any("drop_background_parcels" in c for c in calls)


# ==================================================================================================
# Full NiSpace(...).fit(**kwargs) wiring -- per-role dict + binary_y precedence
# ==================================================================================================

def _make_nsp_binary_y():
    x_arr = np.zeros(SHAPE)
    x_arr[0], x_arr[1], x_arr[2] = 10.0, 20.0, 30.0
    x_img = nib.Nifti1Image(x_arr, AFFINE)
    sub1_img = nib.Nifti1Image(_sub1_arr(), AFFINE)
    return NiSpace(
        x=[x_img], y=[sub1_img],
        x_labels=["ref"], y_labels=["sub1"],
        parcellation=_parc_img(), parcellation_space="MNI152NLin6Asym",
        parcellation_labels=["p1", "p2", "p3"],
        data_space="MNI152NLin6Asym",
        standardize=False, verbose=False, n_proc=1, return_self=True,
        binary_y=True,
    )


def test_binary_y_default_background_value_false_for_y_only():
    nsp = _make_nsp_binary_y()
    nsp.fit()
    Y = nsp.get_y()
    # p3 (all-zero raw data): binary_y default keeps it as real 0.0, not background-excluded
    assert Y.loc["sub1", "p3"] == 0.0


def test_binary_y_backs_off_only_on_explicit_y_dict_key():
    nsp = _make_nsp_binary_y()
    nsp.fit(background_value={"y": 0.0})
    Y = nsp.get_y()
    # explicit {"y": 0.0} overrides the binary_y safety default -> p3 all-zero -> excluded -> NaN
    assert np.isnan(Y.loc["sub1", "p3"])


def test_binary_y_not_backed_off_by_plain_scalar_meant_for_other_roles():
    nsp = _make_nsp_binary_y()
    # plain top-level scalar (not a dict targeting "y") must NOT back off Y's binary_y default
    nsp.fit(background_value=0.0)
    Y = nsp.get_y()
    assert Y.loc["sub1", "p3"] == 0.0


def test_per_role_dict_unspecified_role_falls_back_to_auto():
    """"x" is absent from the dict -> must fall back to the top-level default
    'auto', not to some other implicit value. Verified by comparing against a
    separate fit() using the bare top-level default explicitly (rather than
    hardcoding X's expected numeric output, which would be sensitive to
    exactly what the border-median auto-detection resolves to on this tiny
    synthetic fixture) -- X must come out identical either way, while Y must
    differ (only the {"y": False} call disables exclusion for Y)."""
    nsp1 = _make_nsp()
    nsp1.fit(background_value={"y": False})
    nsp2 = _make_nsp()
    nsp2.fit(background_value="auto")
    np.testing.assert_allclose(nsp1.get_x().values, nsp2.get_x().values, equal_nan=True)
    assert not nsp1.get_y().equals(nsp2.get_y())


def test_input_kwargs_z_independent_of_x():
    """_input_kwargs_z must be its own copy, independently resolved from X --
    regression coverage for the fix that gave Z its own per-role resolution
    (previously X and Z shared one kwargs dict)."""
    x_arr = np.zeros(SHAPE)
    x_arr[0], x_arr[1], x_arr[2] = 10.0, 20.0, 30.0
    x_img = nib.Nifti1Image(x_arr, AFFINE)
    z_img = nib.Nifti1Image(_sub1_arr(), AFFINE)
    y_img = nib.Nifti1Image(_sub1_arr(), AFFINE)
    nsp = NiSpace(
        x=[x_img], y=[y_img], z=[z_img],
        x_labels=["ref"], y_labels=["sub1"], z_labels=["zmap"],
        parcellation=_parc_img(), parcellation_space="MNI152NLin6Asym",
        parcellation_labels=["p1", "p2", "p3"],
        data_space="MNI152NLin6Asym",
        standardize=False, verbose=False, n_proc=1, return_self=True,
    )
    nsp.fit(background_value={"x": 0.0, "z": False})
    Z = nsp.get_z()
    # z uses background_value=False -> p3 (all-zero raw) stays real 0.0
    assert Z.loc["zmap", "p3"] == 0.0
