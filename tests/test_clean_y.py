"""Tests for nispace.core.clean_y and NiSpace.clean_y().

Two parts:
1. Below: general unit coverage of the covariate-normalization helpers and the
   within/between regression engines (_normalize_cov_df, _detect_categoricals,
   _encode, _clean_y_between, _clean_y_within). Offline/network-free. ComBat
   correctness (neuroHarmonize's own numerics) is out of scope -- only
   NiSpace's wiring around it is tested (site detection, disabling when no
   site column).
2. Original content (kept as-is): regression tests for a bug where
   clean_y(how="between", covariates_between=...) without an explicit
   `protect` argument fully demeaned every parcel across subjects (removing
   the intercept, not just the covariate-attributable component). That's
   harmless for group-difference transforms (a shared per-parcel constant
   cancels out of any a-vs-b contrast, e.g. group_colocalization()'s
   hedges/cohen/zscore/meandiff/elemdiff/pairedcohen), but it silently
   destroyed real signal for any pipeline that colocalizes the cleaned maps
   directly -- colocalization()/xsea(), which pass covariates_between with no
   protect (nispace/workflows.py). On real data (see
   _explore/mem_covariate_regression_prototype.ipynb), cleaning against a
   covariate with *zero* true relationship to anything collapsed mean
   per-subject colocalization from ~0.81 to ~0.002.
"""

import numpy as np
import pandas as pd
import pytest

from nispace import NiSpace
from nispace.core.clean_y import (
    _normalize_cov_df,
    _detect_categoricals,
    _encode,
    _clean_y_between,
    _clean_y_within,
)
from nispace.stats.misc import residuals_nan, partial_residuals_nan


# ── _normalize_cov_df ────────────────────────────────────────────────────────

def test_normalize_cov_df_array_gets_generic_names(rng):
    df = _normalize_cov_df(rng.normal(size=(6, 2)), 6, "cov")
    assert df.columns.tolist() == ["cov_0", "cov_1"]


def test_normalize_cov_df_series_uses_lowercased_name(rng):
    s = pd.Series(rng.normal(size=6), name="Age")
    df = _normalize_cov_df(s, 6, "cov")
    assert df.columns.tolist() == ["age"]


def test_normalize_cov_df_dataframe_lowercases_columns(rng):
    df_in = pd.DataFrame({"Age": rng.normal(size=6), "Site": ["a"] * 6})
    df = _normalize_cov_df(df_in, 6, "cov")
    assert df.columns.tolist() == ["age", "site"]


def test_normalize_cov_df_row_mismatch_raises(rng):
    df_in = pd.DataFrame({"age": rng.normal(size=4)})
    with pytest.raises(ValueError, match="rows"):
        _normalize_cov_df(df_in, 6, "cov")


# ── _detect_categoricals ──────────────────────────────────────────────────────

def test_detect_categoricals_splits_by_dtype(rng):
    df = pd.DataFrame({"age": rng.normal(size=4), "sex": ["m", "f", "m", "f"]})
    cat, cont = _detect_categoricals(df)
    assert cat == ["sex"]
    assert cont == ["age"]


def test_detect_categoricals_site_always_categorical_even_if_numeric(rng):
    # "site" forced categorical regardless of dtype (e.g. numeric site codes)
    df = pd.DataFrame({"age": rng.normal(size=4), "site": [1, 1, 2, 2]})
    cat, cont = _detect_categoricals(df)
    assert cat == ["site"]
    assert cont == ["age"]


# ── _encode ───────────────────────────────────────────────────────────────────

def test_encode_one_hot_drop_first_prepends_continuous(rng):
    df = pd.DataFrame({"age": rng.normal(size=4), "site": ["a", "a", "b", "b"]})
    enc = _encode(df, ["age"], ["site"])
    assert enc.columns.tolist() == ["age", "site_b"]
    assert np.array_equal(enc["site_b"].values, [0, 0, 1, 1])


# ── _clean_y_between ──────────────────────────────────────────────────────────

@pytest.fixture
def between_setup(rng):
    n_subjects, n_parcels = 8, 5
    Y = rng.normal(size=(n_subjects, n_parcels))
    cov = pd.DataFrame({"age": rng.normal(size=n_subjects), "sex": ["m", "f"] * 4})
    return Y, cov, n_subjects, n_parcels


def _default_between_kwargs(**overrides):
    kwargs = dict(
        protect=None, combat=False, combat_protect=None, combat_train=None,
        combat_model=None, combat_kwargs={}, plot_design_between=False,
        n_proc=1, dtype=np.float32, verbose=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_clean_y_between_matches_manual_partial_residuals(between_setup):
    Y, cov, n_subjects, n_parcels = between_setup
    Y_out, combat_model, combat_cov = _clean_y_between(
        Y.copy(), cov, n_subjects, **_default_between_kwargs()
    )
    assert combat_model is None and combat_cov is None

    reg_arr = pd.concat(
        [cov[["age"]], pd.get_dummies(cov["sex"], prefix="sex", drop_first=True, dtype=np.float32)],
        axis=1,
    ).values.astype(np.float32)
    protect_arr = np.zeros((n_subjects, 0), dtype=np.float32)
    expected = np.array([
        partial_residuals_nan(reg_arr, protect_arr, Y[:, i].astype(np.float32))
        for i in range(n_parcels)
    ]).T
    assert np.allclose(Y_out, expected, atol=1e-5)


def test_clean_y_between_protect_preserves_protected_effect(between_setup):
    Y, _, n_subjects, n_parcels = between_setup
    cov = pd.DataFrame({"age": np.random.default_rng(2).normal(size=n_subjects)})
    protect = pd.DataFrame({"group": ["a"] * (n_subjects // 2) + ["b"] * (n_subjects // 2)})

    Y_out, _, _ = _clean_y_between(
        Y.copy(), cov, n_subjects, **_default_between_kwargs(protect=protect)
    )
    reg_arr = cov[["age"]].values.astype(np.float32)
    protect_arr = pd.get_dummies(protect["group"], prefix="group", drop_first=True, dtype=np.float32).values
    expected = np.array([
        partial_residuals_nan(reg_arr, protect_arr, Y[:, i].astype(np.float32))
        for i in range(n_parcels)
    ]).T
    assert np.allclose(Y_out, expected, atol=1e-5)


def test_clean_y_between_combat_disabled_without_site_column(between_setup):
    Y, cov, n_subjects, n_parcels = between_setup
    cov_no_site = cov[["age"]]  # no "site" column

    Y_out, combat_model, combat_cov = _clean_y_between(
        Y.copy(), cov_no_site, n_subjects, **_default_between_kwargs(combat=True)
    )
    # combat silently disabled -> falls back to plain regression, no model fitted
    assert combat_model is None
    assert combat_cov is None
    reg_arr = cov_no_site.values.astype(np.float32)
    protect_arr = np.zeros((n_subjects, 0), dtype=np.float32)
    expected = np.array([
        partial_residuals_nan(reg_arr, protect_arr, Y[:, i].astype(np.float32))
        for i in range(n_parcels)
    ]).T
    assert np.allclose(Y_out, expected, atol=1e-5)


def test_clean_y_between_handles_dtype_mismatch(between_setup):
    """Regression test for the dtype-mismatch bug found while writing these
    tests: Y_arr wasn't cast to `dtype` before being combined with the
    already-cast reg_arr/protect_arr, crashing numba's partial_residuals_nan
    whenever Y_arr's dtype differed from the `dtype` kwarg (e.g. float64 Y
    with the default dtype=float32). Not reachable through the public API
    today (NiSpace.fit() always pre-casts self._Y), but this pins the fix at
    the helper level regardless of caller discipline."""
    Y, cov, n_subjects, n_parcels = between_setup
    Y_float64 = Y.astype(np.float64)
    Y_out, _, _ = _clean_y_between(
        Y_float64.copy(), cov, n_subjects, **_default_between_kwargs()
    )
    assert Y_out.dtype == np.float32


# ── _clean_y_within ───────────────────────────────────────────────────────────

def test_clean_y_within_z_without_z_data_raises(rng):
    Y = rng.normal(size=(4, 6))
    with pytest.raises(ValueError, match="Z data"):
        _clean_y_within(Y, "z", None, 4, 6, False, 1, np.float32, False)


def test_clean_y_within_z_shared_across_maps_matches_manual(rng):
    n_maps, n_parcels = 4, 6
    Y = rng.normal(size=(n_maps, n_parcels))
    Z = pd.DataFrame(rng.normal(size=(1, n_parcels)))

    Y_out, used_z, did = _clean_y_within(Y.copy(), "z", Z, n_maps, n_parcels, False, 1, np.float32, False)
    assert used_z is True
    assert did is True

    z_arr = np.array(Z, dtype=np.float32)
    Y_f32 = Y.astype(np.float32)
    expected = np.array([residuals_nan(z_arr[0], Y_f32[i]) for i in range(n_maps)])
    assert np.allclose(Y_out, expected, atol=1e-5)


def test_clean_y_within_custom_array_shared_map(rng):
    n_maps, n_parcels = 4, 6
    Y = rng.normal(size=(n_maps, n_parcels))
    cov = rng.normal(size=(1, n_parcels))

    Y_out, used_z, did = _clean_y_within(Y.copy(), cov, None, n_maps, n_parcels, False, 1, np.float32, False)
    assert used_z is False
    assert did is True


def test_clean_y_within_parcel_mismatch_leaves_y_unchanged(rng):
    n_maps, n_parcels = 4, 6
    Y = rng.normal(size=(n_maps, n_parcels))
    cov_bad = rng.normal(size=(1, n_parcels + 1))  # wrong parcel count

    Y_out, used_z, did = _clean_y_within(Y.copy(), cov_bad, None, n_maps, n_parcels, False, 1, np.float32, False)
    assert did is False
    assert np.array_equal(Y_out, Y.astype(np.float32))


def test_clean_y_within_y_specific_matches_manual(rng):
    n_maps, n_parcels = 4, 6
    Y = rng.normal(size=(n_maps, n_parcels))
    cov_specific = rng.normal(size=(n_maps, n_parcels))

    Y_out, used_z, did = _clean_y_within(
        Y.copy(), cov_specific, None, n_maps, n_parcels, True, 1, np.float32, False
    )
    assert did is True
    cov_f32 = cov_specific.astype(np.float32)
    Y_f32 = Y.astype(np.float32)
    expected = np.array([residuals_nan(cov_f32[i], Y_f32[i]) for i in range(n_maps)])
    assert np.allclose(Y_out, expected, atol=1e-5)


def test_clean_y_within_handles_dtype_mismatch(rng):
    """Regression test for the dtype-mismatch bug in the 'z' string branch:
    wcov_arr = np.array(Z) skipped the dtype cast that the sibling
    array/Series/DataFrame branch applies, crashing numba's residuals_nan
    whenever Z's dtype differed from Y_arr's post-cast dtype."""
    n_maps, n_parcels = 4, 6
    Y = np.random.default_rng(3).normal(size=(n_maps, n_parcels)).astype(np.float64)
    Z = pd.DataFrame(np.random.default_rng(4).normal(size=(1, n_parcels)))  # float64
    Y_out, _, _ = _clean_y_within(Y.copy(), "z", Z, n_maps, n_parcels, False, 1, np.float32, False)
    assert Y_out.dtype == np.float32


def test_between_regression_preserves_intercept(rng):
    """clean_y(how='between') without `protect` must remove only the
    covariate-attributable component, never the per-parcel intercept."""
    n_subj, n_parcels = 25, 40
    Y_true = rng.normal(size=n_parcels)
    Y = Y_true[np.newaxis, :] + rng.normal(scale=0.3, size=(n_subj, n_parcels))
    cov = rng.normal(size=n_subj)  # covariate uncorrelated with Y_true by construction

    parcel_labels = [f"p{i}" for i in range(n_parcels)]
    y_df = pd.DataFrame(Y, index=[f"y{i}" for i in range(n_subj)], columns=parcel_labels)
    x_df = pd.DataFrame(Y_true[np.newaxis, :], index=["x0"], columns=parcel_labels)

    nsp = NiSpace(x=x_df, y=y_df, parcellation=None, standardize=False,
                 n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    nsp.clean_y(how="between", covariates_between=pd.Series(cov, name="cov"), verbose=False)
    Y_clean = nsp.get_y().to_numpy()

    # reference: OLS on [cov | intercept], remove only the cov term (partial_residuals_nan's
    # definition) -- this is the invariant that a full-residual (residuals_nan) regression breaks
    X_design = np.column_stack([cov, np.ones(n_subj)])
    beta = np.linalg.pinv(X_design.T @ X_design) @ (X_design.T @ Y)
    Y_expected = Y - np.outer(cov, beta[0])

    np.testing.assert_allclose(Y_clean, Y_expected, atol=1e-4)


def test_irrelevant_covariate_does_not_destroy_colocalization(rng):
    """An irrelevant (pure-noise) between-covariate must not measurably change
    per-subject colocalization with X after clean_y(how='between'). Direct
    regression test for the colocalization()/xsea()-facing symptom: previously,
    cleaning against a covariate with zero real relationship to anything
    collapsed mean per-subject colocalization from ~0.8 to ~0."""
    n_subj, n_parcels = 30, 100
    Y_true = rng.normal(size=n_parcels)
    Y = Y_true[np.newaxis, :] + rng.normal(scale=0.5, size=(n_subj, n_parcels))
    irrelevant_cov = rng.normal(size=n_subj)

    parcel_labels = [f"p{i}" for i in range(n_parcels)]
    y_df = pd.DataFrame(Y, index=[f"y{i}" for i in range(n_subj)], columns=parcel_labels)
    x_df = pd.DataFrame(Y_true[np.newaxis, :], index=["x0"], columns=parcel_labels)

    def mean_coloc(Y_arr):
        return np.mean([np.corrcoef(Y_arr[i], Y_true)[0, 1] for i in range(n_subj)])

    coloc_before = mean_coloc(Y)
    assert coloc_before > 0.5  # sanity: the synthetic signal is actually detectable pre-cleaning

    nsp = NiSpace(x=x_df, y=y_df, parcellation=None, standardize=False,
                 n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    nsp.clean_y(how="between", covariates_between=pd.Series(irrelevant_cov, name="cov"),
                verbose=False)
    coloc_after = mean_coloc(nsp.get_y().to_numpy())

    # previously this collapsed to near 0 (>95% drop); an irrelevant covariate should
    # leave colocalization essentially unchanged
    assert coloc_after > 0.9 * coloc_before
