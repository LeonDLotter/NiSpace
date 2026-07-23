"""Tests for NiSpace.correlate_within_region() / get_within_region_correlations()
and its core/correlate_within_region.py backend (correlate_within_region_core):
per-parcel, across-subject correlation with a subject-permutation null.

Ground truth is scipy.stats.pearsonr/spearmanr, computed independently per
parcel column -- not through nispace's own colocalize()/pearson(), which
correlate across the opposite (parcel) axis.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from nispace import NiSpace
from nispace.core.correlate_within_region import correlate_within_region_core


def make_cwr_data(rng, n_subj=12, n_parcels=8, signal_parcel=0, signal_r=0.95):
    """X, Y: (n_subj, n_parcels), with a strong injected true correlation at
    `signal_parcel` and independent noise elsewhere."""
    X = rng.normal(size=(n_subj, n_parcels))
    Y = rng.normal(size=(n_subj, n_parcels))
    # construct Y[:, signal_parcel] to have a known, strong correlation with X[:, signal_parcel]
    noise_scale = np.sqrt(1 - signal_r**2) / signal_r if signal_r < 1 else 0.05
    Y[:, signal_parcel] = X[:, signal_parcel] + rng.normal(scale=noise_scale, size=n_subj)
    return X, Y


@pytest.fixture
def cwr_nsp(rng):
    n_subj, n_parcels = 12, 8
    X, Y = make_cwr_data(rng, n_subj=n_subj, n_parcels=n_parcels)
    subj_labels = [f"s{i}" for i in range(n_subj)]
    parcel_labels = [f"p{i}" for i in range(n_parcels)]
    x_df = pd.DataFrame(X, index=subj_labels, columns=parcel_labels)
    y_df = pd.DataFrame(Y, index=subj_labels, columns=parcel_labels)
    nsp = NiSpace(x=x_df, y=y_df, parcellation=None, standardize=False,
                  n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    return nsp, X, Y


# ── core engine: ground truth ───────────────────────────────────────────────

def test_core_pearson_matches_scipy(rng):
    X, Y = make_cwr_data(rng)
    rho, null = correlate_within_region_core(X, Y, method="pearson", n_perm=100, seed=1)
    ref = np.array([sps.pearsonr(X[:, p], Y[:, p])[0] for p in range(X.shape[1])])
    np.testing.assert_allclose(rho, ref, atol=1e-8)
    assert null.shape == (100, X.shape[1])


def test_core_spearman_matches_scipy(rng):
    X, Y = make_cwr_data(rng)
    rho, _ = correlate_within_region_core(X, Y, method="spearman", n_perm=0)
    ref = np.array([sps.spearmanr(X[:, p], Y[:, p])[0] for p in range(X.shape[1])])
    np.testing.assert_allclose(rho, ref, atol=1e-8)


def test_core_nan_column_matches_scipy_after_masking(rng):
    X, Y = make_cwr_data(rng, n_subj=15, n_parcels=6)
    X[0, 2] = np.nan
    Y[3, 2] = np.nan
    rho, null = correlate_within_region_core(X, Y, method="pearson", n_perm=20, seed=2)
    mask = ~np.isnan(X[:, 2]) & ~np.isnan(Y[:, 2])
    ref = sps.pearsonr(X[mask, 2], Y[mask, 2])[0]
    assert np.isclose(rho[2], ref, atol=1e-8)
    assert null.shape == (20, X.shape[1])


def test_core_spearman_mismatched_nan_pattern_matches_pairwise_deletion(rng):
    # regression test: ranking each side's column in isolation (against its own
    # NaN pattern) BEFORE intersecting with the other side is wrong whenever the
    # two NaN patterns differ -- it leaves a numeric gap in the surviving ranks
    # instead of a fresh 1..k ranking of just the jointly-valid subset. Correct
    # (pandas-style pairwise-deletion) behavior: mask incomplete pairs first,
    # THEN rank only what's left.
    n_subj, n_parcels = 10, 4
    X, Y = make_cwr_data(rng, n_subj=n_subj, n_parcels=n_parcels, signal_parcel=0, signal_r=0.9)
    X[1, 2] = np.nan  # X-only NaN
    Y[4, 2] = np.nan  # Y-only NaN, different subject, same parcel -> mismatched pattern
    Y[0, 0] = np.nan  # X has no NaN in parcel 0 at all -> tests the "one side clean" case too

    rho, _ = correlate_within_region_core(X, Y, method="spearman", n_perm=0)
    for p in range(n_parcels):
        mask = ~np.isnan(X[:, p]) & ~np.isnan(Y[:, p])
        ref = sps.spearmanr(X[mask, p], Y[mask, p])[0]
        assert np.isclose(rho[p], ref, atol=1e-8), f"parcel {p}: got {rho[p]}, expected {ref}"


def test_core_injected_signal_is_strong(rng):
    X, Y = make_cwr_data(rng, signal_parcel=0, signal_r=0.95)
    rho, _ = correlate_within_region_core(X, Y, method="pearson", n_perm=0)
    assert rho[0] > 0.85
    assert np.all(np.abs(rho[1:]) < 0.85)


# ── core engine: broadcast / symmetry ───────────────────────────────────────

def test_core_1d_y_broadcast_matches_manual_loop(rng):
    n_subj, n_parcels = 15, 8
    X = rng.normal(size=(n_subj, n_parcels))
    yvec = rng.normal(size=n_subj)
    X[:, 3] = yvec * 2 + rng.normal(scale=0.1, size=n_subj)
    rho, null = correlate_within_region_core(X, yvec, method="pearson", n_perm=50, seed=3)
    ref = np.array([sps.pearsonr(X[:, p], yvec)[0] for p in range(n_parcels)])
    np.testing.assert_allclose(rho, ref, atol=1e-8)
    assert null.shape == (50, n_parcels)


def test_core_column_shaped_y_matches_1d_broadcast(rng):
    # (n_subjects, 1) must behave exactly like (n_subjects,) -- both in the
    # vectorized no-NaN path and the NaN-aware loop fallback, which used to
    # IndexError on a (n_subjects, 1) side since it assumed matching column
    # counts on both inputs
    n_subj, n_parcels = 15, 8
    X = rng.normal(size=(n_subj, n_parcels))
    yvec = rng.normal(size=n_subj)

    rho_1d, null_1d = correlate_within_region_core(X, yvec, method="pearson", n_perm=30, seed=4)
    rho_col, null_col = correlate_within_region_core(X, yvec[:, None], method="pearson",
                                                      n_perm=30, seed=4)
    np.testing.assert_allclose(rho_1d, rho_col, atol=1e-10)
    np.testing.assert_allclose(null_1d, null_col, atol=1e-10)

    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    rho_nan_1d, _ = correlate_within_region_core(X_nan, yvec, method="pearson", n_perm=0)
    rho_nan_col, _ = correlate_within_region_core(X_nan, yvec[:, None], method="pearson", n_perm=0)
    np.testing.assert_allclose(rho_nan_1d, rho_nan_col, atol=1e-10)


def test_core_x_vector_and_y_vector_are_symmetric(rng):
    n_subj, n_parcels = 15, 8
    X = rng.normal(size=(n_subj, n_parcels))
    yvec = rng.normal(size=n_subj)
    rho_a, _ = correlate_within_region_core(X, yvec, method="pearson", n_perm=0)
    rho_b, _ = correlate_within_region_core(yvec, X, method="pearson", n_perm=0)
    np.testing.assert_allclose(rho_a, rho_b, atol=1e-10)


# ── core engine: error paths ────────────────────────────────────────────────

def test_core_both_1d_raises(rng):
    yvec = rng.normal(size=10)
    with pytest.raises(ValueError):
        correlate_within_region_core(yvec, yvec, method="pearson", n_perm=0)


def test_core_mismatched_subject_count_raises(rng):
    X = rng.normal(size=(10, 5))
    Y = rng.normal(size=(12, 5))
    with pytest.raises(ValueError):
        correlate_within_region_core(X, Y, method="pearson", n_perm=0)


def test_core_bad_method_raises(rng):
    X, Y = make_cwr_data(rng)
    with pytest.raises(ValueError):
        correlate_within_region_core(X, Y, method="bogus", n_perm=0)


# ── NiSpace method: ground truth + p-values ─────────────────────────────────

def test_nsp_matches_scipy_full_map_vs_full_map(cwr_nsp):
    nsp, X, Y = cwr_nsp
    nsp.correlate_within_region(method="pearson", n_perm=0)
    rho_df = nsp.get_within_region_correlations(mc_method=None)["stat"]
    ref = np.array([sps.pearsonr(X[:, p], Y[:, p])[0] for p in range(X.shape[1])])
    np.testing.assert_allclose(rho_df.values[0], ref, atol=1e-6)


def test_nsp_no_null_returns_none_p(cwr_nsp):
    nsp, X, Y = cwr_nsp
    nsp.correlate_within_region(method="pearson", n_perm=0)
    out = nsp.get_within_region_correlations(mc_method=None)
    assert out["p"] is None
    assert out["p_corr"] is None
    assert out["stat_type"] == "rho"
    assert out["mc_method"] is None


def test_nsp_default_mc_method_requires_null(cwr_nsp):
    # mc_method defaults to "step_maxT" now -- with no null computed, that default
    # must still raise (not silently fall back to uncorrected results), same as an
    # explicit mc_method="step_maxT" request would
    nsp, X, Y = cwr_nsp
    nsp.correlate_within_region(method="pearson", n_perm=0)
    with pytest.raises(KeyError):
        nsp.get_within_region_correlations()


def test_nsp_default_mc_method_is_step_maxT(cwr_nsp):
    nsp, X, Y = cwr_nsp
    nsp.correlate_within_region(method="pearson", n_perm=200, seed=1)
    out = nsp.get_within_region_correlations()
    assert out["mc_method"] == "step_maxT"
    assert out["p_corr"] is not None


def test_nsp_p_values_in_range_and_signal_is_significant(cwr_nsp):
    nsp, X, Y = cwr_nsp
    nsp.correlate_within_region(method="pearson", n_perm=500, seed=1)
    out = nsp.get_within_region_correlations()
    p = out["p"].values[0]
    assert ((p >= 0) & (p <= 1)).all()
    assert p[0] < 0.05  # injected signal parcel
    assert (p[1:] > 0.05).sum() >= (len(p) - 1) * 0.5  # most non-signal parcels not significant


@pytest.mark.parametrize("mc_method", ["fdr_bh", "bonferroni", "maxT", "step_maxT"])
def test_nsp_corrections_run_and_stay_in_range(cwr_nsp, mc_method):
    nsp, X, Y = cwr_nsp
    nsp.correlate_within_region(method="pearson", n_perm=200, seed=1)
    out = nsp.get_within_region_correlations(mc_method=mc_method)
    assert out["mc_method"] == mc_method
    p_corr = out["p_corr"].values[0]
    assert ((p_corr >= 0) & (p_corr <= 1)).all()
    # corrected p should never be more liberal (smaller) than the uncorrected p
    assert (p_corr >= out["p"].values[0] - 1e-8).all()


def test_nsp_maxT_requires_null(cwr_nsp):
    nsp, X, Y = cwr_nsp
    nsp.correlate_within_region(method="pearson", n_perm=0)
    with pytest.raises(KeyError):
        nsp.get_within_region_correlations(mc_method="maxT")


def test_nsp_meff_correction_rejected(cwr_nsp):
    """meff is deliberately unsupported here (anti-conservative at n_subjects << n_parcels,
    see bench5-1_region_correlation_fpr.ipynb) -- must raise, not silently compute it."""
    nsp, X, Y = cwr_nsp
    nsp.correlate_within_region(method="pearson", n_perm=200, seed=1)
    with pytest.raises(ValueError):
        nsp.get_within_region_correlations(mc_method="meff")


# ── NiSpace method: fully-missing parcel -> NaN p, not a fake-significant one ──

@pytest.fixture
def cwr_nsp_missing_parcel(rng):
    # regression fixture: one parcel is NaN for every subject in Y, so rho is
    # undefined there. A naive '|null| >= |obs|' comparison silently treats
    # 'NaN >= NaN' as False, so the fraction collapses to 0.0 and gets floor-
    # clipped to a fake, tiny "significant" p (e.g. 0.001 at n_perm=1000) instead
    # of staying NaN -- and for plain maxT, that single NaN column used to poison
    # every OTHER parcel's corrected p too (np.max propagates NaN across the row).
    n_subj, n_parcels = 10, 5
    X, Y = make_cwr_data(rng, n_subj=n_subj, n_parcels=n_parcels, signal_parcel=4, signal_r=0.9)
    Y[:, 0] = np.nan
    x_df = pd.DataFrame(X, columns=[f"p{i}" for i in range(n_parcels)])
    y_df = pd.DataFrame(Y, columns=[f"p{i}" for i in range(n_parcels)])
    nsp = NiSpace(x=x_df, y=y_df, parcellation=None, standardize=False,
                  n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    nsp.correlate_within_region(method="pearson", n_perm=1000, seed=1)
    return nsp


def test_nsp_missing_parcel_rho_and_raw_p_are_nan(cwr_nsp_missing_parcel):
    out = cwr_nsp_missing_parcel.get_within_region_correlations(mc_method=None)
    assert np.isnan(out["stat"].values[0, 0])
    assert np.isnan(out["p"].values[0, 0])
    assert not np.isnan(out["stat"].values[0, 1:]).any()
    assert not np.isnan(out["p"].values[0, 1:]).any()


@pytest.mark.parametrize("mc_method", ["fdr_bh", "bonferroni", "maxT", "step_maxT"])
def test_nsp_missing_parcel_p_corr_is_nan_only_for_that_parcel(cwr_nsp_missing_parcel, mc_method):
    out = cwr_nsp_missing_parcel.get_within_region_correlations(mc_method=mc_method)
    p_corr = out["p_corr"].values[0]
    assert np.isnan(p_corr[0])
    assert not np.isnan(p_corr[1:]).any()
    assert ((p_corr[1:] >= 0) & (p_corr[1:] <= 1)).all()
    # the real signal parcel (index 4) must still be recovered as significant --
    # this is what catches the old maxT bug, where the missing column poisoned
    # np.max for every permutation and made p_corr==1/n_perm for ALL parcels
    assert p_corr[4] < 0.05


def test_nsp_missing_parcel_maxT_not_globally_poisoned(cwr_nsp_missing_parcel):
    # a non-signal, non-missing parcel should NOT be artificially significant
    out = cwr_nsp_missing_parcel.get_within_region_correlations(mc_method="maxT")
    p_corr = out["p_corr"].values[0]
    # parcels 1-3 are independent noise; at minimum they shouldn't all be
    # floor-clipped to 1/n_perm the way the old bug forced every parcel to be
    assert not np.allclose(p_corr[1:4], 1.0 / 1000, atol=1e-9)


def test_nsp_missing_parcel_omnibus_excludes_missing_not_nan(cwr_nsp_missing_parcel):
    out = cwr_nsp_missing_parcel.get_within_region_correlations_omnibus(omnibus_stat="absrho")
    assert not np.isnan(out["stat"])
    assert not np.isnan(out["p"])
    rho = cwr_nsp_missing_parcel.get_within_region_correlations(mc_method=None)["stat"].values[0]
    assert np.isclose(out["stat"], np.nanmean(np.abs(rho)))


# ── NiSpace method: omnibus test ────────────────────────────────────────────

@pytest.mark.parametrize("omnibus_stat", ["rho", "absrho", "rho2"])
def test_nsp_omnibus_p_in_range(cwr_nsp, omnibus_stat):
    nsp, X, Y = cwr_nsp
    nsp.correlate_within_region(method="pearson", n_perm=500, seed=1)
    out = nsp.get_within_region_correlations_omnibus(omnibus_stat=omnibus_stat)
    assert out["stat_type"] == omnibus_stat
    assert 0 <= out["p"] <= 1


@pytest.mark.parametrize("omnibus_stat", ["absrho", "rho2"])
def test_nsp_omnibus_signal_is_significant_for_unsigned_stats(cwr_nsp, omnibus_stat):
    # only "absrho"/"rho2" are guaranteed power here: the fixture's single signal
    # parcel among 7 independent-noise parcels isn't enough for "rho" (signed mean)
    # to reliably detect, since the noise parcels' signs aren't controlled -- exactly
    # the sign-heterogeneity tradeoff documented in the omnibus_stat docstring.
    nsp, X, Y = cwr_nsp
    nsp.correlate_within_region(method="pearson", n_perm=500, seed=1)
    out = nsp.get_within_region_correlations_omnibus(omnibus_stat=omnibus_stat)
    assert out["p"] < 0.1


def test_nsp_omnibus_matches_manual_aggregate(cwr_nsp):
    nsp, X, Y = cwr_nsp
    nsp.correlate_within_region(method="pearson", n_perm=200, seed=1)
    rho = nsp.get_within_region_correlations()["stat"].values[0]
    out = nsp.get_within_region_correlations_omnibus(omnibus_stat="absrho")
    assert np.isclose(out["stat"], np.mean(np.abs(rho)))


def test_nsp_omnibus_requires_null(cwr_nsp):
    nsp, X, Y = cwr_nsp
    nsp.correlate_within_region(method="pearson", n_perm=0)
    with pytest.raises(KeyError):
        nsp.get_within_region_correlations_omnibus()


def test_nsp_omnibus_bad_stat_raises(cwr_nsp):
    nsp, X, Y = cwr_nsp
    nsp.correlate_within_region(method="pearson", n_perm=200, seed=1)
    with pytest.raises(ValueError):
        nsp.get_within_region_correlations_omnibus(omnibus_stat="bogus")


# ── NiSpace method: 1D covariate broadcast + symmetry ───────────────────────

def test_nsp_1d_y_covariate_matches_manual_loop(cwr_nsp):
    nsp, X, Y = cwr_nsp
    rng = np.random.default_rng(7)
    yvec = pd.Series(rng.normal(size=X.shape[0]), index=[f"s{i}" for i in range(X.shape[0])])
    nsp.correlate_within_region(Y=yvec, n_perm=0)
    rho_df = nsp.get_within_region_correlations(mc_method=None)["stat"]
    ref = np.array([sps.pearsonr(X[:, p], yvec.values)[0] for p in range(X.shape[1])])
    np.testing.assert_allclose(rho_df.values[0], ref, atol=1e-6)


def test_nsp_x_vector_and_y_vector_symmetric_same_data(cwr_nsp):
    nsp, X, Y = cwr_nsp
    rng = np.random.default_rng(8)
    subj_labels = [f"s{i}" for i in range(X.shape[0])]
    parcel_labels = [f"p{i}" for i in range(X.shape[1])]
    x_df = pd.DataFrame(X, index=subj_labels, columns=parcel_labels)
    yvec = pd.Series(rng.normal(size=X.shape[0]), index=subj_labels)

    nsp.correlate_within_region(X=x_df, Y=yvec, n_perm=0)
    rho_a = nsp.get_within_region_correlations(mc_method=None)["stat"]

    nsp.correlate_within_region(X=yvec, Y=x_df, n_perm=0)
    rho_b = nsp.get_within_region_correlations(mc_method=None)["stat"]

    np.testing.assert_allclose(rho_a.values, rho_b.values, atol=1e-10)


# ── NiSpace method: error paths ─────────────────────────────────────────────

def test_nsp_both_1d_raises(cwr_nsp):
    nsp, X, Y = cwr_nsp
    rng = np.random.default_rng(9)
    yvec = rng.normal(size=X.shape[0])
    with pytest.raises(ValueError):
        nsp.correlate_within_region(X=yvec, Y=yvec, n_perm=0)


def test_nsp_mismatched_subject_count_raises(cwr_nsp):
    nsp, X, Y = cwr_nsp
    rng = np.random.default_rng(10)
    short_vec = rng.normal(size=X.shape[0] - 2)
    with pytest.raises(ValueError):
        nsp.correlate_within_region(X=short_vec, n_perm=0)


def test_nsp_bad_method_raises(cwr_nsp):
    nsp, X, Y = cwr_nsp
    with pytest.raises(ValueError):
        nsp.correlate_within_region(method="bogus", n_perm=0)


def test_nsp_mismatched_labels_warns_not_raises(cwr_nsp, caplog):
    nsp, X, Y = cwr_nsp
    parcel_labels = [f"p{i}" for i in range(X.shape[1])]
    x_df = pd.DataFrame(X, index=[f"other{i}" for i in range(X.shape[0])], columns=parcel_labels)
    y_df = pd.DataFrame(Y, index=[f"s{i}" for i in range(X.shape[0])], columns=parcel_labels)
    with caplog.at_level("WARNING"):
        nsp.correlate_within_region(X=x_df, Y=y_df, n_perm=0)
    assert any("do not match" in r.message for r in caplog.records)


# ── NiSpace method: standardize warning ─────────────────────────────────────

def test_nsp_standardize_warns_when_using_stored_data(caplog):
    rng = np.random.default_rng(12)
    X, Y = make_cwr_data(rng)
    parcel_labels = [f"p{i}" for i in range(X.shape[1])]
    subj_labels = [f"s{i}" for i in range(X.shape[0])]
    x_df = pd.DataFrame(X, index=subj_labels, columns=parcel_labels)
    y_df = pd.DataFrame(Y, index=subj_labels, columns=parcel_labels)
    nsp = NiSpace(x=x_df, y=y_df, parcellation=None, standardize="xy",
                  n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    with caplog.at_level("WARNING"):
        nsp.correlate_within_region(method="pearson", n_perm=0)
    assert any("Z-standardized" in r.message and "X was" in r.message for r in caplog.records)
    assert any("Z-standardized" in r.message and "Y was" in r.message for r in caplog.records)


def test_nsp_standardize_no_warning_when_x_and_y_overridden(caplog):
    rng = np.random.default_rng(13)
    X, Y = make_cwr_data(rng)
    parcel_labels = [f"p{i}" for i in range(X.shape[1])]
    subj_labels = [f"s{i}" for i in range(X.shape[0])]
    x_df = pd.DataFrame(X, index=subj_labels, columns=parcel_labels)
    y_df = pd.DataFrame(Y, index=subj_labels, columns=parcel_labels)
    nsp = NiSpace(x=x_df, y=y_df, parcellation=None, standardize="xy",
                  n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    with caplog.at_level("WARNING"):
        # raw override bypasses the object's standardize pipeline entirely
        nsp.correlate_within_region(X=x_df, Y=y_df, method="pearson", n_perm=0)
    assert not any("Z-standardized" in r.message for r in caplog.records)


# ── NiSpace method: store / return_self ─────────────────────────────────────

def test_nsp_return_self_false_returns_dataframe():
    rng = np.random.default_rng(11)
    X, Y = make_cwr_data(rng)
    parcel_labels = [f"p{i}" for i in range(X.shape[1])]
    subj_labels = [f"s{i}" for i in range(X.shape[0])]
    x_df = pd.DataFrame(X, index=subj_labels, columns=parcel_labels)
    y_df = pd.DataFrame(Y, index=subj_labels, columns=parcel_labels)
    nsp = NiSpace(x=x_df, y=y_df, parcellation=None, standardize=False,
                  n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    out = nsp.correlate_within_region(method="pearson", n_perm=0)
    assert isinstance(out, pd.DataFrame)


def test_nsp_return_self_true_returns_nispace(cwr_nsp):
    nsp, X, Y = cwr_nsp
    nsp._return_self = True
    out = nsp.correlate_within_region(method="pearson", n_perm=0)
    assert isinstance(out, NiSpace)
