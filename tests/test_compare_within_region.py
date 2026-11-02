"""Tests for NiSpace.compare_within_region() / get_within_region_comparisons(): per-parcel group
comparison from group-permutation nulls (shared with permute(what="groups")).

Ground truth: scipy.stats.permutation_test per parcel (independent samples / paired samples),
and correlate_within_region() with a 0/1 group vector (point-biserial r is monotone in Student's
t, as is Hedges' g, so both rank parcels identically).
"""

import logging

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from nispace import NiSpace


N_PARCELS = 12


def _make(rng, n_per_group=12, effect=1.5, n_effect=3, paired=False, return_self=False,
          standardize=False):
    n = 2 * n_per_group
    groups = np.array(["HC"] * n_per_group + ["PAT"] * n_per_group)
    Y = rng.normal(size=(n, N_PARCELS))
    subjects = None
    if paired:
        subjects = np.r_[np.arange(n_per_group), np.arange(n_per_group)].astype(str)
        Y += 2 * rng.normal(size=(n_per_group, N_PARCELS))[
            np.r_[np.arange(n_per_group), np.arange(n_per_group)]]
    Y[groups == "PAT", :n_effect] += effect
    cols = [f"p{i}" for i in range(N_PARCELS)]
    y_df = pd.DataFrame(Y, index=[f"s{i}" for i in range(n)], columns=cols)
    nsp = NiSpace(x=y_df.iloc[:2], y=y_df, parcellation=None, standardize=standardize,
                  n_proc=1, verbose=False, return_self=return_self)
    nsp.fit()
    return nsp, groups, subjects, Y


@pytest.fixture
def unpaired(rng):
    nsp, groups, _, Y = _make(rng)
    nsp.transform_y("hedges(PAT,HC)", groups=groups, verbose=False)
    return nsp, groups, Y


# ── statistic and p-values ────────────────────────────────────────────────────

def test_stat_is_transform_and_p_from_cached_nulls(unpaired):
    nsp, _, _ = unpaired
    stat = nsp.compare_within_region(n_perm=300, seed=1, verbose=False)
    np.testing.assert_allclose(stat.to_numpy(), nsp.get_y(verbose=False).to_numpy(), atol=1e-6)
    res = nsp.get_within_region_comparisons(mc_method=None, verbose=False)
    assert res["stat_type"] == "Hedges' g" and res["p_corr"] is None
    cube = nsp._nulls["groups_null"].data[0].astype(float)
    obs = stat.to_numpy()[0]
    ge = 1 + (cube >= obs).sum(0)
    le = 1 + (cube <= obs).sum(0)
    p_manual = np.minimum(1, 2 * np.minimum(ge, le) / 301)
    np.testing.assert_allclose(res["p"].to_numpy()[0], p_manual, atol=1e-6)


@pytest.mark.skipif(not hasattr(sps, "permutation_test"), reason="scipy >= 1.8 needed")
def test_unpaired_p_matches_scipy_permutation_test(unpaired):
    nsp, groups, Y = unpaired
    nsp.compare_within_region(n_perm=4000, seed=1, verbose=False)
    p = nsp.get_within_region_comparisons(mc_method=None, verbose=False)["p"].to_numpy()[0]
    t = lambda x, y, axis: sps.ttest_ind(x, y, axis=axis).statistic
    ref = sps.permutation_test((Y[groups == "PAT"], Y[groups == "HC"]), t, axis=0,
                               permutation_type="independent", n_resamples=4000,
                               random_state=0).pvalue
    np.testing.assert_allclose(p, ref, atol=0.035)


@pytest.mark.skipif(not hasattr(sps, "permutation_test"), reason="scipy >= 1.8 needed")
def test_paired_p_matches_scipy_permutation_test(rng):
    nsp, groups, subjects, Y = _make(rng, paired=True)
    nsp.transform_y("pairedcohen(PAT,HC)", groups=groups, subjects=subjects, verbose=False)
    nsp.compare_within_region(n_perm=4000, seed=1, verbose=False)
    p = nsp.get_within_region_comparisons(mc_method=None, verbose=False)["p"].to_numpy()[0]

    def dz(x, y, axis):
        d = x - y
        return d.mean(axis=axis) / d.std(axis=axis, ddof=1)
    ref = sps.permutation_test((Y[groups == "PAT"], Y[groups == "HC"]), dz, axis=0,
                               permutation_type="samples", n_resamples=4000,
                               random_state=0).pvalue
    np.testing.assert_allclose(p, ref, atol=0.035)


def test_hedges_ranks_parcels_like_point_biserial_correlation(unpaired):
    nsp, groups, Y = unpaired
    stat = nsp.compare_within_region(n_perm=50, verbose=False).to_numpy()[0]
    rho = nsp.correlate_within_region(X=(groups == "PAT").astype(float), Y=Y, n_perm=0,
                                      store=False, verbose=False).to_numpy()[0]
    assert sps.spearmanr(stat, rho)[0] == pytest.approx(1.0)


@pytest.mark.parametrize("formula", ["zscore(PAT,HC)", "centile(PAT,HC)"])
def test_one_map_per_subject_is_averaged(rng, formula, caplog):
    nsp, groups, _, _ = _make(rng)
    nsp.transform_y(formula, groups=groups, verbose=False)
    with caplog.at_level(logging.INFO, logger="nispace"):
        stat = nsp.compare_within_region(n_perm=100, seed=2, verbose=True)
    np.testing.assert_allclose(stat.to_numpy()[0],
                               nsp.get_y(verbose=False).to_numpy().mean(axis=0), atol=1e-5)
    key = f"ytrans-{formula.lower()}_compwithin"
    null = nsp._nulls["cmp_within"][key]["null_dist"]
    np.testing.assert_allclose(null, nsp._nulls["groups_null"].data.mean(axis=0), atol=1e-5)
    assert "across 12 maps" in caplog.text


@pytest.mark.parametrize("formula", ["elemdiff(PAT,HC)", "prc(PAT,HC)"])
def test_paired_one_map_per_subject_is_averaged(rng, formula):
    nsp, groups, subjects, _ = _make(rng, paired=True)
    nsp._Y = nsp._Y + 20                        # positive ratio scale for prc
    nsp.transform_y(formula, groups=groups, subjects=subjects, verbose=False)
    stat = nsp.compare_within_region(n_perm=100, verbose=False)
    np.testing.assert_allclose(stat.to_numpy()[0],
                               nsp.get_y(verbose=False).to_numpy().mean(axis=0), atol=1e-4)


def test_effect_parcels_are_detected_after_correction(unpaired):
    nsp, _, _ = unpaired
    nsp.compare_within_region(n_perm=2000, seed=1, verbose=False)
    res = nsp.get_within_region_comparisons(verbose=False)
    assert res["mc_method"] == "step_minP"
    p_corr = res["p_corr"].to_numpy()[0]
    assert (p_corr[:3] < 0.05).all()
    assert (p_corr[3:] > 0.05).mean() > 0.8


@pytest.mark.parametrize("mc", ["step_minP", "minP", "step_maxT", "maxT", "fdr_bh", "bonferroni",
                                "holm"])
def test_corrected_p_not_below_raw_p(unpaired, mc):
    nsp, _, _ = unpaired
    nsp.compare_within_region(n_perm=200, verbose=False)
    res = nsp.get_within_region_comparisons(mc_method=mc, verbose=False)
    assert (res["p_corr"].to_numpy() >= res["p"].to_numpy() - 1e-7).all()


def test_meff_is_rejected(unpaired):
    nsp, _, _ = unpaired
    nsp.compare_within_region(n_perm=50, verbose=False)
    with pytest.raises(ValueError, match="meff"):
        nsp.get_within_region_comparisons(mc_method="meff", verbose=False)


def test_nan_parcel_gives_nan_p(rng):
    nsp, groups, _, _ = _make(rng)
    nsp._Y.iloc[:, 4] = np.nan
    nsp.transform_y("hedges(PAT,HC)", groups=groups, verbose=False)
    nsp.compare_within_region(n_perm=100, verbose=False)
    for mc in ["step_minP", "fdr_bh"]:
        res = nsp.get_within_region_comparisons(mc_method=mc, verbose=False)
        assert np.isnan(res["p"].to_numpy()[0, 4]) and np.isnan(res["p_corr"].to_numpy()[0, 4])
        assert not np.isnan(res["p_corr"].to_numpy()[0, :4]).any()


def test_p_tails(unpaired):
    nsp, _, _ = unpaired
    nsp.compare_within_region(n_perm=500, seed=1, p_tails="upper", verbose=False)
    p_up = nsp.get_within_region_comparisons(mc_method=None, verbose=False)["p"].to_numpy()[0]
    nsp.compare_within_region(n_perm=500, seed=1, p_tails="lower", verbose=False)
    p_lo = nsp.get_within_region_comparisons(mc_method=None, verbose=False)["p"].to_numpy()[0]
    assert (p_up[:3] < 0.01).all() and (p_lo[:3] > 0.9).all()
    with pytest.raises(ValueError, match="p_tails"):
        nsp.compare_within_region(n_perm=10, p_tails="both", verbose=False)


# ── invalid inputs ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("formula", ["mean(y)", "y", "zscore(PAT)", "centile(PAT)"])
def test_transforms_without_comparison_raise(rng, formula):
    nsp, groups, _, _ = _make(rng)
    nsp.transform_y(formula, groups=groups if "PAT" in formula else None, verbose=False)
    with pytest.raises(ValueError, match="two groups or sessions"):
        nsp.compare_within_region(n_perm=10, verbose=False)


def test_no_transform_raises(rng):
    nsp, _, _, _ = _make(rng)
    with pytest.raises(ValueError, match="No Y transform"):
        nsp.compare_within_region(n_perm=10, verbose=False)


def test_binary_y_raises(rng):
    Y = (rng.normal(size=(10, N_PARCELS)) > 0).astype(float)
    y_df = pd.DataFrame(Y, columns=[f"p{i}" for i in range(N_PARCELS)])
    nsp = NiSpace(x=y_df.iloc[:2], y=y_df, parcellation=None, standardize=False, n_proc=1,
                  verbose=False, return_self=False, binary_y=True)
    nsp.fit()
    with pytest.raises(ValueError, match="binary"):
        nsp.compare_within_region(n_perm=10, verbose=False)


def test_y_changed_after_transform_raises(unpaired):
    nsp, _, _ = unpaired
    nsp.clean_y("between", covariates_between=pd.DataFrame({"age": np.arange(24.0)}))
    with pytest.raises(ValueError, match="Y changed after transform_y"):
        nsp.compare_within_region(n_perm=10, verbose=False)


def test_standardize_y_warns(rng, caplog):
    nsp, groups, _, _ = _make(rng, standardize="y")
    nsp.transform_y("hedges(PAT,HC)", groups=groups, verbose=False)
    with caplog.at_level(logging.WARNING, logger="nispace"):
        nsp.compare_within_region(n_perm=10, verbose=False)
    assert "z-scored per map" in caplog.text


def test_unknown_kwarg_raises(unpaired):
    nsp, _, _ = unpaired
    with pytest.raises(TypeError, match="Unknown"):
        nsp.compare_within_region(n_perm=10, mc_method="fdr_bh", verbose=False)


# ── sharing nulls with permute(what="groups") ─────────────────────────────────

def test_compare_reuses_nulls_of_permute(unpaired, caplog):
    nsp, _, _ = unpaired
    nsp.colocalize("pearson", verbose=False)
    nsp.permute("groups", n_perm=300, verbose=False)
    cached = nsp._nulls["groups_null"]
    with caplog.at_level(logging.INFO, logger="nispace"):
        nsp.compare_within_region(verbose=True)          # n_perm=None -> cached 300
    assert nsp._nulls["groups_null"] is cached
    assert "Using cached group permutation null maps (300 permutations)" in caplog.text
    assert nsp._nulls["cmp_within"]["ytrans-hedges(pat,hc)_compwithin"]["n_perm"] == 300


def test_permute_reuses_nulls_of_compare(unpaired):
    nsp, _, _ = unpaired
    nsp.compare_within_region(n_perm=300, verbose=False)
    cached = nsp._nulls["groups_null"]
    nsp.colocalize("pearson", verbose=False)
    nsp.permute("groups", n_perm=300, verbose=False)
    assert nsp._nulls["groups_null"] is cached


def test_cache_n_perm_subset_and_regeneration(unpaired):
    nsp, _, _ = unpaired
    nsp.compare_within_region(n_perm=300, seed=1, verbose=False)
    cube = nsp._nulls["groups_null"].data.copy()
    nsp.compare_within_region(n_perm=100, verbose=False)
    key = "ytrans-hedges(pat,hc)_compwithin"
    np.testing.assert_allclose(nsp._nulls["cmp_within"][key]["null_dist"], cube[0, :100])
    nsp.compare_within_region(n_perm=500, verbose=False)
    assert nsp._nulls["groups_null"].n_perm == 500


def test_cache_seed_and_strategy(unpaired):
    nsp, _, _ = unpaired
    nsp.compare_within_region(n_perm=100, seed=1, verbose=False)
    cached = nsp._nulls["groups_null"]
    nsp.compare_within_region(n_perm=100, verbose=False)              # any seed -> reuse
    assert nsp._nulls["groups_null"] is cached
    nsp.compare_within_region(n_perm=100, seed=1, verbose=False)      # same seed -> reuse
    assert nsp._nulls["groups_null"] is cached
    nsp.compare_within_region(n_perm=100, seed=2, verbose=False)      # other seed -> new
    assert nsp._nulls["groups_null"] is not cached
    cached = nsp._nulls["groups_null"]
    nsp.compare_within_region(n_perm=100, groups_strategy="proportional", verbose=False)
    assert nsp._nulls["groups_null"] is not cached


def test_memmap_path_gives_same_results(unpaired, tmp_path):
    nsp, _, _ = unpaired
    nsp.compare_within_region(n_perm=100, seed=1, verbose=False)
    p_ram = nsp.get_within_region_comparisons(verbose=False)["p_corr"].to_numpy()
    nsp.compare_within_region(n_perm=100, seed=3, verbose=False)      # other cache
    nsp.compare_within_region(n_perm=100, seed=1, memmap_path=tmp_path / "nulls.npy",
                              verbose=False)
    assert nsp._nulls["groups_null"].is_memmap
    p_mm = nsp.get_within_region_comparisons(verbose=False)["p_corr"].to_numpy()
    np.testing.assert_allclose(p_mm, p_ram)


def test_covariates_use_freedman_lane(rng):
    nsp, groups, _, _ = _make(rng)
    age = (groups == "PAT") + rng.normal(size=len(groups))
    nsp.clean_y("between", covariates_between=pd.DataFrame({"age": age}),
                protect=pd.DataFrame({"groups": groups}))
    nsp.transform_y("hedges(PAT,HC)", groups=groups, verbose=False)
    nsp.compare_within_region(n_perm=200, seed=1, verbose=False)
    assert nsp._nulls["groups_null_info"]["scheme"] == "freedman-lane"


# ── storage and accessor ──────────────────────────────────────────────────────

def test_return_self_and_store(rng):
    nsp, groups, _, _ = _make(rng, return_self=True)
    nsp.transform_y("hedges(PAT,HC)", groups=groups, verbose=False)
    assert nsp.compare_within_region(n_perm=20, verbose=False) is nsp
    nsp2, groups, _, _ = _make(rng)
    nsp2.transform_y("hedges(PAT,HC)", groups=groups, verbose=False)
    out = nsp2.compare_within_region(n_perm=20, store=False, verbose=False)
    assert isinstance(out, pd.DataFrame)
    with pytest.raises(KeyError, match="No compare_within_region"):
        nsp2.get_within_region_comparisons(verbose=False)


def test_accessor_defaults_to_last_compared_transform(unpaired):
    nsp, groups, _ = unpaired
    nsp.compare_within_region(n_perm=20, verbose=False)
    nsp.transform_y("cohen(PAT,HC)", groups=groups, verbose=False)   # not compared
    res = nsp.get_within_region_comparisons(verbose=False)
    assert res["stat"].index[0] == "hedges(pat,hc)"
    with pytest.raises(KeyError, match="Available"):
        nsp.get_within_region_comparisons(Y_transform="cohen(PAT,HC)", verbose=False)


def test_pickle_round_trip(unpaired, tmp_path):
    nsp, _, _ = unpaired
    nsp.compare_within_region(n_perm=100, verbose=False)
    res = nsp.get_within_region_comparisons(verbose=False)
    nsp.to_pickle(tmp_path / "nsp.pkl", verbose=False)
    nsp2 = NiSpace.from_pickle(tmp_path / "nsp.pkl", verbose=False)
    res2 = nsp2.get_within_region_comparisons(verbose=False)
    pd.testing.assert_frame_equal(res["p_corr"], res2["p_corr"])
