"""Tests for diagnostics.compare_sets(): overlap statistics, observed set-mean-map
similarity (raw, standardized, ranked, weighted, background-mean removed), the
random-pair null, and input handling (DataFrame vs NiSpace, errors)."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom, rankdata
from statsmodels.stats.multitest import multipletests

from nispace import NiSpace
import nispace.diagnostics as diagnostics

N_PARCELS = 40
N_BG = 200


@pytest.fixture
def data(rng):
    """Background of N_BG genes; three sets: A and B overlap by 5, C is a subset of A."""
    cols = [f"p{i}" for i in range(N_PARCELS)]
    bg = pd.DataFrame(rng.normal(size=(N_BG, N_PARCELS)), index=[f"g{i}" for i in range(N_BG)],
                      columns=cols)
    members = {"A": [f"g{i}" for i in range(0, 20)],
               "B": [f"g{i}" for i in range(15, 45)],
               "C": [f"g{i}" for i in range(0, 10)]}
    tuples = [(s, g) for s, genes in members.items() for g in genes]
    X = bg.loc[[g for _, g in tuples]].copy()
    X.index = pd.MultiIndex.from_tuples(tuples, names=["set", "map"])
    return X, bg, members


def _corr(a, b):
    return np.corrcoef(a, b)[0, 1]


def _resid(m, ref):
    A = np.c_[np.ones(len(ref)), ref]
    return m - A @ np.linalg.lstsq(A, m, rcond=None)[0]


def test_overlap_statistics(data):
    X, bg, members = data
    res = diagnostics.compare_sets(X, bg, n_perm=10, seed=0, verbose=False)
    assert list(res.index) == [("A", "B"), ("A", "C"), ("B", "C")]
    row = res.loc[("A", "B")]
    assert (row.n_a, row.n_b, row.n_shared) == (20, 30, 5)
    assert row.jaccard == pytest.approx(5 / 45)
    assert row.overlap_coef == pytest.approx(5 / 20)
    assert row.p_overlap == pytest.approx(hypergeom.sf(4, N_BG, 20, 30))
    assert res.loc[("A", "C")].overlap_coef == 1


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("rank", [False, True])
def test_observed_r_matches_manual(data, standardize, rank):
    X, bg, members = data
    res = diagnostics.compare_sets(X, bg, n_perm=10, standardize=standardize, rank=rank,
                                   seed=0, verbose=False)

    def prep(df):
        a = df.values.astype(float)
        if rank:
            a = np.array([rankdata(r) for r in a])
        if standardize:
            a = (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
        return pd.DataFrame(a, index=df.index)

    bg_p = prep(bg)
    m = {s: bg_p.loc[g].mean().values for s, g in members.items()}
    ref = bg_p.mean().values
    for a, b in res.index:
        assert res.loc[(a, b), "r"] == pytest.approx(_corr(m[a], m[b]), abs=1e-10)
        assert res.loc[(a, b), "r_resid"] == pytest.approx(
            _corr(_resid(m[a], ref), _resid(m[b], ref)), abs=1e-10)


def test_weighted_set_means(data, rng):
    X, bg, members = data
    w = rng.uniform(0.5, 2, size=len(X))
    Xw = X.copy()
    Xw.index = pd.MultiIndex.from_arrays(
        [X.index.get_level_values("set"), X.index.get_level_values("map"), w],
        names=["set", "map", "weight"])
    res = diagnostics.compare_sets(Xw, bg, n_perm=10, standardize=False,
                                   remove_background_mean=False, seed=0, verbose=False)
    assert "r_resid" not in res.columns
    sets = Xw.index.get_level_values("set")
    m = {s: np.average(Xw.values[sets == s], axis=0, weights=w[sets == s]) for s in members}
    assert res.loc[("A", "B"), "r"] == pytest.approx(_corr(m["A"], m["B"]), abs=1e-10)
    # weighted=False ignores the weights
    res_u = diagnostics.compare_sets(Xw, bg, n_perm=10, standardize=False, weighted=False,
                                     seed=0, verbose=False)
    m_u = {s: Xw.values[sets == s].mean(0) for s in members}
    assert res_u.loc[("A", "B"), "r"] == pytest.approx(_corr(m_u["A"], m_u["B"]), abs=1e-10)


def test_null_keeps_overlap(data):
    """Identical sets: with the overlap-matched null, both random sets are always the
    same set (r == 1); with the disjoint null they never are."""
    X, bg, members = data
    tuples = [(s, g) for s in ["A", "A2"] for g in members["A"]]
    Xd = bg.loc[[g for _, g in tuples]].copy()
    Xd.index = pd.MultiIndex.from_tuples(tuples, names=["set", "map"])
    _, nulls = diagnostics.compare_sets(Xd, bg, n_perm=50, null="overlap", seed=0,
                                        return_nulls=True, verbose=False)
    np.testing.assert_allclose(nulls[("A", "A2")]["r"], 1)
    _, nulls = diagnostics.compare_sets(Xd, bg, n_perm=50, null="disjoint", seed=0,
                                        return_nulls=True, verbose=False)
    assert (nulls[("A", "A2")]["r"] < 0.99).all()


@pytest.mark.parametrize("z_method", ["robust", "standard"])
def test_z_vs_null_matches_manual(data, z_method):
    """Same as NiSpace.normalize_colocalizations(): z-score of Fisher-z r vs Fisher-z null."""
    X, bg, _ = data
    res, nulls = diagnostics.compare_sets(X, bg, n_perm=200, z_method=z_method, seed=0,
                                          return_nulls=True, verbose=False)
    for pair in res.index:
        for stat in ["r", "r_resid"]:
            obs, null = np.arctanh(res.loc[pair, stat]), np.arctanh(nulls[pair][stat])
            if z_method == "robust":
                med = np.median(null)
                expected = (obs - med) / (1.4826 * np.median(np.abs(null - med)))
            else:
                expected = (obs - null.mean()) / null.std(ddof=1)
            assert res.loc[pair, f"z_{stat}"] == pytest.approx(expected, rel=1e-6)


def test_null_reproducible_and_independent_of_n_proc(data):
    X, bg, _ = data
    r1 = diagnostics.compare_sets(X, bg, n_perm=100, seed=1, n_proc=1, verbose=False)
    r2 = diagnostics.compare_sets(X, bg, n_perm=100, seed=1, n_proc=2, verbose=False)
    r3 = diagnostics.compare_sets(X, bg, n_perm=100, seed=2, n_proc=1, verbose=False)
    pd.testing.assert_frame_equal(r1, r2)
    assert not r1.r_null_median.equals(r3.r_null_median)


def test_random_sets_not_significant(data, rng):
    """Two disjoint random sets: the observed r lies inside its own null range."""
    _, bg, _ = data
    idx = rng.choice(N_BG, 50, replace=False)
    tuples = [("R1", f"g{i}") for i in idx[:25]] + [("R2", f"g{i}") for i in idx[25:]]
    Xr = bg.loc[[g for _, g in tuples]].copy()
    Xr.index = pd.MultiIndex.from_tuples(tuples, names=["set", "map"])
    res = diagnostics.compare_sets(Xr, bg, n_perm=500, seed=0, verbose=False)
    assert res.p_r.iloc[0] > 0.01
    assert res.r_null_lo.iloc[0] < res.r.iloc[0] < res.r_null_hi.iloc[0]


def test_mc_correction_across_pairs(data):
    """Each p column is corrected across pairs on its own; mc_method=None adds no columns."""
    X, bg, _ = data
    res = diagnostics.compare_sets(X, bg, n_perm=50, mc_method="holm", seed=0, verbose=False)
    p_cols = ["p_overlap", "p_r", "p_r_resid"]
    for p_col in p_cols:
        np.testing.assert_allclose(res[f"{p_col}_corr"],
                                   multipletests(res[p_col], method="holm")[1])
        assert list(res.columns).index(f"{p_col}_corr") == list(res.columns).index(p_col) + 1
    res_none = diagnostics.compare_sets(X, bg, n_perm=50, mc_method=None, seed=0, verbose=False)
    assert not any(c.endswith("_corr") for c in res_none.columns)
    pd.testing.assert_frame_equal(res_none, res.drop(columns=[f"{c}_corr" for c in p_cols]))


def test_nispace_input_matches_dataframe(data):
    X, bg, _ = data
    y = pd.DataFrame(np.ones((1, N_PARCELS)) + np.arange(N_PARCELS), index=["y"],
                     columns=X.columns)
    X_gene = X.copy()
    X_gene.index = X_gene.index.set_names(["set", "gene"])  # NiSpace keeps any level name
    nsp = NiSpace(x=X_gene, y=y, parcellation=None, n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    res_nsp = diagnostics.compare_sets(nsp, bg, n_perm=50, seed=0, verbose=False)
    res_df = diagnostics.compare_sets(X, bg, n_perm=50, seed=0, standardize=True, verbose=False)
    pd.testing.assert_frame_equal(res_nsp, res_df, atol=1e-5)


def test_pairs_subset(data):
    X, bg, _ = data
    res = diagnostics.compare_sets(X, bg, pairs=[("B", "C")], n_perm=10, seed=0, verbose=False)
    assert list(res.index) == [("B", "C")]


@pytest.mark.parametrize("case", ["no_set_level", "background_type", "missing_parcels",
                                  "unknown_pair", "null", "too_large", "mc_method"])
def test_errors(data, case):
    X, bg, _ = data
    kwargs = dict(n_perm=10, verbose=False)
    if case == "no_set_level":
        X = X.droplevel("set")
    elif case == "background_type":
        bg = bg.values
    elif case == "missing_parcels":
        bg = bg.iloc[:, :-1]
    elif case == "unknown_pair":
        kwargs["pairs"] = [("A", "Z")]
    elif case == "null":
        kwargs["null"] = "uniform"
    elif case == "too_large":
        bg = bg.iloc[:40]
        kwargs["null"] = "disjoint"
    elif case == "mc_method":
        kwargs["mc_method"] = "fdr_xyz"
    with pytest.raises((ValueError, TypeError)):
        diagnostics.compare_sets(X, bg, **kwargs)
