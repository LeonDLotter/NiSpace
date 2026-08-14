"""Tests for NiSpace.colocalize() itself (not regional_influence()).

Regression tests for colocalize()-level bugs discovered while building
regional_influence()/regional_contribution():
1. `rank` staleness -- must default to a function of *this* call's method,
   not leak in from whatever a previous, possibly different-method call
   last used.
2. `partialspearman` with Z didn't rank Z, silently computing something
   between partialpearson and the standard partial-Spearman definition.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from nispace import NiSpace
from nispace.core.colocalize import _get_colocalize_fun, _xsea_aggregate


def test_rank_does_not_leak_across_methods(synthetic_nispace):
    """colocalize(method="pearson") run after colocalize(method="spearman")
    must compute true Pearson, not silently inherit rank=True and compute
    Spearman's correlation under the "pearson" label."""
    nsp = synthetic_nispace

    nsp.colocalize(method="pearson", verbose=False)
    pearson_rho = nsp.get_colocalizations(method="pearson").copy()

    nsp.colocalize(method="spearman", verbose=False)
    spearman_rho = nsp.get_colocalizations(method="spearman")

    # re-run pearson after spearman -- must reproduce the original pearson result,
    # not spearman's
    nsp.colocalize(method="pearson", verbose=False)
    pearson_rho_again = nsp.get_colocalizations(method="pearson")

    np.testing.assert_allclose(pearson_rho.to_numpy(), pearson_rho_again.to_numpy())
    assert not np.allclose(pearson_rho.to_numpy(), spearman_rho.to_numpy())


def test_explicit_rank_override_does_not_leak_to_other_methods(synthetic_nispace):
    """An explicit rank=True for one method (e.g. ranked mlr) must not leak into
    a later method that doesn't request it, and must itself survive being
    sandwiched between other calls."""
    nsp = synthetic_nispace

    nsp.colocalize(method="mlr", rank=True, verbose=False)
    assert nsp._coloc_kwargs_by_method["mlr"]["rank"] is True

    nsp.colocalize(method="spearman", verbose=False)
    assert nsp._coloc_kwargs_by_method["spearman"]["rank"] is True

    nsp.colocalize(method="pearson", verbose=False)
    assert nsp._coloc_kwargs_by_method["pearson"]["rank"] is False

    # mlr's own explicit setting must be unaffected by the later calls
    assert nsp._coloc_kwargs_by_method["mlr"]["rank"] is True


def test_partialspearman_ranks_z_too(rng):
    """partialspearman must match the standard definition of partial Spearman
    correlation: rank X, Y, AND Z, then partial-correlate the ranks -- not
    rank only X/Y while leaving Z on its raw scale (which gives a materially
    different number whenever Z isn't already rank-equivalent to raw, e.g.
    a skewed Z)."""
    n_parcels = 200
    x = rng.normal(size=n_parcels)
    z = rng.exponential(size=n_parcels)  # skewed on purpose
    y = 0.5 * x + 0.6 * z + rng.normal(scale=0.8, size=n_parcels)

    parcel_labels = [f"p{i}" for i in range(n_parcels)]
    x_df = pd.DataFrame(x[np.newaxis, :], index=["x0"], columns=parcel_labels)
    y_df = pd.DataFrame(y[np.newaxis, :], index=["y0"], columns=parcel_labels)
    z_df = pd.DataFrame(z[np.newaxis, :], index=["z0"], columns=parcel_labels)

    nsp = NiSpace(x=x_df, y=y_df, z=z_df, parcellation=None, standardize=False,
                 n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    nsp.colocalize(method="partialspearman", verbose=False)
    got_z = nsp.get_colocalizations(method="partialspearman").iloc[0, 0]
    got_rho = np.tanh(got_z)  # colocalize() stores the Fisher-z-transformed rho by default

    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    standard_partial_spearman = (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))

    np.testing.assert_allclose(got_rho, standard_partial_spearman, atol=1e-3)


# ── XSEA gene-to-set aggregation must never average raw rho (2026-07-29) ──────
#
# NOTE: through the public NiSpace.colocalize() API this scenario cannot actually
# arise -- colocalize() forces r_to_z=True (with a warning) whenever xsea=True and
# method is pearson/spearman-family (see api.py's xsea-handling block), so
# _get_colocalize_fun()/_xsea_aggregate() are never reached with xsea=True,
# r_to_z=False for a real user. These tests call the low-level functions directly
# (bypassing that guard) to verify the fix at its source, matching this file's
# convention (test_regional_influence.py does the same for _get_colocalize_fun).

def _make_xsea_gene_data(rng, n_genes=6, n_parcels=40, signal_r=0.9):
    """One set of genes, each independently correlated with y at a known-ish
    strength -- large enough |rho| that raw vs Fisher-z averaging visibly differ."""
    y = rng.normal(size=n_parcels)
    noise_scale = np.sqrt(1 - signal_r**2) / signal_r
    X = np.stack([
        y + rng.normal(scale=noise_scale, size=n_parcels) * (1 + 0.3 * i)
        for i in range(n_genes)
    ])
    return X, y


def test_xsea_live_path_mean_matches_fisher_z_average(rng):
    """_get_colocalize_fun(xsea=True, xsea_method="mean", r_to_z=False)'s aggregated
    per-set "rho" must equal tanh(mean(arctanh(per-gene rho))), not a raw mean."""
    X, y = _make_xsea_gene_data(rng)

    y_coloc_raw = _get_colocalize_fun(
        "pearson", xsea=True, xsea_method="mean", r_to_z=False, dtype=np.float32,
    )
    out = y_coloc_raw({"setA": X}, y)
    per_set_rho = float(out["rho"][0])

    y_coloc_per_gene = _get_colocalize_fun("pearson", r_to_z=False, dtype=np.float32)
    per_gene_rho = np.array([y_coloc_per_gene(X[[i]], y)["rho"][0] for i in range(X.shape[0])])
    expected = np.tanh(np.mean(np.arctanh(per_gene_rho)))
    naive_raw_mean = np.mean(per_gene_rho)

    np.testing.assert_allclose(per_set_rho, expected, atol=1e-5)
    assert not np.isclose(per_set_rho, naive_raw_mean, atol=1e-3)


def test_xsea_live_path_mean_matches_regardless_of_r_to_z(rng):
    """Same gene data, only r_to_z differs -- the aggregated per-set rho must be
    identical (r_to_z=False forces the correction on the fly; r_to_z=True was
    already correct, since per-gene rho is already Fisher-z before averaging)."""
    X, y = _make_xsea_gene_data(rng)

    y_coloc_raw = _get_colocalize_fun(
        "pearson", xsea=True, xsea_method="mean", r_to_z=False, dtype=np.float32,
    )
    y_coloc_z = _get_colocalize_fun(
        "pearson", xsea=True, xsea_method="mean", r_to_z=True, dtype=np.float32,
    )
    out_raw = y_coloc_raw({"setA": X}, y)["rho"][0]
    out_z = np.tanh(y_coloc_z({"setA": X}, y)["rho"][0])  # r_to_z=True result is on the z scale

    np.testing.assert_allclose(out_raw, out_z, atol=1e-5)


def test_xsea_aggregate_rho_scale_matches_fisher_z_average(rng):
    """_xsea_aggregate(xsea_method="mean", rho_scale=True) (the null-precompute fast
    path's aggregator) must match the same Fisher-z-average-then-back-transform."""
    per_gene_rho = rng.uniform(-0.9, 0.9, size=(1, 1, 6)).astype(np.float32)  # (n_Y, n_perm, set_size)

    out = _xsea_aggregate(per_gene_rho, "mean", axis=-1, rho_scale=True)
    expected = np.tanh(np.mean(np.arctanh(per_gene_rho), axis=-1))
    naive_raw_mean = np.mean(per_gene_rho, axis=-1)

    np.testing.assert_allclose(out, expected, atol=1e-5)
    assert not np.allclose(out, naive_raw_mean, atol=1e-3)


def test_xsea_aggregate_median_unaffected_by_rho_scale(rng):
    """Order statistics commute exactly with a monotonic transform -- "median" must
    give bit-identical results regardless of rho_scale."""
    per_gene_rho = rng.uniform(-0.9, 0.9, size=(2, 3, 7)).astype(np.float32)
    out_scaled = _xsea_aggregate(per_gene_rho, "median", axis=-1, rho_scale=True)
    out_plain = _xsea_aggregate(per_gene_rho, "median", axis=-1, rho_scale=False)
    np.testing.assert_allclose(out_scaled, out_plain, atol=1e-5)


def test_xsea_weighted_aggregation_handles_scalar_float_stats(rng):
    # regression: found via a real crash (AttributeError: 'float' object has no
    # attribute 'ndim') in _y_colocalize_xsea's weighted-aggregation branch
    # (core/colocalize.py). Root cause: it checked `_colocs_xsea[0][stat].ndim == 0`
    # to detect "nothing to aggregate, one value per set" stats -- but not every
    # colocalization method returns those as a 0-d ndarray. mlr's "r2" is a plain
    # Python float (confirmed directly: _get_colocalize_fun("mlr")(X, y)["r2"] has no
    # .ndim attribute at all), which crashed immediately. Only reachable for
    # multivariate methods (mlr/dominance/pls/pcr/lasso/ridge/elasticnet) + weighted
    # XSEA aggregation, since univariate methods + xsea now go through the vectorized
    # _xsea_aggregate fast paths instead of this per-permutation closure. Fixed by
    # using np.ndim(...) (works uniformly on floats/np.float64/ndarrays) instead of
    # the .ndim attribute.
    X, y = _make_xsea_gene_data(rng, n_genes=6, n_parcels=20)
    weights = {"setA": np.ones(6, dtype=np.float32)}

    y_coloc = _get_colocalize_fun("mlr", xsea=True, xsea_method="weightedmean",
                                  dtype=np.float32)
    out = y_coloc({"setA": X}, y, weights)  # must not raise AttributeError
    assert "r2" in out
    assert np.ndim(out["r2"]) >= 1  # aggregated to one value per set (1 set here)
    assert np.isfinite(out["r2"][0])
