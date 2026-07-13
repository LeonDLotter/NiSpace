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
