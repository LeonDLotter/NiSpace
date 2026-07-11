"""Shared pytest fixtures.

These are written to be generic and reusable across the whole test suite,
not specific to any one feature -- e.g. any future test of colocalize()/
permute() can reuse ``toy_regression``/``synthetic_nispace`` as-is.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    """Fixed-seed numpy Generator for reproducible synthetic data across tests."""
    return np.random.default_rng(42)


def make_toy_regression(rng, n_parcels=30, n_x=3, n_y=1, noise=0.3,
                        outlier_idx=0, outlier_scale=8.0):
    """Synthetic linear regression data: y = w @ X + noise, with one
    deliberately injected outlier parcel (for ground-truth sanity checks).

    Returns
    -------
    X : ndarray (n_x, n_parcels) -- predictor maps
    Y : ndarray (n_y, n_parcels) -- target map(s)
    w : ndarray (n_x,) -- true weights used to generate Y
    outlier_idx : int -- index of the deliberately injected outlier parcel
    """
    X = rng.normal(size=(n_x, n_parcels))
    w = rng.normal(size=n_x)
    Y = np.zeros((n_y, n_parcels))
    for i in range(n_y):
        Y[i] = w @ X + rng.normal(scale=noise, size=n_parcels)
    Y[:, outlier_idx] += outlier_scale
    return X, Y, w, outlier_idx


@pytest.fixture
def toy_regression(rng):
    """Default-sized synthetic regression fixture (n_parcels=30, n_x=3, one outlier)."""
    return make_toy_regression(rng)


@pytest.fixture
def synthetic_nispace(toy_regression):
    """A fully synthetic, network-free, fitted NiSpace object.

    ``parcellation=None`` with DataFrame x/y input bypasses all parcellation
    machinery (NiSpace.fit() only touches it when self._parc is not None,
    and io.parcellate_data() passes DataFrames/ndarrays through as
    already-parcellated) -- no real Parcellation object, no network or
    data-repo access needed.
    """
    from nispace import NiSpace

    X, Y, w, outlier_idx = toy_regression
    n_x, n_parcels = X.shape
    n_y = Y.shape[0]

    parcel_labels = [f"parcel{i}" for i in range(n_parcels)]
    x_df = pd.DataFrame(X, index=[f"x{i}" for i in range(n_x)], columns=parcel_labels)
    y_df = pd.DataFrame(Y, index=[f"y{i}" for i in range(n_y)], columns=parcel_labels)

    nsp = NiSpace(
        x=x_df, y=y_df, z=None,
        parcellation=None, standardize=False,
        n_proc=1, verbose=False,
        return_self=False,
    )
    nsp.fit()
    return nsp
