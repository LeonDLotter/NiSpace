"""Smoke tests (does it run and return a figure, not pixel-level
correctness) for NiSpace.plot(kind="categorical") -- catplot()/nullplot()/
print_significance() in plotting.py -- using purely synthetic data (no real
brain geometry needed). plot_brain()/brainplot() glass-brain rendering runs on the
fake parcellation in tests/test_plot_brain_labels.py; surface/combined layouts need real
geometry and are tested under tests/integration/.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from nispace import NiSpace


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def permuted_nispace(toy_regression):
    X, Y, w, outlier_idx = toy_regression
    n_x, n_parcels = X.shape
    parcel_labels = [f"parcel{i}" for i in range(n_parcels)]
    x_df = pd.DataFrame(X, index=[f"x{i}" for i in range(n_x)], columns=parcel_labels)
    y_df = pd.DataFrame(Y, index=[f"y{i}" for i in range(Y.shape[0])], columns=parcel_labels)
    nsp = NiSpace(x=x_df, y=y_df, z=None, parcellation=None, standardize=False,
                  n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    nsp.colocalize(method="pearson", verbose=False)
    nsp.permute(what="maps", maps_method="random", n_perm=200, seed=1, verbose=False)
    return nsp


@pytest.fixture
def unpermuted_nispace(toy_regression):
    """Same as permuted_nispace but permute() is never called -- for testing
    that plot() degrades gracefully (no null overlay/annotation) rather than
    crashing when _last_settings["perm"] was never set."""
    X, Y, w, outlier_idx = toy_regression
    n_x, n_parcels = X.shape
    parcel_labels = [f"parcel{i}" for i in range(n_parcels)]
    x_df = pd.DataFrame(X, index=[f"x{i}" for i in range(n_x)], columns=parcel_labels)
    y_df = pd.DataFrame(Y, index=[f"y{i}" for i in range(Y.shape[0])], columns=parcel_labels)
    nsp = NiSpace(x=x_df, y=y_df, z=None, parcellation=None, standardize=False,
                  n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    nsp.colocalize(method="pearson", verbose=False)
    return nsp


def test_plot_categorical_coloc_values_returns_figure(permuted_nispace):
    fig, ax, plot = permuted_nispace.plot(kind="categorical", show=False, verbose=False)
    assert isinstance(fig, plt.Figure)


def test_plot_categorical_p_values(permuted_nispace):
    fig, ax, plot = permuted_nispace.plot(kind="categorical", values="p", show=False, verbose=False)
    assert isinstance(fig, plt.Figure)


def test_plot_categorical_z_values_requires_normalize_first(permuted_nispace):
    permuted_nispace.normalize_colocalizations(verbose=False)
    fig, ax, plot = permuted_nispace.plot(kind="categorical", values="z", show=False, verbose=False)
    assert isinstance(fig, plt.Figure)


def test_plot_kind_brain_raises_not_implemented(permuted_nispace):
    with pytest.raises(NotImplementedError):
        permuted_nispace.plot(kind="brain", show=False, verbose=False)


def test_plot_categorical_sort_by_truncates_with_n_categories(permuted_nispace):
    fig, ax, plot = permuted_nispace.plot(
        kind="categorical", sort_by="coloc", n_categories=2, show=False, verbose=False
    )
    assert isinstance(fig, plt.Figure)


def test_plot_before_any_permute_degrades_gracefully(unpermuted_nispace):
    """Regression test for the _last_settings["perm"] fix: before "perm" was
    added as a default key, plot() crashed on _get_last(perm=...) with a
    generic Exception rather than reaching its own pre-existing graceful
    degradation (disabling null overlay/annotation with a warning)."""
    fig, ax, plot = unpermuted_nispace.plot(kind="categorical", show=False, verbose=False)
    assert isinstance(fig, plt.Figure)


def test_data_to_volume_matches_per_label_assignment():
    """_data_to_volume (brainplot's per-parcel -> voxel projection) against per-label
    assignment: unsorted labels, fewer values than labels, non-finite values -> bg_value."""
    import nibabel as nib
    from nispace.plotting import _data_to_volume

    rng = np.random.default_rng(0)
    parc = rng.integers(0, 9, size=(6, 7, 5)).astype(np.int16)
    ref = nib.Nifti1Image(parc, np.eye(4))
    labels = np.array([5.0, 1.0, 8.0, 3.0, 2.0])
    data = np.array([1.5, np.nan, -2.0, np.inf])   # one value short of the labels
    for bg in (np.nan, 0.0):
        expected = np.full(parc.shape, bg, dtype=np.float32)
        for lab, v in zip(labels, data):
            expected[parc == lab] = v if np.isfinite(v) else bg
        out = _data_to_volume(data, ref, labels, bg).get_fdata()
        np.testing.assert_array_equal(out, expected)
    assert np.isnan(_data_to_volume([], ref, labels).get_fdata()).all()

