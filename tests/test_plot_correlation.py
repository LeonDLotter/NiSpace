"""NiSpace.plot(kind="correlation"): parcel-wise scatter of one Y row vs. X maps.

Runs on the synthetic fake parcellation (conftest.py). The cortex/subcortex colouring test
needs a real combined parcellation: tests/integration/test_plot_correlation_combined.py."""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from nispace import NiSpace

N_PARCELS = 24  # fake_parcellation


@pytest.fixture
def _nsp(fake_parcellation):
    def make(x, y, z=None):
        return NiSpace(x=x, y=y, z=z, parcellation=fake_parcellation, verbose=False).fit()
    return make


def _panel_r(ax):
    """Correlation printed in a panel's corner annotation."""
    return float(re.search(r"=\s*(-?\d+\.\d+)", ax.texts[0].get_text()).group(1))


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_annotated_r_matches_colocalize(rng, method, _nsp):
    x = pd.DataFrame(rng.normal(size=(3, N_PARCELS)), index=["a", "b", "c"])
    y = rng.normal(size=(1, N_PARCELS))
    nsp = _nsp(x, y)
    nsp.colocalize(method, r_to_z=False)
    expected = nsp.get_colocalizations(method).iloc[0]
    fig, axes = nsp.plot(kind="correlation", show=False)
    assert len(axes) == 3
    for ax, lab in zip(axes, ["a", "b", "c"]):
        assert _panel_r(ax) == pytest.approx(expected[lab], abs=0.006)


def test_partial_plots_residuals_and_matches_partialspearman(rng, _nsp):
    x = pd.DataFrame(rng.normal(size=(2, N_PARCELS)), index=["a", "b"])
    y = rng.normal(size=(1, N_PARCELS))
    nsp = _nsp(x, y, z=rng.normal(size=(1, N_PARCELS)))
    nsp.colocalize("partialspearman", r_to_z=False)
    expected = nsp.get_colocalizations("partialspearman").iloc[0]
    fig, axes = nsp.plot(kind="correlation", show=False)
    assert "residual" in axes[0].get_ylabel()
    for ax, lab in zip(axes, ["a", "b"]):
        assert _panel_r(ax) == pytest.approx(expected[lab], abs=0.006)
        # residuals of ranks are centred
        xy = ax.collections[0].get_offsets()
        assert abs(np.asarray(xy)[:, 0].mean()) < 1e-6


def test_x_maps_order_and_single_y_required(rng, _nsp):
    x = pd.DataFrame(rng.normal(size=(3, N_PARCELS)), index=["a", "b", "c"])
    y = pd.DataFrame(rng.normal(size=(2, N_PARCELS)), index=["y0", "y1"])
    nsp = _nsp(x, y)
    nsp.colocalize("spearman")
    with pytest.raises(ValueError, match="one Y row"):
        nsp.plot(kind="correlation", show=False)
    fig, axes = nsp.plot(kind="correlation", Y_maps="y1", X_maps=["c", "a"], show=False)
    assert [ax.get_xlabel() for ax in axes] == ["c", "a"]


def test_non_pairwise_method_raises(rng, _nsp):
    nsp = _nsp(rng.normal(size=(3, N_PARCELS)), rng.normal(size=(1, N_PARCELS)))
    with pytest.raises(ValueError, match="pairwise"):
        nsp.plot(kind="correlation", method="mlr", show=False)


def test_p_value_annotated_after_permutation(rng, _nsp):
    x = pd.DataFrame(rng.normal(size=(2, N_PARCELS)), index=["a", "b"])
    nsp = _nsp(x, rng.normal(size=(1, N_PARCELS)))
    nsp.colocalize("spearman")
    nsp.permute("maps", n_perm=50, seed=1)
    nsp.correct_p(mc_method="fdr_bh")
    # label as in the categorical plot: the mc_method string as given
    fig, axes = nsp.plot(kind="correlation", mc_method="fdr_bh", show=False)
    p_unc = nsp.get_p_values("spearman", permute_what="maps").iloc[0]
    for ax, lab in zip(axes, ["a", "b"]):
        lines = ax.texts[0].get_text().split("\n")
        assert lines[1].startswith(r"$p_{\mathrm{uncorrected}}$")
        assert float(lines[1].split("= ")[1]) == pytest.approx(p_unc[lab], abs=0.0006)
        assert lines[2].startswith(r"$p_{\mathrm{fdr-bh}}$")
    fig, axes = nsp.plot(kind="correlation", mc_method="uncorrected", show=False)
    assert all(len(ax.texts[0].get_text().split("\n")) == 2 for ax in axes)
