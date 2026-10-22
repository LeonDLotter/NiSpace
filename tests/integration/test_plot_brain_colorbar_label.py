"""plot_brain(colorbar_label="auto"): labels only values whose meaning is known."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from nispace import NiSpace
from nispace.api import _CBAR_LABELS_X, _CBAR_LABELS_Y

N_DK = 68
_ALL_LABELS = set(_CBAR_LABELS_Y.values()) | set(_CBAR_LABELS_X.values())


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _titles(fig):
    return {a.get_title() for a in fig.axes}


def _nsp(rng, parc="DesikanKilliany", n=N_DK, n_x=3):
    y = pd.DataFrame(rng.normal(size=(6, n)), index=[f"s{i}" for i in range(6)])
    x = pd.DataFrame(rng.normal(size=(n_x, n)), index=[f"x{i}" for i in range(n_x)])
    return NiSpace(x=x, y=y, parcellation=parc, verbose=False).fit()


def _paired(nsp, formula):
    nsp.transform_y(formula, groups=np.array([0, 0, 0, 1, 1, 1]),
                    subjects=np.array([1, 2, 3, 1, 2, 3]))


@pytest.mark.parametrize("formula,label", [("pairedcohen(a,b)", "Paired Cohen's d"),
                                           ("elemdiff(a,b)", "Difference"),
                                           ("zscore(a,b)", "Z score")])
def test_known_y_transform_labelled(rng, formula, label):
    nsp = _nsp(rng)
    _paired(nsp, formula)
    fig, _ = nsp.plot_brain(data="Y", Y_transform=formula, maps=0 if "pairedcohen" not in
                            formula else None, n_max=10, show=False)
    assert label in _titles(fig)


def test_unknown_transform_and_raw_data_unlabelled(rng):
    nsp = _nsp(rng)
    # raw Y (no transform run yet)
    fig, _ = nsp.plot_brain(data="Y", maps="s0", show=False)
    assert not (_titles(fig) & _ALL_LABELS)
    # raw X
    fig, _ = nsp.plot_brain(data="X", maps="x0", show=False)
    assert not (_titles(fig) & _ALL_LABELS)
    # a passed DataFrame
    fig, _ = nsp.plot_brain(data=nsp.get_y().iloc[[0]], show=False)
    assert not (_titles(fig) & _ALL_LABELS)
    # a transform without a known unit
    _paired(nsp, "center(a,b)")
    fig, _ = nsp.plot_brain(data="Y", Y_transform="center(a,b)", maps=0, show=False)
    assert not (_titles(fig) & _ALL_LABELS)


def test_x_reduction_labelled(rng):
    nsp = _nsp(rng, n_x=4)
    nsp.reduce_x("pca", n_components=2)
    fig, _ = nsp.plot_brain(data="X", X_reduction="pca", show=False)
    assert "PCA score" in _titles(fig)


def test_explicit_labels_win(rng):
    nsp = _nsp(rng)
    _paired(nsp, "pairedcohen(a,b)")
    fig, _ = nsp.plot_brain(data="Y", colorbar_label="", show=False)
    assert not (_titles(fig) & _ALL_LABELS)
    fig, _ = nsp.plot_brain(data="Y", colorbar_label="my label", show=False)
    assert "my label" in _titles(fig)


@pytest.mark.parametrize("parc,n,kind", [("DesikanKilliany", N_DK, "surface"),
                                         ("Schaefer100+TianS1", 116, None)])
def test_label_in_surface_and_combined_layouts(rng, parc, n, kind):
    nsp = _nsp(rng, parc=parc, n=n)
    _paired(nsp, "pairedcohen(a,b)")
    fig, _ = nsp.plot_brain(data="Y", kind=kind, show=False)
    assert "Paired Cohen's d" in _titles(fig)
