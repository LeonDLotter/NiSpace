"""plot_brain()/brainplot() map titles and colorbar labels, on the fake parcellation
(conftest.py; glass-brain layout, fully offline).

- colorbar_label="auto" labels only values whose meaning is known.
- title_fontsize sets the map title size; title_kwargs["fontsize"] overrides it.

Surface and combined (cx+sc) layouts need real geometry:
tests/integration/test_plot_brain_colorbar_label.py and test_plot_brain_real.py.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from nispace import NiSpace
from nispace.api import _CBAR_LABELS_X, _CBAR_LABELS_Y
from nispace.plotting import brainplot

N_PARCELS = 24  # fake_parcellation
_ALL_LABELS = set(_CBAR_LABELS_Y.values()) | set(_CBAR_LABELS_X.values())


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


@pytest.fixture
def _nsp(rng, fake_parcellation):
    def make(n_x=3):
        y = pd.DataFrame(rng.normal(size=(6, N_PARCELS)), index=[f"s{i}" for i in range(6)])
        x = pd.DataFrame(rng.normal(size=(n_x, N_PARCELS)), index=[f"x{i}" for i in range(n_x)])
        return NiSpace(x=x, y=y, parcellation=fake_parcellation, verbose=False).fit()
    return make


def _titles(fig):
    return {a.get_title() for a in fig.axes}


def _paired(nsp, formula):
    nsp.transform_y(formula, groups=np.array([0, 0, 0, 1, 1, 1]),
                    subjects=np.array([1, 2, 3, 1, 2, 3]))


# --- colorbar_label="auto" ---

@pytest.mark.parametrize("formula,label", [("pairedcohen(a,b)", "Paired Cohen's d"),
                                           ("elemdiff(a,b)", "Difference"),
                                           ("zscore(a,b)", "Z score")])
def test_known_y_transform_labelled(_nsp, formula, label):
    nsp = _nsp()
    _paired(nsp, formula)
    fig, _ = nsp.plot_brain(data="Y", Y_transform=formula, maps=0 if "pairedcohen" not in
                            formula else None, n_max=10, show=False)
    assert label in _titles(fig)


def test_unknown_transform_and_raw_data_unlabelled(_nsp):
    nsp = _nsp()
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


def test_x_reduction_labelled(_nsp):
    nsp = _nsp(n_x=4)
    nsp.reduce_x("pca", n_components=2)
    fig, _ = nsp.plot_brain(data="X", X_reduction="pca", show=False)
    assert "PCA score" in _titles(fig)


def test_explicit_labels_win(_nsp):
    nsp = _nsp()
    _paired(nsp, "pairedcohen(a,b)")
    fig, _ = nsp.plot_brain(data="Y", colorbar_label="", show=False)
    assert not (_titles(fig) & _ALL_LABELS)
    fig, _ = nsp.plot_brain(data="Y", colorbar_label="my label", show=False)
    assert "my label" in _titles(fig)


# --- title_fontsize ---

def _title_sizes(fig, labels):
    return [t.get_fontsize() for t in fig.texts if t.get_text() in labels]


@pytest.fixture
def _x(rng):
    return pd.DataFrame(rng.normal(size=(2, N_PARCELS)), index=["map0", "map1"])


def test_brainplot_title_fontsize(_x, fake_parcellation):
    fig, _ = brainplot(_x, parcellation=fake_parcellation, ncols=2, title_fontsize=23)
    sizes = _title_sizes(fig, set(_x.index))
    assert len(sizes) == 2 and all(s == 23 for s in sizes)


def test_brainplot_default_title_size_unchanged(_x, fake_parcellation):
    fig, _ = brainplot(_x, parcellation=fake_parcellation, ncols=2)
    ref = plt.matplotlib.font_manager.FontProperties(size="large").get_size_in_points()
    assert _title_sizes(fig, set(_x.index)) == [ref, ref]


def test_title_kwargs_fontsize_takes_precedence(_x, fake_parcellation):
    fig, _ = brainplot(_x, parcellation=fake_parcellation, ncols=2, title_fontsize=23,
                       title_kwargs={"fontsize": 9})
    assert _title_sizes(fig, set(_x.index)) == [9, 9]


def test_plot_brain_forwards_title_fontsize(rng, _x, fake_parcellation):
    y = pd.DataFrame(rng.normal(size=(1, N_PARCELS)), index=["y"])
    nsp = NiSpace(x=_x, y=y, parcellation=fake_parcellation, verbose=False).fit()
    fig, _ = nsp.plot_brain(data="X", ncols=2, title_fontsize=19, show=False)
    assert _title_sizes(fig, set(_x.index)) == [19, 19]
