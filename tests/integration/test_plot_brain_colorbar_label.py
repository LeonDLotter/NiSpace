"""plot_brain(colorbar_label="auto") in the surface and combined (cx+sc) layouts, which need
real geometry. The label logic itself is tested on the fake parcellation:
tests/test_plot_brain_labels.py."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from nispace import NiSpace

N_DK = 68


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


@pytest.mark.parametrize("parc,n,kind", [("DesikanKilliany", N_DK, "surface"),
                                         ("Schaefer100+TianS1", 116, None)])
def test_label_in_surface_and_combined_layouts(rng, parc, n, kind):
    nsp = _nsp(rng, parc=parc, n=n)
    _paired(nsp, "pairedcohen(a,b)")
    fig, _ = nsp.plot_brain(data="Y", kind=kind, show=False)
    assert "Paired Cohen's d" in _titles(fig)
