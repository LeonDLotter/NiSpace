"""Smoke test for NiSpace.plot_brain() (surface rendering via brainplot())
against real parcellation geometry -- the actual image-rendering path that
tests/test_plotting_smoke.py's synthetic kind="categorical" tests can't
reach (brainplot() needs a real fitted Parcellation with loadable surface/
volume template assets). Well-formedness only, not pixel-level correctness.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from nispace import NiSpace
from nispace.datasets import fetch_reference


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_plot_brain_surface_smoke():
    x_df = fetch_reference("rsn", parcellation="Yan100", print_references=False, verbose=False)
    y = x_df.iloc[[0]].copy()
    y.index = ["y0"]
    nsp = NiSpace(x=x_df.iloc[1:4], y=y, parcellation="Yan100", verbose=False,
                  n_proc=1, return_self=False)
    nsp.fit()

    fig, axes = nsp.plot_brain(data="Y", show=False, verbose=False)
    assert isinstance(fig, plt.Figure)
