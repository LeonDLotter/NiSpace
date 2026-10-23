"""NiSpace.plot(kind="correlation") on a real combined (cortex+subcortex) parcellation.
The other kind="correlation" tests run on the fake parcellation: tests/test_plot_correlation.py."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nispace import NiSpace


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def test_combined_parcellation_coloured_by_compartment(rng):
    n = 116  # Schaefer100 + TianS1
    nsp = NiSpace(x=rng.normal(size=(1, n)), y=rng.normal(size=(1, n)),
                  parcellation="Schaefer100+TianS1", verbose=False).fit()
    nsp.colocalize("spearman")
    fig, axes = nsp.plot(kind="correlation", show=False)
    labels = [t.get_text() for t in axes[0].get_legend().get_texts()]
    assert sorted(labels) == ["Cortex", "Subcortex"]
    n_sc = len(np.asarray(axes[0].collections[1].get_offsets()))
    assert n_sc == 16
