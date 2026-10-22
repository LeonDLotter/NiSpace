"""brainplot / plot_brain(title_fontsize=...): map title size, overridable via title_kwargs."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from nispace import NiSpace
from nispace.plotting import brainplot

N_DK = 68


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _title_sizes(fig, labels):
    return [t.get_fontsize() for t in fig.texts if t.get_text() in labels]


def _x(rng, n_x=2):
    return pd.DataFrame(rng.normal(size=(n_x, N_DK)), index=[f"map{i}" for i in range(n_x)])


def test_brainplot_title_fontsize(rng):
    x = _x(rng)
    fig, _ = brainplot(x, parcellation="DesikanKilliany", ncols=2, title_fontsize=23)
    sizes = _title_sizes(fig, set(x.index))
    assert len(sizes) == 2 and all(s == 23 for s in sizes)


def test_brainplot_default_title_size_unchanged(rng):
    x = _x(rng)
    fig, _ = brainplot(x, parcellation="DesikanKilliany", ncols=2)
    ref = plt.matplotlib.font_manager.FontProperties(size="large").get_size_in_points()
    assert _title_sizes(fig, set(x.index)) == [ref, ref]


def test_title_kwargs_fontsize_takes_precedence(rng):
    x = _x(rng)
    fig, _ = brainplot(x, parcellation="DesikanKilliany", ncols=2, title_fontsize=23,
                       title_kwargs={"fontsize": 9})
    assert _title_sizes(fig, set(x.index)) == [9, 9]


def test_plot_brain_forwards_title_fontsize(rng):
    x = _x(rng)
    y = pd.DataFrame(rng.normal(size=(1, N_DK)), index=["y"])
    nsp = NiSpace(x=x, y=y, parcellation="DesikanKilliany", verbose=False).fit()
    fig, _ = nsp.plot_brain(data="X", ncols=2, title_fontsize=19, show=False)
    assert _title_sizes(fig, set(x.index)) == [19, 19]
