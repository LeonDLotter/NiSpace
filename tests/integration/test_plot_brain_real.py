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


def _colorbar_limits(fig):
    return [(a._colorbar.norm.vmin, a._colorbar.norm.vmax)
            for a in fig.axes if getattr(a, "_colorbar", None) is not None]


def test_brainplot_combined_respects_shared_colorscale():
    """Combined (cx+sc) plots used to force one shared colorscale across maps. Now
    shared_colorscale controls sharing across maps, while each map's limits still come
    from its full row (cx AND sc parcels), so its cx and sc panels share one scale."""
    import numpy as np
    import pandas as pd
    from nispace.plotting import brainplot

    cols = fetch_reference("tpm", maps=["gm"], parcellation="Yan200+TianS2",
                           print_references=False, verbose=False).columns
    n_cx, n_sc = 200, len(cols) - 200
    # sc ranges above cx ranges: per-map vmax must come from the sc part
    a = np.r_[np.linspace(1, 2, n_cx), np.linspace(3, 4, n_sc)]
    b = np.r_[np.linspace(10, 20, n_cx), np.linspace(30, 40, n_sc)]
    df = pd.DataFrame([a, b], index=["a", "b"], columns=cols)

    fig, _ = brainplot(df, parcellation="Yan200+TianS2", shared_colorscale=False)
    np.testing.assert_allclose(_colorbar_limits(fig), [(1, 4), (10, 40)])
    plt.close(fig)

    fig, _ = brainplot(df, parcellation="Yan200+TianS2", shared_colorscale=True)
    np.testing.assert_allclose(_colorbar_limits(fig), [(1, 40), (1, 40)])


def test_brainplot_surface_title_anchored_to_drawn_content():
    """Surface titles sit 6 pt above the drawn brain (via _drawn_top), not at the
    axes-relative position that left them floating far above the meshes. A silent
    fallback to the old placement would pass a plain smoke test, so check positions."""
    from nispace.plotting import brainplot, _drawn_top

    df = fetch_reference("tpm", maps=["gm"], parcellation="Yan200",
                         print_references=False, verbose=False)
    fig, axes = brainplot(df, parcellation="Yan200", kind="surface", title=["gm"])
    title = [t for t in fig.texts if t.get_text() == "gm"][0]

    fig.draw_without_rendering()
    top = _drawn_top(fig, axes[0])
    assert top is not None
    title_bottom = fig.transFigure.inverted().transform((0, title.get_window_extent().y0))[1]
    pad_pt = (title_bottom - top) * fig.get_figheight() * 72
    assert pad_pt == pytest.approx(6, abs=0.5)
    pos = axes[0].get_position()
    assert title_bottom < pos.y0 + 1.02 * pos.height   # below the old axes-relative spot


def test_brainplot_title_y_override_keeps_axes_relative_placement():
    from nispace.plotting import brainplot

    df = fetch_reference("tpm", maps=["gm"], parcellation="Yan200",
                         print_references=False, verbose=False)
    fig, axes = brainplot(df, parcellation="Yan200", kind="surface", title=["gm"],
                          title_kwargs={"y": 1.1})
    title = [t for t in fig.texts if t.get_text() == "gm"][0]
    pos = axes[0].get_position()
    assert title.get_position()[1] == pytest.approx(pos.y0 + 1.1 * pos.height)
