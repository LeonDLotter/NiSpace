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
    with pytest.raises(ValueError, match="Y_maps='all'"):
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


def test_p_value_annotated_with_kminp(rng, _nsp):
    x = pd.DataFrame(rng.normal(size=(3, N_PARCELS)), index=["a", "b", "c"])
    nsp = _nsp(x, rng.normal(size=(1, N_PARCELS)))
    nsp.colocalize("spearman")
    nsp.permute("maps", n_perm=50, seed=1)
    nsp.correct_p(mc_method="kminP")
    p_k = nsp.get_p_values("spearman", permute_what="maps", mc_method="kminP(1)").iloc[0]
    fig, axes = nsp.plot(kind="correlation", mc_method="kminP", show=False)
    for ax, lab in zip(axes, ["a", "b", "c"]):
        line = ax.texts[0].get_text().split("\n")[2]
        assert line.startswith(r"$p_{\mathrm{kminP}}$")
        assert float(line.split("= ")[1]) == pytest.approx(p_k[lab], abs=0.0006)
    fig, ax, _ = nsp.plot(kind="categorical", values="p", mc_method="kminP", show=False)


# ── one regression line per Y row ────────────────────────────────────────────────────────

def _case_control(rng, _nsp, n_per_group=6):
    """X: 2 maps; Y: 2 * n_per_group subject maps of groups "a" and "b"; "b" correlates with X map "x1"."""
    x = pd.DataFrame(rng.normal(size=(2, N_PARCELS)), index=["x1", "x2"])
    n = 2 * n_per_group
    y = rng.normal(size=(n, N_PARCELS))
    y[n_per_group:] += 1.5 * x.loc["x1"].to_numpy()
    groups = np.array(["a"] * n_per_group + ["b"] * n_per_group)
    nsp = _nsp(x, pd.DataFrame(y, index=[f"s{i}" for i in range(n)]))
    nsp.transform_y("cohen(b,a)", groups=groups)
    return nsp, x, y, groups


def _legend_texts(fig):
    """Texts of the (single) legend: a figure legend, or an axes legend."""
    legends = list(fig.legends) + [a.get_legend() for a in fig.axes if a.get_legend()]
    assert len(legends) == 1
    return [t.get_text() for t in legends[0].get_texts()]


def _line_slopes(ax):
    """Slopes of all 2-point lines in a panel, in drawing order."""
    out = []
    for line in ax.get_lines():
        xd, yd = line.get_xdata(), line.get_ydata()
        out.append((yd[1] - yd[0]) / (xd[1] - xd[0]))
    return np.array(out)


@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_lines_slopes_are_per_row_correlations(rng, _nsp, method):
    """Line mode with ranks (spearman) and standardize: each line's slope is that row's
    correlation; the bold group lines are the Fisher-z mean per group."""
    nsp, x, y, groups = _case_control(rng, _nsp)
    nsp.colocalize(method, Y_transform=False, r_to_z=False)
    r = nsp.get_colocalizations(method, Y_transform=False)
    fig, axes = nsp.plot(kind="correlation", method=method, Y_transform=False, Y_maps="all",
                         plot_kwargs={"line_labels": "groups", "ranks": method == "spearman",
                                      "standardize": True}, show=False)
    assert len(axes) == 2
    for ax, lab in zip(axes, ["x1", "x2"]):
        slopes = _line_slopes(ax)
        n_y = len(y)
        # subject lines drawn group by group ("a" first), then each group's mean line
        sub_a, mean_a = slopes[:6], slopes[6]
        sub_b, mean_b = slopes[7:13], slopes[13]
        np.testing.assert_allclose(np.r_[sub_a, sub_b], r[lab].to_numpy(), atol=1e-6)
        for sub, mean in ((sub_a, mean_a), (sub_b, mean_b)):
            assert mean == pytest.approx(np.tanh(np.arctanh(sub).mean()), abs=1e-6)
        assert len(slopes) == n_y + 2
    # group legend from transform_y(); group "b" lines carry the X-map-"x1" signal
    assert _legend_texts(fig) == ["a (n = 6)", "b (n = 6)"]
    assert _line_slopes(axes[0])[13] > _line_slopes(axes[0])[6]
    assert "z-scored" in axes[0].get_ylabel()
    assert ("rank" in axes[0].get_ylabel()) == (method == "spearman")


def test_lines_ranks_option(rng, _nsp):
    """ranks=False under spearman: lines are fitted on (z-scored) raw values -> slopes are
    Pearson correlations of the raw maps, not the plotted spearman rho."""
    nsp, x, y, _ = _case_control(rng, _nsp)
    nsp.colocalize("spearman", Y_transform=False, r_to_z=False)
    fig, axes = nsp.plot(kind="correlation", Y_transform=False, Y_maps="all", X_maps="x1",
                         plot_kwargs={"ranks": False, "standardize": True, "mean_line": False},
                         show=False)
    pearson = [np.corrcoef(x.loc["x1"], yi)[0, 1] for yi in y]
    np.testing.assert_allclose(_line_slopes(axes[0]), pearson, atol=1e-6)
    assert "rank" not in axes[0].get_ylabel()
    # and ranks=True works for pearson, too (Spearman slopes)
    from scipy.stats import spearmanr
    nsp.colocalize("pearson", Y_transform=False, r_to_z=False)
    fig, axes = nsp.plot(kind="correlation", method="pearson", Y_transform=False, Y_maps="all",
                         X_maps="x1", plot_kwargs={"ranks": True, "standardize": True,
                                                   "mean_line": False}, show=False)
    np.testing.assert_allclose(_line_slopes(axes[0]),
                               [spearmanr(x.loc["x1"], yi)[0] for yi in y], atol=1e-6)


@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_lines_default_plots_data_as_they_are(rng, _nsp, method):
    """Default: no ranks, no z-scoring -- lines are OLS fits on the values as fitted, so
    offsets and scales stay visible; the mean line is the mean slope and intercept."""
    nsp, x, y, _ = _case_control(rng, _nsp)
    nsp.colocalize(method, Y_transform=False, r_to_z=False)
    fig, axes = nsp.plot(kind="correlation", method=method, Y_transform=False, Y_maps="all",
                         X_maps="x1", show=False)
    assert "rank" not in axes[0].get_ylabel() and "z-scored" not in axes[0].get_ylabel()
    # the fitted (standardized) maps, as colocalized
    xf = nsp.get_x().iloc[0].to_numpy()
    fits = [np.polyfit(xf, yi, 1) for yi in nsp.get_y(Y_transform=False).to_numpy()]
    slopes = _line_slopes(axes[0])
    np.testing.assert_allclose(slopes[:-1], [f[0] for f in fits], atol=1e-6)
    assert slopes[-1] == pytest.approx(np.mean([f[0] for f in fits]), abs=1e-6)
    mean_line = axes[0].get_lines()[-1]
    xd, yd = mean_line.get_xdata(), mean_line.get_ydata()
    icept = np.mean([f[1] for f in fits])
    assert yd[0] == pytest.approx(icept + slopes[-1] * xd[0], abs=1e-6)


def test_lines_groups_option_points_and_errors(rng, _nsp):
    nsp, x, y, _ = _case_control(rng, _nsp)
    nsp.colocalize("spearman", Y_transform=False, r_to_z=False)
    labels = ["g1"] * 4 + ["g2"] * 8
    fig, axes = nsp.plot(kind="correlation", Y_transform=False, Y_maps="all", X_maps="x1",
                         plot_kwargs={"groups": labels, "points": True}, show=False)
    assert _legend_texts(fig) == ["g1 (n = 4)", "g2 (n = 8)"]
    assert len(axes[0].collections) == len(y)  # one point cloud per row
    # a list of Y maps selects rows (and subsets groups given per Y row)
    fig, axes = nsp.plot(kind="correlation", Y_transform=False, Y_maps=["s0", "s2", "s11"],
                         X_maps="x1", plot_kwargs={"groups": labels}, show=False)
    assert _legend_texts(fig) == ["g1 (n = 2)", "g2 (n = 1)"]
    with pytest.raises(ValueError, match="groups has 3 labels"):
        nsp.plot(kind="correlation", Y_transform=False, Y_maps="all", X_maps="x1",
                 plot_kwargs={"groups": ["a", "b", "c"]}, show=False)
    # several Y rows without an explicit selection: still an error (explicit opt-in)
    with pytest.raises(ValueError, match="Y_maps='all'"):
        nsp.plot(kind="correlation", Y_transform=False, show=False)


def test_lines_partial_slopes_are_partial_correlations(rng, _nsp):
    x = pd.DataFrame(rng.normal(size=(1, N_PARCELS)), index=["a"])
    y = rng.normal(size=(5, N_PARCELS))
    nsp = _nsp(x, y, z=rng.normal(size=(1, N_PARCELS)))
    nsp.colocalize("partialspearman", r_to_z=False)
    expected = nsp.get_colocalizations("partialspearman").iloc[:, 0].to_numpy()
    fig, axes = nsp.plot(kind="correlation", Y_maps="all",
                         plot_kwargs={"mean_line": False, "standardize": True}, show=False)
    np.testing.assert_allclose(_line_slopes(axes[0]), expected, atol=1e-6)
    assert "residual" in axes[0].get_ylabel()


def test_lines_subject_wise_transform_keeps_group_labels(rng, _nsp):
    """zscore(b,a) returns one map per group-"b" subject (vs. group "a"): the lines are the
    b subjects only, labelled with their group from transform_y()."""
    nsp, x, y, groups = _case_control(rng, _nsp)
    nsp.transform_y("zscore(b,a)", groups=groups)
    nsp.colocalize("spearman", Y_transform="zscore(b,a)", r_to_z=False)
    fig, axes = nsp.plot(kind="correlation", Y_transform="zscore(b,a)", Y_maps="all",
                         X_maps="x1", plot_kwargs={"line_labels": "groups"}, show=False)
    assert _legend_texts(fig) == ["b (n = 6)"]
    # default for <= 20 rows: one legend entry per map (the subjects), plus the overall mean
    fig, axes = nsp.plot(kind="correlation", Y_transform="zscore(b,a)", Y_maps="all",
                         X_maps="x1", show=False)
    assert _legend_texts(fig) == [f"s{i}" for i in range(6, 12)] + ["mean"]


def test_lines_per_map_default_legend_and_mean(rng, _nsp):
    """<= 20 Y maps without groups (e.g. ENIGMA maps): own colour and legend entry per map,
    legend only once (last panel), one overall dashed mean line = Fisher-z mean slope."""
    x = pd.DataFrame(rng.normal(size=(3, N_PARCELS)), index=["x1", "x2", "x3"])
    names = [f"enigma_{d}" for d in ("adhd", "asd", "bd", "mdd", "ocd", "scz")]
    y = pd.DataFrame(rng.normal(size=(6, N_PARCELS)), index=names)
    nsp = _nsp(x, y)
    nsp.colocalize("spearman", r_to_z=False)
    r = nsp.get_colocalizations("spearman")
    fig, axes = nsp.plot(kind="correlation", Y_maps="all",
                         plot_kwargs={"ranks": True, "standardize": True}, show=False)
    assert _legend_texts(fig) == names + ["mean"]   # one legend, right of the figure
    assert fig.legends
    for ax, lab in zip(axes, ["x1", "x2", "x3"]):
        lines = ax.get_lines()
        slopes = _line_slopes(ax)
        np.testing.assert_allclose(slopes[:6], r[lab].to_numpy(), atol=1e-6)
        assert slopes[6] == pytest.approx(np.tanh(np.arctanh(slopes[:6]).mean()), abs=1e-6)
        assert len({tuple(l.get_color()) for l in lines[:6]}) == 6   # distinct colours
        assert lines[6].get_linestyle() == "--"
    with pytest.raises(ValueError, match="line_labels"):
        nsp.plot(kind="correlation", Y_maps="all", plot_kwargs={"line_labels": "x"}, show=False)


def test_lines_more_than_20_maps_fall_back_to_groups(rng, _nsp):
    x = pd.DataFrame(rng.normal(size=(1, N_PARCELS)), index=["x1"])
    n = 22
    y = pd.DataFrame(rng.normal(size=(n, N_PARCELS)), index=[f"s{i}" for i in range(n)])
    nsp = _nsp(x, y)
    nsp.transform_y("cohen(b,a)", groups=np.array(["a"] * 11 + ["b"] * 11))
    nsp.colocalize("spearman", Y_transform=False, r_to_z=False)
    fig, axes = nsp.plot(kind="correlation", Y_transform=False, Y_maps="all", show=False)
    assert _legend_texts(fig) == ["a (n = 11)", "b (n = 11)"]
    # explicit per-map labels beyond 20 still work (continuous colormap)
    fig, axes = nsp.plot(kind="correlation", Y_transform=False, Y_maps="all",
                         plot_kwargs={"line_labels": "maps"}, show=False)
    assert len(_legend_texts(fig)) == n + 1


@pytest.mark.parametrize("y_rows", [1, 4])
def test_square_panels_by_default(rng, _nsp, y_rows):
    x = pd.DataFrame(rng.normal(size=(2, N_PARCELS)), index=["x1", "x2"])
    nsp = _nsp(x, rng.normal(size=(y_rows, N_PARCELS)))
    nsp.colocalize("spearman")
    y_maps = "all" if y_rows > 1 else None
    fig, axes = nsp.plot(kind="correlation", Y_maps=y_maps, show=False)
    assert all(ax.get_box_aspect() == 1 for ax in axes)
    fig, axes = nsp.plot(kind="correlation", Y_maps=y_maps, plot_kwargs={"box_aspect": None},
                         show=False)
    assert all(ax.get_box_aspect() is None for ax in axes)
