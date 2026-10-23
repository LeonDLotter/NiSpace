import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

import logging
lgr = logging.getLogger(__name__)
from ..plotting import catplot, nullplot, nice_stats_labels, nice_target_label, print_significance


def _plot_categorical(colocs_df, stat, nulls_dict=None, p_df=None, pc_df=None,
                      values="coloc", mc_method=None,
                      sort=False, sort_order=None,
                      annot_p=True, z_method="robust",
                      title=None, fig=None, ax=None, figsize=None,
                      kwargs=None, null_kwargs=None, clean_labels=True, rho_is_z=False):

    # kwargs
    kwargs = {} if kwargs is None else kwargs
    null_kwargs = {} if null_kwargs is None else null_kwargs

    # column names
    colocs_df = colocs_df.copy()
    if p_df is not None:
        p_df = p_df.copy()
    if pc_df is not None:
        pc_df = pc_df.copy()

    # apply pre-computed column sort order (positional indices)
    if sort_order is not None and colocs_df.shape[1] > 1:
        colocs_df = colocs_df.iloc[:, sort_order]
        if p_df is not None:
            p_df = p_df.iloc[:, sort_order]
        if pc_df is not None:
            pc_df = pc_df.iloc[:, sort_order]
        if nulls_dict is not None and stat in nulls_dict:
            orig_keys = list(nulls_dict[stat].keys())
            nulls_dict = dict(nulls_dict)
            nulls_dict[stat] = {orig_keys[i]: nulls_dict[stat][orig_keys[i]] for i in sort_order}
        sort = False  # pre-sorted — disable catplot's own sorting
    
    # things to do for one-column results (everything with r2)
    if colocs_df.shape[1] == 1:
        colocs_df.columns = ["Combined reference maps"]
        if p_df is not None:
            p_df.columns = ["Combined reference maps"]
        if pc_df is not None:
            pc_df.columns = ["Combined reference maps"]
        
    # things to do for multi-column results
    else:
        # if labels should be cleaned:
        if clean_labels:
            
            # check if columns are multiindex 
            if isinstance(colocs_df.columns, pd.MultiIndex):
                if "map" not in colocs_df.columns.names:
                    lgr.warning("Cannot plot clean X labels without named X MultiIndex columns (minimum: 'map')!")
                if all([s in colocs_df.columns.names for s in ["set", "map"]]):   
                    #colocs_df = colocs_df[colocs_df.columns.sortlevel("set", "map")[0]]
                    X_sets = colocs_df.columns.get_level_values("set")
                    X_labels = colocs_df.columns.get_level_values("map")
                elif "set" in colocs_df.columns.names:
                    colocs_df = colocs_df[colocs_df.columns.sortlevel("set")[0]]
                    X_sets = colocs_df.columns.get_level_values("set")
                    X_labels = colocs_df.columns.copy().droplevel("set").to_flat_index()
                elif "map" in colocs_df.columns.names:
                    X_labels = colocs_df.columns.get_level_values("map")
            # if not, just use the columns
            else:
                X_labels = colocs_df.columns
                
            # check if pet labels, if yes make nice string
            # str(): labels may be non-strings (e.g. integer positions of an unnamed X index)
            if all([s in str(X_labels[0]) for s in ["target-", "tracer-", "pub-"]]):
                tmp = []
                for l in X_labels:
                    l_split = l.split("_")
                    tmp.append(f"{nice_target_label(l_split[0].split('-')[1])} ({l_split[4].split('-')[1].capitalize()}, "
                               f"n = {l_split[2].split('-')[1]})")
                X_labels = tmp
            else:  # plain labels that are target names (e.g. "5HT2a") get the display form too
                X_labels = [nice_target_label(l) if isinstance(l, str) else l for l in X_labels]
            # check if brainmap labels, if yes make nice string
            # if all([s in X_labels[0] for s in ["domain-", "n-"]]):
            #     tmp = []
            #     for l in X_labels:
            #         l_split = l.split("_")
            #         tmp.append(f"{l_split[0].split('-')[1]} (n = {l_split[-1].split('-')[1]})")
            #     X_labels = tmp
                
        # if labels are not to be cleaned, convert potential multi-idc to string
        else:
            X_labels = colocs_df.columns.to_flat_index()
            
        # set new column names but keep the set->map assignment as indices (!)
        new_cols = [str(l) for l in X_labels]
        colocs_df.columns = new_cols
        if p_df is not None:
            p_df.columns = new_cols
        if pc_df is not None:
            pc_df.columns = new_cols
        
    # melt df
    colocs_df_melt = colocs_df \
        .assign(Y=colocs_df.index.to_flat_index()).reset_index(drop=True) \
        .melt(
            id_vars=["Y"],
            var_name="X",
            value_name=stat
        )
    
    # null data
    if nulls_dict:
        tmp = []
        for c_nulls, c_colocs in zip(nulls_dict[stat].keys(), colocs_df.columns):
            null_vals = (
                nulls_dict[stat][c_nulls] if isinstance(nulls_dict[stat], dict) else nulls_dict[stat]
            ).mean(axis=0)  # → 1D array of null permutation values
            if values == "z":
                # transform to z-space using the same method as the observed z-scores
                med = np.nanmedian(null_vals)
                if z_method == "robust":
                    scale = 1.4826 * np.nanmedian(np.abs(null_vals - med))
                else:
                    scale = np.nanstd(null_vals, ddof=1)
                null_vals = (null_vals - med) / scale if scale > 0 else (null_vals - med)
            tmp.append(pd.DataFrame({"X": c_colocs, stat: null_vals}))
        nulls_df_melt = pd.concat(tmp)
    
    # default args
    if values == "z":
        stat_label = f"{nice_stats_labels(stat)} (null-normalized Z)"
        legend_title = f"{nice_stats_labels(stat)} (Z)"
    elif values == "p":
        stat_label = r"$-\log_{10}(p)$"
        legend_title = stat_label
    else:
        stat_label = nice_stats_labels(stat)
        if stat == "rho" and rho_is_z:   # stored on the Fisher-z scale (colocalize(r_to_z=True))
            stat_label = f"$z$({stat_label})"
        legend_title = stat_label
    if title in ["", None, False]:
        title = None
    elif title == True:
        title = stat_label
    catplot_kwargs = {
        "legend": {"kwargs": {"title": legend_title}},
        "color_how": "cont",
        "color_which": "auto",
        "sort_categories": sort
    }
    nullplot_kwargs = {
        "legend": {"kwargs": {"title": legend_title}},
        "color_which": "Greys",
        "bands": {"alpha": 0.15, "edgealpha": 0.5, "label_prefix": f"Null perc. "},
        "median_line": {"label": "Null median"}
    }

    # one X:
    if colocs_df.shape[1] == 1:
        catplot_kwargs["categorical_axis"] = "x"
        catplot_kwargs["labels"] = {"x": "", "y": stat_label, "title": title}
        nullplot_kwargs["categorical_axis"] = "x"
        nullplot_kwargs["violins"] = {"plot": True, "legend": "brief", "label": f"Null distr."}
        nullplot_kwargs["bands"] = {"plot": False}
        nullplot_kwargs["median_line"] = {"plot": False}
        
    # multiple X:
    else:
        catplot_kwargs["categorical_axis"] = "y"
        catplot_kwargs["labels"] = {"x": stat_label, "y": "", "title": title}
        nullplot_kwargs["categorical_axis"] = "y"
        
    # one Y:
    if colocs_df.shape[0] == 1:
        catplot_kwargs["bars"] = {"plot": True, "label": stat_label, "linewidth": 1}
        catplot_kwargs["scatters"] = {"plot": False}
        catplot_kwargs["errorbars"] = {"plot": False}
        catplot_kwargs["legend"] |= {"plot": True}
        catplot_kwargs["dots"] = {"plot": False}
        
    # multiple Y: suppress individual dot/errorbar labels — combined entry added at the end
    else:
        catplot_kwargs["dots"] = {"label": False}
        catplot_kwargs["errorbars"] = {"label": False}

    # combine with custom input 
    for k, v in kwargs.items():
        if k in catplot_kwargs and isinstance(v, dict):
            catplot_kwargs[k] = catplot_kwargs[k] | v
        else:
            catplot_kwargs[k] = v
    nullplot_kwargs = nullplot_kwargs | null_kwargs
    
    # make sure that both have same axis orientation
    nullplot_kwargs["categorical_axis"] = catplot_kwargs["categorical_axis"]
    
    # create figure/ax
    if not (ax or fig):
        if not figsize:
            # ~0.24 in per X category on top of a small base: few categories (e.g. 3 components)
            # stay flat instead of tall relative to the width; with 1-2 categories the legend may
            # be taller than the plot (fine)
            n_elements = colocs_df.shape[1]
            figsize = (1.0 + 0.24 * n_elements, 5)
            if catplot_kwargs["categorical_axis"] != "x":
                figsize = np.flip(figsize)
        fig, ax = plt.subplots(1, figsize=figsize)
    
    # zero-line for coloc mode (passed into catplot)
    axis = catplot_kwargs["categorical_axis"]
    cont_on_y = (axis == "x")
    plot_h0line = values == "coloc" and stat in ["beta", "rho"] and cont_on_y
    plot_v0line = values == "coloc" and stat in ["beta", "rho"] and not cont_on_y

    # collect threshold specs for p mode — drawn after render with limit check
    # force=True: always draw; extend axis if the position is outside seaborn's range
    _grey = "dimgrey"
    if values == "p":
        _ref_specs = [
            {"pos": -np.log10(0.05),  "ls": ":",  "lw": 0.8, "label": "$p = .05$",  "sym": False, "force": True},
            {"pos": -np.log10(0.01),  "ls": "--", "lw": 0.8, "label": "$p = .01$",  "sym": False, "force": False},
            {"pos": -np.log10(0.001), "ls": "-.", "lw": 0.8, "label": "$p = .001$", "sym": False, "force": False},
        ]
    else:
        _ref_specs = []

    # plot
    plot = catplot(
        fig, ax, colocs_df_melt, categorical_var="X", continuous_var=stat,
        **catplot_kwargs,
        hline=dict(plot=plot_h0line),
        vline=dict(plot=plot_v0line),
    )

    if nulls_dict is not None:
        if "plot" in locals():
            nullplot_kwargs = nullplot_kwargs | {
                "labels": {
                    "title": ax.get_title(),
                    "x": ax.get_xlabel(),
                    "y": ax.get_ylabel(),
                    "category_order": ax.get_yticklabels()
                    if catplot_kwargs["categorical_axis"] == "y" else ax.get_xticklabels()
                }
            }
        plot = nullplot(fig, ax, nulls_df_melt, categorical_var="X", continuous_var=stat,
                        **nullplot_kwargs)

    # draw reference lines after render — forced lines extend the axis if needed
    if _ref_specs:
        lims = list(ax.get_ylim() if cont_on_y else ax.get_xlim())
        set_lims = ax.set_ylim if cont_on_y else ax.set_xlim
        draw_fn = ax.axhline if cont_on_y else ax.axvline
        _ref_handles, _ref_labels = [], []
        _labels_seen = set()
        for spec in _ref_specs:
            force = spec.get("force", False)
            positions = ([spec["pos"], -spec["pos"]] if spec.get("sym") and spec["pos"] != 0
                         else [spec["pos"]])
            for pos in positions:
                in_range = lims[0] <= pos <= lims[1]
                if in_range or force:
                    if not in_range:
                        span = lims[1] - lims[0]
                        if pos < lims[0]:
                            lims[0] = pos - 0.05 * span
                        else:
                            lims[1] = pos + 0.05 * span
                        set_lims(lims)
                    h = draw_fn(pos, color=_grey, linewidth=spec["lw"],
                                linestyle=spec["ls"], zorder=-100)
                    label = spec["label"]
                    if label and label not in _labels_seen:
                        _ref_handles.append(h)
                        _ref_labels.append(label)
                        _labels_seen.add(label)
        if _ref_handles:
            leg = ax.get_legend()
            if leg is not None:
                # move_legend_fig_to_ax builds the legend via internal box extension,
                # so legend_handles is empty; read the children boxes directly instead
                old_boxes = leg.get_children()[0].get_children()[:]
                ex_title = leg.get_title().get_text()
                leg_kwargs = {"title": ex_title, "loc": leg._loc}
                if leg._bbox_to_anchor is not None:
                    try:
                        pts = leg._bbox_to_anchor._bbox.get_points()
                        norms = np.max(np.abs(pts), axis=1)
                        leg_kwargs["bbox_to_anchor"] = tuple(pts[norms.argmax()])
                    except Exception:
                        leg_kwargs["bbox_to_anchor"] = (1.03, 0.5)
                new_leg = ax.legend(_ref_handles, _ref_labels, **leg_kwargs)
                new_leg.get_children()[0].get_children()[:0] = old_boxes
            else:
                ax.legend(handles=_ref_handles, labels=_ref_labels)

    # significance annotations
    if annot_p is not False and p_df is not None and values not in ("p",):
        if p_df.shape[0] > 1:
            lgr.warning(
                "p_df has one row per Y subject — significance annotation requires a single "
                "aggregated p-value row (set pooled_p=True or pass a pre-aggregated "
                "p_df). Skipping significance annotation."
            )
        else:
            _p_row = p_df.iloc[0]
            _q_row = pc_df.iloc[0] if pc_df is not None else None

            # align all series to current colocs_df column order (after label cleaning)
            _p_vals     = _p_row.reindex(colocs_df.columns).values
            _q_vals     = _q_row.reindex(colocs_df.columns).values if _q_row is not None else None
            _coloc_data = colocs_df.reindex(columns=colocs_df.columns).values  # (n_Y, n_X)

            _sig_handles, _sig_labels = print_significance(
                ax,
                p_values=_p_vals,
                q_values=_q_vals,
                coloc_values=_coloc_data,
                mode=annot_p,
                mc_method=mc_method,
                categorical_axis=catplot_kwargs["categorical_axis"],
            )

            if _sig_handles:
                leg = ax.get_legend()
                if leg is not None:
                    old_boxes = leg.get_children()[0].get_children()[:]
                    ex_title = leg.get_title().get_text()
                    leg_kwargs = {"title": ex_title, "loc": leg._loc}
                    if leg._bbox_to_anchor is not None:
                        try:
                            pts = leg._bbox_to_anchor._bbox.get_points()
                            norms = np.max(np.abs(pts), axis=1)
                            leg_kwargs["bbox_to_anchor"] = tuple(pts[norms.argmax()])
                        except Exception:
                            leg_kwargs["bbox_to_anchor"] = (1.03, 0.5)
                    new_leg = ax.legend(_sig_handles, _sig_labels, **leg_kwargs)
                    new_leg.get_children()[0].get_children()[:0] = old_boxes
                else:
                    ax.legend(handles=_sig_handles, labels=_sig_labels)

    # combined dot + CI legend entry for multi-Y mode (HandlerTuple overlays both icons)
    if colocs_df.shape[0] > 1:
        _dot_proxy = mpl.lines.Line2D(
            [], [], marker="o", linestyle="none", color="k",
            markerfacecolor=(1, 1, 1, 0.8), markeredgewidth=1, markersize=5)
        _eb_proxy = mpl.lines.Line2D([], [], color="k", linewidth=1)
        leg = ax.get_legend()
        if leg is not None:
            old_boxes = leg.get_children()[0].get_children()[:]
            ex_title = leg.get_title().get_text()
            leg_kwargs = {
                "title": ex_title, "loc": leg._loc,
                "handler_map": {tuple: HandlerTuple(ndivide=1, pad=0)}
            }
            if leg._bbox_to_anchor is not None:
                try:
                    pts = leg._bbox_to_anchor._bbox.get_points()
                    norms = np.max(np.abs(pts), axis=1)
                    leg_kwargs["bbox_to_anchor"] = tuple(pts[norms.argmax()])
                except Exception:
                    leg_kwargs["bbox_to_anchor"] = (1.03, 0.5)
            new_leg = ax.legend([(_dot_proxy, _eb_proxy)], ["Observed mean\n(95% CI)"], **leg_kwargs)
            new_leg.get_children()[0].get_children().extend(old_boxes)
        else:
            ax.legend(handles=[(_dot_proxy, _eb_proxy)], labels=["Observed mean\n(95% CI)"],
                      handler_map={tuple: HandlerTuple(ndivide=1, pad=0)})

    return fig, ax, plot



# CORRELATION (SCATTER) PLOT =======================================================================

_COMPARTMENT_COLORS = {"Cortex": "#4C72B0", "Subcortex": "#DD8452"}


def _clean_map_label(label):
    """Readable map label: 'target-D23_tracer-flb457_n-55_..._pub-sandiego2015' -> 'D23
    (Sandiego2015, n = 55)' (same convention as the categorical plot); else the label itself."""
    lab = str(label[-1] if isinstance(label, tuple) else label)
    if all(s in lab for s in ["target-", "tracer-", "pub-"]):
        parts = dict(p.split("-", 1) for p in lab.split("_") if "-" in p)
        return (f"{nice_target_label(parts['target'])} ({parts['pub'].capitalize()}, "
                f"n = {parts.get('n', '?')})")
    return nice_target_label(lab)


def _format_p(p):
    return f"{p:.0e}".replace("e-0", "e-") if p < 0.001 else f"{p:.3f}"


# line spacing of wrapped (two-line) panel titles and x labels, e.g. "MOR\n(Kantonen2020, ...)"
_LABEL_LINESPACING = 1.4


def _suptitle(fig, title):
    """Figure title with some distance to the panel titles below (constrained layout reserves
    the space of the trailing empty line)."""
    fig.suptitle(f"{title}\n", linespacing=0.9)


def _correlation_panels(n, fig, ax, figsize, ncols, sharey, box_aspect=1, multiline=False):
    """n panels: the given axes, or a new grid (unused cells hidden). `box_aspect`: height/width
    of each panel's plot box (default 1: square; None: free). `multiline`: panel labels wrap to
    two lines -> more height per row. Returns (fig, axes, ncols)."""
    if ax is not None:
        axes = np.atleast_1d(ax).ravel()
        if len(axes) < n:
            lgr.critical_raise(f"Got {len(axes)} axes for {n} X maps.", ValueError)
        if box_aspect is not None:
            for a in axes[:n]:
                a.set_box_aspect(box_aspect)
        return (fig if fig is not None else axes[0].figure), axes, None
    ncols = ncols or min(n, 4)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        # compact: text and markers read larger; a wrapped label line adds height per row
        if n == 1:  # a single panel reads too small at the grid cell size
            figsize = (3.8, 3.65 if multiline else 3.25)
        else:
            figsize = (2.8 * ncols, (3.05 if multiline else 2.65) * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                             sharey=sharey, constrained_layout=True)
    try:  # a bit more room between panels than the constrained-layout default (0.02)
        fig.get_layout_engine().set(wspace=0.08, hspace=0.1)
    except AttributeError:  # matplotlib < 3.6
        pass
    axes = axes.ravel()
    for a in axes[n:]:
        a.axis("off")
    if box_aspect is not None:
        for a in axes[:n]:
            a.set_box_aspect(box_aspect)
    return fig, axes, ncols


def _panels_legend(fig, axes, n, own_grid, **kwargs):
    """Legend of the last used panel, always on the right: outside the figure for an own grid
    (so it doesn't widen a grid column), next to the last panel for axes passed by the caller."""
    handles, labels = axes[n - 1].get_legend_handles_labels()
    if not handles:
        return
    # borderpad: padding inside the (invisible) legend frame = distance to the panels, in font
    # sizes (matplotlib default 0.4; borderaxespad would pull an "outside" legend inwards)
    kw = dict(frameon=False, fontsize=8, handlelength=1.5, borderpad=1.6)
    kw.update(kwargs)
    if own_grid:
        try:
            fig.legend(handles, labels, loc="outside right center", **kw)
        except ValueError:  # matplotlib < 3.7: no "outside" placement -> at the right edge
            fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
                       bbox_transform=fig.transFigure, **kw)
        return
    axes[n - 1].legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), **kw)


def _plot_correlation(x_vals, y_vals, x_labels, y_label, r_values, stat_label,
                      p_values=None, groups=None, x_axis_label=None,
                      y_axis_label=None, title=None, fig=None, ax=None, figsize=None,
                      ncols=None, line=True, kwargs=None):
    """Parcel-wise scatter of one Y row against one or more X maps (one panel per X map).

    x_vals : (n_x, n_parcels) array; y_vals : (n_parcels,) or (n_x, n_parcels) array (partial
    residuals differ per X map). NaN parcels are dropped per panel. r_values: one value per X map;
    p_values: list of (label, values per X map), one annotation line each (NaN skipped). groups : (n_parcels,) labels used for point colours
    ("Cortex"/"Subcortex" use the compartment colours). Returns (fig, axes).
    """
    kwargs = {} if kwargs is None else dict(kwargs)
    scatter_kws = dict(s=12, alpha=0.5, linewidths=0)
    scatter_kws.update(kwargs.pop("scatter_kws", {}))
    line_kws = dict(color="black", lw=1.2)
    line_kws.update(kwargs.pop("line_kws", {}))

    x_vals = np.atleast_2d(np.asarray(x_vals, dtype=float))
    y_vals = np.asarray(y_vals, dtype=float)
    if y_vals.ndim == 1:
        y_vals = np.tile(y_vals, (x_vals.shape[0], 1))
    n = x_vals.shape[0]

    fig, axes, ncols = _correlation_panels(n, fig, ax, figsize, ncols, kwargs.pop("sharey", True),
                                           kwargs.pop("box_aspect", 1),
                                           any(" (" in str(l) for l in x_labels))

    groups = None if groups is None else np.asarray(groups)
    for i in range(n):
        a = axes[i]
        xv, yv = x_vals[i], y_vals[i]
        keep = np.isfinite(xv) & np.isfinite(yv)
        if groups is not None:
            for g in pd.unique(groups[keep]):
                sel = keep & (groups == g)
                a.scatter(xv[sel], yv[sel], color=_COMPARTMENT_COLORS.get(g), label=str(g),
                          **scatter_kws)
        else:
            a.scatter(xv[keep], yv[keep], color="#4C72B0", **scatter_kws)
        if line and keep.sum() > 2:
            b1, b0 = np.polyfit(xv[keep], yv[keep], 1)
            xs = np.array([xv[keep].min(), xv[keep].max()])
            a.plot(xs, b0 + b1 * xs, **line_kws)
        txt = f"{stat_label} = {r_values[i]:.2f}"
        for p_lab, p_val in (p_values or []):
            if np.isfinite(p_val[i]):
                txt += f"\n{p_lab} = {_format_p(p_val[i])}"
        # annotation in the corner the regression line leaves free
        right = r_values[i] < 0
        a.text(0.97 if right else 0.03, 0.97, txt, transform=a.transAxes, va="top",
               ha="right" if right else "left", fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))
        # "MOR (Kantonen2020, n = 204)" -> two lines: compact panels would let labels collide
        a.set_xlabel(x_axis_label or str(x_labels[i]).replace(" (", "\n(", 1),
                     linespacing=_LABEL_LINESPACING)
        if n > 1 and x_axis_label:  # "MOR (Kantonen2020, n = 204)" -> two lines, as the x labels
            a.set_title(str(x_labels[i]).replace(" (", "\n(", 1), fontsize=10,
                        linespacing=_LABEL_LINESPACING)
        if i % (len(axes) if ax is not None else (ncols or 1)) == 0 or ax is not None:
            a.set_ylabel(y_axis_label or y_label)
        a.spines[["top", "right"]].set_visible(False)
    if groups is not None:  # outside the data area: points fill all of it
        _panels_legend(fig, axes, n, ax is None, handletextpad=0.2, markerscale=1.3,
                       handlelength=None)
    if title:
        _suptitle(fig, title)
    return fig, axes[:n]


def _fisher_mean(r):
    """Mean of correlations on the Fisher-z scale (never average raw correlations)."""
    r = np.clip(np.asarray(r, dtype=float), -0.999999, 0.999999)
    return float(np.tanh(np.nanmean(np.arctanh(r))))


# line mode: up to this many Y rows, each line gets its own colour and legend entry by default
# (tab20: distinct colours up to 20)
_LINE_LABEL_MAX_MAPS = 20


def _row_colors(n, palette=None, labels=None):
    """One colour per row: palette (list or dict by label), else tab10 / tab20 / viridis."""
    if isinstance(palette, dict):
        return [palette.get(l, palette.get(str(l))) for l in labels]
    if palette is not None:
        return [palette[i % len(palette)] for i in range(n)]
    if n <= 10:
        return [plt.get_cmap("tab10")(i) for i in range(n)]
    if n <= 20:
        return [plt.get_cmap("tab20")(i) for i in range(n)]
    return [plt.get_cmap("viridis")(v) for v in np.linspace(0, 0.95, n)]


def _plot_correlation_lines(x_vals, y_vals, r_values, x_labels, stat_label, groups=None,
                            row_labels=None, x_axis_label=None, y_axis_label=None, title=None,
                            fig=None, ax=None, figsize=None, ncols=None, points=False,
                            mean_line=True, standardized=True, kwargs=None):
    """One regression line per Y row against one or more X maps (one panel per X map).

    x_vals, y_vals : (n_x, n_y, n_parcels) arrays of the plotted values (NaN parcels dropped
    per line). r_values : (n_x, n_y) correlations of the colocalization method, summarized
    (Fisher-z mean) in the corner annotation.

    row_labels : (n_y,) labels -> one colour and legend entry per line, plus one overall mean
    line (black, dashed). Else groups : (n_y,) labels colouring the lines, with a bold mean
    line and mean correlation per group. Mean lines use the Fisher-z mean slope if
    `standardized` (the slope is then a correlation), else the mean slope and intercept.
    Returns (fig, axes).
    """
    kwargs = {} if kwargs is None else dict(kwargs)
    by_row = row_labels is not None
    line_kws = dict(lw=1.2, alpha=0.9) if by_row else dict(lw=0.8, alpha=0.35)
    line_kws.update(kwargs.pop("line_kws", {}))
    mean_kws = dict(lw=2.5, color="black", ls="--") if by_row else dict(lw=2.5)
    mean_kws.update(kwargs.pop("mean_line_kws", {}))
    scatter_kws = dict(s=4, alpha=0.15, linewidths=0)
    scatter_kws.update(kwargs.pop("scatter_kws", {}))
    palette = kwargs.pop("palette", None)

    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    r_values = np.asarray(r_values, dtype=float)
    n, n_y = x_vals.shape[:2]
    if by_row:
        # every row its own "group": its colour, its legend entry; one overall mean
        row_labels = [str(l) for l in row_labels]
        row_colors = _row_colors(n_y, palette, row_labels)
        sets = [(None, np.arange(n_y))]
    else:
        if groups is None:
            groups = np.full(n_y, None, dtype=object)
        groups = np.asarray(groups, dtype=object)
        labels = list(pd.unique(groups))
        if palette is None:
            colors = {g: ("#4C72B0" if g is None else f"C{i}") for i, g in enumerate(labels)}
        elif isinstance(palette, dict):
            colors = {g: palette.get(g, palette.get(str(g))) for g in labels}
        else:
            colors = {g: palette[i % len(palette)] for i, g in enumerate(labels)}
        sets = [(g, np.flatnonzero(groups == g) if g is not None else np.arange(n_y))
                for g in labels]

    fig, axes, ncols = _correlation_panels(n, fig, ax, figsize, ncols, kwargs.pop("sharey", True),
                                           kwargs.pop("box_aspect", 1),
                                           any(" (" in str(l) for l in x_labels))
    for i in range(n):
        a = axes[i]
        txt = []
        for g, sel in sets:
            slopes, icepts, x_lo, x_hi = [], [], [], []
            for j in sel:
                xv, yv = x_vals[i, j], y_vals[i, j]
                keep = np.isfinite(xv) & np.isfinite(yv)
                if keep.sum() < 3:
                    continue
                color = row_colors[j] if by_row else colors[g]
                if points:
                    a.scatter(xv[keep], yv[keep], color=color, **scatter_kws)
                b1, b0 = np.polyfit(xv[keep], yv[keep], 1)
                xs = np.array([xv[keep].min(), xv[keep].max()])
                a.plot(xs, b0 + b1 * xs, color=color,
                       label=(row_labels[j] if by_row and i == n - 1 else None), **line_kws)
                slopes.append(b1); icepts.append(b0)
                x_lo.append(xs[0]); x_hi.append(xs[1])
            if mean_line and slopes:
                b1 = _fisher_mean(slopes) if standardized else float(np.mean(slopes))
                b0 = 0.0 if standardized else float(np.mean(icepts))
                xs = np.array([min(x_lo), max(x_hi)])
                if by_row:
                    lab, kws = ("mean" if i == n - 1 else None), mean_kws
                else:
                    lab = None if g is None else f"{g} (n = {len(slopes)})"
                    kws = dict(mean_kws, color=colors[g])
                a.plot(xs, b0 + b1 * xs, label=lab, **kws)
            name = "mean " if g is None else f"{g}: mean "
            txt.append(f"{name}{stat_label} = {_fisher_mean(r_values[i, sel]):.2f}")
        right = _fisher_mean(r_values[i]) < 0
        a.text(0.97 if right else 0.03, 0.97, "\n".join(txt), transform=a.transAxes, va="top",
               ha="right" if right else "left", fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))
        a.set_xlabel(x_axis_label or str(x_labels[i]).replace(" (", "\n(", 1),
                     linespacing=_LABEL_LINESPACING)
        if n > 1 and x_axis_label:  # "MOR (Kantonen2020, n = 204)" -> two lines, as the x labels
            a.set_title(str(x_labels[i]).replace(" (", "\n(", 1), fontsize=10,
                        linespacing=_LABEL_LINESPACING)
        if i % (len(axes) if ax is not None else (ncols or 1)) == 0 or ax is not None:
            a.set_ylabel(y_axis_label or "Y")
        a.spines[["top", "right"]].set_visible(False)
    if by_row or (labels != [None] and mean_line):
        _panels_legend(fig, axes, n, ax is None)
    if title:
        _suptitle(fig, title)
    return fig, axes[:n]
