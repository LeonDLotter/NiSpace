import inspect
import matplotlib as mpl        
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import PatchCollection
import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as sno
from sklearn.preprocessing import minmax_scale
from nilearn.plotting import view_surf as view_surf_nilearn
from nilearn.plotting import plot_surf_stat_map, plot_surf_contours, plot_glass_brain, plot_stat_map
from nilearn.image import new_img_like
from neuromaps import images

import logging
lgr = logging.getLogger(__name__)
from .utils.utils import vect_to_vol_arr, set_log
from .datasets import fetch_parcellation, fetch_template, parcellation_lib, template_lib
from ._patches import apply_surface_plot_patches
apply_surface_plot_patches()


def nice_stats_labels(string, add_dollars=True):
    replace_dict = {
        "r2": "R^2",
        "R2": "R^2",
        "beta": "Beta",
        "spearman": "Spearman's Rho",
        "pearson": "Pearson's Rho",
        "partialpearson": "Partial Pearson's Rho",
        "partialspearman": "Partial Spearman's Rho",
        "mi": "Mutual Information",
        "mlr": "MLR",
        "slr": "SLR",
        "pls": "PLS",
        "pcr": "PCR",
        "dominance": "Dominance Analysis",
        "individual": "Individual R^2",
        "total": "Total R^2",
        "ridge": "Ridge",
        "lasso": "Lasso",
        "elasticnet": "ElasticNet",
        "meandiff": "Mean Difference",
        "elemdiff": "Elementwise Difference",
        "zscore": "Z score",
        "pairedcohen": "Paired Cohen's d",
        "pairedhedges": "Paired Hedges' g",
        "cohen": "Cohen's d",
        "hedges": "Hedges' g",
        "psc": "PSC",
        "md": "MD",
        "rho": "Rho",
        "ci": "95% CI",
        "mean": "Mean",
        "median": "Median",
        "sd": "SD",
        "std": "STD",
        "groups": "Groups",
        "sets": "Sets",
        "xmaps": "X maps",
        "ymaps": "X maps",
        "xymaps": "X and Y maps"
    }
    result = replace_dict.get(string, string)
    if add_dollars:
        return "$" + result.replace(' ', r'\ ') + "$"
    return result


def hide_empty_axes(axes):
    
    [ax.axis("off") for ax in axes.ravel() if ax.axis() == (0.0, 1.0, 0.0, 1.0)]


def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)


def move_legend_fig_to_ax(fig, ax, loc, bbox_to_anchor=None, no_legend_error=False, **kwargs):
        # copied from GitHub user thuiop
        # https://github.com/mwaskom/seaborn/issues/3247#issuecomment-1420731692 

        if hasattr(fig, "legends"):
            if len(fig.legends) > 0:
            
                old_fig_legend = fig.legends[-1]
                old_fig_boxes = old_fig_legend.get_children()[0].get_children()
                
                if ax.legend_:
                    old_ax_legend = ax.legend_
                    old_ax_boxes = old_ax_legend.get_children()[0].get_children()
                

                legend_kws = inspect.signature(mpl.legend.Legend).parameters
                props = {
                    k: v for k, v in old_fig_legend.properties().items() if k in legend_kws
                }

                props.pop("bbox_to_anchor")
                title = props.pop("title")
                if "title" in kwargs:
                    title.set_text(kwargs.pop("title"))
                title_kwargs = {k: v for k, v in kwargs.items() if k.startswith("title_")}
                for key, val in title_kwargs.items():
                    title.set(**{key[6:]: val})
                    kwargs.pop(key)
                kwargs.setdefault("frameon", old_fig_legend.legendPatch.get_visible())

                # Remove the old legend and create the new one
                props.update(kwargs)
                fig.legends = []
                new_legend = ax.legend(
                    [], [], loc=loc, bbox_to_anchor=bbox_to_anchor, **props
                )
                new_legend.get_children()[0].get_children().extend(old_fig_boxes)
                
                if "old_ax_legend" in locals():
                    new_legend.get_children()[0].get_children().extend(old_ax_boxes)

        else:
            if no_legend_error:
                raise ValueError("Figure has no legend attached.")
            else:
                pass
            

def linewidth_from_data_units(linewidth, axis, reference='x'):
    """
    Convert a linewidth in data units to linewidth in points.

    Parameters
    ----------
    linewidth: float
        Linewidth in data units of the respective reference-axis
    axis: matplotlib axis
        The axis which is used to extract the relevant transformation
        data (data limits and size must not change afterwards)
    reference: string
        The axis that is taken as a reference for the data width.
        Possible values: 'x' and 'y'. Defaults to 'y'.

    Returns
    -------
    linewidth: float
        Linewidth in points
    """
    fig = axis.get_figure()
    if reference == 'x':
        length = fig.bbox_inches.width * axis.get_position().width
        value_range = np.diff(axis.get_xlim())
    elif reference == 'y':
        length = fig.bbox_inches.height * axis.get_position().height
        value_range = np.diff(axis.get_ylim())
    # Convert length to points
    length *= 72
    # Scale linewidth to value range
    return linewidth * (length / value_range)


def print_significance(ax, p_values, q_values=None, coloc_values=None,
                       mode=True, mc_method=None,
                       symbols=("☆", "★"), symbol_size=12, text_size=8,
                       pad_frac=0.02, bold_labels=True, categorical_axis="y"):
    """Annotate a categorical axis plot with significance markers or p-value text.

    Parameters
    ----------
    mode : True / "stars" / "text" / "both" / False
        True/"stars": draw ☆/★ symbols. "text": draw numeric p-value. "both": both.
    mc_method : str or None
        Name of the MC correction method, used for legend labels.

    Returns
    -------
    (handles, labels) : lists for legend integration.
    """
    if mode is False:
        return [], []

    mode = "stars" if mode is True else mode
    p_values = np.array(p_values).flatten()
    q_values = p_values if q_values is None else np.array(q_values).flatten()
    has_q = not np.array_equal(p_values, q_values)

    # continuous axis direction
    cont_on_x = (categorical_axis == "y")  # bars extend horizontally
    get_lim = ax.get_xlim if cont_on_x else ax.get_ylim
    set_lim = ax.set_xlim if cont_on_x else ax.set_ylim

    # coloc_values: 1D (means) for bar mode, 2D (n_Y × n_X) for scatter/point mode
    _cv = np.atleast_2d(np.array(coloc_values, dtype=float))  # always (n_Y, n_X)
    col_means = np.nanmean(_cv, axis=0)

    # For multi-Y: position annotations at the 95% CI edge rather than the mean
    # so markers don't overlap with errorbars.  Single-Y: CI == mean, so this is a no-op.
    if _cv.shape[0] > 1:
        from scipy import stats as _stats
        n_valid = np.sum(~np.isnan(_cv), axis=0).astype(float)
        se = np.nanstd(_cv, ddof=1, axis=0) / np.sqrt(np.maximum(n_valid, 1))
        t_crit = _stats.t.ppf(0.975, df=np.maximum(n_valid - 1, 1))
        ci_margin = t_crit * se
        col_ci_upper = col_means + ci_margin
        col_ci_lower = col_means - ci_margin
    else:
        col_ci_upper = col_means
        col_ci_lower = col_means

    span = get_lim()[1] - get_lim()[0]
    pad = pad_frac * span

    # tick label objects — one per X element (categorical axis)
    labs = ax.get_yticklabels() if categorical_axis == "y" else ax.get_xticklabels()

    # build lookup maps aligned to tick order
    col_means_map    = {}
    col_ci_upper_map = {}
    col_ci_lower_map = {}
    p_map = {}
    q_map = {}
    for i, lab in enumerate(labs):
        if i < len(col_means):
            col_means_map[lab.get_text()]    = col_means[i]
            col_ci_upper_map[lab.get_text()] = col_ci_upper[i]
            col_ci_lower_map[lab.get_text()] = col_ci_lower[i]
            p_map[lab.get_text()] = p_values[i]
            q_map[lab.get_text()] = q_values[i]

    def _fmt_p(pval):
        return "< 0.001" if pval < 0.001 else f"{pval:.3f}"

    handles, labels_out = [], []

    # --- pass 1: collect annotations and apply label weights ---
    _to_draw = []  # (cat_coord, pos, mean, ha, ann_str, size)
    _n_total  = sum(1 for txt in (lab.get_text() for lab in labs) if txt in p_map)
    _n_p_sig  = sum(1 for txt in p_map if p_map[txt] < 0.05)
    _n_q_sig  = sum(1 for txt in q_map if q_map[txt] < 0.05) if has_q else 0
    _any_open   = False  # any ☆ drawn
    _any_filled = False  # any ★ drawn
    _any_text   = False  # any number drawn

    _mc_str = f"p_{mc_method}" if mc_method else "p_corrected"
    _q_str = f"{_n_q_sig}/{_n_total} {_mc_str} < 0.05" if has_q else f"0/{_n_total} {_mc_str} < 0.05 (no correction applied)"
    lgr.info(
        f"Significance annotation: {_n_p_sig}/{_n_total} p_uncorrected < 0.05, {_q_str}"
    )

    for lab in labs:
        txt = lab.get_text()
        if txt not in p_map:
            continue

        p = p_map[txt]
        q = q_map[txt]
        mean     = col_means_map[txt]
        ci_upper = col_ci_upper_map[txt]
        ci_lower = col_ci_lower_map[txt]

        # bold tick label — only for MC-corrected significance
        if bold_labels:
            lab.set_weight("bold" if (has_q and q < 0.05) else "normal")

        ann_parts = []
        if mode in ("stars", "both"):
            if has_q and q < 0.05:
                ann_parts.append(symbols[1])
                _any_filled = True
            elif p < 0.05:
                ann_parts.append(symbols[0])
                _any_open = True
        if mode == "text":
            if p < 0.05:
                ann_parts.append(_fmt_p(q if (has_q and q < 0.05) else p))
                _any_text = True
        elif mode == "both":
            if p < 0.05:
                ann_parts.append(_fmt_p(p))  # always uncorrected alongside stars
                _any_text = True
        if not ann_parts:
            continue

        # anchor at CI edge (for multi-Y) or mean (for single-Y / bar mode)
        anchor = ci_upper if mean >= 0 else ci_lower
        pos = anchor + pad if mean >= 0 else anchor - pad
        ha = "left" if mean >= 0 else "right"
        l_pos = lab.get_position()
        cat_coord = l_pos[0] if categorical_axis == "x" else l_pos[1]
        size = symbol_size if mode == "stars" else text_size
        _to_draw.append((cat_coord, pos, mean, ha, " ".join(ann_parts), size))

    # --- axis limit expansion ---
    # seaborn auto-margins are ~5% of data range; stars need 4×pad to reliably
    # exceed that; text strings are wider so use 8×pad for text/both modes
    _exp = (8 if mode in ("text", "both") else 4) * pad
    if _to_draw:
        curr = list(get_lim())
        for _, pos, mean, *_ in _to_draw:
            if mean >= 0:
                curr[1] = max(curr[1], pos + _exp)
            else:
                curr[0] = min(curr[0], pos - _exp)
        set_lim(curr)

    # --- pass 2: draw ---
    for cat_coord, pos, mean, ha, ann_str, size in _to_draw:
        if cont_on_x:
            ax.text(pos, cat_coord, ann_str, ha=ha, va="center", size=size, clip_on=False)
        else:
            ax.text(cat_coord, pos, ann_str, ha="center",
                    va="bottom" if mean >= 0 else "top", size=size, clip_on=False)

    # build legend proxy handles using Line2D with custom math-text markers
    _p_unc = r"$p_{\mathrm{uncorrected}}$"
    if mc_method:
        _mc_sub = mc_method.replace("_", "-")
        _p_mc = rf"$p_{{\mathrm{{{_mc_sub}}}}}$"
    else:
        _p_mc = _p_unc
    _star_size = symbol_size * 0.75
    if mode in ("stars", "both"):
        if _any_open:
            h_open = mpl.lines.Line2D(
                [], [], linestyle="none", marker=f"${symbols[0]}$",
                markersize=_star_size, markeredgewidth=0.1, color="k"
            )
            handles.append(h_open)
            labels_out.append(f"{_p_unc} < 0.05")
        if _any_filled:
            h_filled = mpl.lines.Line2D(
                [], [], linestyle="none", marker=f"${symbols[1]}$",
                markersize=_star_size, markeredgewidth=0.1, color="k"
            )
            handles.append(h_filled)
            labels_out.append(f"{_p_mc} < 0.05")
    if mode in ("text", "both") and _any_text:
        h_txt = mpl.lines.Line2D(
            [], [], linestyle="none", marker=r"$0.001$",
            markersize=24, markeredgewidth=0, color="k"
        )
        handles.append(h_txt)
        labels_out.append(f"{_p_unc}")

    return handles, labels_out


def pivot_brainspan_result(brainspan_vector):
    if brainspan_vector.shape != (80,):
        raise ValueError(f"brainspan_vector must have shape (80,), not shape {brainspan_vector.shape}.")
    elif not isinstance(brainspan_vector, pd.Series):
        raise ValueError(f"brainspan_vector must be a pandas Series, not a {type(brainspan_vector)}.")
    return (
        brainspan_vector.to_frame()
        .assign(stage=lambda x: x.index.str.split("-").str[0], 
                region=lambda x: x.index.str.split("-").str[1])
        .assign(stage=lambda x: pd.Categorical(x.stage, x.stage.unique()), 
                region=lambda x: pd.Categorical(x.region, x.region.unique()))
        .pivot_table(index="stage", columns="region", values=brainspan_vector.name, observed=False)
    )



def catplot(fig, ax, data_long, categorical_var="variable", continuous_var="value", group_var=None,
            categorical_axis="x", sort_categories=False, category_order=None,
            color_how="continuous", color_which="auto", color_center=None,
            labels=None,
            limits=None,
            bars=None,
            violins=None,
            scatters=None,
            dots=None,
            errorbars=None,
            hline=None,
            vline=None,
            hlines=None,
            vlines=None,
            legend=None
            ):   
    
    # defaults, overwrite with user input
    bars = dict(
        plot=False, label=True,
        width=0.5, agg_method="mean", dodge_width=0.5, 
        kwargs={"zorder": 10, "ec": "k", "lw": 0.7}
    ) | ({} if bars is None else bars)
    violins = dict(
        plot=False, label=False,
        kwargs={"zorder": 20, "density_norm": "width", "cut": 0, "inner": "quart",
                "fill": False, "edgecolor": "k", "linewidth": 0.7}
    ) | ({} if violins is None else violins)
    scatters = dict(
        plot=True, label=False,
        size="auto", jitter_width=0.5, dodge_width=0.5,
        kwargs={"zorder": 30, "linewidth": 0.2, "alpha": 0.2}
    ) | ({} if scatters is None else scatters) 
    dots = dict(
        plot=True, label=True,
        agg_method="mean", size=7, color="k", dodge_width=0.5, 
        kwargs={"zorder": 90, "facecolor": (1,1,1,0.8), "lw": 1}
    ) | ({} if dots is None else dots)
    errorbars = dict(
        plot=True, label=True,
        agg_method="ci", dodge_width=0.5, color="k", 
        kwargs={"zorder": 100}
    ) | ({} if errorbars is None else errorbars)
    hline = dict(
        plot=False, y=[0], color="k", linewidth=1, linestyle="--", zorder=-100, kwargs={}
    ) | ({} if hline is None else hline)
    vline = dict(
        plot=False, x=[0], color="k", linewidth=1, linestyle="--", zorder=-100, kwargs={}
    ) | ({} if vline is None else vline)
    legend = dict(
        plot=True,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        nice_labels=True,
        kwargs={}
    ) | ({} if legend is None else legend)
    labels = dict() | ({} if labels is None else labels)
    limits = dict() | ({} if limits is None else limits)
    
    # orientation    
    if categorical_axis == "x":
        xy = dict(x=categorical_var, y=continuous_var)
    elif categorical_axis == "y":
        xy = dict(y=categorical_var, x=continuous_var)
    else:
        raise ValueError("categorical_axis must be 'x' or 'y'!")
    
    # sort
    if sort_categories and (len(data_long[categorical_var].unique()) > 1) and not category_order:
        if sort_categories not in ["mean", "median"]:
            sort_categories = "mean"  
        category_order = data_long[[categorical_var, continuous_var]] \
            .groupby(categorical_var).apply(sort_categories) \
            .sort_values(continuous_var, ascending=False) \
            .index.to_list()
            
    if category_order:
        data_long[categorical_var] = pd.Categorical(data_long[categorical_var], category_order, ordered=True)
        
    # color by continuous variable
    if "color" in limits.keys():
        data_range_lim = limits.pop("color")
    if color_how:
        if color_how.lower().startswith("cont") and data_long.shape[0] > 1:
            color_var = continuous_var
            data_min, data_max = data_long[continuous_var].min(), data_long[continuous_var].max()
            if color_which == "auto":
                if (data_min > 0 and data_max < 0) or (data_min < 0 and data_max > 0):
                    color_which = "icefire"
                    color_center = True
                else:
                    color_which = "inferno"
            if color_center:
                data_range = np.max(np.abs([data_min, data_max]))
                data_range = (-data_range, data_range)
            else:
                data_range = (data_min, data_max)
            if "data_range_lim" in locals():
                data_range = list(data_range)
                for i, lim in enumerate(data_range_lim):
                    data_range[i] = lim if lim is not None else data_range[i]
                data_range = tuple(data_range)
                
        # color by categorical variable
        elif color_how.lower().startswith("cat"):
            color_var = categorical_var
            color_which = "Spectral"    
            
        # else no coloring
        else:
            color_var = None    
    else:
        color_var = None   
            
    # label handler
    def handle_label(label, agg_method=None):
        if not label:
            return None
        elif label == True:
            return agg_method if not legend["nice_labels"] else nice_stats_labels(agg_method, False)
        else:
            return label
    
    ## PLOT OBJECT
    plot = sno.Plot(data=data_long, **xy, fill=group_var)
    
    ## SCATTER
    if scatters["plot"] and data_long.shape[0] > 1:
        if scatters["size"] == "auto":
            n_max = data_long.groupby(categorical_var, observed=False).count().max().values[0]
            n_cat = len(data_long[categorical_var].unique())
            scatters["size"] = max(1.5, min(5, 5 / (0.1 * n_max**0.5) / ((0.3 * n_cat**0.5) if n_cat > 1 else 1)))
        plot = plot.add(
            sno.Dots(pointsize=scatters["size"], artist_kws=scatters["kwargs"]), 
            sno.Jitter(x=scatters["jitter_width"] if categorical_axis == "x" else None,
                       y=scatters["jitter_width"] if categorical_axis == "y" else None),
            *(sno.Dodge(gap=scatters["dodge_width"])) if group_var else (),
            color=color_var,
            label=handle_label(scatters["label"]),
            legend=False if color_var==categorical_var else True
        )
        
    ## VIOLINS
    if violins["plot"] and data_long.shape[0] > 1:
        sns.violinplot(
            data=data_long, **xy, hue=group_var,   
            label=handle_label(violins["label"]),
            **violins["kwargs"]         
        )
        
    ## BARS
    if bars["plot"]:
        if not color_var:
            plot = plot.add(
                sno.Bar(width=bars["width"], artist_kws=bars["kwargs"]), 
                sno.Agg(func=bars["agg_method"]),
                *(sno.Dodge(gap=bars["dodge_width"])) if group_var else (),
                label=handle_label(bars["label"], bars["agg_method"])
            )
        else:
            tmp = data_long[[categorical_var, continuous_var]] \
                .groupby(categorical_var, observed=False).apply(bars["agg_method"])
            plot = plot.add(
                sno.Bar(width=bars["width"], artist_kws=bars["kwargs"]), 
                data=tmp,
                **xy,
                color=color_var,
                legend=False if color_var==categorical_var else True,
                label=None
            )
        
    ## DOTS
    if dots["plot"]:
        plot = plot.add(
            sno.Dots(pointsize=dots["size"], color=dots["color"], artist_kws=dots["kwargs"]), 
            sno.Agg(func=dots["agg_method"]),
            *(sno.Dodge(gap=dots["dodge_width"])) if group_var else (),
            label=handle_label(dots["label"], dots["agg_method"])
        )
    
    ## ERRORBARS
    if errorbars["plot"] and data_long.shape[0] > 1:
        plot = plot.add( 
            sno.Range(color=errorbars["color"], artist_kws=errorbars["kwargs"]), 
            sno.Est(errorbar=errorbars["agg_method"]),
            *(sno.Dodge(gap=errorbars["dodge_width"])) if group_var else (),
            label=handle_label(errorbars["label"], errorbars["agg_method"])
        )
        
    ## HORIZONTAL AND VERTICAL LINES
    for hvline, xy in zip([hline, vline], ["y", "x"]):
        if hvline["plot"]:
            if isinstance(hvline[xy], (int, float)):
                hvline[xy] = [hvline[xy]]
            for hvline_xy in hvline[xy]:
                kws = dict(c=hvline["color"], lw=hvline["linewidth"], ls=hvline["linestyle"],
                           zorder=hvline["zorder"],
                           **hvline["kwargs"])
                if xy == "y":
                    ax.axhline(hvline_xy, **kws)
                else:
                    ax.axvline(hvline_xy, **kws)

    ## MULTI-STYLE REFERENCE LINES (hlines / vlines — list of per-line dicts)
    _ref_handles, _ref_labels = [], []
    for specs, draw_fn in [(hlines, ax.axhline), (vlines, ax.axvline)]:
        if not specs:
            continue
        for spec in specs:
            pos = spec.get("y", spec.get("x", 0))
            kws = dict(
                color=spec.get("color", "dimgrey"),
                linewidth=spec.get("linewidth", 1),
                linestyle=spec.get("linestyle", "--"),
                zorder=spec.get("zorder", -100),
            )
            h = draw_fn(pos, **kws)
            label = spec.get("label")
            if label:
                _ref_handles.append(h)
                _ref_labels.append(label)
    if _ref_handles:
        ax.legend(handles=_ref_handles, labels=_ref_labels)

    ## LIMITS
    plot = plot.limit(**limits)
                   
    ## COLOR
    if color_var == continuous_var:
        plot = plot.scale(
            color=sno.Continuous(color_which, norm=data_range)
        )
    elif color_var == categorical_var:
        plot = plot.scale(
            color=sno.Nominal(color_which)
        )
        
    ## FINALIZE
    plot = plot.label(**labels).on(ax).plot()

    ## LEGEND
    if legend["plot"]:
        move_legend_fig_to_ax(fig, ax, loc=legend["loc"], bbox_to_anchor=legend["bbox_to_anchor"],
                              **legend["kwargs"])
    else:
        fig.legends[-1].set_visible(False)
        
    return plot


def nullplot(fig, ax, data_long, categorical_var="variable", continuous_var="value",
             categorical_axis="x", category_order=None,
             color_which="viridis_r",
             quantiles_below_median=[0.01, 0.05, 0.25],
             bands=None,
             median_line=None,
             violins=None,
             labels=None,
             limits=None,
             legend=None
             ):   
    bands = dict(
        plot=True, label=True, label_prefix="Null percentile ",
        alpha=0.1,
        edgewidth=1,
        edgestyle="-",
        edgealpha=0.3,
        kwargs={"zorder": -200}
    ) | ({} if bands is None else bands)
    median_line = dict(
        plot=True, label="Null percentile 50",
        alpha=0.6,
        kwargs={"zorder": -150}
    ) | ({} if median_line is None else median_line)
    violins = dict(
        plot=False, label="Null distribution",
        legend=None,
        kwargs={"zorder": 100, "density_norm": "width", "cut": 0, "inner": "quart",
                "fill": False, "edgecolor": "k", "linewidth": 1}
    ) | ({} if violins is None else violins)
    legend = dict(
        plot=True,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        nice_labels=True,
        kwargs={}
    ) | ({} if legend is None else legend)
    labels = dict() | ({} if labels is None else labels)
    limits = dict() | ({} if limits is None else limits)
    
    # orientation   
    if categorical_axis not in ["x", "y"]: 
        raise ValueError("categorical_axis must be 'x' or 'y'!") 
    continuous_axis = "y" if categorical_axis == "x" else "x"
    
    # correct label
    labels = {
        continuous_axis: continuous_var
     } | labels
    
    # data
    #data_long[categorical_var] = pd.Categorical([str(c) for c in data_long[categorical_var]])
    
    # percentiles
    quantiles_below_median = np.array(quantiles_below_median)
    quantiles = np.concatenate([quantiles_below_median, np.array([0.5]), 1 - quantiles_below_median])
    quantiles.sort()
    data_quant = pd.concat(
        [data_long.groupby(categorical_var).quantile(q).rename(columns={continuous_var: q}) \
         for q in quantiles], 
        axis=1
    ).reset_index()
        
    # color
    if color_which:
        try:
            colors = sns.color_palette(color_which, len(quantiles_below_median) + 1)# [1:-1]
        except KeyError:
            colors = [color_which] * len(quantiles_below_median) 
    else:
        colors = ["0.3"] * len(quantiles_below_median)
    
    ## PLOT OBJECT
    plot = sno.Plot() # data=data_long, **xy
    
    ## AREAS
    for i_q, q in enumerate(quantiles[quantiles < 0.5]):
        
        if bands["plot"]:
            xy = {
                categorical_axis: categorical_var,
                f"{continuous_axis}min": q,
                f"{continuous_axis}max": 1 - q
            }
            plot = plot.add(
                sno.Band(color=colors[i_q], alpha=bands["alpha"], edgewidth=bands["edgewidth"],
                         edgestyle=bands["edgestyle"], edgealpha=bands["edgealpha"], 
                         artist_kws=bands["kwargs"]),
                data=data_quant,
                **xy,
                label=f"{bands['label_prefix']}{q*100:.0f}/{(1-q)*100:.0f}" if bands["label"] else None
            )
    
    # MEDIAN
    if median_line["plot"]:
        xy = {
            categorical_axis: categorical_var,
            continuous_axis: 0.5,
        }
        plot = plot.add(
            sno.Line(color=colors[-1], alpha=median_line["alpha"], ), #artist_kws=lines["kwargs"]
            data=data_quant,
            **xy,
            label=median_line['label'] if median_line["label"] else None
        )
        
    # VIOLINS
    if violins["plot"]:
        xy = {
            categorical_axis: categorical_var,
            continuous_axis: continuous_var,
        }
        sns.violinplot(
            data=data_long, **xy, 
            label=violins["label"],
            legend=violins["legend"],
            **violins["kwargs"]         
        )
    
    # FINALIZE
    plot = plot.limit(**limits).label(**labels).on(ax).plot()
    
    ## LEGEND
    if legend["plot"]:
        move_legend_fig_to_ax(fig, ax, loc=legend["loc"], bbox_to_anchor=legend["bbox_to_anchor"],
                              **legend["kwargs"])
    else:
        fig.legends[-1].set_visible(False)

    return plot


def heatmap(ax,
            data_colors=None,
            data_sizes=None, 
            data_shapes=None,
            mapping_shapes=None,
            annotation=None,
            mask=None,
            cmap="auto",
            symmetric_cmap=True,
            size_scale=(0.2, 1),
            shape="square",
            color="tab:blue",
            edgecolor="k",
            linewidth=0.075,
            square=True,
            spines=False,
            spinewidth=0.1,
            spinecolor="k",
            xy_pad=0.1,
            xtick_labels=None,
            ytick_labels=None,
            legend_orientation="vertical",
            legend_colors=True,
            legend_colors_kwargs=None,
            legend_sizes=True,
            legend_sizes_kwargs=None,
            legend_shapes=True,
            legend_shapes_kwargs=None,
            ):
    
    # kwargs
    legend_colors_kwargs = {} if legend_colors_kwargs is None else legend_colors_kwargs
    legend_sizes_kwargs = {} if legend_sizes_kwargs is None else legend_sizes_kwargs
    legend_shapes_kwargs = {} if legend_shapes_kwargs is None else legend_shapes_kwargs
    
    # input arrays
    arrays = [data_colors, data_sizes, data_shapes, annotation, mask]
    arrays = [arr for arr in arrays if arr is not None]
    if len(arrays) == 0:
        raise ValueError("No input arrays provided.")
    if not all([isinstance(arr, (np.ndarray, pd.DataFrame)) for arr in arrays]):
        raise ValueError("All input arrays must be 2d arrays or dataframes.")
    shapes = [arr.shape for arr in arrays]
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            if shapes[i] != shapes[j]:
                raise ValueError("All input arrays must have the same shape.")
    
    # x labels
    autolabels = {"x": False, "y": False}
    if xtick_labels is None:
        autolabels["x"] = True
        for arr in arrays:
            if isinstance(arr, pd.DataFrame):
                xtick_labels = arr.columns.to_list()
                break
    if xtick_labels is None:
        xtick_labels = list(range(arrays[0].shape[1]))
    if len(xtick_labels) != arrays[0].shape[1]:
        raise ValueError("xtick_labels must have the same length as the number of columns in the input array(s).")
    xtick_labels = [str(x) for x in xtick_labels]
        
    # y labels
    if ytick_labels is None:
        autolabels["y"] = True
        for arr in arrays:
            if isinstance(arr, pd.DataFrame):
                ytick_labels = arr.index.to_list()
                break
    if ytick_labels is None:
        ytick_labels = list(range(arrays[0].shape[0]))
    if len(ytick_labels) != arrays[0].shape[0]:
        raise ValueError("ytick_labels must have the same length as the number of rows in the input array(s).")
    ytick_labels = [str(y) for y in ytick_labels]
    
    # make indices
    x_idc = np.arange(len(xtick_labels))
    y_idc = np.arange(len(ytick_labels))
    x_idc_2d, y_idc_2d = np.meshgrid(x_idc, y_idc)
    data_x = x_idc_2d.flatten()
    data_y = y_idc_2d.flatten()
    
    # plotting sizes
    if data_sizes is not None:
        data_sizes = np.array(data_sizes).flatten("C")
        if not np.issubdtype(data_sizes.dtype, np.number):
            raise ValueError("data_sizes array must be numeric.")
        data_sizes_input = data_sizes.copy()
        data_sizes = minmax_scale(data_sizes, size_scale)
    else:
        data_sizes = np.ones(len(data_x)) * size_scale[1]
        
    # plotting colors
    if data_colors is not None:
        data_colors = np.array(data_colors).flatten("C")
        if not np.issubdtype(data_colors.dtype, np.number):
            raise ValueError("data_colors array must be numeric.")
        if symmetric_cmap:
            vmax = np.nanmax(np.abs(data_colors))
            vmin = -vmax
            if cmap in ["auto", None, False]:
                cmap = "icefire"
        else:
            vmin, vmax = np.nanmin(data_colors), np.nanmax(data_colors)
            if cmap in ["auto", None, False]:
                cmap = "magma"
        if isinstance(cmap, str):
            cmap = mpl.colormaps[cmap]
    else:
        data_colors = np.ones((len(data_x)))
        vmin, vmax = 1, 1
        cmap = mpl.colors.ListedColormap([color])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)   
 
    # plotting shapes   
    if data_shapes is not None:
        data_shapes = np.array(data_shapes).flatten("C")
        data_shapes_unique = np.unique(data_shapes)
        if data_shapes_unique[~pd.isnull(data_shapes_unique)].shape[0] > 3:
            raise ValueError("For data_shapes, maximally 3 unique values are supported.")
        if mapping_shapes is None:
            mapping_shapes = {val: shape for val, shape in zip(data_shapes_unique, ["s", "o", "D"])}
        if not all([shape in ["circle", "o", "square", "s", "diamond", "D", ""] for shape in mapping_shapes.values()]):
            raise ValueError("mapping_shapes values must be 'circle'/'o', 'square'/'s', or 'diamond'/'D'")
        data_shapes = [mapping_shapes[val] if ~pd.isnull(val) else "" for val in data_shapes]
    else:
        data_shapes = [shape] * len(data_x)
        
    # mask
    if mask is not None:
        mask = np.array(mask).flatten("C")
        if not np.issubdtype(mask.dtype, bool):
            raise ValueError("mask array must be boolean.")
    else:
        mask = [True] * len(data_x)
    
    # plot
    elements = []
    if linewidth in [0, None, False]:
        linewidth = 0
    if spinewidth in [0, None, False]:
        spinewidth = 0
    lw = linewidth_from_data_units(linewidth / len(xtick_labels), ax)
    sw = linewidth_from_data_units(spinewidth / len(xtick_labels), ax)
    kwargs_base = {
        "lw": lw,
        "ec": edgecolor,
        "joinstyle": "bevel"
    }
    for mask, x, y, color, shape, size in zip(mask, data_x, data_y, data_colors, data_shapes, data_sizes):
        if mask and ~np.isnan(color) and ~np.isnan(size) and shape!="":
            if shape in ["o", "circle"]:
                fun = plt.Circle
                kwargs = {
                    "xy": (x, y),
                    "radius": size / 2,
                    **kwargs_base
                }
            elif shape in ["s", "square"]:
                fun = plt.Rectangle
                kwargs = {
                    "xy": (x - size / 2, y - size / 2),
                    "width": size,
                    "height": size,
                    **kwargs_base
                }
            elif shape in ["diamond", "D"]:
                fun = plt.Rectangle
                size *= 0.7
                kwargs = {
                    "xy": (x - size / 2, y - size / 2),
                    "width": size,
                    "height": size,
                    "angle": 45,
                    "rotation_point": "center",
                    **kwargs_base
                }
            elements.append(fun(**kwargs))

    collection = PatchCollection(elements, cmap=cmap, norm=norm, match_original=True)
    if data_colors is not None:
        collection.set_array(data_colors)
    ax.add_collection(collection)
    
    # padding at the outside of x and y axes
    xy_lims = 0.5
    ax.set_xlim(-xy_lims - xy_pad, len(xtick_labels) - xy_lims + xy_pad)
    ax.set_ylim(len(ytick_labels) - xy_lims + xy_pad, -xy_lims - xy_pad)
    
    # spines
    for spine in ax.spines.values():
        if spines:
            spine.set_linewidth(sw)
            spine.set_color(spinecolor)
        else:
            spine.set_visible(False)
            
    # labels
    ax.set_xticks(x_idc, xtick_labels, rotation=45, ha="right", va="center", rotation_mode="anchor")
    ax.set_yticks(y_idc, ytick_labels)
    if autolabels["x"]:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if autolabels["y"] and len(ytick_labels) > 5:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # layout
    if square:
        ax.set_aspect("equal")
        
    # legends
    # colors
    if legend_colors and np.unique(data_colors).shape[0] > 1:
        if "cax" in legend_colors_kwargs:
            cax = legend_colors_kwargs.pop("cax")
        else:
            cax = ax.inset_axes((0, 1.1, 1/3, 0.05) if legend_orientation == "horizontal" else (1.05, 2/3, 0.05, 1/3))   
        legend_colors_kwargs = {
            "label": "Colors",
            "orientation": legend_orientation,
            "cax": cax
        } | legend_colors_kwargs
        plt.colorbar(collection, **legend_colors_kwargs)
    # sizes
    if legend_sizes and np.unique(data_sizes).shape[0] > 1:
        legend_sizes_kwargs = {
            "title": "Sizes",
            "labelspacing": 1,
            "bbox_to_anchor": (0.5, 1.05) if legend_orientation == "horizontal" else (1.03, 0.45),
            "ncol": 2 if legend_orientation == "horizontal" else 1,
            "loc": "lower center" if legend_orientation == "horizontal" else "center left",
            "fmt": ".2f"
        } | legend_sizes_kwargs
        fmt = legend_sizes_kwargs.pop('fmt')
        handles = []
        for size, size_label in zip(np.linspace(data_sizes.min(), data_sizes.max(), 5),
                                    np.linspace(data_sizes_input.min(), data_sizes_input.max(), 5)):
            handles.append(
                mpl.lines.Line2D(
                    [0], [0], 
                    color=(0,0,0,0), 
                    marker="s", 
                    markerfacecolor="k", 
                    markeredgewidth=0.5,
                    markeredgecolor="w",
                    markersize=linewidth_from_data_units(size, ax),
                    label=f"{size_label:{fmt}}"
                )
            )
        lax = ax.inset_axes((0,0,1,1))
        lax.axis("off")
        lax.legend(handles=handles, **legend_sizes_kwargs)
    # shapes
    if legend_shapes and np.unique(data_shapes).shape[0] > 1:
        legend_shapes_kwargs = {
            "title": "Shapes",
            "labelspacing": 1,
            "bbox_to_anchor": (0.5 + 0.33, 1.05) if legend_orientation == "horizontal" else (1.03, 0),
            "loc": "lower center" if legend_orientation == "horizontal" else "lower left",
        } | legend_shapes_kwargs
        handles = []
        for val, shape in mapping_shapes.items():
            handles.append(
                mpl.lines.Line2D(
                    [0], [0], 
                    color=(0,0,0,0), 
                    marker=shape, 
                    markerfacecolor="k", 
                    markeredgewidth=0.5,
                    markeredgecolor="w",
                    markersize=linewidth_from_data_units(0.5 if shape in ["D", "diamond"] else 0.7, ax),
                    label=val
                )
            )
        lax = ax.inset_axes((0,0,1,1))
        lax.axis("off")
        lax.legend(handles=handles, **legend_shapes_kwargs)
        
    return ax, collection

def view_surf(data=None, parcellation=None, hemi="L", template="fsaverage", replace_nan=0,
              template_kwargs=None, parcellation_kwargs=None,
              verbose=False, **kwargs):
    set_log(lgr, verbose)
    
    # kwargs 
    template_kwargs = {} if template_kwargs is None else template_kwargs
    parcellation_kwargs = {} if parcellation_kwargs is None else parcellation_kwargs
    
    # parcellation and data
    if data is None and parcellation is None:
        raise ValueError("Either data or parcellation must be provided")
    
    # template
    if not isinstance(template, str):
        raise NotImplementedError(f"For now, template must be a string: {list(template_lib.keys())}")
    else:
        space = template
        template = fetch_template(template, hemi=hemi, **template_kwargs, verbose=verbose)
        template = images.load_gifti(template)
        template_arr = template.agg_data()
        
    # parcellation
    if parcellation is not None:
        if not isinstance(parcellation, str):
            raise NotImplementedError(f"For now, parcellation must be a string: {list(parcellation_lib.keys())}")
        else:
            parc, labels = fetch_parcellation(parcellation, space=space, return_loaded=True, **parcellation_kwargs)
            parc_arr = parc[0 if hemi == "L" else 1].agg_data()
            labels = [l for l in labels if f"hemi-{hemi}" in l]
    if data is None:
        data = np.trim_zeros(np.unique(parc_arr))
        
    # data
    if not isinstance(data, (list, pd.Series, pd.DataFrame, np.ndarray)):
        raise ValueError(f"Data must be a list, pd.Series, pd.DataFrame or np.array, not {type(data)}")
    else:
        data = np.squeeze(np.array(data))
    if data.ndim > 1:
        raise ValueError(f"Data must be a 1D array, not {data.ndim}D")
    
    # check dimensions
    if len(data) == len(template_arr):
        data_arr = data     
    elif "parc_arr" in locals():
        if len(data) == len(labels):
            data_arr = vect_to_vol_arr(data, parc_arr, np.trim_zeros(np.unique(parc_arr)))
            # replace nan
            if replace_nan is not False:
                data_arr = np.nan_to_num(data_arr, replace_nan)
            
        else:
            raise ValueError(f"Data length ({len(data)}) must match number of parcels ({len(labels)})")
    else:
        raise ValueError("Data must match either the shape of the template or the number of parcels"
                         f"template: {template_arr.shape}, data: {data_arr.shape}")
    
    # plot
    return view_surf_nilearn(surf_map=data_arr, surf_mesh=template_arr, **{"cmap": "RdBu_r"} | kwargs)


# ==============================================================================================
# BRAIN PLOT
# ==============================================================================================

def _auto_vmin_vmax(data_flat, symmetric, vmin=None, vmax=None):
    """Compute display range from a flat data array, optionally centering at 0."""
    data_flat = data_flat[np.isfinite(data_flat)]
    v_min = float(vmin) if vmin is not None else float(np.nanmin(data_flat))
    v_max = float(vmax) if vmax is not None else float(np.nanmax(data_flat))
    if symmetric:
        lim = max(abs(v_min), abs(v_max))
        return -lim, lim
    return v_min, v_max


def _load_fslr_assets(surf_mesh="inflated"):
    """Load fslr32k surface geometry, sulcal background, and medial wall mask via neuromaps.

    Returns
    -------
    surf_geom : (lh_GiftiImage, rh_GiftiImage)
    bg_data   : (sulc_lh_array, sulc_rh_array)
    medial    : (medial_lh_array, medial_rh_array)
    """
    # TODO: ADJUST THIS TO ONLY LOAD VIA NISPACE FETCHERS
    from neuromaps.datasets import fetch_fslr
    fslr = fetch_fslr(density="32k")
    valid = [k for k in ("pial", "inflated", "midthickness", "veryinflated") if k in fslr]
    if surf_mesh not in fslr:
        raise ValueError(
            f"surf_mesh='{surf_mesh}' not available for fsLR. Choose from: {valid}"
        )
    surf_lh = images.load_gifti(str(fslr[surf_mesh].L))
    surf_rh = images.load_gifti(str(fslr[surf_mesh].R))
    sulc_lh = images.load_gifti(str(fslr["sulc"].L)).agg_data()
    sulc_rh = images.load_gifti(str(fslr["sulc"].R)).agg_data()
    medial_lh = images.load_gifti(str(fslr["medial"].L)).agg_data()
    medial_rh = images.load_gifti(str(fslr["medial"].R)).agg_data()
    return (surf_lh, surf_rh), (sulc_lh, sulc_rh), (medial_lh, medial_rh)


def _load_fsaverage_assets(surf_mesh="pial"):
    """Load fsaverage surface geometry, sulcal background, and medial wall mask via neuromaps.

    Returns
    -------
    surf_geom : (lh_GiftiImage, rh_GiftiImage)
    bg_data   : (sulc_lh_array, sulc_rh_array)
    medial    : (medial_lh_array, medial_rh_array)
    """
    # TODO: ADJUST THIS TO ONLY LOAD VIA NISPACE FETCHERS
    from neuromaps.datasets import fetch_fsaverage
    fsavg = fetch_fsaverage(density="41k")
    valid = [k for k in ("pial", "inflated", "white") if k in fsavg]
    if surf_mesh not in fsavg:
        lgr.warning(
            f"surf_mesh='{surf_mesh}' not available for fsaverage. "
            f"Choose from: {valid}. Falling back to 'pial'."
        )
        surf_mesh = "pial"
    surf_lh = images.load_gifti(str(fsavg[surf_mesh].L))
    surf_rh = images.load_gifti(str(fsavg[surf_mesh].R))
    sulc_lh = images.load_gifti(str(fsavg["sulc"].L)).agg_data()
    sulc_rh = images.load_gifti(str(fsavg["sulc"].R)).agg_data()
    medial_lh = images.load_gifti(str(fsavg["medial"].L)).agg_data()
    medial_rh = images.load_gifti(str(fsavg["medial"].R)).agg_data()
    return (surf_lh, surf_rh), (sulc_lh, sulc_rh), (medial_lh, medial_rh)


def _data_to_surf_verts(data_lh, data_rh, parc_arr_lh, parc_arr_rh, medial=None):
    """Map per-parcel 1-D data to per-vertex arrays for both hemispheres.

    Medial wall vertices are set to NaN when a medial wall mask is provided
    (fslr32k: medial[h]==0 marks the medial wall).
    """
    idc_lh = np.trim_zeros(np.unique(parc_arr_lh)).astype(np.float64)
    idc_rh = np.trim_zeros(np.unique(parc_arr_rh)).astype(np.float64)
    vert_lh = vect_to_vol_arr(
        data_lh.astype(np.float64), parc_arr_lh.astype(np.float64), idc_lh
    )
    vert_rh = vect_to_vol_arr(
        data_rh.astype(np.float64), parc_arr_rh.astype(np.float64), idc_rh
    )
    if medial is not None:
        vert_lh = np.where(medial[0] == 0, np.nan, vert_lh)
        vert_rh = np.where(medial[1] == 0, np.nan, vert_rh)
    return vert_lh, vert_rh


def _render_surf_row(ax, fig, vert_lh, vert_rh, parc_arr_lh, parc_arr_rh,
                     surf_geom, bg_data, views, cmap, vmin, vmax,
                     symmetric_cmap, bg_on_data, darkness, threshold, alpha,
                     plot_contours, zoom, black_bg=False, **kwargs):
    """Render one brain map onto surfaces inside *ax* via n inset 3-D subaxes."""
    surf_lh, surf_rh = surf_geom
    n_views = len(views)

    # Bake alpha into the colormap once before the per-view loop.
    # With bg_on_data=False the colormap alpha is preserved; mix_colormaps
    # then produces: stat_color*alpha + bg_color*(1-alpha) — the correct blend.
    # When alpha==1 we fall through to the normal bg_on_data behaviour.
    import matplotlib.colors as _mcolors
    _plot_cmap = cmap
    _plot_bg_on_data = bg_on_data
    if alpha is not None and float(alpha) != 1.0:
        _a = float(alpha)
        _cmap_obj = (mpl.colormaps[cmap] if isinstance(cmap, str)
                     else cmap if isinstance(cmap, _mcolors.Colormap)
                     else _mcolors.LinearSegmentedColormap.from_list("_c", cmap))
        _rgba = _cmap_obj(np.linspace(0, 1, 256))
        _rgba[:, 3] = _a
        _plot_cmap = _mcolors.ListedColormap(_rgba)
        _plot_bg_on_data = False  # let colormap alpha drive the blend

    for i, view_str in enumerate(views):
        hemi_str, view_name = view_str.split("_", 1)
        is_left = hemi_str == "left"
        surf      = surf_lh      if is_left else surf_rh
        bg        = bg_data[0]   if (bg_data is not None and is_left) else (
                    bg_data[1]   if bg_data is not None else None)
        vert      = vert_lh      if is_left else vert_rh
        parc_arr  = parc_arr_lh  if is_left else parc_arr_rh

        ax_3d = ax.inset_axes([i / n_views, 0, 1 / n_views, 1], projection="3d")

        plot_surf_stat_map(
            surf_mesh=surf.agg_data(),
            stat_map=vert,
            bg_map=bg,
            hemi=hemi_str,
            view=view_name,
            cmap=_plot_cmap,
            vmin=vmin,
            vmax=vmax,
            symmetric_cbar=symmetric_cmap,
            bg_on_data=_plot_bg_on_data,
            darkness=darkness,
            threshold=threshold,
            colorbar=False,
            axes=ax_3d,
            figure=fig,
            **kwargs,
        )
        if plot_contours:
            # levels/labels/colors must all have the same length; computing
            # levels explicitly avoids nilearn's default [1:]-skip which
            # drops label 1 when the gifti has no background (label 0).
            _levels = list(np.trim_zeros(np.unique(parc_arr)).astype(int))
            plot_surf_contours(
                surf_mesh=surf.agg_data(),
                roi_map=parc_arr,
                hemi=hemi_str,
                view=view_name,
                levels=_levels,
                labels=[None] * len(_levels),
                colors=["k"] * len(_levels),
                axes=ax_3d,
                figure=fig,
            )
        if black_bg:
            ax_3d.set_facecolor("black")
            for _pane in (ax_3d.xaxis.pane, ax_3d.yaxis.pane, ax_3d.zaxis.pane):
                _pane.fill = True
                _pane.set_facecolor("black")
                _pane.set_edgecolor("black")
        ax_3d.set_box_aspect(ax_3d.get_box_aspect(), zoom=zoom)


def _data_to_volume(data, ref_nii, labels_in_img, bg_value=np.nan):
    """Project per-parcel *data* onto a NIfTI volume using *labels_in_img* as the mapping.

    Voxels outside all parcels and NaN data entries are filled with *bg_value*
    (default NaN → nilearn converts these to 0 via nan_to_num).
    """
    parc_arr = ref_nii.get_fdata()
    stat_arr = np.full(parc_arr.shape, bg_value, dtype=np.float32)
    for i, label in enumerate(labels_in_img):
        if i < len(data):
            val = float(data[i])
            stat_arr[parc_arr == label] = bg_value if not np.isfinite(val) else val
    return new_img_like(ref_nii, stat_arr, copy_header=True)


def _render_vol_row(ax, fig, stat_nii, bg_img, kind, display_mode, cut_coords,
                    cmap, vmin, vmax, symmetric_cmap, threshold, alpha,
                    draw_cross, colorbar, dim="auto", colorbar_label="", colorbar_inset=None,
                    draw_brain=True, zoom=1.0, black_bg=False,
                    **kwargs):
    """Render one brain map as a glass brain or anatomical slices into *ax*."""
    if kind == "glass":
        _axes_before = set(id(a) for a in fig.axes)
        plot_glass_brain(
            stat_nii,
            figure=fig, axes=ax,
            display_mode=display_mode,
            transparency=alpha,  # controls stat map overlay opacity
            plot_abs=False,
            cmap=cmap, vmin=vmin, vmax=vmax,
            symmetric_cbar=symmetric_cmap,
            threshold=threshold,
            colorbar=False,
            annotate=False,
            black_bg=black_bg,
            **kwargs,
        )
        _new_axes = [a for a in fig.axes if id(a) not in _axes_before]
        if not draw_brain:
            for _sub_ax in _new_axes:
                for _patch in _sub_ax.patches:
                    _patch.set_visible(False)
        if zoom != 1.0:
            for _sub_ax in _new_axes:
                for _getter, _setter in (
                    (_sub_ax.get_xlim, _sub_ax.set_xlim),
                    (_sub_ax.get_ylim, _sub_ax.set_ylim),
                ):
                    _lo, _hi = _getter()
                    _center = (_lo + _hi) / 2
                    _half = (_hi - _lo) / (2 * zoom)
                    _setter(_center - _half, _center + _half)
        if colorbar:
            _norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            _sm = plt.cm.ScalarMappable(cmap=cmap, norm=_norm)
            _sm.set_array([])
            _inset = colorbar_inset if colorbar_inset is not None else [1.02, 0.15, 0.02, 0.7]
            _cax = ax.inset_axes(_inset)
            fig.colorbar(_sm, cax=_cax)
            if colorbar_label:
                _cax.set_title(colorbar_label, fontsize="medium")
    else:  # slice
        plot_stat_map(
            stat_nii,
            bg_img=bg_img,
            cut_coords=cut_coords,
            draw_cross=draw_cross,
            figure=fig, axes=ax,
            cmap=cmap, vmin=vmin, vmax=vmax,
            symmetric_cbar=symmetric_cmap,
            colorbar=False,
            threshold=threshold,
            display_mode=display_mode,
            black_bg=black_bg,
            annotate=False,
            transparency=alpha,
            dim=dim,
            **kwargs,
        )
        if colorbar:
            _norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            _sm = plt.cm.ScalarMappable(cmap=cmap, norm=_norm)
            _sm.set_array([])
            _inset = colorbar_inset if colorbar_inset is not None else [1.04, 0.25, 0.02, 0.5]
            _cax = ax.inset_axes(_inset)
            fig.colorbar(_sm, cax=_cax)
            if colorbar_label:
                _cax.set_title(colorbar_label, fontsize="medium")


def brainplot(
    data,
    parcellation=None,
    kind=None,
    space=None,
    surf_mesh="inflated",
    views=None,
    plot_contours=False,
    zoom=None,
    bg_on_data=True,
    darkness=0.7,
    dim="auto",
    threshold="auto",
    alpha=0.8,
    display_mode=None,
    cut_coords=None,
    bg_img=None,
    draw_cross=False,
    draw_brain=True,
    black_bg=False,
    cmap=None,
    vmin=None,
    vmax=None,
    shared_colorscale=False,
    symmetric_cmap="auto",
    colorbar=True,
    colorbar_label="",
    colorbar_inset=None,
    ncols=1,
    title="auto",
    hspace=None,
    wspace=None,
    figsize=None,
    fig=None,
    axes=None,
    title_kwargs=None,
    verbose=False,
    **kwargs,
):
    """Plot brain maps onto surfaces or anatomical volumes.

    Parameters
    ----------
    data : pd.DataFrame, pd.Series, array-like, NIfTI image, or GIfTI pair
        Tabular input: shape (n_maps, n_parcels) — requires *parcellation*.
        NIfTI image (``nib.Nifti1Image`` or path): 3D volume for glass/slice
        plots — *parcellation* must be ``None``.
        GIfTI pair: ``(lh, rh)`` tuple or list of such tuples for surface
        plots, where each element is a ``nib.GiftiImage``, a path, or a
        vertex-data array — *parcellation* must be ``None``.
    parcellation : Parcellation, str, Path, NIfTI image, GIfTI image, tuple, or None
        Parcellation to use for mapping tabular data onto the brain.
        Required when *data* is tabular; must be ``None`` for NIfTI/GIfTI image input.
        Accepted forms:

        * ``Parcellation`` — a fitted NiSpace Parcellation object.
        * ``str`` — NiSpace library name (e.g. ``"Schaefer400"``) **or** a
          file-system path to a NIfTI/GIfTI parcellation image.
        * ``pathlib.Path`` — path to a NIfTI or GIfTI parcellation image.
        * ``nib.Nifti1Image`` — volumetric parcellation image.
        * ``nib.GiftiImage`` or ``tuple`` — surface parcellation; pass a
          ``(lh, rh)`` tuple where each element is a ``GiftiImage`` or path.
    kind : {"surface", "glass", "slice", "combined"}, optional
        Rendering mode. Defaults to ``"surface"`` when *parcellation* is a
        GIfTI image/tuple, ``"surface"`` for surface-only Parcellation objects,
        and ``"glass"`` otherwise. Use ``"combined"`` for surface+glass side-by-side
        (not yet implemented); pass ``"glass"`` or ``"slice"`` to render any
        parcellation as a plain MNI volume.
    space : str, optional
        Parcellation space to use for rendering. Defaults to fslr32k for
        surface plots and MNI152NLin2009cAsym for volume plots.
    surf_mesh : {"inflated", "pial", "midthickness", "veryinflated"}
        Surface geometry for fslr32k. Ignored for non-surface plots.
    views : list of str, optional
        Surface views as "<hemi>_<perspective>" strings.
        Default: ["left_lateral", "left_medial", "right_medial", "right_lateral"].
    plot_contours : bool
        Draw black parcel borders on surface plots.
    zoom : float, optional
        Zoom factor applied to each brain panel. Defaults to ``1.5`` for
        surface plots and ``1.0`` for glass/slice (no zoom).
    bg_on_data : bool
        Overlay sulcal shading on top of the statistical map.
    darkness : float
        Darkness of sulcal background shading (0 = bright, 1 = dark).
    dim : float or "auto"
        Dimming factor for the anatomical background in slice plots.
        Roughly −2 (more contrast) to +2 (dimmer). Default ``"auto"``.
    threshold : float, "auto", or None
        Values with absolute value below threshold are not displayed.
        Default ``"auto"`` sets threshold to ``min(|data|) / 2``, masking
        background zeros while keeping all parcel values visible. Pass
        ``None`` to disable thresholding entirely.
    alpha : float, optional
        Transparency of the statistical map. Default 0.8 for all kinds.
    display_mode : str, optional
        Nilearn display mode for glass/slice. Defaults to "lyrz" for glass
        (left, posterior, right, top) and "ortho" for slice.
    cut_coords : int or list, optional
        Number of auto-cuts or explicit coordinates for slice plots.
    bg_img : NIfTI image or path, optional
        Background anatomical for slice plots. Auto-fetched from NiSpace
        templates when None.
    draw_cross : bool
        Draw crosshair lines at slice positions.
    draw_brain : bool
        Draw the glass brain outline (grey wireframe). Set to ``False`` to
        show only the statistical overlay without the brain silhouette.
        Only applies to ``kind="glass"``.
    black_bg : bool
        Use a black figure and axes background. For glass/slice this is
        forwarded to nilearn's ``black_bg`` parameter; for surface it sets
        the figure patch and 3-D axes pane colors to black.
    cmap : str, optional
        Colormap name. Defaults to "RdBu_r" when the colorscale is symmetric
        and "viridis" otherwise (see symmetric_cmap).
    vmin, vmax : float, optional
        Color scale limits. None = auto-computed per map.
    shared_colorscale : bool
        Compute vmin/vmax across all maps (one shared colorbar). Always
        True for combined parcellations.
    symmetric_cmap : bool or "auto"
        Center color limits at 0. True → RdBu_r, False → viridis (unless
        cmap is set explicitly). "auto" (default) detects from the data:
        symmetric if values span both sides of zero, sequential otherwise.
    colorbar : bool
        Show colorbar. When shared, one horizontal bar is placed at the
        bottom center of the figure.
    colorbar_label : str
        Title string displayed above the colorbar. Empty string (default)
        means no title.
    colorbar_inset : list of float, optional
        Position of the colorbar axes as ``[x0, y0, width, height]`` in
        parent-axes coordinates. Defaults to ``[1.02, 0.15, 0.02, 0.7]``
        for glass brain and ``[1.04, 0.25, 0.02, 0.5]`` for surface/slice.
    ncols : int
        Number of map columns in the subplot grid.
    title : str, list, or bool, optional
        Map title above each brain. ``"auto"`` (default) uses the map's index
        label for tabular input and shows no title for NIfTI/GIfTI image input
        (where no meaningful label is available). Pass a string, a list of
        strings (one per map), or ``False``/``None`` to disable.
    hspace : float
        Vertical spacing between map rows (passed to GridSpec/subplots).
    wspace : float, optional
        Horizontal spacing between map columns (fraction of axes width). Defaults
        to 0.1 for glass/slice and 0.02 for surface; auto-increased for multi-column
        layouts to leave room for per-column colorbars.
    figsize : tuple, optional
        Figure size in inches. Auto-computed if None.
    fig : matplotlib.Figure, optional
        Existing figure to draw into.
    axes : list of matplotlib.Axes, optional
        Axes to draw into (must match layout when fig is provided).
    title_kwargs : dict, optional
        Extra keyword arguments passed to ``fig.text()`` for the title.
        Standard matplotlib text params (``fontsize``, ``color``, etc.) are
        accepted; defaults are ``{"fontsize": "large", "fontweight": "bold"}``.
        A special ``"y"`` key sets the title y-position as a fraction of the
        axes height (default: ``0.92`` inside the axes for surface, ``1.02``
        above for others).
    verbose : bool
        Enable verbose logging.
    **kwargs
        Extra keyword arguments forwarded to the underlying nilearn function
        (``plot_surf_stat_map``, ``plot_glass_brain``, or ``plot_stat_map``
        depending on *kind*).

    Returns
    -------
    fig : matplotlib.Figure
    axes_list : list of matplotlib.Axes
    """
    from matplotlib.gridspec import GridSpec
    set_log(lgr, verbose)
    lgr.warning("Brain plotting in NiSpace is experimental. "
                "If things look off, feel free to raise a GitHub issue!")

    if views is None:
        views = ["left_lateral", "left_medial", "right_medial", "right_lateral"]

    # -- detect input mode and normalise --
    import nibabel as _nib
    import pathlib as _pl

    def _load_gifti_arr(x):
        if isinstance(x, np.ndarray):
            return x.astype(np.float32)
        img = _nib.load(str(x)) if not isinstance(x, _nib.GiftiImage) else x
        return img.darrays[0].data.astype(np.float32)

    _is_nifti_like = isinstance(data, _nib.Nifti1Image) or (
        isinstance(data, (str, _pl.Path)) and str(data).endswith((".nii", ".nii.gz"))
    )
    _is_gifti_pair = isinstance(data, tuple) and len(data) == 2
    _is_gifti_list = (
        isinstance(data, list) and len(data) > 0
        and isinstance(data[0], tuple) and len(data[0]) == 2
    )

    _parc_is_gifti_input = (
        isinstance(parcellation, (_nib.GiftiImage, tuple))
        or (isinstance(parcellation, (str, _pl.Path))
            and str(parcellation).endswith((".gii", ".gii.gz")))
    )

    if _is_nifti_like:
        _img_mode = "nifti"
    elif _is_gifti_pair or _is_gifti_list:
        _img_mode = "gifti"
    else:
        _img_mode = None  # tabular

    # -- validate parcellation / image combination --
    if _img_mode is None and parcellation is None:
        raise ValueError(
            "'parcellation' is required when 'data' is a tabular array. "
            "Pass a Parcellation object or a parcellation name/path."
        )
    if _img_mode is not None and parcellation is not None:
        raise ValueError(
            "Set parcellation=None when passing a NIfTI or GIfTI image directly. "
            "'parcellation' is only used for tabular (parcellated) input."
        )

    # tabular normalisation
    if _img_mode is None:
        if isinstance(data, pd.Series):
            data = pd.DataFrame([data.values], index=[data.name or "map"])
        elif isinstance(data, np.ndarray):
            data = pd.DataFrame(np.atleast_2d(data))
        elif not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(np.atleast_2d(np.asarray(data)))

    # -- validate / load parcellation (tabular path only) --
    if _img_mode is None:
        from .core.parcellation import Parcellation as _Parc
        if not isinstance(parcellation, _Parc):
            if isinstance(parcellation, str):
                from .datasets import _check_parcellation, fetch_parcellation as _fp
                try:
                    _name = _check_parcellation(parcellation, force_str=True, raise_not_found=True)
                    parcellation = _fp(parcellation=_name)
                except (ValueError, AssertionError):
                    try:
                        parcellation = _Parc.from_path(source=parcellation, space=space)
                    except Exception as _e:
                        raise ValueError(
                            f"'{parcellation}' is neither a NiSpace library parcellation "
                            "nor a valid file path. Available library parcellations: "
                            f"{', '.join(k for k in parcellation_lib if 'alias' not in parcellation_lib[k])}"
                        ) from _e
            elif isinstance(parcellation, (_pl.Path, _nib.Nifti1Image, _nib.GiftiImage, tuple)):
                try:
                    parcellation = _Parc.from_path(source=parcellation, space=space)
                except Exception as _e:
                    raise ValueError(
                        f"Could not load parcellation from path/image: {_e}"
                    ) from _e
            else:
                raise TypeError(
                    "'parcellation' must be a Parcellation instance or a parcellation "
                    f"name/path, not {type(parcellation).__name__}."
                )

        if data.empty:
            raise ValueError("'data' is empty.")
        if not np.issubdtype(data.values.dtype, np.number):
            raise ValueError(
                f"'data' must contain numeric values, got dtype '{data.values.dtype}'."
            )
        _n_parc = len(parcellation._labels) if parcellation._labels is not None else None
        if _n_parc is not None and data.shape[1] != _n_parc:
            raise ValueError(
                f"'data' has {data.shape[1]} columns but parcellation "
                f"'{parcellation._name}' has {_n_parc} parcels."
            )
        _all_nan = data.isnull().all(axis=1)
        if _all_nan.any():
            _bad = list(data.index[_all_nan])
            lgr.warning(f"brainplot: {len(_bad)} map(s) are all-NaN and will appear blank: {_bad}")

        n_maps           = len(data)
        _parc_is_combined = parcellation._is_combined
    else:
        n_maps            = 0  # set below after image preprocessing
        _parc_is_combined = False

    # -- validate kind and auto-detect for image input --
    _kind_user_set = kind is not None
    if kind is None:
        kind = "surface" if _parc_is_gifti_input else "glass"
    # post-load: surface-only Parcellation object passed directly (no MNI space available)
    if (_img_mode is None and not _kind_user_set and kind == "glass"
            and not any("mni" in s.lower() for s in parcellation.spaces)):
        kind = "surface"
        lgr.info("brainplot: surface-only parcellation → kind auto-set to 'surface'")
    if kind not in ("surface", "glass", "slice", "combined"):
        raise ValueError(f"kind='{kind}' must be 'surface', 'glass', 'slice', or 'combined'.")
    if _img_mode == "nifti" and kind in ("surface", "combined"):
        kind = "glass"
        lgr.info("brainplot: NIfTI input → kind auto-set to 'glass'")
    elif _img_mode == "gifti" and kind != "surface":
        kind = "surface"
        lgr.info("brainplot: GIfTI input → kind auto-set to 'surface'")
    if _parc_is_combined and kind == "surface":
        raise ValueError(
            "kind='surface' is not supported for combined (cx+sc) parcellations. "
            "Use kind='glass' or kind='slice' to render the full MNI volume."
        )
    if kind == "combined" and not _parc_is_combined:
        raise ValueError(
            "kind='combined' requires a combined (cx+sc) parcellation."
        )
    if kind == "combined":
        raise NotImplementedError(
            "kind='combined' is not yet implemented. Combined plots will integrate "
            "cortical surface rendering with subcortical volumetric rendering, but "
            "this feature is still in development. Use kind='glass' or kind='slice' "
            "to render the full MNI volume instead."
        )

    # is_combined rendering path only active when explicitly requested
    is_combined = _parc_is_combined and kind == "combined"
    # default zoom per kind
    if zoom is None:
        zoom = 1.5 if kind == "surface" else 1.0
    # default display_mode and cut_coords per kind
    if display_mode is None:
        if kind == "glass":
            display_mode = "lyrz"
        elif kind in ("slice", "combined"):
            display_mode = "z"
        else:
            display_mode = "z"  # surface: unused
    if cut_coords is None and kind == "slice":
        cut_coords = 5

    # -- preprocess image inputs / resolve threshold --
    _vol_bg       = 0.0   # background fill for _data_to_volume (tabular path)
    _stat_niis    = None  # list of pre-built NIfTI images  (nifti path)
    _gifti_pairs  = None  # list of (lh_arr, rh_arr) tuples (gifti path)
    _img_all_vals = None  # flat finite array for vmin/vmax  (image paths)

    if _img_mode == "nifti":
        _nii_in = _nib.load(str(data)) if not isinstance(data, _nib.Nifti1Image) else data
        _arr    = _nii_in.get_fdata()
        if _arr.ndim == 4:
            raise ValueError("4D NIfTI is not supported; pass a single 3D volume.")
        _finite_nz  = _arr[np.isfinite(_arr) & (_arr != 0)]
        _min_abs    = float(np.min(np.abs(_finite_nz))) if len(_finite_nz) else 0.0
        if threshold == "auto":
            threshold = float(np.float32(_min_abs / 2)) if _min_abs > 0 else None
            lgr.info(f"brainplot: threshold='auto' → {threshold}")
        _stat_niis    = [new_img_like(_nii_in, np.nan_to_num(_arr, nan=0.0), copy_header=True)]
        _img_all_vals = _arr[np.isfinite(_arr)].flatten()
        n_maps        = 1

    elif _img_mode == "gifti":
        _pairs_raw = [data] if _is_gifti_pair else data
        _gifti_pairs = [(_load_gifti_arr(p[0]), _load_gifti_arr(p[1])) for p in _pairs_raw]
        _img_all_vals = np.concatenate([np.concatenate(p) for p in _gifti_pairs])
        _finite_nz  = _img_all_vals[np.isfinite(_img_all_vals) & (_img_all_vals != 0)]
        _min_abs    = float(np.min(np.abs(_finite_nz))) if len(_finite_nz) else 0.0
        if threshold == "auto":
            threshold = float(np.float32(_min_abs / 2)) if _min_abs > 0 else None
            lgr.info(f"brainplot: threshold='auto' → {threshold}")
        n_maps = len(_gifti_pairs)
        plot_contours = False  # no parcellation array available for contours

    else:  # tabular
        if threshold == "auto":
            _vals = data.values.flatten()
            _finite_nz = _vals[np.isfinite(_vals) & (_vals != 0)]
            _min_abs = float(np.min(np.abs(_finite_nz))) if len(_finite_nz) else 0.0
            threshold = float(np.float32(_min_abs / 2)) if _min_abs > 0 else None
            lgr.info(f"brainplot: threshold='auto' → {threshold}")

    # -- resolve spaces --
    def _is_surf(s):  return any(k in s.lower() for k in ("fslr", "fsaverage", "fsa"))
    def _is_mni(s):   return "mni" in s.lower()

    surf_space = mni_space = None

    if _img_mode is None:  # tabular: resolve spaces from parcellation
        if is_combined:
            cx_surf_spaces = list(parcellation._cx_surface.keys())
            if not cx_surf_spaces:
                raise ValueError(
                    f"Combined parcellation '{parcellation._name}' has no cx surface data. "
                    "kind='surface' requires surface data for cortex."
                )
            _cx_pref = space if (space and not _is_mni(space)) else None
            surf_space = (_cx_pref if _cx_pref in cx_surf_spaces
                          else next((s for s in cx_surf_spaces if "fslr" in s.lower()),
                                    cx_surf_spaces[0]))
            _mni_spaces = [s for s in parcellation.spaces if _is_mni(s)]
            if not _mni_spaces:
                raise ValueError(
                    f"Combined parcellation '{parcellation._name}' has no MNI space "
                    "for subcortex projection."
                )
            mni_space = next((s for s in _mni_spaces if "2009" in s), _mni_spaces[0])

        elif kind == "surface":
            _surf_spaces = [s for s in parcellation.spaces if _is_surf(s)]
            if not _surf_spaces:
                raise ValueError(
                    f"Parcellation '{parcellation._name}' has no surface space. "
                    f"Available: {parcellation.spaces}"
                )
            if space is not None and space not in _surf_spaces:
                raise ValueError(
                    f"space='{space}' not available for parcellation '{parcellation._name}'. "
                    f"Available surface spaces: {_surf_spaces}"
                )
            surf_space = (space if space in _surf_spaces
                          else next((s for s in _surf_spaces if "fslr" in s.lower()),
                                    _surf_spaces[0]))

        else:  # glass / slice
            _mni_spaces = [s for s in parcellation.spaces if _is_mni(s)]
            if not _mni_spaces:
                raise ValueError(
                    f"Parcellation '{parcellation._name}' has no MNI space. "
                    f"Available: {parcellation.spaces}"
                )
            if space is not None and space not in _mni_spaces:
                raise ValueError(
                    f"space='{space}' not available for parcellation '{parcellation._name}'. "
                    f"Available MNI spaces: {_mni_spaces}"
                )
            mni_space = (space if space in _mni_spaces
                         else next((s for s in _mni_spaces if "2009" in s), _mni_spaces[0]))

        _primary = mni_space if mni_space is not None else surf_space
        if (_primary is not None
                and _primary in parcellation.spaces
                and parcellation._idc_byhemi_dict.get(_primary) is None):
            parcellation.set_active_space(_primary)

    lgr.info(
        f"brainplot: kind='{kind}', img_mode='{_img_mode}', "
        f"surf_space='{surf_space}', mni_space='{mni_space}', surf_mesh='{surf_mesh}'"
    )

    # -- load surface assets --
    surf_geom = bg_data = medial_data = None
    parc_arr_lh = parc_arr_rh = None

    if kind == "surface" or is_combined:
        if _img_mode == "gifti":
            # GIfTI passthrough: load geometry from templates (assume fslr32k)
            _use_fslr = space is None or "fslr" in (space or "").lower()
            if _use_fslr:
                surf_geom, bg_data, medial_data = _load_fslr_assets(surf_mesh)
            else:
                surf_geom, bg_data, medial_data = _load_fsaverage_assets(surf_mesh)
        else:
            if surf_space and "fslr" in surf_space.lower():
                surf_geom, bg_data, medial_data = _load_fslr_assets(surf_mesh)
            else:
                surf_geom, bg_data, medial_data = _load_fsaverage_assets(surf_mesh)

            if is_combined:
                cx_entry = parcellation._cx_surface.get(surf_space)
                if cx_entry is None:
                    raise ValueError(
                        f"No cx surface data for space '{surf_space}'. "
                        f"Available: {list(parcellation._cx_surface.keys())}"
                    )
                if cx_entry.get("image") is None:
                    from .io import load_img as _load_img
                    cx_entry["image"] = _load_img(cx_entry["img_paths"])
                    parcellation._cx_surface[surf_space] = cx_entry
                parc_arr_lh = cx_entry["image"][0].agg_data()
                parc_arr_rh = cx_entry["image"][1].agg_data()
            else:
                parc_img = parcellation.get_image(surf_space)
                parc_arr_lh = parc_img[0].agg_data()
                parc_arr_rh = parc_img[1].agg_data()

    # -- load volume assets --
    mni_nii = bg_img_nii = None

    if _img_mode is None and (kind in ("glass", "slice") or is_combined):
        mni_nii = parcellation.get_image(mni_space)
        if kind == "slice":
            if bg_img is not None:
                bg_img_nii = bg_img
            else:
                _bg_space = mni_space
                for _desc in ("brain", "T1w"):
                    try:
                        bg_img_nii = fetch_template(_bg_space, desc=_desc, verbose=False)
                        break
                    except Exception:
                        continue
                if bg_img_nii is None:
                    lgr.warning(
                        "Could not auto-fetch MNI background image; "
                        "slice plots will use nilearn's default."
                    )

    elif _img_mode == "nifti" and kind == "slice" and bg_img is None:
        _bg_space = space if space is not None else "MNI152NLin2009cAsym"
        for _desc in ("brain", "T1w"):
            try:
                bg_img_nii = fetch_template(_bg_space, desc=_desc, verbose=False)
                break
            except Exception:
                continue
        if bg_img_nii is None:
            lgr.warning(
                "Could not auto-fetch MNI background image; "
                "slice plots will use nilearn's default."
            )

    # -- vmin / vmax --
    force_shared = is_combined
    use_shared   = shared_colorscale or force_shared
    _global_vmin = _global_vmax = None

    _flat_vals = (
        _img_all_vals if _img_mode is not None else data.values.flatten()
    )

    # -- resolve symmetric_cmap and cmap --
    if symmetric_cmap == "auto":
        _clean = _flat_vals[np.isfinite(_flat_vals)]
        symmetric_cmap = bool(np.any(_clean < 0) and np.any(_clean > 0))
    if cmap is None:
        cmap = "RdBu_r" if symmetric_cmap else "viridis"

    if use_shared:
        _global_vmin, _global_vmax = _auto_vmin_vmax(
            _flat_vals, symmetric_cmap, vmin, vmax
        )

    # Pre-resolve integer cut_coords to explicit positions so all maps use
    # identical slice locations (nilearn would pick different cuts per image).
    if kind == "slice" and isinstance(cut_coords, int):
        from nilearn.plotting import find_cut_slices as _find_cuts
        _ref_img = (_stat_niis[0] if _img_mode == "nifti"
                    else mni_nii if mni_nii is not None else None)
        if _ref_img is not None:
            cut_coords = list(_find_cuts(_ref_img, direction=display_mode,
                                         n_cuts=cut_coords))
            lgr.info(f"brainplot: cut_coords resolved to {cut_coords}")

    def _vminmax(row_vals):
        if use_shared:
            return _global_vmin, _global_vmax
        return _auto_vmin_vmax(row_vals, symmetric_cmap, vmin, vmax)

    # -- figure layout --
    n_rows_grid  = int(np.ceil(n_maps / ncols))
    n_cols_grid  = min(n_maps, ncols)
    n_views      = len(views)
    if kind == "slice":
        _n_panels = len(cut_coords) if hasattr(cut_coords, "__len__") else int(cut_coords)
    elif kind == "glass":
        _n_panels = len(display_mode)
    else:  # surface / combined
        _n_panels = n_views
    _has_shared_cbar = False   # colorbars always shown per-row on the right
    _cbar_h      = 0.0
    if wspace is None:
        _base_wspace = 0.1 if kind in ("glass", "slice") else 0.02
        _wspace = _base_wspace if n_cols_grid == 1 else max(_base_wspace, 0.15)
    else:
        _wspace = wspace
    def _fmt_label(lbl):
        """Format a single label string; prettify PET-style keys."""
        s = str(lbl)
        if all(k in s for k in ("target-", "tracer-", "pub-")):
            p = s.split("_")
            try:
                target = p[0].split("-")[1]
                n      = p[2].split("-")[1]
                pub    = p[4].split("-")[1].capitalize()
                return f"{target} ({pub}, n = {n})"
            except (IndexError, ValueError):
                pass
        return s

    # Normalise title to a per-map list (or None to disable).
    if title in (False, None, ""):
        _titles = None
    elif title == "auto":
        if _img_mode is None:
            _titles = [
                _fmt_label(
                    " | ".join(str(v) for v in idx) if isinstance(idx, tuple) else idx
                )
                for idx in data.index
            ]
        else:
            # volume / gifti input — no meaningful label available
            _titles = None
    elif isinstance(title, (list, tuple)):
        if len(title) != n_maps:
            raise ValueError(
                f"title has {len(title)} entries but {n_maps} maps are being plotted."
            )
        _titles = [_fmt_label(t) for t in title]
    else:
        _titles = [str(title)] * n_maps
    _has_title = _titles is not None
    _multirow = n_rows_grid > 1
    if hspace is None:
        if _has_title and _multirow:
            hspace = 0.05 if kind == "surface" else 0.35
        else:
            hspace = -0.1 if kind == "surface" else 0.05

    # Surface titles sit inside the axes; only add extra row height for multi-row.
    if _has_title and _multirow:
        _title_h = 0.3
    else:
        _title_h = 0.0
    if figsize is None:
        _panel_w = 1.8 if kind == "glass" else (1.4 if kind == "slice" else 2.0)
        _w = n_cols_grid * _n_panels * _panel_w
        if kind in ("surface", "combined") or is_combined:
            _surf_row_h = 2.2 if n_rows_grid == 1 else 1.8
            _h = n_rows_grid * (_surf_row_h + _title_h)
            if is_combined:
                _h += n_rows_grid * 0.8
        else:
            _h = n_rows_grid * (1.8 + _title_h)
        figsize = (_w, _h + _cbar_h)

    if fig is not None and axes is not None:
        # caller provides their own layout — normalise to flat list
        _axes_flat = np.asarray(axes).flatten().tolist()
        _n_expected = n_rows_grid * n_cols_grid
        if len(_axes_flat) != _n_expected:
            raise ValueError(
                f"'axes' has {len(_axes_flat)} elements but the grid requires "
                f"{_n_expected} ({n_rows_grid} rows × {n_cols_grid} cols)."
            )
        if not all(isinstance(a, mpl.axes.Axes) for a in _axes_flat):
            raise TypeError("All elements of 'axes' must be matplotlib Axes instances.")
        _axes_arr = np.array(_axes_flat).reshape(n_rows_grid, n_cols_grid)
        _own_fig  = False
    elif fig is not None and axes is None:
        # caller provided fig only — create axes inside it
        _own_fig = True
        if is_combined:
            _hr = []
            for _ in range(n_rows_grid):
                _hr += [2.2, 1.0]
            _gs = GridSpec(
                n_rows_grid * 2, n_cols_grid,
                figure=fig,
                height_ratios=_hr,
                hspace=hspace, wspace=_wspace,
            )
            _axes_arr = None
        else:
            _axes_arr = np.array(
                fig.subplots(n_rows_grid, n_cols_grid,
                             gridspec_kw={"hspace": hspace, "wspace": _wspace})
            ).reshape(n_rows_grid, n_cols_grid)
    else:
        # create everything from scratch
        _own_fig = True
        if is_combined:
            fig = plt.figure(figsize=figsize)
            _hr = []
            for _ in range(n_rows_grid):
                _hr += [2.2, 1.0]
            _gs = GridSpec(
                n_rows_grid * 2, n_cols_grid,
                figure=fig,
                height_ratios=_hr,
                hspace=hspace, wspace=_wspace,
            )
            _axes_arr = None  # combined axes created per-map below
        else:
            fig, _axes_arr = plt.subplots(
                n_rows_grid, n_cols_grid,
                figsize=figsize, squeeze=False,
                gridspec_kw={"hspace": hspace, "wspace": _wspace},
            )

    # Resolve default colorbar inset coords per kind
    if colorbar_inset is None:
        if kind == "glass":
            colorbar_inset = [1.02, 0.15, 0.02, 0.7]
        elif is_combined:  # kind == "combined"
            colorbar_inset = [1.04, 0.25, 0.03, 0.5]
        elif kind == "slice":
            colorbar_inset = [1.04, 0.2, 0.02, 0.6]
        else:  # surface
            colorbar_inset = [1.04, 0.25, 0.02, 0.5]

    if black_bg:
        fig.patch.set_facecolor("black")

    axes_out = []
    # Colorbars and titles are added after all brains so they render on top.
    _pending_cbars  = []  # [(ax, v_min, v_max)]
    _pending_titles = []  # [(ax, title_str)]

    for i in range(n_maps):
        ri    = i // n_cols_grid
        ci    = i % n_cols_grid

        if _img_mode is None:
            row_vals = data.iloc[i].values.astype(np.float64)
            v_min, v_max = _vminmax(row_vals)
        else:
            row_vals = None
            v_min, v_max = _vminmax(_img_all_vals)

        if is_combined:
            ax_s = fig.add_subplot(_gs[ri * 2,     ci])
            ax_v = fig.add_subplot(_gs[ri * 2 + 1, ci])
            ax_s.set_axis_off()
            ax_v.set_axis_off()
            if black_bg:
                ax_s.set_facecolor("black")
                ax_v.set_facecolor("black")
            axes_out += [ax_s, ax_v]
        else:
            ax_main = _axes_arr[ri, ci]
            ax_main.set_axis_off()
            if black_bg:
                ax_main.set_facecolor("black")
            axes_out.append(ax_main)

        # ---- title (deferred) ----
        if _has_title:
            _pending_titles.append((ax_s if is_combined else ax_main, _titles[i]))

        # ---- GIfTI passthrough: surface ----
        if _img_mode == "gifti":
            vert_lh, vert_rh = _gifti_pairs[i]
            _render_surf_row(
                ax_main, fig, vert_lh, vert_rh,
                parc_arr_lh, parc_arr_rh,
                surf_geom, bg_data, views,
                cmap, v_min, v_max,
                symmetric_cmap, bg_on_data, darkness, threshold, alpha,
                plot_contours, zoom, black_bg=black_bg,
                **kwargs,
            )
            if colorbar:
                _pending_cbars.append((ax_main, v_min, v_max))

        # ---- NIfTI passthrough: glass / slice ----
        elif _img_mode == "nifti":
            _render_vol_row(
                ax_main, fig, _stat_niis[i], bg_img_nii or bg_img,
                kind, display_mode, cut_coords,
                cmap, v_min, v_max,
                symmetric_cmap, threshold, alpha, draw_cross,
                colorbar=False, dim=dim, draw_brain=draw_brain, zoom=zoom,
                black_bg=black_bg,
                **kwargs,
            )
            if colorbar:
                _pending_cbars.append((ax_main, v_min, v_max))

        # ---- tabular: surface (non-combined) ----
        elif kind == "surface" and not is_combined:
            idc = parcellation.get_idc_byhemi(surf_space)
            vert_lh, vert_rh = _data_to_surf_verts(
                row_vals[idc["L"]], row_vals[idc["R"]],
                parc_arr_lh, parc_arr_rh, medial_data,
            )
            _render_surf_row(
                ax_main, fig, vert_lh, vert_rh,
                parc_arr_lh, parc_arr_rh,
                surf_geom, bg_data, views,
                cmap, v_min, v_max,
                symmetric_cmap, bg_on_data, darkness, threshold, alpha,
                plot_contours, zoom, black_bg=black_bg,
                **kwargs,
            )
            if colorbar:
                _pending_cbars.append((ax_main, v_min, v_max))

        # ---- tabular: combined cx surface + sc slices ----
        elif is_combined:
            n_cx = len(parcellation._labels) - parcellation._get_sc_n_parcels()
            # Surface labels are identical to cx portion labels in the combined volume
            # (1-based), so label - 1 is the 0-based position in row_vals.
            _idc_lh_surf = np.trim_zeros(np.unique(parc_arr_lh)).astype(int)
            _idc_rh_surf = np.trim_zeros(np.unique(parc_arr_rh)).astype(int)
            vert_lh, vert_rh = _data_to_surf_verts(
                row_vals[_idc_lh_surf - 1],
                row_vals[_idc_rh_surf - 1],
                parc_arr_lh, parc_arr_rh, medial_data,
            )
            _render_surf_row(
                ax_s, fig, vert_lh, vert_rh,
                parc_arr_lh, parc_arr_rh,
                surf_geom, bg_data, views,
                cmap, v_min, v_max,
                symmetric_cmap, bg_on_data, darkness, threshold, alpha,
                plot_contours, zoom, black_bg=black_bg,
                **kwargs,
            )
            _mni_arr     = mni_nii.get_fdata()
            _all_labels  = np.trim_zeros(np.unique(_mni_arr))
            _sc_labels   = _all_labels[n_cx:]
            _sc_stat     = _data_to_volume(row_vals[n_cx:], mni_nii, _sc_labels, _vol_bg)
            _render_vol_row(
                ax_v, fig, _sc_stat, None,
                "glass", "lyrz", None,
                cmap, v_min, v_max,
                symmetric_cmap, threshold, alpha, draw_cross,
                colorbar=False, dim=dim, draw_brain=draw_brain, zoom=zoom,
                black_bg=black_bg,
                **kwargs,
            )
            if colorbar:
                _pending_cbars.append((ax_s, v_min, v_max))

        # ---- tabular: glass / slice (non-combined) ----
        else:
            _mni_arr    = mni_nii.get_fdata()
            _all_labels = np.trim_zeros(np.unique(_mni_arr))
            _stat_nii   = _data_to_volume(row_vals, mni_nii, _all_labels, _vol_bg)
            _render_vol_row(
                ax_main, fig, _stat_nii, bg_img_nii,
                kind, display_mode, cut_coords,
                cmap, v_min, v_max,
                symmetric_cmap, threshold, alpha, draw_cross,
                colorbar=False, dim=dim, draw_brain=draw_brain, zoom=zoom,
                black_bg=black_bg,
                **kwargs,
            )
            if colorbar:
                _pending_cbars.append((ax_main, v_min, v_max))

    # Add colorbars last so they render on top of all brain axes (important for
    # multi-column layouts where adjacent brains would otherwise cover them).
    # Bake alpha into the colorbar colormap so it matches the rendered data.
    import matplotlib.colors as _mcolors
    _cbar_cmap = cmap
    if alpha is not None and float(alpha) != 1.0:
        _a = float(alpha)
        _cmap_obj = (mpl.colormaps[cmap] if isinstance(cmap, str)
                     else cmap if isinstance(cmap, _mcolors.Colormap)
                     else _mcolors.LinearSegmentedColormap.from_list("_c", cmap))
        _rgba = _cmap_obj(np.linspace(0, 1, 256))
        _rgba[:, 3] = _a
        _cbar_cmap = _mcolors.ListedColormap(_rgba)

    for (_cax_ax, _vmin, _vmax) in _pending_cbars:
        _norm = mpl.colors.Normalize(vmin=_vmin, vmax=_vmax)
        _sm   = plt.cm.ScalarMappable(cmap=_cbar_cmap, norm=_norm)
        _sm.set_array([])
        _pos  = _cax_ax.get_position()
        _cax  = fig.add_axes([
            _pos.x1 + (colorbar_inset[0] - 1.0) * _pos.width,
            _pos.y0 + colorbar_inset[1] * _pos.height,
            colorbar_inset[2] * _pos.width,
            colorbar_inset[3] * _pos.height,
        ])
        fig.colorbar(_sm, cax=_cax)
        if black_bg:
            _cax.yaxis.set_tick_params(color="white", labelcolor="white")
            for _sp in _cax.spines.values():
                _sp.set_edgecolor("white")
        if colorbar_label:
            _cax.set_title(colorbar_label, fontsize="medium",
                           color="white" if black_bg else "black")

    # Add titles last so they render on top of all brain axes.
    # y is expressed as a fraction of the axes height added to the axes top.
    # Surface/glass: 1.02 in single-row; surface bumps to 1.05 in multi-row.
    # Slice: 1.06 (single and multi) — more white space above slice panels.
    if kind == "surface":
        _title_y = 1.05 if _multirow else 1.02
        _title_va = "top"
    elif kind == "slice":
        _title_y = 1.06
        _title_va = "bottom"
    else:  # glass
        _title_y = 1.02
        _title_va = "bottom"
    _title_kw = {"fontsize": "large", "fontweight": "bold"}
    if black_bg:
        _title_kw["color"] = "white"
    if title_kwargs:
        _title_kw.update(title_kwargs)
    _title_y_override = _title_kw.pop("y", None)
    for (_tax, _tstr) in _pending_titles:
        _pos = _tax.get_position()
        _ty = _pos.y0 + (_title_y_override if _title_y_override is not None else _title_y) * _pos.height
        fig.text(
            _pos.x0 + _pos.width / 2, _ty, _tstr,
            ha="center", va=_title_va,
            **_title_kw,
        )

    # -- hide unused subplot cells --
    if not is_combined and _own_fig:
        for j in range(n_maps, n_rows_grid * n_cols_grid):
            _axes_arr[j // n_cols_grid, j % n_cols_grid].set_visible(False)

    return fig, axes_out