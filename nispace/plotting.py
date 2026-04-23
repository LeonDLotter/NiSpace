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
from neuromaps import images

from . import lgr
from .utils.utils import vect_to_vol_arr
from .datasets import fetch_parcellation, fetch_template, parcellation_lib, template_lib


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
    for k in replace_dict:
        if add_dollars:
            k_replace = "$" + replace_dict[k].replace(' ', '\ ') + "$"
        else:
            k_replace = replace_dict[k]
        string = string.replace(k, k_replace)
    return string


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
            scatters["size"] = 5 / (0.1 * n_max**0.5) / ((0.3 * n_cat**0.5) if n_cat > 1 else 1)
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
    lgr.setLevel(verbose)
    
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