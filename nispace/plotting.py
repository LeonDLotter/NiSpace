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


def nice_stats_labels(string, add_dollars=True):
    replace_dict = {
        "r2": "R^2",
        "R2": "R^2",
        "beta": "Beta",
        "spearman": "Spearman's Rho",
        "pearson": "Pearson's Rho",
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


def catplot(fig, ax, data_long, categorical_var="variable", continuous_var="value", group_var=None,
            categorical_axis="x", sort_categories=False, category_order=None,
            color_how="continuous", color_which="auto", color_center=None,
            labels={},
            limits={},
            bars={},
            violins={},    
            scatters={},
            dots={},
            errorbars={},  
            hline={},
            vline={},
            legend={}
            ):   
    
    # defaults, overwrite with user input
    bars = dict(
        plot=False, label=True,
        width=0.5, agg_method="mean", dodge_width=0.5, 
        kwargs={"zorder": 10, "ec": "k", "lw": 0.7}
    ) | bars
    violins = dict(
        plot=False, label=False,
        kwargs={"zorder": 20, "density_norm": "width", "cut": 0, "inner": "quart",
                "fill": False, "edgecolor": "k", "linewidth": 0.7}
    ) | violins
    scatters = dict(
        plot=True, label=False,
        size="auto", jitter_width=0.5, dodge_width=0.5,
        kwargs={"zorder": 30, "linewidth": 0.2, "alpha": 0.2}
    ) | scatters 
    dots = dict(
        plot=True, label=True,
        agg_method="mean", size=7, color="k", dodge_width=0.5, 
        kwargs={"zorder": 90, "facecolor": (1,1,1,0.8), "lw": 1}
    ) | dots
    errorbars = dict(
        plot=True, label=True,
        agg_method="ci", dodge_width=0.5, color="k", 
        kwargs={"zorder": 100}
    ) | errorbars
    hline = dict(
        plot=False, y=[0], color="k", linewidth=1, linestyle="--", zorder=-100, kwargs={}
    ) | hline
    vline = dict(
        plot=False, x=[0], color="k", linewidth=1, linestyle="--", zorder=-100, kwargs={}
    ) | vline
    legend = dict(
        plot=True,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        nice_labels=True,
        kwargs={}
    ) | legend
    
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
             bands={},
             median_line={},
             violins={},
             labels={},
             limits={},
             legend={}
             ):   
    bands = dict(
        plot=True, label=True, label_prefix="Null percentile ",
        alpha=0.1,
        edgewidth=1,
        edgestyle="-",
        edgealpha=0.3,
        kwargs={"zorder": -200}
    ) | bands
    median_line = dict(
        plot=True, label="Null percentile 50",
        alpha=0.6,
        kwargs={"zorder": -150}
    ) | median_line
    violins = dict(
        plot=False, label="Null distribution",
        legend=None,
        kwargs={"zorder": 100, "density_norm": "width", "cut": 0, "inner": "quart",
                "fill": False, "edgecolor": "k", "linewidth": 1}
    ) | violins
    legend = dict(
        plot=True,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        nice_labels=True,
        kwargs={}
    ) | legend
    
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
            legend_colors_kwargs={},
            legend_sizes=True,
            legend_sizes_kwargs={},
            legend_shapes=True,
            legend_shapes_kwargs={},
            ):
    
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
            raise ValueError("mapping_shapes keys must be 'circle'/'o', 'square'/'s', or 'diamond'/'D'")
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
        legend_colors_kwargs = {
            "label": "Colors",
            "orientation": legend_orientation,
            "cax": ax.inset_axes((0, 1.1, 1/3, 0.05) if legend_orientation == "horizontal" else (1.05, 2/3, 0.05, 1/3))
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

