import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .. import lgr
from ..plotting import catplot, nullplot, nice_stats_labels, print_significance


def _plot_categorical(colocs_df, stat, nulls_dict=None, p_df=None, pc_df=None,
                      values="coloc", mc_method=None,
                      sort=False, sort_order=None,
                      annot_p=True,
                      title=None, fig=None, ax=None, figsize=None,
                      kwargs=None, null_kwargs=None, clean_labels=True):

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
            if all([s in X_labels[0] for s in ["target-", "tracer-", "pub-"]]):
                tmp = []
                for l in X_labels:
                    l_split = l.split("_")
                    tmp.append(f"{l_split[0].split('-')[1]} ({l_split[4].split('-')[1].capitalize()}, "
                               f"n = {l_split[2].split('-')[1]})")
                X_labels = tmp
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
            tmp.append(
                pd.DataFrame({
                    "X": c_colocs,
                    stat: (nulls_dict[stat][c_nulls] if colocs_df.shape[1] > 1 else nulls_dict[stat]).mean(axis=0)
                })
            )     
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
        "median_line": {"label": "Null Median"}
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
        
    # multiple Y:
    # anything?

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
            n_elements = colocs_df.shape[1]
            figsize=(1.5 + 0.2 * n_elements, 5)
            if catplot_kwargs["categorical_axis"] != "x":
                figsize = np.flip(figsize)
        fig, ax = plt.subplots(1, figsize=figsize)
    
    # zero-line for coloc mode (passed into catplot)
    axis = catplot_kwargs["categorical_axis"]
    cont_on_y = (axis == "x")
    plot_h0line = values == "coloc" and stat in ["beta", "rho"] and cont_on_y
    plot_v0line = values == "coloc" and stat in ["beta", "rho"] and not cont_on_y

    # collect threshold specs for z/p modes — drawn after render with limit check
    # sym=True:   draw both +pos and -pos, one legend entry per threshold
    # force=True: always draw; extend axis if the position is outside seaborn's range
    _grey = "dimgrey"
    if values == "z":
        _ref_specs = [
            {"pos": 0,    "ls": "-",  "lw": 1.2, "label": "null mean",        "sym": False, "force": True},
            {"pos": 1.96, "ls": ":",  "lw": 0.8, "label": "$Z = 1.96$",  "sym": True,  "force": True},
            {"pos": 2.58, "ls": "--", "lw": 0.8, "label": "$Z = 2.58$",  "sym": True,  "force": False},
            {"pos": 3.29, "ls": "-.", "lw": 0.8, "label": "$Z = 3.29$", "sym": True,  "force": False},
        ]
    elif values == "p":
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
                        leg_kwargs["bbox_to_anchor"] = (1.0, 0.5)
                new_leg = ax.legend(_ref_handles, _ref_labels, **leg_kwargs)
                new_leg.get_children()[0].get_children()[:0] = old_boxes
            else:
                ax.legend(handles=_ref_handles, labels=_ref_labels)

    # significance annotations
    if annot_p is not False and p_df is not None and values not in ("p",):
        if p_df.shape[0] > 1:
            lgr.warning(
                "p_df has one row per Y subject — significance annotation requires a single "
                "aggregated p-value row (set p_from_average_y=True or pass a pre-aggregated "
                "p_df). Skipping significance annotation."
            )
        else:
            _p_row = p_df.iloc[0]
            _q_row = pc_df.iloc[0] if pc_df is not None else None
            _col_means = colocs_df.mean(axis=0)

            # align all series to current colocs_df column order (after label cleaning)
            _p_vals  = _p_row.reindex(colocs_df.columns).values
            _q_vals  = _q_row.reindex(colocs_df.columns).values if _q_row is not None else None
            _means   = _col_means.values

            _sig_handles, _sig_labels = print_significance(
                ax,
                p_values=_p_vals,
                q_values=_q_vals,
                coloc_values=_means,
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
                            leg_kwargs["bbox_to_anchor"] = (1.0, 0.5)
                    new_leg = ax.legend(_sig_handles, _sig_labels, **leg_kwargs)
                    new_leg.get_children()[0].get_children()[:0] = old_boxes
                else:
                    ax.legend(handles=_sig_handles, labels=_sig_labels)

    return fig, ax, plot

