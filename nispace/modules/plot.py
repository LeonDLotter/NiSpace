import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .. import lgr
from ..plotting import catplot, nullplot, nice_stats_labels


def _plot_categorical(colocs_df, stat, nulls_dict=None, p_df=None, pc_df=None, 
                      sort=False, title=None, fig=None, ax=None, figsize=None, 
                      kwargs={}, null_kwargs={}, clean_labels=True):
       
    # column names
    colocs_df = colocs_df.copy()
    
    # things to do for one-column results (everything with r2)
    if colocs_df.shape[1] == 1:
        colocs_df.columns = ["Combined reference maps"]
        
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
            if all([s in X_labels[0] for s in ["domain-", "n-"]]):
                tmp = []
                for l in X_labels:
                    l_split = l.split("_")
                    tmp.append(f"{l_split[0].split('-')[1]} (n = {l_split[-1].split('-')[1]})")
                X_labels = tmp
                
        # if labels are not to be cleaned, convert potential multi-idc to string
        else:
            X_labels = colocs_df.columns.to_flat_index()
            
        # set new column names but keep the set->map assignment as indices (!)
        colocs_df.columns = [str(l) for l in X_labels]
        
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
    stat_label = nice_stats_labels(stat)
    if title in ["", None, False]:
        title = None
    elif title == True:
        title = stat_label
    catplot_kwargs = {
        "legend": {"kwargs": {"title": stat_label}},
        "color_how": "cont",
        "color_which": "auto",
        "sort_categories": sort
    }
    nullplot_kwargs = {
        "legend": {"kwargs": {"title": stat_label}},
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
    
    # zero lines 
    plot_h0line = True if stat in ["beta", "rho"] and catplot_kwargs["categorical_axis"] == "x" else False
    plot_v0line = True if stat in ["beta", "rho"] and catplot_kwargs["categorical_axis"] == "y" else False
    
    # plot
    plot = catplot(
        fig, ax, colocs_df_melt, categorical_var="X", continuous_var=stat, 
        **catplot_kwargs,
        hline=dict(plot=plot_h0line), 
        vline=dict(plot=plot_v0line)
    )
    
    if nulls_dict is not None:
        if "plot" in locals():
            nullplot_kwargs = nullplot_kwargs | {
                "labels": {
                    "title": ax.get_title(),
                    "x": ax.get_xlabel(),
                    "y": ax.get_ylabel(),
                    "category_order": ax.get_yticklabels()
                }
            }
        plot = nullplot(fig, ax, nulls_df_melt, categorical_var="X", continuous_var=stat,
                        **nullplot_kwargs)
        
    # if p_df is not None:
    #     p = colocs_df.reset_index(names="Y").melt(
    #         id_vars=["Y"],
    #         var_name="X",
    #         value_name=stat
    #     )
    #     print(p)
    
    return fig, ax, plot

