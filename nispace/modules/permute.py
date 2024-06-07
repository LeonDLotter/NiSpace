import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .. import lgr
from ..nulls import generate_null_maps
from ..stats.misc import null_to_p, zscore_df
from ..utils import set_log
from ..modules.colocalize import _get_coloc_stats
from ..modules.constants import _P_TAILS


def _get_null_maps(data_obs, nispace_nulls, null_maps=None, use_existing_maps=True, standardize=True,
                   n_perm=1000, null_method="moran",
                   dist_mat=None, parc=None, parc_kwargs=None, centroids=False,
                   seed=None, n_proc=-1, dtype=np.float32):
    
    # case null maps given
    if null_maps is not None:
        if not isinstance(null_maps, dict):
            lgr.warning("Provided null maps are not a dictionary. Will re-generate.")
            null_maps = None
        else:
            lgr.info(f"Using provided null maps.")
        
    # case null maps not given but existing
    elif (null_maps is None) & (use_existing_maps==True):
        try:
            permute, null_method_stored, null_maps = \
                [nispace_nulls[k] for k in ["maps_null_which", "maps_null_method", "maps_null"]]
            lgr.info(f"Found existing null maps.")
        except:
            lgr.info("No null maps found.")
            
    # check existing null maps
    if null_maps is not None:
        if not all([x in null_maps.keys() for x in data_obs.index]):
            lgr.warning("Not all X/Y variables in null maps. Will re-generate.")
            null_maps = None
        if null_method_stored != null_method:
            lgr.warning("Null method changed. Will re-generate.")
            null_maps = None
        else:
            if any(np.array([null_maps[x].shape[0] for x in data_obs.index]) < n_perm):
                lgr.warning(f"Number of null maps < n_perm ({n_perm}). Will re-generate.")
                null_maps = None
                
    # datatype
    if null_maps is not None:
        for k in null_maps.keys():
            null_maps[k] = null_maps[k].astype(dtype)
    
    # case null maps not given & not existing
    if null_maps is None:
        lgr.info(f"Generating null maps (n = {n_perm}, null_method = '{null_method}').")
        
        # null data for all maps 
        null_maps, dist_mat = generate_null_maps(
            method=null_method,
            data=data_obs, 
            parcellation=parc,
            parc_space=parc_kwargs["space"], 
            parc_hemi=parc_kwargs["hemi"], 
            #parc_density=parc_kwargs["density"], 
            n_nulls=n_perm, 
            centroids=centroids, 
            dist_mat=dist_mat, 
            n_proc=n_proc, 
            seed=seed, 
        )
            
    # standardize
    if standardize:
        lgr.info("Z-standardizing null maps.")
        null_maps = {k: zscore_df(null_maps[k], along="rows", force_df=False) for k in null_maps.keys()}
    
    return null_maps


def _get_exact_p_values(method, colocs_obs, colocs_null, 
                        xsea_aggr=None, p_tails=None, 
                        verbose=True, dtype=np.float32):
    verbose = set_log(lgr, verbose)
    
    ## get list of the current method's results types
    stats = _get_coloc_stats(method, permuted_only=True)
    if method == "mlr":
        if "individual" not in colocs_obs.keys():
            stats.remove("individual")
        
    ## define p tails
    # defaults
    p_tails_default = _P_TAILS[method]
    if xsea_aggr is not None:
        if "abs" in xsea_aggr:
            p_tails_default = {k: "upper" for k in p_tails_default.keys()}
    if p_tails is None:
        p_tails = p_tails_default
    # if provided check validity
    else:
        # if there's only one stat for method, p_tails can be a string else must be dict
        if isinstance(p_tails, str):
            if len(p_tails_default) == 1:
                p_tails = {stats[0]: p_tails}
            else:
                lgr.warning(f"'p_tails' can only be a string if method has only one outcome stat. "
                            "Using default settings.")
                p_tails = p_tails_default
        # check if dict with one entry for each result type
        elif isinstance(p_tails, dict):
            if not all([stat in p_tails for stat in stats]):
                lgr.warning(f"If 'p_tail' dict is provided, it must contain one entry for each "
                            f"stat ({stats}), you provided: {p_tails}! Using defaults.")
                p_tails = p_tails_default
        # wrong type
        else:
            lgr.warning(f"'p_tails' must be of type dict or string, not {type(p_tails)}.")
            p_tails = p_tails_default
        # check if only contains valid entries
        tails = set([tail for tail in [p_tails[k] for k in p_tails]])
        if any([tail not in ["two", "upper", "lower"] for tail in tails]):
            lgr.error(f"Provided 'p_tails' values can only be one of ['two', 'upper', 'lower'], "
                      f"you provided: {tails}!")
        
    # calculate exact p values
    lgr.info(f"Calculating exact p-values (tails = {p_tails}).")
    # iterate results metrics
    p_data, p_data_norm = dict(), dict()
    for stat in stats:
        p = np.zeros(colocs_obs[stat].shape, dtype=dtype)
        p_norm = p.copy()
        # iterate predictors (columns)
        for x in range(p.shape[1]):
            # iterate targets (rows)
            for y in range(p.shape[0]):
                obs = colocs_obs[stat][y, x]
                null = [colocs_null[i][stat][y, x] for i in range(len(colocs_null))]
                # get p value
                p[y, x] = null_to_p(obs, null, tail=p_tails[stat], fit_norm=False)
                p_norm[y, x] = null_to_p(obs, null, tail=p_tails[stat], fit_norm=True)
        # save data
        p_data[stat] = p
        p_data_norm[stat] = p_norm
        
    # return
    return p_data, p_data_norm


def _get_correct_mc_method(mc_method):
    
    if mc_method=="fdr":
        mc_method = "fdr_bh"
    elif mc_method in ["fdrbh", "fdrby", "fdrtsbh", "fdrtsbky"]:
        mc_method = f"{mc_method[:3]}_{mc_method[3:]}" 
    elif mc_method=="holmsidak":
        mc_method = "holm-sidak"
    elif mc_method=="simeshochberg":
        mc_method = "simes-hochberg"
    else:
        mc_method = mc_method
    
    return mc_method