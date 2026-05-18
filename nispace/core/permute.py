import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import logging
lgr = logging.getLogger(__name__)
from ..nulls import generate_null_maps
from ..stats.misc import null_to_p, zscore_df
from ..utils.utils import set_log
from .colocalize import _get_coloc_stats
from .constants import _P_TAILS


def _get_null_maps(data_obs, nispace_nulls, null_maps=None, use_existing_maps=True, standardize=True,
                   n_perm=1000, null_method="moran",
                   dist_mat=None, spin_mat=None, parc=None, centroids=False, parc_resample=2,
                   lr_mirror_dist_mat=False, split_hemi=None, split_cxsc=False,
                   cx_sc_minmax_scale=False,
                   parc_name=None,
                   seed=None, n_proc=-1, dtype=np.float32, verbose=True, **kwargs):

    # case null maps given
    _custom = False
    if null_maps is not None:
        if not isinstance(null_maps, dict):
            lgr.warning("Provided null maps are not a dictionary. Will re-generate.")
            null_maps = None
        else:
            lgr.info(f"Using provided null maps.")
            _custom = True

    # case null maps not given but existing
    elif (null_maps is None) & (use_existing_maps==True):
        null_method_stored = None
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
        else:
            if any(np.array([null_maps[x].shape[0] for x in data_obs.index]) < n_perm):
                lgr.warning(f"Number of null maps < n_perm ({n_perm}). Will re-generate.")
                null_maps = None
        if not _custom and null_method_stored != null_method:
            lgr.warning("Null method changed. Will re-generate.")
            null_maps = None

    # datatype
    if null_maps is not None:
        for k in null_maps.keys():
            null_maps[k] = null_maps[k].astype(dtype)

    # case null maps not given & not existing
    if null_maps is None:
        lgr.info(f"Generating null maps (n = {n_perm}, null_method = '{null_method}').")

        if parc is not None:
            idc_lh     = parc._idc_byhemi["L"]
            idc_rh     = parc._idc_byhemi["R"]
            parc_img   = parc._image_obj
            parc_space_ = parc._space
            parc_hemi_  = parc._hemi
            parc_sym    = parc._symmetric
        else:
            idc_lh = idc_rh = None
            parc_img = parc_space_ = parc_hemi_ = None
            parc_sym = False

        # for spin methods: resolve surface image and cached spin matrix
        from ..nulls import _SPIN_METHODS
        if null_method in _SPIN_METHODS:
            if spin_mat is None:
                try:
                    spin_mat = nispace_nulls.get("maps_spin", None)
                except Exception:
                    pass
            if parc is not None:
                # get surface image (may differ from the active MNI image for MNI-primary parcs)
                surf_img, surf_spin_mat, surf_space = parc.get_surface_for_spins()
                if surf_img is not None:
                    parc_img     = surf_img
                    parc_space_  = surf_space
                    parc_hemi_   = ("L", "R")
                    # for combined parcellations use cx-only hemisphere indices
                    if parc._is_combined and parc._cx_idc_lh is not None:
                        idc_lh = parc._cx_idc_lh
                        idc_rh = parc._cx_idc_rh
                        # TODO: combined spin+moran: after spin fills cx null maps, run moran
                        # separately for sc parcels (parc_idc_sc) and merge both into the output.
                        # Currently parc_idc_sc=None below, so sc parcels get no null variation.
                    if spin_mat is None and surf_spin_mat is not None:
                        spin_mat = surf_spin_mat
                else:
                    lgr.warning(
                        f"Spin method '{null_method}' requested but no surface data found for "
                        f"parcellation '{parc._name}'. Falling back to 'moran'."
                    )
                    null_method = "moran"

        # null data for all maps
        null_maps, result_mat = generate_null_maps(
            method=null_method,
            data=data_obs,
            parcellation=parc_img,
            parc_space=parc_space_,
            parc_hemi=parc_hemi_,
            parc_symmetric=parc_sym,
            parc_resample=parc_resample,
            n_nulls=n_perm,
            centroids=centroids,
            dist_mat=dist_mat,
            spin_mat=spin_mat,
            parc_idc_lh=idc_lh,
            parc_idc_rh=idc_rh,
            parc_idc_sc=None,
            lr_mirror_dist_mat=lr_mirror_dist_mat,
            split_hemi=split_hemi,
            split_cxsc=split_cxsc,
            cx_sc_minmax_scale=cx_sc_minmax_scale,
            parc_name=parc_name,
            dtype=dtype,
            n_proc=n_proc,
            seed=seed,
            verbose=verbose,
            **kwargs
        )

        # cache spin or dist mat
        if null_method in _SPIN_METHODS:
            nispace_nulls["maps_spin"] = result_mat
        else:
            dist_mat = result_mat
            
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
    p_data = dict()
    for stat in stats:
        p = np.zeros(colocs_obs[stat].shape, dtype=dtype)
        # iterate predictors (columns)
        for x in range(p.shape[1]):
            # iterate targets (rows)
            for y in range(p.shape[0]):
                obs = colocs_obs[stat][y, x]
                null = [colocs_null[i][stat][y, x] for i in range(len(colocs_null))]
                p[y, x] = null_to_p(obs, null, tail=p_tails[stat])
        p_data[stat] = p

    # return (also expose resolved p_tails for storage by caller)
    return p_data, p_tails


_MC_METHOD_ALIASES = {
    # shorthand
    "fdr": "fdr_bh",
    "bonf": "bonferroni",
    # stripped variants of multi-word statsmodels methods (underscore/dash removed)
    "fdrbh": "fdr_bh",
    "fdrby": "fdr_by",
    "fdrtsbh": "fdr_tsbh",
    "fdrtsbky": "fdr_tsbky",
    "holmsidak": "holm-sidak",
    "simeshochberg": "simes-hochberg",
    # empirical methods — case-insensitive aliases
    "meff": "meff_galwey",
    "meff_galwey": "meff_galwey",
    "meffgalwey": "meff_galwey",
    "meff_liji": "meff_li_ji",
    "meffliji": "meff_li_ji",
    "meff_li_ji": "meff_li_ji",
    "maxt": "maxT",
    "maxT": "maxT",
    "step_maxt": "step_maxT",
    "stepmaxt": "step_maxT",
    "step_maxT": "step_maxT",
}

# empirical methods handled internally — NOT passed to statsmodels
_EMPIRICAL_MC_METHODS = {"meff_galwey", "meff_li_ji", "maxT", "step_maxT"}

def _get_correct_mc_method(mc_method):
    return _MC_METHOD_ALIASES.get(mc_method, mc_method)