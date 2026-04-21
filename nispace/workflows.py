import numpy as np
import pandas as pd
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt

from . import lgr, NiSpace
from .utils.utils import set_log
from .modules.constants import (_PARC_DEFAULT, 
                                _COLLECT_DEFAULT,
                                _COLOC_METHODS)
from .datasets import fetch_reference, reference_lib, _check_parcellation

def _workflow_base(x, y, z, x_collection, #x_load_nulls,
                   space,
                   data_space,
                   parcellation_space, 
                   parcellation, parcellation_labels,
                   standardize,
                   colocalization_method,
                   n_proc, verbose,
                   nispace_object, 
                   fetch_x_kwargs,
                   init_kwargs,
                   ):
    """Base workflow for colocalization, group comparison, and GSEA.
    Will load X data, initialize NiSpace object """
    
    status = {fun: False for fun in ["init", "fit"]}
    
    # check colocalization method
    if isinstance(colocalization_method, (list, tuple)):
       if not all(method in _COLOC_METHODS for method in colocalization_method):
           raise lgr.critical_raise("'colocalization_method' must be one or a list of "
                                    f"{list(_COLOC_METHODS.keys())} not {colocalization_method}!",
                                    ValueError)
    else:
        raise lgr.critical_raise("'colocalization_method' must be a string or a list of strings!",
                                 TypeError)
        
    # check if nispace object provided
    if nispace_object is not None:
        lgr.info("NiSpace object provided. Validating.")
        nsp = nispace_object
        if isinstance(nispace_object, NiSpace):
            if nsp._check_fit():
                lgr.info("Fitted NiSpace object provided, ignoring 'x', 'y', and 'z'.")
                status["init"] = True
                status["fit"] = True
            else:
                lgr.info("NiSpace object provided but .fit() was not run. Running.")
                status["init"] = True
        else:
            lgr.critical_raise(f"Argument 'nispace_object' must be of type NiSpace not {type(nsp)}!")
            
    # space: data_space and parcellation_space default to the shared space arg
    data_space = space if not data_space else data_space
    parcellation_space = space if not parcellation_space else parcellation_space

    ## INIT
    if not status["init"]:

        # resolve integrated parcellation name (needed for fetch_reference below)
        parc_integrated = None
        if isinstance(parcellation, str):
            parc_integrated = _check_parcellation(parcellation, force_str=True, raise_not_found=False)

        # y
        if y is None:
            lgr.error("You must provide 'y' data: (list) of volumetric/surface or pre-parcellated data!")

        # x
        if isinstance(x, str):
            x = x.lower()
            if x in reference_lib:
                lgr.info(f"Loading integrated {x} dataset as X data.")
                if x_collection is None or not isinstance(x_collection, str):
                    x_collection = _COLLECT_DEFAULT[x]
                    lgr.info(f"Using collection {x_collection}.")
                fetch_x_kwargs = dict(
                    dataset=x,
                    collection=x_collection,
                    standardize_parcellated=False,
                    parcellation=parc_integrated,
                    verbose=verbose
                ) | fetch_x_kwargs
                x = fetch_reference(**fetch_x_kwargs)
                if isinstance(x, tuple):
                    x, null_maps = x
                else:
                    null_maps = None
            else:
                lgr.error(f"'x' must be one of: '{list(reference_lib.keys())}' not '{x}'!")
        else:
            null_maps = None

        # init — pass parcellation as-is; api.fit() resolves integrated names and spaces
        init_kwargs = dict(
            x=x,
            y=y,
            z=z,
            standardize=standardize,
            parcellation=parcellation,
            parcellation_labels=parcellation_labels,
            parcellation_space=parcellation_space,
            data_space=data_space,
            n_proc=n_proc,
            verbose=verbose,
            return_self=True,
        ) | init_kwargs
        nsp = NiSpace(**init_kwargs)
    
    ## FIT
    if not status["fit"]:
        nsp.fit()
        status["fit"] = True
        
    ## RETURN status, NiSpace object, pre-loaded nulls
    return status, nsp, null_maps
       
       
def simple_colocalization(y,
                          x="PET",
                          z=None,
                          x_collection=None,
                          standardize="xz",
                          space="MNI152NLin2009cAsym",
                          data_space=None,
                          parcellation_space=None,
                          parcellation=_PARC_DEFAULT,
                          parcellation_labels=None,
                          y_covariates=None,
                          colocalization_method="spearman",
                          mc_method="meff",
                          p_from_average_y=False,
                          plot=True,
                          combat=False,
                          n_perm=10000,
                          seed=None,
                          #x_load_nulls=True,
                          n_proc=1,
                          verbose=True,
                          nispace_object=None,
                          fetch_x_kwargs=None,
                          init_kwargs=None,
                          clean_y_kwargs=None,
                          colocalize_kwargs=None,
                          permute_kwargs=None,
                          correct_p_kwargs=None,
                          plot_kwargs=None):
    """Simple colocalization workflow.
    
    Parameters
    ----------
    y : array-like or pandas DataFrame or list
        Input Y data to colocalize with X. Can be a numpy array, pandas DataFrame,
        (list of) path(s) to a file(s) or list of image objects.
    x : str or array-like, default="PET"
        Input X data. Can be a string indicating a reference dataset ("PET", "mRNA", ...), 
        or inputs types as listed for y.
    z : array-like or None, default=None
        Optional confound data to regress out. Can be "gm", or input types as listed for y.
    x_collection : str or None, default=None
        If x is a string reference dataset, specifies which collection to use.
    standardize : str, default="xz"
        Which data to standardize. Can contain "x", "y", and/or "z".
    parcellation : str or int, default=_PARC_DEFAULT
        Brain parcellation to use. Can be a string name or integer ID.
    parcellation_labels : array-like or None, default=None
        Optional labels for the parcellation regions.
    y_covariates : array-like or None, default=None
        Optional covariates to regress from Y data.
    colocalization_method : str or list, default="spearman"
        Method(s) to use for colocalization. Can be "spearman", "pearson", etc.
    p_from_average_y : bool, default=False
        Whether to compute p-values from averaged Y values.
    plot : bool, default=True
        Whether to generate visualization plots.
    combat : bool, default=False
        Whether to apply ComBat harmonization.
    n_perm : int, default=10000
        Number of permutations for null distribution.
    seed : int or None, default=None
        Random seed for reproducibility.
    n_proc : int, default=-1
        Number of processes for parallel computation. -1 uses all CPUs.
    verbose : bool, default=True
        Whether to print progress messages.
    nispace_object : NiSpace or None, default=None
        Optional pre-initialized NiSpace object to use.
    fetch_x_kwargs : dict, default={}
        Additional arguments for fetching X data.
    init_kwargs : dict, default={}
        Additional arguments for NiSpace initialization.
    clean_y_kwargs : dict, default={}
        Additional arguments for Y data cleaning.
    colocalize_kwargs : dict, default={}
        Additional arguments for colocalization.
    permute_kwargs : dict, default={}
        Additional arguments for permutation testing.
    correct_p_kwargs : dict, default={}
        Additional arguments for p-value correction.
    plot_kwargs : dict, default={}
        Additional arguments for plotting.

    Returns
    -------
    colocs : dict or array
        Colocalization values for each method.
    p_values : dict or array
        Uncorrected p-values for each method.
    p_fdr_values : dict or array
        FDR-corrected p-values for each method.
    nsp : NiSpace
        The NiSpace object containing all results.
    """
    verbose = set_log(lgr, verbose)
    # kwarg dicts
    fetch_x_kwargs = {} if fetch_x_kwargs is None else fetch_x_kwargs
    init_kwargs = {} if init_kwargs is None else init_kwargs
    clean_y_kwargs = {} if clean_y_kwargs is None else clean_y_kwargs
    colocalize_kwargs = {} if colocalize_kwargs is None else colocalize_kwargs
    permute_kwargs = {} if permute_kwargs is None else permute_kwargs
    correct_p_kwargs = {} if correct_p_kwargs is None else correct_p_kwargs
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    
    ## COMMON FUNCTIONS: COLOC METHOD VALIDATION, DATA LOADING, INIT,
    if isinstance(colocalization_method, str):
        colocalization_method = [colocalization_method]
    status, nsp, null_maps = _workflow_base(
        x=x, y=y, z=z, 
        x_collection=x_collection, 
        space=space,
        data_space=data_space,
        parcellation_space=parcellation_space,
        #x_load_nulls=x_load_nulls,
        standardize=standardize,
        parcellation=parcellation,
        parcellation_labels=parcellation_labels,
        colocalization_method=colocalization_method,
        n_proc=n_proc, 
        verbose=verbose,
        nispace_object=nispace_object, 
        fetch_x_kwargs=fetch_x_kwargs,
        init_kwargs=init_kwargs
    )
    status = status | {fun: False for fun in ["clean_y", "colocalize", "permute", "correct_p"]}   
    
    ## CLEAN Y
    if not status["clean_y"] and y_covariates is not None:
        clean_y_kwargs = dict(
            how="between",
            covariates_between=y_covariates,
            combat=combat,
        ) | clean_y_kwargs
        nsp.clean_y(**clean_y_kwargs)
        status["clean_y"] = True
    
    ## COLOCALIZE
    # xsea must be same for colocalization and permutation
    if colocalize_kwargs.get("xsea", False) or permute_kwargs.get("xsea", False):
        colocalize_kwargs["xsea"] = True
        permute_kwargs["xsea"] = True
    if not status["colocalize"]:
        for method in colocalization_method:
            colocalize_kwargs_curr = dict(
                method=method,
                regress_z=True,
            ) | colocalize_kwargs
            nsp.colocalize(**colocalize_kwargs_curr)
        status["colocalize"] = True
        
    ## PERMUTE
    if not status["permute"]:
        for method in colocalization_method:
            permute_kwargs_curr = dict(
                what="maps",
                maps_which="X",
                maps_nulls=null_maps,
                method=method,
                p_from_average_y_coloc=p_from_average_y,
                n_perm=n_perm,
                seed=seed,
            ) | permute_kwargs
            nsp.permute(**permute_kwargs_curr)
        permuted = nsp._get_last(perm=None)
        status["permute"] = True  
    
    ## CORRECT
    # normalize mc_method to list; explicit override inside correct_p_kwargs takes precedence
    mc_methods = ([mc_method] if isinstance(mc_method, str) else list(mc_method))
    if "mc_method" in correct_p_kwargs:
        mc_methods = [correct_p_kwargs.pop("mc_method")]
    if not status["correct_p"]:
        for mc_m in mc_methods:
            nsp.correct_p(**{"mc_method": mc_m} | correct_p_kwargs)
        status["correct_p"] = True

    ## VIZ
    if plot:
        for method in colocalization_method:
            plot_kwargs_curr = dict(
                method=method,
                permute_what=permuted,
            ) | plot_kwargs
            nsp.plot(**plot_kwargs_curr)

    ## RETURN
    colocs = {method: nsp.get_colocalizations(method)
              for method in colocalization_method}
    p_values = {method: nsp.get_p_values(method, permuted)
                for method in colocalization_method}
    p_fdr_values = {
        mc_m: {method: nsp.get_p_values(method, permuted, mc_method=mc_m)
               for method in colocalization_method}
        for mc_m in mc_methods
    }
    if len(colocalization_method) == 1:
        k = colocalization_method[0]
        colocs, p_values = colocs[k], p_values[k]
        p_fdr_values = {mc_m: p_fdr_values[mc_m][k] for mc_m in mc_methods}
    if len(mc_methods) == 1:
        p_fdr_values = p_fdr_values[mc_methods[0]]

    return colocs, p_values, p_fdr_values, nsp
    
        
def group_comparison(y, design,
                     x="PET",
                     z=None,
                     x_collection=None,
                     standardize="xz",
                     space="MNI152NLin2009cAsym",
                     data_space=None,
                     parcellation_space=None,
                     parcellation=_PARC_DEFAULT,
                     parcellation_labels=None,
                     colocalization_method="spearman",
                     comparison_method=None,
                     mc_method="meff",
                     paired=False,
                     plot_design_between=True,
                     combat=False,
                     plot=True,
                     n_perm=10000,
                     seed=None,
                     n_proc=1,
                     verbose=True,
                     nispace_object=None,
                     fetch_x_kwargs=None,
                     init_kwargs=None,
                     clean_y_kwargs=None,
                     transform_y_kwargs=None,
                     colocalize_kwargs=None,
                     permute_kwargs=None,
                     correct_p_kwargs=None,
                     plot_kwargs=None):
    verbose = set_log(lgr, verbose)
    # kwarg dicts
    fetch_x_kwargs = {} if fetch_x_kwargs is None else fetch_x_kwargs
    init_kwargs = {} if init_kwargs is None else init_kwargs
    clean_y_kwargs = {} if clean_y_kwargs is None else clean_y_kwargs
    transform_y_kwargs = {} if clean_y_kwargs is None else clean_y_kwargs
    colocalize_kwargs = {} if colocalize_kwargs is None else colocalize_kwargs
    permute_kwargs = {} if permute_kwargs is None else permute_kwargs
    correct_p_kwargs = {} if correct_p_kwargs is None else correct_p_kwargs
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs

    ## COMMON FUNCTIONS: DATA LOADING, INIT, YCOLOC METHOD VALIDATION
    if isinstance(colocalization_method, str):
        colocalization_method = [colocalization_method]
    status, nsp, _ = _workflow_base(
        x=x, y=y, z=z, 
        x_collection=x_collection, 
        #x_load_nulls=False,
        space=space,
        data_space=data_space,
        parcellation_space=parcellation_space,
        standardize=standardize,
        parcellation=parcellation,
        parcellation_labels=parcellation_labels,
        colocalization_method=colocalization_method,
        n_proc=n_proc, 
        verbose=verbose,
        nispace_object=nispace_object, 
        fetch_x_kwargs=fetch_x_kwargs,
        init_kwargs=init_kwargs
    )
    status = status | {fun: False for fun in ["clean_y", "transform_y", "colocalize", "permute", "correct_p"]}   
      
    ## DESIGN MATRIX HANDLING
    # ensure dtype and format
    # 1d
    if isinstance(design, (list, tuple)) or \
        (isinstance(design, (np.ndarray, pd.Series)) and design.ndim==1):
        if paired:
            lgr.critical_raise("If paired==True, design must have two columns: 'group' and 'subjects'.",
                               ValueError)
        else:
            lgr.info("1d array provided for design. Assuming this to be dummy-coded groups!")
            design = pd.DataFrame(
                {"groups": np.array(design)}, 
                index=nsp._y_lab
            )
    # 2darray
    elif isinstance(design, np.ndarray) and design.ndim==2:
        if paired:
            lgr.info("2d array provided for design with paired==True. Assuming first column "
                     "to be group labels, second column to be subjects, and remaining to be covariates.")
            design = pd.DataFrame(
                design, 
                columns=["groups", "subjects"] + [f"V{i}" for i in range(design.shape[1] - 2)],
                index=nsp._y_lab
            )
        else:
            lgr.info("2d array provided for design. Assuming first column to be group labels, "
                     "second column to be subjects.")
            design = pd.DataFrame(
                design, 
                columns=["groups"] + [f"V{i}" for i in range(design.shape[1] - 1)],
                index=nsp._y_lab
            )
    # dataframe
    elif isinstance(design, pd.DataFrame):
        lgr.info("DataFrame provided for design. Expecting 'groups' and, if paired==True, 'subjects' columns.")
        if paired:
            if "groups" not in design.columns and "subjects" not in design.columns:
                lgr.critical_raise("If a DataFrame is passed for design with paired==True, "
                                   "it must have a 'groups' and a 'subjects' column.",
                                   KeyError)
        else:
            if "groups" not in design.columns:
                lgr.critical_raise("If a DataFrame is passed for design, it must have a 'groups' column.",
                                   KeyError)
    # unrecognized type
    else:
        lgr.critical_raise("'design' must be a list, ndarray, Series, or DataFrame!",
                           TypeError)
    # check dimensions
    lgr.info(f"Design matrix of shape {design.shape}. Assuming {design.shape[0]} subjects/maps.")
    if design.shape[0] != len(y):
        lgr.critical_raise(f"The number of rows in design matrix {design.shape[0]} must equal "
                           f"the length of the y data {len(y)}!",
                           ValueError)
    
    ## CLEAN Y
    if not status["clean_y"] and \
        ((not paired and design.shape[1] > 1) or (paired and design.shape[1] > 2)):
        if not paired:
            y_covariates = design.iloc[:, 1:]       # exclude groups
            combat_protect = design[["groups"]]
        else:
            y_covariates = design.iloc[:, 2:]       # exclude groups and subjects
            combat_protect = design[["groups", "subjects"]]
        clean_y_kwargs = dict(
            how="between",
            covariates_between=y_covariates,
            combat=combat,
            combat_protect=combat_protect if combat else None,
            plot_design_between=plot_design_between
        ) | clean_y_kwargs
        nsp.clean_y(**clean_y_kwargs)
        status["clean_y"] = True

    ## TRANSFORM
    if not status["transform_y"]:
        if comparison_method is None and not paired:
            comparison_method = "hedges(a,b)"
        elif comparison_method is None and paired:
            comparison_method = "pairedcohen(a,b)"
        transform_y_kwargs = dict(
            transform=comparison_method,
            groups=design["groups"],
            subjects=design["subjects"] if paired else None,
        ) | transform_y_kwargs
        nsp.transform_y(**transform_y_kwargs)
        status["transform_y"] = True
    
    ## COLOCALIZE
    if not status["colocalize"]:
        for method in colocalization_method:
            colocalize_kwargs_curr = dict(
                method=method,
                Y_transform=comparison_method,
                regress_z=True,
                verbose=verbose,
            ) | colocalize_kwargs
            nsp.colocalize(**colocalize_kwargs_curr)
        status["colocalize"] = True
        
    ## PERMUTE
    if not status["permute"]:
        for method in colocalization_method:
            permute_kwargs_curr = dict(
                what="groups",
                method=method,
                Y_transform=comparison_method,
                groups_paired=paired, 
                groups_strategy="proportional",
                n_perm=n_perm,
                seed=seed,
                verbose=verbose,
            ) | permute_kwargs
            nsp.permute(**permute_kwargs_curr)
        permute_what = "groups"
        status["permute"] = True  
    
    ## CORRECT
    mc_methods = ([mc_method] if isinstance(mc_method, str) else list(mc_method))
    if "mc_method" in correct_p_kwargs:
        mc_methods = [correct_p_kwargs.pop("mc_method")]
    if not status["correct_p"]:
        for mc_m in mc_methods:
            nsp.correct_p(**{"mc_method": mc_m, "verbose": verbose} | correct_p_kwargs)
        status["correct_p"] = True

    ## VIZ
    if plot:
        for method in colocalization_method:
            plot_kwargs_curr = dict(
                method=method,
                permute_what=permute_what,
                Y_transform=comparison_method,
                verbose=verbose,
            ) | plot_kwargs
            nsp.plot(**plot_kwargs_curr)

    ## RETURN
    colocs = {method: nsp.get_colocalizations(method, Y_transform=comparison_method)
              for method in colocalization_method}
    p_values = {method: nsp.get_p_values(method, permute_what, Y_transform=comparison_method)
                for method in colocalization_method}
    p_fdr_values = {
        mc_m: {method: nsp.get_p_values(method, permute_what, Y_transform=comparison_method,
                                        mc_method=mc_m)
               for method in colocalization_method}
        for mc_m in mc_methods
    }
    if len(colocalization_method) == 1:
        k = colocalization_method[0]
        colocs, p_values = colocs[k], p_values[k]
        p_fdr_values = {mc_m: p_fdr_values[mc_m][k] for mc_m in mc_methods}
    if len(mc_methods) == 1:
        p_fdr_values = p_fdr_values[mc_methods[0]]

    return colocs, p_values, p_fdr_values, nsp
    

def simple_xsea(y,
                x="mRNA",
                z=None,
                x_collection=None,
                x_background=None,
                standardize="xz",
                space="MNI152NLin2009cAsym",
                data_space=None,
                parcellation_space=None,
                parcellation=_PARC_DEFAULT,
                parcellation_labels=None,
                y_covariates=None,
                colocalization_method="spearman",
                mc_method="meff",
                xsea_aggregation_method="mean",
                permute_sets=False,
                p_from_average_y=False,
                plot=True,
                combat=False,
                n_perm=10000,
                seed=None,
                n_proc=1,
                verbose=True,
                nispace_object=None,
                fetch_x_kwargs=None,
                init_kwargs=None,
                clean_y_kwargs=None,
                colocalize_kwargs=None,
                permute_kwargs=None,
                correct_p_kwargs=None,
                plot_kwargs=None):
    verbose = set_log(lgr, verbose)
    # kwarg dicts
    fetch_x_kwargs = {} if fetch_x_kwargs is None else fetch_x_kwargs
    init_kwargs = {} if init_kwargs is None else init_kwargs
    clean_y_kwargs = {} if clean_y_kwargs is None else clean_y_kwargs
    colocalize_kwargs = {} if colocalize_kwargs is None else colocalize_kwargs
    permute_kwargs = {} if permute_kwargs is None else permute_kwargs
    correct_p_kwargs = {} if correct_p_kwargs is None else correct_p_kwargs
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    
    # GET THE BACKGROUND
    if permute_sets:
        if x_background is None and isinstance(x, str):
            lgr.info("Trying to fetch background X dataset.")
            if x.lower() in reference_lib:
                try:
                    x_background = fetch_reference(x.lower(), parcellation=parcellation, print_references=False)
                except:
                    x_background = None
        if x_background is None:
            lgr.warning(f"Could not fetch background dataset for input x!")
    
    ## We go the easy way and just call .simple_colocalization() with some kwargs:
    colocs, p_values, p_fdr_values, nsp = simple_colocalization(
        y=y,
        x=x, z=z,
        x_collection=x_collection,
        standardize=standardize,
        parcellation=parcellation,
        parcellation_labels=parcellation_labels,
        y_covariates=y_covariates,
        colocalization_method=colocalization_method,
        mc_method=mc_method,
        p_from_average_y=p_from_average_y,
        plot=plot,
        combat=combat,
        n_perm=n_perm,
        seed=seed,
        n_proc=n_proc,
        verbose=verbose,
        nispace_object=nispace_object,
        fetch_x_kwargs=fetch_x_kwargs,
        init_kwargs=init_kwargs,
        clean_y_kwargs=clean_y_kwargs,
        colocalize_kwargs={
            "xsea_aggregation_method": xsea_aggregation_method,
            "xsea": True
        } | colocalize_kwargs,
        permute_kwargs={
            "what": "maps" if not permute_sets else "sets",
            "maps_which": "Y",
            "sets_X_background": x_background if permute_sets else None,
        } | permute_kwargs,
        correct_p_kwargs=correct_p_kwargs,
        plot_kwargs=plot_kwargs
    )
    
    return colocs, p_values, p_fdr_values, nsp
