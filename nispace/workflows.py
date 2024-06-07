import numpy as np
import pandas as pd
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt

from . import lgr, NiSpace
from .utils import set_log
from .modules.constants import (_PARCS, _PARCS_NICE, _PARCS_DEFAULT, 
                                _DSETS, _DSETS_NICE, _COLLECT_DEFAULT,
                                _COLOC_METHODS)
from .datasets import fetch_reference

def _workflow_base(x, y, z, x_collection, #x_load_nulls,
                   standardize,
                   parcellation, parcellation_labels,
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
            lgr.critical(f"Argument 'nispace_object' must be of type NiSpace not {type(nsp)}!")
          
    ## INIT
    if not status["init"]:
        
        # check provided data
        # parcellation
        if isinstance(parcellation, str):
            if parcellation.lower() in _PARCS:
                lgr.info(f"Using integrated parcellation {parcellation}.")
                parc_integrated = parcellation
            else:
                parc_integrated = None
            
        # y
        if y is None:
            lgr.error("You must provide 'y' data: (list) of volumetric/surface or pre-parcellated data!")
            
        # x
        if isinstance(x, str):
            x = x.lower()
            if x in _DSETS:
                lgr.info(f"Loading integrated {x} dataset as X data.")
                if x_collection is None or not isinstance(x_collection, str):
                    x_collection = _COLLECT_DEFAULT[x]
                    lgr.info(f"Using collection {x_collection}.")
                fetch_x_kwargs = dict(
                    dataset=x,
                    collection=x_collection,
                    standardize_parcellated=False,
                    parcellation=parc_integrated,
                    return_nulls=False,
                    verbose=verbose
                ) | fetch_x_kwargs
                x = fetch_reference(**fetch_x_kwargs)
                if isinstance(x, tuple):
                    x, null_maps = x
                else:
                    null_maps = None
            else:
                lgr.error(f"'x' must be one of: '{_DSETS_NICE}' not '{x}'!")
        else:
            null_maps = None
        
        # init
        init_kwargs = dict(
            x=x,
            y=y,
            z=z,
            standardize=standardize,
            parcellation=parcellation,
            parcellation_labels=parcellation_labels,
            n_proc=n_proc,
            verbose=verbose,
        ) | init_kwargs
        nsp = NiSpace(**init_kwargs)
    
    ## FIT
    if not status["fit"]:
        nsp.fit()
        status["fit"] = True
        
    ## RETURN status, NiSpace object, pre-loaded nulls
    return status, nsp, null_maps
       
       
def simple_colocalization(y, 
                          x="PET", z="gm", 
                          x_collection=None,
                          standardize="xz",
                          parcellation=_PARCS_DEFAULT,
                          parcellation_labels=None,
                          y_covariates=None,
                          colocalization_method="spearman",
                          p_from_average_y=False,
                          plot=True,
                          combat=False,
                          n_perm=10000,
                          #x_load_nulls=True,
                          n_proc=-1,
                          verbose=True,
                          nispace_object=None, 
                          fetch_x_kwargs={},
                          init_kwargs={},
                          clean_y_kwargs={},
                          colocalize_kwargs={},
                          permute_kwargs={},
                          correct_p_kwargs={},
                          plot_kwargs={}):
    verbose = set_log(lgr, verbose)
    lgr.info("*** NiSpace Workflows: Simple Colocalization ***")
    
    ## COMMON FUNCTIONS: COLOC METHOD VALIDATION, DATA LOADING, INIT,
    if isinstance(colocalization_method, str):
        colocalization_method = [colocalization_method]
    status, nsp, null_maps = _workflow_base(
        x=x, y=y, z=z, 
        x_collection=x_collection, 
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
    if not status["colocalize"]:
        for method in colocalization_method:
            colocalize_kwargs = dict(
                method=method,
                Z_regression=True,
            ) | colocalize_kwargs
            nsp.colocalize(**colocalize_kwargs)
        status["colocalize"] = True
        
    ## PERMUTE
    if not status["permute"]:
        for method in colocalization_method:
            permute_kwargs = dict(
                    what="maps",
                    maps_which="X",
                    maps_nulls=null_maps,
                    method=method,
                    p_from_average_y_coloc=p_from_average_y,
                    n_perm=n_perm,
                ) | permute_kwargs
            nsp.permute(**permute_kwargs)
        permuted = f"{permute_kwargs['maps_which']}{permute_kwargs['what']}".lower()
        status["permute"] = True  
    
    ## CORRECT
    if not status["correct_p"]:
        correct_p_kwargs = dict(
        ) | correct_p_kwargs
        nsp.correct_p(**correct_p_kwargs)
        status["correct_p"] = True
        
    ## VIZ
    if plot:
        for method in colocalization_method:
            plot_kwargs = dict(
                method=method,
                permute_what=permuted,
            ) | plot_kwargs
            nsp.plot(**plot_kwargs)
        
    ## RETURN
    colocs = {method: nsp.get_colocalizations(method) 
              for method in colocalization_method}
    p_values = {method: nsp.get_p_values(method, permuted) 
                for method in colocalization_method}
    p_fdr_values = {method: nsp.get_p_values(method, permuted, mc_method="fdr_bh") 
                    for method in colocalization_method}
    if len(colocalization_method)==1:
        colocs, p_values, p_fdr_values = (colocs[colocalization_method[0]], 
                                          p_values[colocalization_method[0]], 
                                          p_fdr_values[colocalization_method[0]])
        
    return colocs, p_values, p_fdr_values, nsp
    
        
def group_comparison(y, design, 
                     x="PET", z="gm", 
                     x_collection=None,
                     standardize="xz",
                     parcellation=_PARCS_DEFAULT,
                     parcellation_labels=None,
                     colocalization_method="spearman",
                     group_comparison=None,
                     paired=False,
                     plot_design=True,
                     combat=False,
                     plot=True,
                     n_perm=10000,
                     n_proc=-1,
                     verbose=True,
                     nispace_object=None, 
                     fetch_x_kwargs={},
                     init_kwargs={},
                     clean_y_kwargs={},
                     transform_y_kwargs={},
                     colocalize_kwargs={},
                     permute_kwargs={},
                     correct_p_kwargs={},
                     plot_kwargs={}):
    verbose = set_log(lgr, verbose)
    lgr.info("*** NiSpace Workflows: Group Comparison ***")

    ## COMMON FUNCTIONS: DATA LOADING, INIT, YCOLOC METHOD VALIDATION
    if isinstance(colocalization_method, str):
        colocalization_method = [colocalization_method]
    status, nsp, _ = _workflow_base(
        x=x, y=y, z=z, 
        x_collection=x_collection, 
        #x_load_nulls=False,
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
                {"groups": design}, 
                index=y.index
            )
    # 2darray
    elif isinstance(design, np.ndarray) and design.ndim==2:
        if paired:
            lgr.info("2d array provided for design with paired==True. Assuming first column "
                     "to be group labels, second column to be subjects, and remaining to be covariates.")
            design = pd.DataFrame(
                design, 
                columns=["groups", "subjects"] + [f"V{i}" for i in range(design.shape[1] - 2)],
                index=y.index
            )
        else:
            lgr.info("2d array provided for design. Assuming first column to be group labels, "
                     "second column to be subjects.")
            design = pd.DataFrame(
                design, 
                columns=["groups"] + [f"V{i}" for i in range(design.shape[1] - 1)],
                index=y.index
            )
    # dataframe
    elif isinstance(design, pd.DataFrame):
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
    if design.shape[0] != y.shape[0]:
        lgr.critical_raise(f"The number of rows in design matrix {design.shape[0]} must equal "
                           f"the number of rows in y data {y.shape[0]}!",
                           ValueError)
    # plot
    if plot_design:
        print(design.head(5))
        plot_design_matrix(design)
        plt.title("Design matrix")
        plt.ylabel("Y maps")
        plt.show()
    
    ## CLEAN Y
    if not status["clean_y"] and \
        ((not paired and design.shape[1] > 1) or (paired and design.shape[1] > 2)):
        if combat and not paired:
            y_covariates = design
            combat_keep = ["groups"]
        elif combat and paired:
            y_covariates = design
            combat_keep = ["groups", "subjects"]
        elif not combat and not paired:
            y_covariates = design.iloc[:, 1:]
            combat_keep = None
        elif not combat and paired:
            y_covariates = design.iloc[:, 2:]
            combat_keep = None
        clean_y_kwargs = dict(
            how="between",
            covariates_between=y_covariates,
            combat=combat,
            combat_keep=combat_keep,
        ) | clean_y_kwargs
        nsp.clean_y(**clean_y_kwargs)
        status["clean_y"] = True

    ## TRANSFORM
    if not status["transform_y"]:
        if group_comparison is None and not paired:
            group_comparison = "hedges(a,b)"
        elif group_comparison is None and paired:
            group_comparison = "pairedcohen(a,b)"
        transform_y_kwargs = dict(
            transform=group_comparison,
            groups=design["groups"],
            subjects=design["subjects"] if paired else None,
        ) | transform_y_kwargs
        nsp.transform_y(**transform_y_kwargs)
        status["transform_y"] = True
    
    ## COLOCALIZE
    if not status["colocalize"]:
        for method in colocalization_method:
            colocalize_kwargs = dict(
                method=method,
                Y_transform=group_comparison,
                Z_regression=True,
                verbose=verbose,
            ) | colocalize_kwargs
            nsp.colocalize(**colocalize_kwargs)
        status["colocalize"] = True
        
    ## PERMUTE
    if not status["permute"]:
        for method in colocalization_method:
            permute_kwargs = dict(
                    what="groups",
                    method=method,
                    Y_transform=group_comparison,
                    groups_perm_paired=paired, 
                    groups_perm_strategy="proportional",
                    n_perm=n_perm,
                    verbose=verbose,
                ) | permute_kwargs
            nsp.permute(**permute_kwargs)
        permute_what = "groups"
        status["permute"] = True  
    
    ## CORRECT
    if not status["correct_p"]:
        correct_p_kwargs = dict(
            verbose=verbose,
        ) | correct_p_kwargs
        nsp.correct_p(**correct_p_kwargs)
        status["correct_p"] = True
        
    ## VIZ
    if plot:
        for method in colocalization_method:
            plot_kwargs = dict(
                method=method,
                permute_what=permute_what,
                Y_transform=group_comparison,
                verbose=verbose,
            ) | plot_kwargs
            nsp.plot(**plot_kwargs)
        
    ## RETURN
    colocs = {method: nsp.get_colocalizations(method, Y_transform=group_comparison) 
              for method in colocalization_method}
    p_values = {method: nsp.get_p_values(method, permute_what, Y_transform=group_comparison) 
                for method in colocalization_method}
    p_fdr_values = {method: nsp.get_p_values(method, permute_what, Y_transform=group_comparison, 
                                             mc_method="fdr_bh") 
                    for method in colocalization_method}
    if len(colocalization_method)==1:
        colocs, p_values, p_fdr_values = (colocs[colocalization_method[0]], 
                                          p_values[colocalization_method[0]], 
                                          p_fdr_values[colocalization_method[0]])
        
    return colocs, p_values, p_fdr_values, nsp
    

def simple_xsea():
    lgr.critical_raise("X-set enrichment analysis is not yet implemented!", 
                       NotImplementedError)



