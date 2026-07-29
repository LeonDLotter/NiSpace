import numpy as np
import pandas as pd
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt

import logging
lgr = logging.getLogger(__name__)
from . import NiSpace
from .utils.utils import set_log
from .core.constants import _PARC_DEFAULT, _COLOC_METHODS, _SPACE_DEFAULT_VOL
from .datasets import fetch_reference, reference_lib, _check_parcellation

_DEPR_POOLED_P = (
    "'p_from_average_y' is deprecated and will be removed in the first "
    "non-dev release. Use 'pooled_p' instead."
)
_DEPR_RETURN_TUPLE = (
    "Returning a tuple (colocs, p_values, pc_values, nsp) from workflow functions is "
    "deprecated and will be removed in the first non-dev release. "
    "Set 'return_nispace_only=True' and use 'nsp.get_colocalizations()' and "
    "'nsp.get_p_values()' to access results."
)
_DEPR_FUNC_NAME = (
    "'{old}' is deprecated and will be removed in the first non-dev release. "
    "Use '{new}()' instead."
)


def _workflow_base(x, y, z, x_collection, #x_load_nulls,
                   space,
                   data_space,
                   parcellation_space,
                   parcellation, parcellation_labels,
                   parcellation_hemi,
                   standardize,
                   colocalization_method,
                   n_proc, verbose,
                   nispace_object,
                   fetch_x_kwargs,
                   init_kwargs,
                   fit_kwargs,
                   ):
    """Base workflow for colocalization, group comparison, and GSEA.
    Will load X data, initialize NiSpace object """
    
    status = {fun: False for fun in ["init", "fit"]}
    null_maps = None

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
                    x_collection = reference_lib[x].get("default_collection")
                    if x_collection is not None:
                        lgr.info(f"Using default collection '{x_collection}'.")
                fetch_x_kwargs = dict(
                    dataset=x,
                    collection=x_collection,
                    standardize_parcellated=False,
                    parcellation=parc_integrated,
                    hemi=parcellation_hemi,
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
            parcellation_hemi=parcellation_hemi,
            parcellation_space=parcellation_space,
            data_space=data_space,
            n_proc=n_proc,
            verbose=verbose,
            return_self=True,
        ) | init_kwargs
        nsp = NiSpace(**init_kwargs)
    
    ## FIT
    if not status["fit"]:
        nsp.fit(**fit_kwargs)
        status["fit"] = True
        
    ## RETURN status, NiSpace object, pre-loaded nulls
    return status, nsp, null_maps
       
       
def colocalization(y,
                   x="PET",
                   z=None,
                   x_collection=None,
                   standardize="xz",
                   space=_SPACE_DEFAULT_VOL,
                   data_space=None,
                   parcellation_space=None,
                   parcellation=_PARC_DEFAULT,
                   parcellation_labels=None,
                   parcellation_hemi=["L", "R"],
                   y_covariates=None,
                   colocalization_method=None,
                   mc_method="meff",
                   normalize_colocalizations=True,
                   pooled_p=False,
                   p_from_average_y=None,  # TODO (first non-dev release): remove
                   plot=True,
                   combat=False,
                   binary_y=False,
                   n_perm=10000,
                   seed=None,
                   n_proc=1,
                   verbose=True,
                   nispace_object=None,
                   fetch_x_kwargs=None,
                   init_kwargs=None,
                   fit_kwargs=None,
                   clean_y_kwargs=None,
                   colocalize_kwargs=None,
                   permute_kwargs=None,
                   correct_p_kwargs=None,
                   plot_kwargs=None,
                   return_nispace_only=False):
    """Colocalization workflow.

    Parameters
    ----------
    y : array-like or pandas DataFrame or list
        Input Y data to colocalize with X. Can be a numpy array, pandas DataFrame,
        (list of) path(s) to a file(s) or list of image objects.
    x : str or array-like, default="PET"
        Input X data. Can be a string indicating a reference dataset ("PET", "mRNA", ...),
        or input types as listed for y.
    z : array-like or None, default=None
        Optional confound data to regress out. Can be "gm", or input types as listed for y.
    x_collection : str or None, default=None
        If x is a string reference dataset, specifies which collection to use.
    standardize : str, default="xz"
        Which data to standardize. Can contain "x", "y", and/or "z".
    space : str, default=_SPACE_DEFAULT_VOL ("MNI152NLin6Asym")
        Default template space for both the data images and the parcellation. Used
        to resolve ``data_space``/``parcellation_space`` when those are not given.
    data_space : str or None, default=None
        Template space of the input data images. Falls back to ``space`` if falsy.
    parcellation_space : str or None, default=None
        Template space of the parcellation. Falls back to ``space`` if falsy.
    parcellation : str or int, default=_PARC_DEFAULT
        Brain parcellation to use. Can be a string name or integer ID.
    parcellation_labels : array-like or None, default=None
        Optional labels for the parcellation regions.
    parcellation_hemi : list of str, default=["L", "R"]
        Hemispheres to include. Forwarded to ``NiSpace`` initialization and, if
        ``x`` is a reference dataset string, to :func:`~nispace.datasets.fetch_reference`.
    y_covariates : array-like or None, default=None
        Optional covariates to regress from Y data. If given, :meth:`NiSpace.clean_y`
        is run with ``how="between"`` before colocalization.
    colocalization_method : str or list, default=None
        Method(s) to use for colocalization. When ``None``, defaults to
        ``"pearson"`` if ``binary_y=True``, otherwise ``"spearman"``. See
        :meth:`NiSpace.colocalize` for the full list of supported methods.
    mc_method : str or list, default="meff"
        Multiple-comparisons correction method(s), forwarded to :meth:`NiSpace.correct_p`.
        If a list, each method is applied and stored separately. An explicit
        ``"mc_method"`` key inside ``correct_p_kwargs`` overrides this entirely.
    normalize_colocalizations : bool, default=True
        Whether to call :meth:`NiSpace.normalize_colocalizations` after correction
        (z-scores observed colocalizations against their null distribution). Failures
        are caught and logged as a warning rather than raised.
    pooled_p : str or bool, default=False
        How to aggregate across Y maps before computing p-values. ``False`` (default)
        computes one p-value per Y×X pair. ``"mean"`` or ``"median"`` averages
        colocalizations across Y maps first and returns one p-value per X map.
        ``"auto"`` uses ``False`` for a single Y map and ``"mean"`` otherwise.
    p_from_average_y : str or bool, optional
        Deprecated. Use ``pooled_p`` instead.
    plot : bool, default=True
        Whether to generate visualization plots.
    combat : bool, default=False
        Whether to apply ComBat harmonization. Only relevant if ``y_covariates`` is given.
    binary_y : bool, default=False
        Set if Y is binary (e.g. binary lesion/cluster masks). Forces
        ``NiSpace(binary_y=True)``, which prevents Y from being z-scored regardless
        of ``standardize`` and drives the ``colocalization_method=None`` dynamic
        default toward ``"pearson"``. Not relevant to permutation here, since this
        function always permutes ``what="maps"`` (the ``binary_y`` restriction on
        ``permute(what="groups")`` only applies to :func:`group_colocalization`).
    n_perm : int, default=10000
        Number of permutations for null distribution.
    seed : int or None, default=None
        Random seed for reproducibility.
    n_proc : int, default=1
        Number of processes for parallel computation.
    verbose : bool, default=True
        Whether to print progress messages.
    nispace_object : NiSpace or None, default=None
        Optional pre-initialized NiSpace object to use.
    fetch_x_kwargs : dict, optional
        Additional arguments for fetching X data.
    init_kwargs : dict, optional
        Additional arguments for NiSpace initialization.
    fit_kwargs : dict, optional
        Additional arguments for ``NiSpace.fit()``.
    clean_y_kwargs : dict, optional
        Additional arguments for Y data cleaning.
    colocalize_kwargs : dict, optional
        Additional arguments for colocalization.
    permute_kwargs : dict, optional
        Additional arguments for permutation testing.
    correct_p_kwargs : dict, optional
        Additional arguments for p-value correction.
    plot_kwargs : dict, optional
        Additional arguments for plotting.
    return_nispace_only : bool, default=False
        If True, return only the NiSpace object. Use ``nsp.get_colocalizations()`` and
        ``nsp.get_p_values()`` to access results. Setting False is deprecated and will
        be removed in the first non-dev release.

    Returns
    -------
    nsp : NiSpace
        The NiSpace object containing all results (when ``return_nispace_only=True``).
    colocs, p_values, pc_values, nsp : tuple
        Deprecated. Returned when ``return_nispace_only=False`` (current default).
    """
    verbose = set_log(lgr, verbose)
    # TODO (first non-dev release): remove p_from_average_y parameter
    if p_from_average_y is not None:
        lgr.warning(_DEPR_POOLED_P)
        pooled_p = p_from_average_y
    # kwarg dicts
    fetch_x_kwargs = {} if fetch_x_kwargs is None else fetch_x_kwargs
    init_kwargs = {} if init_kwargs is None else dict(init_kwargs)
    if binary_y:
        init_kwargs.setdefault("binary_y", True)
    if colocalization_method is None:
        colocalization_method = "pearson" if binary_y else "spearman"
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs
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
        parcellation_hemi=parcellation_hemi,
        colocalization_method=colocalization_method,
        n_proc=n_proc,
        verbose=verbose,
        nispace_object=nispace_object,
        fetch_x_kwargs=fetch_x_kwargs,
        init_kwargs=init_kwargs,
        fit_kwargs=fit_kwargs
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
                pooled_p=pooled_p,
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

    ## ZSCORE
    if normalize_colocalizations:
        try:
            nsp.normalize_colocalizations()
        except Exception as e:
            lgr.warning(f"normalize_colocalizations() failed: {e}")

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
    pc_values = {
        mc_m: {method: nsp.get_p_values(method, permuted, mc_method=mc_m)
               for method in colocalization_method}
        for mc_m in mc_methods
    }
    if len(colocalization_method) == 1:
        k = colocalization_method[0]
        colocs, p_values = colocs[k], p_values[k]
        pc_values = {mc_m: pc_values[mc_m][k] for mc_m in mc_methods}
    if len(mc_methods) == 1:
        pc_values = pc_values[mc_methods[0]]

    # TODO (first non-dev release): remove return_nispace_only parameter; always return nsp only;
    #   remove colocs/p_values/pc_values construction block above and the if/else here
    if not return_nispace_only:
        lgr.warning(_DEPR_RETURN_TUPLE)
        return colocs, p_values, pc_values, nsp
    return nsp


def group_colocalization(y, design,
                         x="PET",
                         z=None,
                         x_collection=None,
                         standardize="xz",
                         space=_SPACE_DEFAULT_VOL,
                         data_space=None,
                         parcellation_space=None,
                         parcellation=_PARC_DEFAULT,
                         parcellation_labels=None,
                         parcellation_hemi=["L", "R"],
                         colocalization_method="spearman",
                         comparison_method=None,
                         mc_method="meff",
                         normalize_colocalizations=True,
                         pooled_p=False,
                         p_from_average_y=None,  # TODO (first non-dev release): remove
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
                         fit_kwargs=None,
                         clean_y_kwargs=None,
                         transform_y_kwargs=None,
                         colocalize_kwargs=None,
                         permute_kwargs=None,
                         correct_p_kwargs=None,
                         plot_kwargs=None,
                         return_nispace_only=False):
    """Group-comparison colocalization workflow.

    Compares Y maps between two groups of individual subjects/observations
    (e.g. patients vs. controls), reduces the comparison to a single effect-size
    map via :meth:`NiSpace.transform_y`, and colocalizes that map with X while
    testing significance via group-label permutation.

    Parameters
    ----------
    y : array-like or pandas DataFrame or list
        Input Y data: one map per individual subject/observation (not group-level
        summary maps). Passed straight through to ``NiSpace.fit``.
    design : list, array-like, or pandas DataFrame
        Group (and, if ``paired=True``, subject) labels, one row per row of ``y``
        (row count must match ``len(y)``, else raises ``ValueError``). Accepted forms:

        - 1-D list/array/Series: dummy-coded group labels. Raises ``ValueError`` if
          ``paired=True`` (a 1-D input cannot carry subject IDs).
        - 2-D ``ndarray``: columns ``["groups"(, "subjects"), V0, V1, ...]``
          (subjects column only if ``paired=True``).
        - ``DataFrame``: must have a ``"groups"`` column (and a ``"subjects"``
          column if ``paired=True``).

        Any columns beyond the mandatory group(+subject) column(s) are treated as
        Y covariates and automatically regressed out via :meth:`NiSpace.clean_y`
        (see ``clean_y_kwargs`` below).
    x : str or array-like, default="PET"
        Input X data. Can be a string indicating a reference dataset ("PET", "mRNA", ...),
        or input types as listed for y.
    z : array-like or None, default=None
        Optional confound data to regress out. Can be "gm", or input types as listed for y.
    x_collection : str or None, default=None
        If x is a string reference dataset, specifies which collection to use.
    standardize : str, default="xz"
        Which data to standardize. Can contain "x", "y", and/or "z".
    space : str, default=_SPACE_DEFAULT_VOL ("MNI152NLin6Asym")
        Default template space for both the data images and the parcellation. Used
        to resolve ``data_space``/``parcellation_space`` when those are not given.
    data_space : str or None, default=None
        Template space of the input data images. Falls back to ``space`` if falsy.
    parcellation_space : str or None, default=None
        Template space of the parcellation. Falls back to ``space`` if falsy.
    parcellation : str or int, default=_PARC_DEFAULT
        Brain parcellation to use. Can be a string name or integer ID.
    parcellation_labels : array-like or None, default=None
        Optional labels for the parcellation regions.
    parcellation_hemi : list of str, default=["L", "R"]
        Hemispheres to include. Forwarded to ``NiSpace`` initialization and, if
        ``x`` is a reference dataset string, to :func:`~nispace.datasets.fetch_reference`.
    colocalization_method : str or list, default="spearman"
        Method(s) to use for colocalization. Unlike :func:`colocalization`, this is a
        static default (no ``binary_y``-driven dynamic default) — this function does
        not support binary Y, since :meth:`NiSpace.transform_y` (always run here) and
        group-label permutation are both incompatible with ``binary_y=True``. See
        :meth:`NiSpace.colocalize` for the full list of supported methods.
    comparison_method : str or None, default=None
        Formula passed to :meth:`NiSpace.transform_y` to reduce Y to a single
        group-comparison effect-size map. When ``None``, defaults to ``"hedges(a,b)"``
        if ``paired=False``, otherwise ``"pairedcohen(a,b)"``. This transform is then
        reused as ``Y_transform`` for colocalization, permutation, plotting, and
        result retrieval.
    mc_method : str or list, default="meff"
        Multiple-comparisons correction method(s), forwarded to :meth:`NiSpace.correct_p`.
        If a list, each method is applied and stored separately. An explicit
        ``"mc_method"`` key inside ``correct_p_kwargs`` overrides this entirely.
    normalize_colocalizations : bool, default=True
        Whether to call :meth:`NiSpace.normalize_colocalizations` after correction.
        Failures are caught and logged as a warning rather than raised.
    pooled_p : str or bool, default=False
        Present for signature symmetry with :func:`colocalization`, but **not a free
        choice here**: for group-label permutation (``what="groups"``),
        :meth:`NiSpace.permute` always answers a group-level question and forces
        ``pooled_p="mean"`` regardless of what is passed (with a warning on conflict).
    p_from_average_y : str or bool, optional
        Deprecated. Use ``pooled_p`` instead.
    paired : bool, default=False
        Whether groups are paired/matched by subject (e.g. pre/post, or matched
        case-control pairs). Governs ``design`` parsing rules, the dynamic default of
        ``comparison_method``, and is passed to :meth:`NiSpace.permute` as
        ``groups_paired``.
    plot_design_between : bool, default=True
        Whether to plot the between-subject design matrix (diagnostic only). Only
        takes effect when ``design`` has covariate columns that trigger
        :meth:`NiSpace.clean_y`.
    combat : bool, default=False
        Whether to apply ComBat harmonization. Only relevant when ``design`` has
        covariate columns that trigger :meth:`NiSpace.clean_y`.
    plot : bool, default=True
        Whether to generate visualization plots.
    n_perm : int, default=10000
        Number of permutations for null distribution.
    seed : int or None, default=None
        Random seed for reproducibility.
    n_proc : int, default=1
        Number of processes for parallel computation.
    verbose : bool, default=True
        Whether to print progress messages.
    nispace_object : NiSpace or None, default=None
        Optional pre-initialized NiSpace object to use.
    fetch_x_kwargs : dict, optional
        Additional arguments for fetching X data.
    init_kwargs : dict, optional
        Additional arguments for NiSpace initialization.
    fit_kwargs : dict, optional
        Additional arguments for ``NiSpace.fit()``.
    clean_y_kwargs : dict, optional
        Additional arguments for Y data cleaning. Auto-triggered based on ``design``
        having covariate columns beyond ``"groups"``(+``"subjects"``) — unlike
        :func:`colocalization`, there is no separate ``y_covariates`` flag here.
    transform_y_kwargs : dict, optional
        Additional arguments for :meth:`NiSpace.transform_y`. Always runs (no
        conditional gate other than a pre-fitted ``nispace_object``).
    colocalize_kwargs : dict, optional
        Additional arguments for colocalization.
    permute_kwargs : dict, optional
        Additional arguments for permutation testing. Note ``what="groups"`` is
        forced last in the internal merge and cannot be overridden here.
    correct_p_kwargs : dict, optional
        Additional arguments for p-value correction.
    plot_kwargs : dict, optional
        Additional arguments for plotting.
    return_nispace_only : bool, default=False
        If True, return only the NiSpace object. Use ``nsp.get_colocalizations()`` and
        ``nsp.get_p_values()`` to access results. Setting False is deprecated and will
        be removed in the first non-dev release.

    Returns
    -------
    nsp : NiSpace
        The NiSpace object containing all results (when ``return_nispace_only=True``).
    colocs, p_values, pc_values, nsp : tuple
        Deprecated. Returned when ``return_nispace_only=False`` (current default).
    """
    verbose = set_log(lgr, verbose)
    # TODO (first non-dev release): remove p_from_average_y parameter
    if p_from_average_y is not None:
        lgr.warning(_DEPR_POOLED_P)
        pooled_p = p_from_average_y
    # kwarg dicts
    fetch_x_kwargs = {} if fetch_x_kwargs is None else fetch_x_kwargs
    init_kwargs = {} if init_kwargs is None else init_kwargs
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs
    clean_y_kwargs = {} if clean_y_kwargs is None else clean_y_kwargs
    transform_y_kwargs = {} if transform_y_kwargs is None else transform_y_kwargs
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
        parcellation_hemi=parcellation_hemi,
        colocalization_method=colocalization_method,
        n_proc=n_proc,
        verbose=verbose,
        nispace_object=nispace_object,
        fetch_x_kwargs=fetch_x_kwargs,
        init_kwargs=init_kwargs,
        fit_kwargs=fit_kwargs
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
            if "groups" not in design.columns or "subjects" not in design.columns:
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
            protect=combat_protect,
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
                method=method,
                Y_transform=comparison_method,
                groups_paired=paired,
                groups_strategy="shuffle",
                pooled_p=pooled_p,
                n_perm=n_perm,
                seed=seed,
                verbose=verbose,
            ) | permute_kwargs | {"what": "groups"}
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

    ## ZSCORE
    if normalize_colocalizations:
        try:
            nsp.normalize_colocalizations()
        except Exception as e:
            lgr.warning(f"normalize_colocalizations() failed: {e}")

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
    pc_values = {
        mc_m: {method: nsp.get_p_values(method, permute_what, Y_transform=comparison_method,
                                        mc_method=mc_m)
               for method in colocalization_method}
        for mc_m in mc_methods
    }
    if len(colocalization_method) == 1:
        k = colocalization_method[0]
        colocs, p_values = colocs[k], p_values[k]
        pc_values = {mc_m: pc_values[mc_m][k] for mc_m in mc_methods}
    if len(mc_methods) == 1:
        pc_values = pc_values[mc_methods[0]]

    # TODO (first non-dev release): remove return_nispace_only parameter; always return nsp only;
    #   remove colocs/p_values/pc_values construction block above and the if/else here
    if not return_nispace_only:
        lgr.warning(_DEPR_RETURN_TUPLE)
        return colocs, p_values, pc_values, nsp
    return nsp


def paired_colocalization(y,
                          x,
                          z=None,
                          standardize="xz",
                          space=_SPACE_DEFAULT_VOL,
                          data_space=None,
                          parcellation_space=None,
                          parcellation=_PARC_DEFAULT,
                          parcellation_labels=None,
                          parcellation_hemi=["L", "R"],
                          colocalization_method="spearman",
                          pooled_p="mean",
                          plot=True,
                          n_perm=10000,
                          seed=None,
                          n_proc=1,
                          verbose=True,
                          nispace_object=None,
                          init_kwargs=None,
                          fit_kwargs=None,
                          colocalize_kwargs=None,
                          permute_kwargs=None,
                          plot_kwargs=None):
    """Within-pair colocalization workflow (SPICE test).

    Tests whether within-pair correspondence between two brain map modalities
    (correlation across parcels per pair, averaged over pairs) is significantly
    greater than between-pair correspondence. The null distribution is built by
    permuting pair labels on the precomputed N×N colocalization matrix.

    "Pairs" can be subjects (structure vs. function per person), studies (one
    map per study in a meta-analytic context), neurochemical targets (tracers
    grouped by target), or any other unit for which matched maps exist in both
    modalities.

    Reference: :cite:`weinstein2021`.

    Parameters
    ----------
    y : array-like, DataFrame, or list
        Modality A — N brain maps, one per pair. Same ordering as ``x``
        is required; matching is done positionally.
    x : array-like, DataFrame, or list
        Modality B — N brain maps in the same pair order as ``y``.
        Unlike other workflow functions, ``x`` is expected to be individual-level
        data, not a reference dataset string — this is a usage contract (there is
        no ``x_collection`` parameter here and no dedicated runtime check forbidding
        a string), not an enforced validation.
    z : array-like, DataFrame, or list, optional
        Covariate maps to partial out. Required for partial correlation methods
        (``"partialspearman"``, ``"partialpearson"``); ignored otherwise.
    standardize : str, default="xz"
        Which data to z-standardize (parcels). Can contain "x", "y", "z".
    space : str
        Image space for parcellation and data loading.
    data_space : str or None
        Override for the data image space.
    parcellation_space : str or None
        Override for the parcellation space.
    parcellation : str or int, default=_PARC_DEFAULT
        Brain parcellation to use.
    parcellation_labels : array-like or None
        Optional subset of parcellation region labels.
    parcellation_hemi : list, default=["L", "R"]
        Hemispheres to include.
    colocalization_method : str, default="spearman"
        Colocalization method. Produces an N×N correlation matrix.
    pooled_p : str, default="mean"
        Aggregation used for the within-pair statistic: ``"mean"`` or
        ``"median"`` of the N diagonal entries. ``"auto"`` resolves to
        ``"mean"``. ``False`` is not supported and falls back to ``"mean"``.
        For ``colocalization_method`` in {"pearson", "spearman",
        "partialpearson", "partialspearman"}, the diagonal entries are always
        aggregated on the Fisher-z scale -- if ``colocalize_kwargs`` overrides
        ``r_to_z=False``, the values are transformed on the fly for this
        aggregation step only (the underlying stored ``"rho"`` stays raw);
        averaging raw correlation coefficients is never a valid choice here.
    plot : bool, default=True
        Whether to generate a plot after permutation, via ``nsp.plot(permute_what="pairs")``
        (relies on :meth:`NiSpace.plot`'s own ``kind="categorical"`` default — not
        passed explicitly here).
    n_perm : int, default=10000
        Number of subject-label permutations for the null distribution.
    seed : int or None
        Random seed for reproducibility.
    n_proc : int, default=1
        Parallel workers (passed to NiSpace init; not used by the fast
        vectorised pair permutation itself).
    verbose : bool, default=True
        Whether to print progress messages.
    nispace_object : NiSpace or None
        Pre-fitted NiSpace object to reuse; skips init/fit when provided.
    init_kwargs : dict, optional
        Extra keyword arguments for :class:`NiSpace` initialisation.
    fit_kwargs : dict, optional
        Extra keyword arguments for :meth:`NiSpace.fit`.
    colocalize_kwargs : dict, optional
        Extra keyword arguments for :meth:`NiSpace.colocalize`.
    permute_kwargs : dict, optional
        Extra keyword arguments for :meth:`NiSpace.permute`.
    plot_kwargs : dict, optional
        Extra keyword arguments for :meth:`NiSpace.plot`.

    Returns
    -------
    nsp : NiSpace
        Fitted NiSpace object. Use :meth:`~NiSpace.get_colocalizations` to
        retrieve the N×N pairwise matrix, :meth:`~NiSpace.get_p_values` for
        the within-pair vs. between-pair p-value (shape 1×1), and
        :meth:`~NiSpace.plot` to re-draw the visualization.
    """
    verbose = set_log(lgr, verbose)
    init_kwargs = {} if init_kwargs is None else dict(init_kwargs)
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs
    colocalize_kwargs = {} if colocalize_kwargs is None else colocalize_kwargs
    permute_kwargs = {} if permute_kwargs is None else permute_kwargs
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs

    if isinstance(colocalization_method, str):
        colocalization_method = [colocalization_method]

    status, nsp, _ = _workflow_base(
        x=x, y=y, z=z,
        x_collection=None,
        space=space,
        data_space=data_space,
        parcellation_space=parcellation_space,
        standardize=standardize,
        parcellation=parcellation,
        parcellation_labels=parcellation_labels,
        parcellation_hemi=parcellation_hemi,
        colocalization_method=colocalization_method,
        n_proc=n_proc,
        verbose=verbose,
        nispace_object=nispace_object,
        fetch_x_kwargs={},
        init_kwargs=init_kwargs,
        fit_kwargs=fit_kwargs,
    )
    status = status | {fun: False for fun in ["colocalize", "permute"]}

    ## COLOCALIZE
    if not status["colocalize"]:
        for method in colocalization_method:
            nsp.colocalize(**dict(method=method) | colocalize_kwargs)
        status["colocalize"] = True

    ## PERMUTE (SPICE — operates on the precomputed N×N matrix, no re-colocalization)
    if not status["permute"]:
        for method in colocalization_method:
            permute_kwargs_curr = dict(
                what="pairs",
                method=method,
                pooled_p=pooled_p,
                n_perm=n_perm,
                seed=seed,
            ) | permute_kwargs
            nsp.permute(**permute_kwargs_curr)
        status["permute"] = True

    ## VIZ
    if plot:
        for method in colocalization_method:
            plot_kwargs_curr = dict(
                method=method,
                permute_what="pairs",
            ) | plot_kwargs
            nsp.plot(**plot_kwargs_curr)

    return nsp


def correlate_within_region(x,
                            y,
                            standardize=False,
                            space=_SPACE_DEFAULT_VOL,
                            data_space=None,
                            parcellation_space=None,
                            parcellation=_PARC_DEFAULT,
                            parcellation_labels=None,
                            parcellation_hemi=["L", "R"],
                            method="pearson",
                            n_perm=1000,
                            seed=None,
                            n_proc=1,
                            verbose=True,
                            nispace_object=None,
                            init_kwargs=None,
                            fit_kwargs=None,
                            correlate_kwargs=None):
    """Per-parcel, across-subject correlation workflow.

    For each parcel independently, tests whether maps with a higher X value at
    that parcel also have a higher Y value at that parcel, across the set of
    X/Y maps (e.g. subjects) -- the transpose of :func:`colocalization` (which
    correlates across parcels, within a map).

    One of ``x``/``y`` may be a 1D, subject-length vector (or an equivalent
    single-column DataFrame/(n_subjects, 1) array) instead of full brain-map
    data (e.g. an external covariate like age) -- it is broadcast against
    every parcel of the other (2D) side, exactly like
    :meth:`NiSpace.correlate_within_region`'s own ``X=``/``Y=`` override
    contract. Internally, :meth:`NiSpace.fit` needs real 2D map data (it has
    no notion of "this is a subject-length covariate, not a map"; a bare 1D
    vector passed to ``fit()`` directly would be mis-parsed as a single
    already-parcellated map and fail on a parcel-count mismatch) -- so the 2D
    side is used to fit the object, and the 1D side is passed through as a
    direct override afterward. This is transparent to the caller; both
    ``x``/``y`` can simply be passed as given. At least one of ``x``/``y``
    must be 2D.

    The null distribution is built by permuting map identity (not a spatial/
    spin null); the same permutation is applied consistently across all
    parcels within one iteration.

    Parameters
    ----------
    x : array-like or DataFrame
        Modality A -- N brain maps (N x n_parcels), or a length-N 1D subject
        covariate if ``y`` is 2D (see above). Same ordering as ``y`` is
        required; matching is done positionally. Unlike other workflow
        functions, ``x`` is expected to be individual-level data here, not a
        reference dataset string.
    y : array-like, DataFrame, or list
        Modality B, in the same map order as ``x`` -- or a length-N 1D
        subject covariate if ``x`` is 2D (see above).
    standardize : str or bool, default=False
        Which data to z-standardize (parcels). Can contain "x", "y". Defaults to ``False``:
        this z-scores each *map* across its own parcels, which distorts the across-map axis
        this function actually correlates (see :meth:`NiSpace.correlate_within_region`). 
    space : str
        Image space for parcellation and data loading.
    data_space : str or None
        Override for the data image space.
    parcellation_space : str or None
        Override for the parcellation space.
    parcellation : str or int, default=_PARC_DEFAULT
        Brain parcellation to use.
    parcellation_labels : array-like or None
        Optional subset of parcellation region labels.
    parcellation_hemi : list, default=["L", "R"]
        Hemispheres to include.
    method : {"pearson", "spearman"}, default "pearson"
    n_perm : int, default 1000
        Number of map-identity permutations for the null distribution.
    seed : int or None
        Random seed for reproducibility.
    n_proc : int, default 1
        Parallel workers (passed to NiSpace init).
    verbose : bool, default True
        Whether to print progress messages.
    nispace_object : NiSpace or None
        Pre-fitted NiSpace object to reuse; skips init/fit when provided.
    init_kwargs : dict, optional
        Extra keyword arguments for :class:`NiSpace` initialisation.
    fit_kwargs : dict, optional
        Extra keyword arguments for :meth:`NiSpace.fit`.
    correlate_kwargs : dict, optional
        Extra keyword arguments for :meth:`NiSpace.correlate_within_region`,
        e.g. ``{"r_to_z": False}`` for raw (non-Fisher-z) correlations --
        Fisher-z is the default there, to align with the rest of the toolbox
        (:func:`colocalization`'s own default).

    Returns
    -------
    nsp : NiSpace
        Fitted NiSpace object. Use
        :meth:`~NiSpace.get_within_region_correlations` to retrieve the
        per-parcel correlation, p-values, and (optionally corrected) results,
        or :meth:`~NiSpace.get_within_region_correlations_omnibus` for a
        single global test across all parcels instead.
    """
    verbose = set_log(lgr, verbose)
    init_kwargs = {} if init_kwargs is None else dict(init_kwargs)
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs
    correlate_kwargs = {} if correlate_kwargs is None else correlate_kwargs

    # 1D-covariate detection: a Series/1D array/list, or a single-column DataFrame/
    # (n_subjects, 1) array, is a subject-length covariate, never a single brain map
    # (this function always needs an across-subject axis, which a lone map doesn't
    # have) -- so this is unambiguous, not a heuristic guess
    def _is_1d(v):
        if isinstance(v, pd.DataFrame):
            return v.shape[1] == 1
        arr = v.values if isinstance(v, pd.Series) else np.asarray(v)
        return arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1)

    def _squeeze_1d(v):
        """(n_subjects, 1) DataFrame/array -> true 1D, matching correlate_within_region_core."""
        if isinstance(v, pd.DataFrame):
            return v.iloc[:, 0]
        if isinstance(v, pd.Series):
            return v
        arr = np.asarray(v)
        return arr[:, 0] if arr.ndim == 2 else arr

    x_is_1d, y_is_1d = _is_1d(x), _is_1d(y)
    if x_is_1d and y_is_1d:
        lgr.critical_raise("At least one of 'x'/'y' must be full brain-map data "
                           "(N maps x n_parcels) -- both were given as 1D vectors.",
                           ValueError)

    # fit() needs real 2D map data to build parcellation metadata from; when one side
    # is a 1D covariate, fit on the 2D side for both x and y, then pass the covariate
    # through as a direct correlate_within_region(X=/Y=) override below (bypassing
    # fit()/get_x()/get_y() entirely for that side, same as the object-level method)
    fit_x, fit_y = x, y
    if y_is_1d:
        fit_y = x
    elif x_is_1d:
        fit_x = y

    status, nsp, _ = _workflow_base(
        x=fit_x, y=fit_y, z=None,
        x_collection=None,
        space=space,
        data_space=data_space,
        parcellation_space=parcellation_space,
        standardize=standardize,
        parcellation=parcellation,
        parcellation_labels=parcellation_labels,
        parcellation_hemi=parcellation_hemi,
        colocalization_method=[method],
        n_proc=n_proc,
        verbose=verbose,
        nispace_object=nispace_object,
        fetch_x_kwargs={},
        init_kwargs=init_kwargs,
        fit_kwargs=fit_kwargs,
    )
    status = status | {"correlate_within_region": False}

    ## CORRELATE WITHIN REGION
    if not status["correlate_within_region"]:
        overrides = {}
        if x_is_1d:
            overrides["X"] = _squeeze_1d(x)
        if y_is_1d:
            overrides["Y"] = _squeeze_1d(y)
        nsp.correlate_within_region(
            **dict(method=method, n_perm=n_perm, seed=seed) | overrides | correlate_kwargs
        )
        status["correlate_within_region"] = True

    return nsp


def xsea(y,
         x="mRNA",
         z=None,
         x_collection=None,
         x_background=None,
         standardize="xz",
         space=_SPACE_DEFAULT_VOL,
         data_space=None,
         parcellation_space=None,
         parcellation=_PARC_DEFAULT,
         parcellation_labels=None,
         parcellation_hemi=["L", "R"],
         y_covariates=None,
         colocalization_method=None,
         mc_method="meff",
         normalize_colocalizations=True,
         xsea_aggregation_method="mean",
         permute_sets=False,
         pooled_p=False,
         p_from_average_y=None,  # TODO (first non-dev release): remove
         plot=True,
         combat=False,
         binary_y=False,
         n_perm=10000,
         seed=None,
         n_proc=1,
         verbose=True,
         nispace_object=None,
         fetch_x_kwargs=None,
         init_kwargs=None,
         fit_kwargs=None,
         clean_y_kwargs=None,
         colocalize_kwargs=None,
         permute_kwargs=None,
         correct_p_kwargs=None,
         plot_kwargs=None,
         return_nispace_only=False):
    """Set enrichment analysis (XSEA) workflow for group-level map(s).

    Equivalent to :func:`colocalization` with X treated as sets (e.g. gene sets)
    rather than individual maps: colocalizations are computed per set (aggregated
    via ``xsea_aggregation_method``) and Y is permuted to build the null
    distribution, unless ``permute_sets=True`` (see below). All parameters not
    listed below behave exactly as in :func:`colocalization`.

    Parameters
    ----------
    y : array-like or pandas DataFrame or list
        Input Y data to test for set enrichment. Can be a numpy array, pandas
        DataFrame, (list of) path(s) to a file(s) or list of image objects.
    x : str or array-like, default="mRNA"
        Input X data, expected to carry set membership (e.g. a "set"/gene-set
        MultiIndex level). Can be a string reference dataset name or input types
        as listed for y.
    z : array-like or None, default=None
        Optional confound data to regress out. Can be "gm", or input types as listed for y.
    x_collection : str or None, default=None
        If x is a string reference dataset, specifies which collection to use.
    x_background : array-like or None, default=None
        Pool of "background" X maps used as the null when ``permute_sets=True``
        (X-set-membership permutation). If ``None`` and ``x`` is a reference dataset
        string, an attempt is made to auto-fetch the full reference collection as
        the background (silently falls back to ``None`` with a warning on failure).
        Ignored if ``permute_sets=False``.
    standardize : str, default="xz"
        Which data to standardize. Can contain "x", "y", and/or "z".
    space : str, default=_SPACE_DEFAULT_VOL ("MNI152NLin6Asym")
        Default template space for both the data images and the parcellation. Used
        to resolve ``data_space``/``parcellation_space`` when those are not given.
    data_space : str or None, default=None
        Template space of the input data images. Falls back to ``space`` if falsy.
    parcellation_space : str or None, default=None
        Template space of the parcellation. Falls back to ``space`` if falsy.
    parcellation : str or int, default=_PARC_DEFAULT
        Brain parcellation to use. Can be a string name or integer ID.
    parcellation_labels : array-like or None, default=None
        Optional labels for the parcellation regions.
    parcellation_hemi : list of str, default=["L", "R"]
        Hemispheres to include. Forwarded to ``NiSpace`` initialization and, if
        ``x`` is a reference dataset string, to :func:`~nispace.datasets.fetch_reference`.
    y_covariates : array-like or None, default=None
        Optional covariates to regress from Y data. If given, :meth:`NiSpace.clean_y`
        is run with ``how="between"`` before colocalization.
    colocalization_method : str or list, default=None
        Method(s) to use for colocalization. When ``None``, defaults to
        ``"pearson"`` if ``binary_y=True``, otherwise ``"spearman"``.
    mc_method : str or list, default="meff"
        Multiple-comparisons correction method(s), forwarded to :meth:`NiSpace.correct_p`.
        An explicit ``"mc_method"`` key inside ``correct_p_kwargs`` overrides this entirely.
    normalize_colocalizations : bool, default=True
        Whether to call :meth:`NiSpace.normalize_colocalizations` after correction.
        Failures are caught and logged as a warning rather than raised.
    xsea_aggregation_method : str, default="mean"
        How to aggregate individual X maps within each set before colocalization.
        One of ``"mean"``, ``"median"``, ``"absmean"``, ``"absmedian"``,
        ``"weightedmean"``, ``"weightedabsmean"`` (weighted variants require a
        ``"weight"`` MultiIndex level on X). See :meth:`NiSpace.colocalize`.
    permute_sets : bool, default=False
        If ``True``, switches the permutation null from ``what="maps"`` (permuting
        Y, the default) to ``what="sets"`` (X-set-membership permutation using
        ``x_background`` as the pool of maps to reassign to sets).
    pooled_p : str or bool, default=False
        How to aggregate across Y maps before computing p-values. Same semantics
        as in :func:`colocalization` (fully relevant here).
    p_from_average_y : str or bool, optional
        Deprecated. Use ``pooled_p`` instead.
    plot : bool, default=True
        Whether to generate visualization plots.
    combat : bool, default=False
        Whether to apply ComBat harmonization. Only relevant if ``y_covariates`` is given.
    binary_y : bool, default=False
        Set if Y is binary. Forces ``NiSpace(binary_y=True)`` and drives the
        ``colocalization_method=None`` dynamic default toward ``"pearson"``. See
        :func:`colocalization` for the full behavior.
    n_perm : int, default=10000
        Number of permutations for null distribution.
    seed : int or None, default=None
        Random seed for reproducibility.
    n_proc : int, default=1
        Number of processes for parallel computation.
    verbose : bool, default=True
        Whether to print progress messages.
    nispace_object : NiSpace or None, default=None
        Optional pre-initialized NiSpace object to use.
    fetch_x_kwargs : dict, optional
        Additional arguments for fetching X data.
    init_kwargs : dict, optional
        Additional arguments for NiSpace initialization.
    fit_kwargs : dict, optional
        Additional arguments for ``NiSpace.fit()``.
    clean_y_kwargs : dict, optional
        Additional arguments for Y data cleaning.
    colocalize_kwargs : dict, optional
        Additional arguments for colocalization. Pre-seeded with
        ``xsea=True``/``xsea_aggregation_method``; explicit caller keys override.
    permute_kwargs : dict, optional
        Additional arguments for permutation testing. Pre-seeded with ``what``,
        ``maps_which="Y"``, ``sets_X_background``; explicit caller keys override.
    correct_p_kwargs : dict, optional
        Additional arguments for p-value correction.
    plot_kwargs : dict, optional
        Additional arguments for plotting.
    return_nispace_only : bool, default=False
        If True, return only the NiSpace object. Use ``nsp.get_colocalizations()`` and
        ``nsp.get_p_values()`` to access results. Setting False is deprecated and will
        be removed in the first non-dev release.

    Returns
    -------
    nsp : NiSpace
        The NiSpace object containing all results (when ``return_nispace_only=True``).
    colocs, p_values, pc_values, nsp : tuple
        Deprecated. Returned when ``return_nispace_only=False`` (current default).
    """
    verbose = set_log(lgr, verbose)
    # TODO (first non-dev release): remove p_from_average_y parameter
    if p_from_average_y is not None:
        lgr.warning(_DEPR_POOLED_P)
        pooled_p = p_from_average_y
    # kwarg dicts
    fetch_x_kwargs = {} if fetch_x_kwargs is None else fetch_x_kwargs
    init_kwargs = {} if init_kwargs is None else dict(init_kwargs)
    if binary_y:
        init_kwargs.setdefault("binary_y", True)
    if colocalization_method is None:
        colocalization_method = "pearson" if binary_y else "spearman"
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs
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
                    x_background = fetch_reference(x.lower(), parcellation=parcellation,
                                                   hemi=parcellation_hemi, print_references=False)
                except:
                    x_background = None
        if x_background is None:
            lgr.warning(f"Could not fetch background dataset for input x!")

    return colocalization(
        y=y,
        x=x, z=z,
        x_collection=x_collection,
        standardize=standardize,
        space=space,
        data_space=data_space,
        parcellation_space=parcellation_space,
        parcellation=parcellation,
        parcellation_labels=parcellation_labels,
        parcellation_hemi=parcellation_hemi,
        y_covariates=y_covariates,
        colocalization_method=colocalization_method,
        mc_method=mc_method,
        normalize_colocalizations=normalize_colocalizations,
        pooled_p=pooled_p,
        plot=plot,
        combat=combat,
        n_perm=n_perm,
        seed=seed,
        n_proc=n_proc,
        verbose=verbose,
        nispace_object=nispace_object,
        fetch_x_kwargs=fetch_x_kwargs,
        init_kwargs=init_kwargs,
        fit_kwargs=fit_kwargs,
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
        plot_kwargs=plot_kwargs,
        return_nispace_only=return_nispace_only,
    )


def group_xsea(y, design,
               x="mRNA",
               z=None,
               x_collection=None,
               standardize="xz",
               space=_SPACE_DEFAULT_VOL,
               data_space=None,
               parcellation_space=None,
               parcellation=_PARC_DEFAULT,
               parcellation_labels=None,
               parcellation_hemi=["L", "R"],
               colocalization_method="spearman",
               comparison_method=None,
               mc_method="meff",
               normalize_colocalizations=True,
               xsea_aggregation_method="mean",
               pooled_p=False,
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
               fit_kwargs=None,
               clean_y_kwargs=None,
               transform_y_kwargs=None,
               colocalize_kwargs=None,
               permute_kwargs=None,
               correct_p_kwargs=None,
               plot_kwargs=None,
               return_nispace_only=False):
    """Group-comparison set enrichment analysis (XSEA) workflow.

    Equivalent to :func:`group_colocalization` with X treated as sets (e.g. gene
    sets): Y is reduced to a single group-comparison effect-size map via
    :meth:`NiSpace.transform_y`, then colocalized with X sets (aggregated via
    ``xsea_aggregation_method``), with group-label permutation for p-values.
    All parameters not listed below behave exactly as in :func:`group_colocalization`
    (this function is a thin wrapper that injects ``xsea=True`` into
    ``colocalize_kwargs`` and forwards everything else unchanged).

    Parameters
    ----------
    y : array-like or pandas DataFrame or list
        Input Y data: one map per individual subject/observation.
    design : list, array-like, or pandas DataFrame
        Group (and, if ``paired=True``, subject) labels. See :func:`group_colocalization`
        for the full accepted forms and validation rules.
    x : str or array-like, default="mRNA"
        Input X data, expected to carry set membership (e.g. a "set"/gene-set
        MultiIndex level). Can be a string reference dataset name or input types
        as listed for y.
    z : array-like or None, default=None
        Optional confound data to regress out. Can be "gm", or input types as listed for y.
    x_collection : str or None, default=None
        If x is a string reference dataset, specifies which collection to use.
    standardize : str, default="xz"
        Which data to standardize. Can contain "x", "y", and/or "z".
    space : str, default=_SPACE_DEFAULT_VOL ("MNI152NLin6Asym")
        Default template space for both the data images and the parcellation. Used
        to resolve ``data_space``/``parcellation_space`` when those are not given.
    data_space : str or None, default=None
        Template space of the input data images. Falls back to ``space`` if falsy.
    parcellation_space : str or None, default=None
        Template space of the parcellation. Falls back to ``space`` if falsy.
    parcellation : str or int, default=_PARC_DEFAULT
        Brain parcellation to use. Can be a string name or integer ID.
    parcellation_labels : array-like or None, default=None
        Optional labels for the parcellation regions.
    parcellation_hemi : list of str, default=["L", "R"]
        Hemispheres to include.
    colocalization_method : str or list, default="spearman"
        Method(s) to use for colocalization. Static default — no ``binary_y`` support
        here, same reasoning as :func:`group_colocalization`.
    comparison_method : str or None, default=None
        Formula passed to :meth:`NiSpace.transform_y` to reduce Y to a single
        group-comparison effect-size map. Defaults to ``"hedges(a,b)"`` (unpaired) or
        ``"pairedcohen(a,b)"`` (paired).
    mc_method : str or list, default="meff"
        Multiple-comparisons correction method(s), forwarded to :meth:`NiSpace.correct_p`.
    normalize_colocalizations : bool, default=True
        Whether to call :meth:`NiSpace.normalize_colocalizations` after correction.
    xsea_aggregation_method : str, default="mean"
        How to aggregate individual X maps within each set before colocalization.
        One of ``"mean"``, ``"median"``, ``"absmean"``, ``"absmedian"``,
        ``"weightedmean"``, ``"weightedabsmean"``. This is the only parameter this
        function adds beyond :func:`group_colocalization`'s own signature, injected
        into ``colocalize_kwargs`` together with ``xsea=True``.
    pooled_p : str or bool, default=False
        Present for signature symmetry, but not a free choice: group-label
        permutation always forces ``pooled_p="mean"``, same as
        :func:`group_colocalization`. Unlike :func:`xsea`, this function does not
        inject anything into ``permute_kwargs`` — XSEA-ness is inherited
        automatically inside :meth:`NiSpace.permute` from ``colocalize_kwargs["xsea"]``,
        since there is no maps/sets axis to choose for group-label permutation.
    paired : bool, default=False
        Whether groups are paired/matched by subject.
    plot_design_between : bool, default=True
        Whether to plot the between-subject design matrix (diagnostic only).
    combat : bool, default=False
        Whether to apply ComBat harmonization.
    plot : bool, default=True
        Whether to generate visualization plots.
    n_perm : int, default=10000
        Number of permutations for null distribution.
    seed : int or None, default=None
        Random seed for reproducibility.
    n_proc : int, default=1
        Number of processes for parallel computation.
    verbose : bool, default=True
        Whether to print progress messages.
    nispace_object : NiSpace or None, default=None
        Optional pre-initialized NiSpace object to use.
    fetch_x_kwargs : dict, optional
        Additional arguments for fetching X data.
    init_kwargs : dict, optional
        Additional arguments for NiSpace initialization.
    fit_kwargs : dict, optional
        Additional arguments for ``NiSpace.fit()``.
    clean_y_kwargs : dict, optional
        Additional arguments for Y data cleaning. Auto-triggered based on ``design``
        having covariate columns.
    transform_y_kwargs : dict, optional
        Additional arguments for :meth:`NiSpace.transform_y`.
    colocalize_kwargs : dict, optional
        Additional arguments for colocalization. Pre-seeded with
        ``xsea=True``/``xsea_aggregation_method``; explicit caller keys override.
    permute_kwargs : dict, optional
        Additional arguments for permutation testing (unmodified by this function;
        ``what="groups"`` is forced by :func:`group_colocalization`).
    correct_p_kwargs : dict, optional
        Additional arguments for p-value correction.
    plot_kwargs : dict, optional
        Additional arguments for plotting.
    return_nispace_only : bool, default=False
        If True, return only the NiSpace object. Setting False is deprecated and
        will be removed in the first non-dev release.

    Returns
    -------
    nsp : NiSpace
        The NiSpace object containing all results (when ``return_nispace_only=True``).
    colocs, p_values, pc_values, nsp : tuple
        Deprecated. Returned when ``return_nispace_only=False`` (current default).

    Notes
    -----
    There is no ``binary_y`` parameter here, consistent with :func:`group_colocalization`:
    :meth:`NiSpace.transform_y` (always run internally) is incompatible with binary Y,
    and group-label permutation raises for ``binary_y=True``.
    """
    return group_colocalization(
        y=y, design=design,
        x=x, z=z,
        x_collection=x_collection,
        standardize=standardize,
        space=space,
        data_space=data_space,
        parcellation_space=parcellation_space,
        parcellation=parcellation,
        parcellation_labels=parcellation_labels,
        parcellation_hemi=parcellation_hemi,
        colocalization_method=colocalization_method,
        comparison_method=comparison_method,
        mc_method=mc_method,
        normalize_colocalizations=normalize_colocalizations,
        pooled_p=pooled_p,
        paired=paired,
        plot_design_between=plot_design_between,
        combat=combat,
        plot=plot,
        n_perm=n_perm,
        seed=seed,
        n_proc=n_proc,
        verbose=verbose,
        nispace_object=nispace_object,
        fetch_x_kwargs=fetch_x_kwargs,
        init_kwargs=init_kwargs,
        fit_kwargs=fit_kwargs,
        clean_y_kwargs=clean_y_kwargs,
        transform_y_kwargs=transform_y_kwargs,
        colocalize_kwargs={"xsea": True, "xsea_aggregation_method": xsea_aggregation_method}
                          | (colocalize_kwargs or {}),
        permute_kwargs=permute_kwargs,
        correct_p_kwargs=correct_p_kwargs,
        plot_kwargs=plot_kwargs,
        return_nispace_only=return_nispace_only,
    )


def nimare_colocalization(y,
                          x="PET",
                          z=None,
                          x_collection=None,
                          standardize="xz",
                          space=_SPACE_DEFAULT_VOL,
                          data_space=None,
                          parcellation_space=None,
                          parcellation=_PARC_DEFAULT,
                          parcellation_labels=None,
                          parcellation_hemi=["L", "R"],
                          y_covariates=None,
                          colocalization_method=None,
                          mc_method="meff",
                          normalize_colocalizations=True,
                          pooled_p=False,
                          plot=True,
                          binary_y=False,
                          nimare_nulls=None,
                          n_perm=10000,
                          seed=None,
                          n_proc=1,
                          verbose=True,
                          nispace_object=None,
                          fetch_x_kwargs=None,
                          init_kwargs=None,
                          fit_kwargs=None,
                          clean_y_kwargs=None,
                          colocalize_kwargs=None,
                          permute_kwargs=None,
                          correct_p_kwargs=None,
                          plot_kwargs=None,
                          return_nispace_only=False):
    """NiMARE colocalization workflow.

    Convenience wrapper around :func:`colocalization` for Y maps derived from
    NiMARE meta-analyses (continuous ALE stat maps or binary cluster-coverage
    maps from :func:`~nispace.helpers.get_binary_cluster_map`).

    Sets the following defaults relative to :func:`colocalization`:

    * ``background_value={"y": False}`` in ``fit_kwargs`` — zero-valued voxels
      in ALE maps are meaningful (no activation there), not missing data. Scoped
      to Y only: X (and Z, if given) keep the normal ``'auto'`` background
      handling, since only Y is guaranteed to be an ALE/cluster-coverage map here.
    * ``maps_which="Y"`` in ``permute_kwargs`` when ``nimare_nulls`` is provided,
      so Y is permuted with coordinate-sampling null maps; otherwise ``"X"``

    For binary cluster maps also pass ``binary_y=True`` and
    ``colocalization_method="pearson"``.

    Parameters
    ----------
    y : NIfTI image, array-like, or list
        ALE stat map (``result.get_map("stat")``) or binary cluster-coverage map
        (from :func:`~nispace.helpers.get_binary_cluster_map`).
    x : str or array-like, default="PET"
        Reference X maps. Same as :func:`colocalization`.
    colocalization_method : str or list, default=None
        Colocalization method. When ``None``, defaults to ``"pearson"`` if
        ``binary_y=True`` (approximate point-biserial), otherwise ``"spearman"``.
    binary_y : bool, default=False
        Set ``True`` when Y is a binary or fractional cluster-coverage map.
        Prevents z-scoring of Y and raises warnings for ranked methods or group
        permutation. See :class:`~nispace.NiSpace` for details.
    nimare_nulls : dict or None, default=None
        Coordinate-sampling null maps from
        :func:`~nispace.helpers.null_maps_from_nimare`. When provided, sets
        ``maps_nulls=nimare_nulls`` and ``maps_which="Y"`` in ``permute_kwargs``
        so Y is permuted with the NiMARE null distribution. Explicit entries in
        ``permute_kwargs`` take precedence.

    Returns
    -------
    nsp : NiSpace
        (when ``return_nispace_only=True``)
    colocs, p_values, pc_values, nsp : tuple
        Deprecated. Returned when ``return_nispace_only=False``.

    Notes
    -----
    All other parameters are identical to :func:`colocalization`.
    """
    init_kwargs = {} if init_kwargs is None else dict(init_kwargs)
    if binary_y:
        init_kwargs.setdefault("binary_y", True)
    if colocalization_method is None:
        colocalization_method = "pearson" if binary_y else "spearman"

    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
    fit_kwargs.setdefault("background_value", {"y": False})

    permute_kwargs = {} if permute_kwargs is None else dict(permute_kwargs)
    if nimare_nulls is not None:
        permute_kwargs.setdefault("maps_nulls", nimare_nulls)
        permute_kwargs.setdefault("maps_which", "Y")
    else:
        permute_kwargs.setdefault("maps_which", "X")

    return colocalization(
        y=y, x=x, z=z, x_collection=x_collection,
        standardize=standardize,
        space=space,
        data_space=data_space,
        parcellation_space=parcellation_space,
        parcellation=parcellation,
        parcellation_labels=parcellation_labels,
        parcellation_hemi=parcellation_hemi,
        y_covariates=y_covariates,
        colocalization_method=colocalization_method,
        mc_method=mc_method,
        normalize_colocalizations=normalize_colocalizations,
        pooled_p=pooled_p,
        plot=plot,
        n_perm=n_perm,
        seed=seed,
        n_proc=n_proc,
        verbose=verbose,
        nispace_object=nispace_object,
        fetch_x_kwargs=fetch_x_kwargs,
        init_kwargs=init_kwargs,
        fit_kwargs=fit_kwargs,
        clean_y_kwargs=clean_y_kwargs,
        colocalize_kwargs=colocalize_kwargs,
        permute_kwargs=permute_kwargs,
        correct_p_kwargs=correct_p_kwargs,
        plot_kwargs=plot_kwargs,
        return_nispace_only=return_nispace_only,
    )


def nimare_xsea(y,
                x="mRNA",
                z=None,
                x_collection=None,
                x_background=None,
                standardize="xz",
                space=_SPACE_DEFAULT_VOL,
                data_space=None,
                parcellation_space=None,
                parcellation=_PARC_DEFAULT,
                parcellation_labels=None,
                parcellation_hemi=["L", "R"],
                y_covariates=None,
                colocalization_method=None,
                mc_method="meff",
                normalize_colocalizations=True,
                xsea_aggregation_method="mean",
                permute_sets=False,
                pooled_p=False,
                plot=True,
                binary_y=False,
                nimare_nulls=None,
                n_perm=10000,
                seed=None,
                n_proc=1,
                verbose=True,
                nispace_object=None,
                fetch_x_kwargs=None,
                init_kwargs=None,
                fit_kwargs=None,
                clean_y_kwargs=None,
                colocalize_kwargs=None,
                permute_kwargs=None,
                correct_p_kwargs=None,
                plot_kwargs=None,
                return_nispace_only=False):
    """NiMARE X-set enrichment analysis (XSEA) workflow.

    Convenience wrapper around :func:`xsea` for Y maps derived from NiMARE
    meta-analyses. Equivalent to :func:`nimare_colocalization` with XSEA enabled.

    Sets ``background_value={"y": False}`` in ``fit_kwargs`` (scoped to Y only --
    X/Z keep normal ``'auto'`` background handling) and routes ``nimare_nulls``
    into ``permute_kwargs``. See :func:`nimare_colocalization` for full details
    on NiMARE-specific parameters.

    Parameters
    ----------
    y : NIfTI image, array-like, or list
        ALE stat map or binary cluster-coverage map.
    x : str or array-like, default="mRNA"
        Reference X maps (gene-set collections for XSEA).
    binary_y : bool, default=False
        Set ``True`` for binary or fractional cluster-coverage Y maps.
    nimare_nulls : dict or None, default=None
        Coordinate-sampling null maps from
        :func:`~nispace.helpers.null_maps_from_nimare`. When provided, sets
        ``maps_nulls=nimare_nulls`` in ``permute_kwargs``.

    Returns
    -------
    nsp : NiSpace
        (when ``return_nispace_only=True``)
    colocs, p_values, pc_values, nsp : tuple
        Deprecated. Returned when ``return_nispace_only=False``.

    Notes
    -----
    All other parameters are identical to :func:`xsea`.

    Unlike :func:`nimare_colocalization`, this function never touches
    ``maps_which`` — :func:`xsea` always permutes ``maps_which="Y"`` regardless of
    ``nimare_nulls``, since X is a fixed reference gene-set collection in XSEA, not
    something meaningfully permuted map-by-map. So ``nimare_nulls=None`` here does
    **not** fall back to X-permutation the way :func:`nimare_colocalization` does —
    Y is still permuted, just with freshly generated standard spatial nulls
    (typically Moran) instead of NiMARE coordinate-sampling nulls, which is
    functionally equivalent to a plain :func:`xsea` call.
    """
    init_kwargs = {} if init_kwargs is None else dict(init_kwargs)
    if binary_y:
        init_kwargs.setdefault("binary_y", True)
    if colocalization_method is None:
        colocalization_method = "pearson" if binary_y else "spearman"

    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
    fit_kwargs.setdefault("background_value", {"y": False})

    permute_kwargs = {} if permute_kwargs is None else dict(permute_kwargs)
    if nimare_nulls is not None:
        permute_kwargs.setdefault("maps_nulls", nimare_nulls)

    return xsea(
        y=y, x=x, z=z, x_collection=x_collection,
        x_background=x_background,
        standardize=standardize,
        space=space,
        data_space=data_space,
        parcellation_space=parcellation_space,
        parcellation=parcellation,
        parcellation_labels=parcellation_labels,
        parcellation_hemi=parcellation_hemi,
        y_covariates=y_covariates,
        colocalization_method=colocalization_method,
        mc_method=mc_method,
        normalize_colocalizations=normalize_colocalizations,
        xsea_aggregation_method=xsea_aggregation_method,
        permute_sets=permute_sets,
        pooled_p=pooled_p,
        plot=plot,
        n_perm=n_perm,
        seed=seed,
        n_proc=n_proc,
        verbose=verbose,
        nispace_object=nispace_object,
        fetch_x_kwargs=fetch_x_kwargs,
        init_kwargs=init_kwargs,
        fit_kwargs=fit_kwargs,
        clean_y_kwargs=clean_y_kwargs,
        colocalize_kwargs=colocalize_kwargs,
        permute_kwargs=permute_kwargs,
        correct_p_kwargs=correct_p_kwargs,
        plot_kwargs=plot_kwargs,
        return_nispace_only=return_nispace_only,
    )


# ==============================================================================
# DEPRECATION WRAPPERS — old function names kept for backward compatibility
# TODO (first non-dev release): remove these wrappers
# ==============================================================================

def simple_colocalization(y, x="PET", z=None, x_collection=None, standardize="xz",
                          space="MNI152NLin2009cAsym", data_space=None,
                          parcellation_space=None, parcellation=_PARC_DEFAULT,
                          parcellation_labels=None, parcellation_hemi=["L", "R"],
                          y_covariates=None, colocalization_method="spearman",
                          mc_method="meff", normalize_colocalizations=True,
                          p_from_average_y=False, plot=True, combat=False,
                          n_perm=10000, seed=None, n_proc=1, verbose=True,
                          nispace_object=None, fetch_x_kwargs=None, init_kwargs=None,
                          fit_kwargs=None, clean_y_kwargs=None, colocalize_kwargs=None,
                          permute_kwargs=None, correct_p_kwargs=None, plot_kwargs=None,
                          return_nispace_only=False):
    """Deprecated wrapper for :func:`colocalization`.

    .. deprecated:: dev
       Use :func:`colocalization` instead. Will be removed in the first non-dev release.
    """
    lgr.warning(_DEPR_FUNC_NAME.format(old="simple_colocalization", new="colocalization"))
    return colocalization(
        y=y, x=x, z=z, x_collection=x_collection, standardize=standardize,
        space=space, data_space=data_space, parcellation_space=parcellation_space,
        parcellation=parcellation, parcellation_labels=parcellation_labels,
        parcellation_hemi=parcellation_hemi, y_covariates=y_covariates,
        colocalization_method=colocalization_method, mc_method=mc_method,
        normalize_colocalizations=normalize_colocalizations,
        pooled_p=p_from_average_y,
        plot=plot, combat=combat, n_perm=n_perm, seed=seed, n_proc=n_proc,
        verbose=verbose, nispace_object=nispace_object, fetch_x_kwargs=fetch_x_kwargs,
        init_kwargs=init_kwargs, fit_kwargs=fit_kwargs, clean_y_kwargs=clean_y_kwargs,
        colocalize_kwargs=colocalize_kwargs, permute_kwargs=permute_kwargs,
        correct_p_kwargs=correct_p_kwargs, plot_kwargs=plot_kwargs,
        return_nispace_only=return_nispace_only,
    )


def group_comparison(y, design, x="PET", z=None, x_collection=None, standardize="xz",
                     space="MNI152NLin2009cAsym", data_space=None,
                     parcellation_space=None, parcellation=_PARC_DEFAULT,
                     parcellation_labels=None, parcellation_hemi=["L", "R"],
                     colocalization_method="spearman", comparison_method=None,
                     mc_method="meff", normalize_colocalizations=True, p_from_average_y=True,
                     paired=False, plot_design_between=True, combat=False,
                     plot=True, n_perm=10000, seed=None, n_proc=1, verbose=True,
                     nispace_object=None, fetch_x_kwargs=None, init_kwargs=None,
                     fit_kwargs=None, clean_y_kwargs=None, transform_y_kwargs=None,
                     colocalize_kwargs=None, permute_kwargs=None,
                     correct_p_kwargs=None, plot_kwargs=None,
                     return_nispace_only=False):
    """Deprecated wrapper for :func:`group_colocalization`.

    .. deprecated:: dev
       Use :func:`group_colocalization` instead. Will be removed in the first non-dev release.
    """
    lgr.warning(_DEPR_FUNC_NAME.format(old="group_comparison", new="group_colocalization"))
    return group_colocalization(
        y=y, design=design, x=x, z=z, x_collection=x_collection, standardize=standardize,
        space=space, data_space=data_space, parcellation_space=parcellation_space,
        parcellation=parcellation, parcellation_labels=parcellation_labels,
        parcellation_hemi=parcellation_hemi, colocalization_method=colocalization_method,
        comparison_method=comparison_method, mc_method=mc_method,
        normalize_colocalizations=normalize_colocalizations, pooled_p=p_from_average_y,
        paired=paired, plot_design_between=plot_design_between, combat=combat,
        plot=plot, n_perm=n_perm, seed=seed, n_proc=n_proc, verbose=verbose,
        nispace_object=nispace_object, fetch_x_kwargs=fetch_x_kwargs,
        init_kwargs=init_kwargs, fit_kwargs=fit_kwargs, clean_y_kwargs=clean_y_kwargs,
        transform_y_kwargs=transform_y_kwargs, colocalize_kwargs=colocalize_kwargs,
        permute_kwargs=permute_kwargs, correct_p_kwargs=correct_p_kwargs,
        plot_kwargs=plot_kwargs, return_nispace_only=return_nispace_only,
    )


def simple_xsea(y, x="mRNA", z=None, x_collection=None, x_background=None,
                standardize="xz", space="MNI152NLin2009cAsym", data_space=None,
                parcellation_space=None, parcellation=_PARC_DEFAULT,
                parcellation_labels=None, parcellation_hemi=["L", "R"],
                y_covariates=None, colocalization_method="spearman",
                mc_method="meff", normalize_colocalizations=True,
                xsea_aggregation_method="mean", permute_sets=False,
                p_from_average_y=False, plot=True, combat=False,
                n_perm=10000, seed=None, n_proc=1, verbose=True,
                nispace_object=None, fetch_x_kwargs=None, init_kwargs=None,
                fit_kwargs=None, clean_y_kwargs=None, colocalize_kwargs=None,
                permute_kwargs=None, correct_p_kwargs=None, plot_kwargs=None,
                return_nispace_only=False):
    """Deprecated wrapper for :func:`xsea`.

    .. deprecated:: dev
       Use :func:`xsea` instead. Will be removed in the first non-dev release.
    """
    lgr.warning(_DEPR_FUNC_NAME.format(old="simple_xsea", new="xsea"))
    return xsea(
        y=y, x=x, z=z, x_collection=x_collection, x_background=x_background,
        standardize=standardize, space=space, data_space=data_space,
        parcellation_space=parcellation_space, parcellation=parcellation,
        parcellation_labels=parcellation_labels, parcellation_hemi=parcellation_hemi,
        y_covariates=y_covariates, colocalization_method=colocalization_method,
        mc_method=mc_method, normalize_colocalizations=normalize_colocalizations,
        xsea_aggregation_method=xsea_aggregation_method, permute_sets=permute_sets,
        pooled_p=p_from_average_y,
        plot=plot, combat=combat, n_perm=n_perm, seed=seed, n_proc=n_proc,
        verbose=verbose, nispace_object=nispace_object, fetch_x_kwargs=fetch_x_kwargs,
        init_kwargs=init_kwargs, fit_kwargs=fit_kwargs, clean_y_kwargs=clean_y_kwargs,
        colocalize_kwargs=colocalize_kwargs, permute_kwargs=permute_kwargs,
        correct_p_kwargs=correct_p_kwargs, plot_kwargs=plot_kwargs,
        return_nispace_only=return_nispace_only,
    )
