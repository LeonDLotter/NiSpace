"""
Diagnostic / exploratory functions that operate on an already-fitted :class:`NiSpace`
object, passed explicitly as the first argument rather than being bound methods on the
class. These are read-only: they never mutate ``nsp`` and never persist their result on
it (no ``store=``, no getters) -- call them again if you need the result again. They
exist to further probe/validate a colocalization result already computed via
``nsp.colocalize()`` (and, for functions that use permutation, ``nsp.permute()``), not to
extend the core pipeline itself.

- :func:`regional_influence` -- leave-one-region-out sensitivity of a colocalization result.
- :func:`regional_contribution` -- additive per-region decomposition of a colocalization result.
- :func:`local_colocalization` -- "searchlight" style colocalization recomputed within each
  region's k-nearest-neighbor window. Works on any ``nsp`` that has run ``colocalize()``,
  whether that used raw Y or a ``transform_y()`` group-contrast map -- there is no separate
  "group" variant, this function auto-detects the pathway (see its docstring).
- :func:`spatial_cross_validate` -- Hansen-et-al.-style distance-dependent cross-validation
  (GitHub issue #74).
- :func:`compare_sets` -- overlap and set-mean-map similarity between X sets, against
  size-matched random set pairs. Needs only set-structured X and a background, not
  ``colocalize()``.
- :func:`shared_axes` -- whether Y and X share known spatial axes (tissue, geometry,
  compartment, cortical hierarchy; :func:`nispace.datasets.fetch_axes`) and whether the Moran
  null is degenerate (dominated by one eigenvector). Needs only ``fit()``.

``regional_influence``/``regional_contribution`` moved here from :class:`NiSpace` (formerly
``NiSpace.regional_influence()``/``.regional_contribution()``) -- this is a clean break, no
backward-compat wrapper was kept on the class.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import rankdata, hypergeom
from joblib import Parallel, delayed
from tqdm import tqdm
from contextlib import contextmanager

import logging
lgr = logging.getLogger(__name__)

from .core.colocalize import _get_coloc_stats, _rank_regress
from .core.permute import _resolve_permute_mode_settings
from .core.region_influence import _get_region_influence_fun, _sort_region_influence, _pool_region_influence
from .core.region_contribution import (_get_region_contribution_fun, _sort_region_contribution,
                                       _CONTRIBUTION_METHODS)
from .stats.coloc import rank2d
from .stats.effectsize import rzscore_nan, zscore_nan
from .stats.misc import (maxT_correction, step_maxT_correction, rho_to_z, z_to_rho, zscore_df,
                         null_to_p)
from .utils.utils import set_log, _quiet, print_arg_pairs


@contextmanager
def _silence_repeated_calls():
    """More robust than :func:`_quiet` alone for a tight loop of many
    ``nsp.colocalize()`` calls: ``_quiet()`` only raises the shared ``"nispace"`` root
    logger's level, but several ``NiSpace`` getters (``get_x``/``get_y``/
    ``get_colocalizations``/...) have a "capture the current effective level, then
    ``lgr.setLevel(captured)`` to restore it afterward" pattern -- that ``setLevel()``
    call sets an explicit level directly on the ``"nispace.api"`` *child* logger, which
    then stops inheriting from the root entirely, so ``_quiet()``'s root-level change
    no longer reaches it (a known, pre-existing bug -- see project memory "Logging
    level stuck-at-60 bug"). Sidesteps it here by also directly silencing
    ``"nispace.api"`` for the duration, restoring its own prior level afterward.
    """
    api_lgr = logging.getLogger("nispace.api")
    old_level = api_lgr.level
    api_lgr.setLevel(60)
    try:
        with _quiet():
            yield
    finally:
        api_lgr.setLevel(old_level)


def _resolve_coloc_data(nsp, method, X, Y, Z, X_reduction, Y_transform, xsea,
                        zy_matched, dtype, verbose):
    """Reproduce colocalize()'s X/Y/Z resolution + rank/Z-regression prep for a method
    that was already run via ``nsp.colocalize()``, so the exact same (post rank/
    Z-regression) arrays are reproduced. rank/regress_z/zy_matched are read back
    verbatim from ``nsp._coloc_kwargs_by_method[method]`` (the fully-resolved, per-method
    values colocalize() itself stored) rather than re-derived here -- re-deriving these
    independently from last-settings + method-name heuristics is what caused a staleness
    bug for `rank` in colocalize() itself; reading colocalize()'s own resolved values
    avoids that whole class of divergence, and correctly reproduces a deliberate
    override too (e.g. ``colocalize(method="mlr", rank=True)``).

    Shared by regional_influence()/regional_contribution()/local_colocalization() --
    deliberately not unified with colocalize()'s own (richer) X/Y/Z block, which also
    auto-runs transform_y(), mutates XSEA state, and handles the regularized-regression
    CV-split branch; see the TODO comment in NiSpace.colocalize().

    Returns
    -------
    X, Y : pandas.DataFrame
        The (possibly re-fetched) input DataFrames, for their .columns/.index.
    X_arr, Y_arr : np.ndarray or dict of np.ndarray
        Resolved, ranked/Z-regressed arrays ready for a colocalize()-style function.
    X_weights : dict of np.ndarray or None
        Per-set weights, only set for XSEA with a "weighted" aggregation method.
    coloc_kwargs : dict
        ``nsp._coloc_kwargs_by_method[method]``, for callers that need e.g. adj_r2/r_to_z/
        xsea_method on top of what this function already consumes.
    """
    coloc_kwargs = nsp._coloc_kwargs_by_method[method]
    xsea_aggregation_method = coloc_kwargs.get("xsea_method", "mean")

    # X
    if X is None:
        if not X_reduction:
            X = nsp._X
        else:
            with _quiet():
                X = nsp.get_x(X_reduction=X_reduction)
    X_arr = np.array(X, dtype=dtype)
    X_weights = None
    if xsea:
        if (not isinstance(X, pd.DataFrame) or not isinstance(X.index, pd.MultiIndex)
                or "set" not in X.index.names):
            lgr.critical_raise("XSEA requires X data to have a MultiIndex with a 'set' level!",
                               ValueError)
        X_arr = {set_name: np.array(set_X, dtype=dtype)
                 for set_name, set_X in X.groupby(level="set", sort=False)}
        if "weighted" in xsea_aggregation_method:
            X_weights = {set_name: np.array(set_X.index.get_level_values("weight"), dtype=dtype)
                         for set_name, set_X in X.groupby(level="set", sort=False)}

    # Y
    if Y is None:
        if not Y_transform:
            Y = nsp._Y
        else:
            if not nsp._check_transform(ytrans=Y_transform, raise_error=True):
                lgr.critical_raise(f"Y transform '{Y_transform}' was not run before "
                                   "colocalize(). Did you run colocalize() first?",
                                   KeyError)
            with _quiet():
                Y = nsp.get_y(Y_transform=Y_transform)
    Y_arr = np.array(Y, dtype=dtype)

    # rank / regress_z / zy_matched -- read back verbatim, see docstring above
    rank = coloc_kwargs.get("rank", "spearman" in method)
    zy_matched = coloc_kwargs.get("zy_matched", zy_matched)
    regress_z = coloc_kwargs.get("regress_z", "")
    if Z is None:
        Z = nsp._Z
    Z_arr = np.array(Z, dtype=dtype) if regress_z else None
    # standard partial Spearman correlation ranks X, Y, AND Z before partial-correlating
    # (not just X/Y) -- without this, Z stays on its raw scale while X/Y are ranked
    if rank and Z_arr is not None:
        Z_arr = rank2d(Z_arr.T).T

    if rank or regress_z:
        X_arr = _rank_regress(arr=X_arr, rank=rank, regress="x" in regress_z, z=Z_arr,
                              zy_matched=zy_matched, verbose=verbose)
        Y_arr = _rank_regress(arr=Y_arr, rank=rank, regress="y" in regress_z, z=Z_arr,
                              zy_matched=zy_matched, verbose=verbose)

    return X, Y, X_arr, Y_arr, X_weights, coloc_kwargs


def regional_influence(nsp, method=None, stat=None, engine="auto", signed=False,
                       X_reduction=None, Y_transform=None, xsea=None,
                       regress_z=True, zy_matched=False,
                       X=None, Y=None, Z=None, pooled=None,
                       n_proc=None, verbose=None, force_dict=False):
    """
    Estimate, per region, the true leave-one-out sensitivity of a colocalization
    result: ``|stat_full| - |stat_loo|`` for the region excluded (or the signed
    ``stat_full - stat_loo`` if ``signed=True``), not an approximation (either
    computed exactly via closed-form case-deletion identities -- engine="analytic"
    -- or by literally rerunning colocalize() with the region excluded --
    engine="bruteforce"). Requires ``nsp.colocalize()`` to have been run first with the
    same method (reuses its stored settings/closure).

    Reports a stat_full/stat_loo delta per region rather than Cook's distance/
    DFFITS/leverage: those answer a classical outlier-flagging question; this
    answers "how much does the reported effect change without this region", which
    is what's needed here. Developed and first applied in :cite:`lotter2024`.

    The default (``signed=False``) takes the absolute value of the full-data and
    LOO stat before differencing. This is a no-op for mlr/dominance/pls/pcr/mi/slr
    (their stat -- R^2 or MI -- is already >= 0, no direction to speak of), but for
    the correlation methods (pearson/spearman/partialpearson/partialspearman) it
    makes the default homogeneous with the other methods: every method's default
    answers "does this region strengthen or weaken the association" without regard
    to direction. ``signed=True`` recovers the original directional delta for the
    correlation methods -- positive means the region pulls the correlation toward
    +1, negative toward -1, regardless of the sign of the observed correlation
    itself (a region can oppose the overall trend and still pull toward +1).

    For pearson/spearman/partialpearson/partialspearman, both the full-data and
    LOO ``rho`` are Fisher-z-transformed (``numpy.arctanh``) before differencing
    whenever the original ``colocalize()`` call for this method used
    ``r_to_z=True`` (the default) -- so the reported delta is on the
    variance-stabilized scale by default, not a raw-rho difference. This is not
    a separate setting here; it always matches the stored ``colocalize()``
    result's own ``r_to_z``.

    Parameters
    ----------
    nsp : NiSpace
        A fitted NiSpace object that has already run ``colocalize()``.
    method : str, optional
        Colocalization method. Defaults to the last method used in ``nsp.colocalize()``.
        Supported: pearson, spearman, partialpearson, partialspearman, mi, slr, mlr,
        dominance, pls, pcr. Not supported: lasso, ridge, elasticnet (their
        colocalization closures capture a fixed-size CV split/regularization mask
        sized to the original number of regions, which would misalign against
        region-excluded data).
    stat : str, optional
        Which colocalization stat to compute influence for. Defaults to the
        method's primary stat.
    engine : {"auto", "analytic", "bruteforce"}, default "auto"
        "analytic" is only available for pearson/spearman/partialpearson/
        partialspearman/mlr; "auto" picks it for those and falls back to
        "bruteforce" otherwise. "bruteforce" reruns colocalize() once per excluded
        region and can be slow for many regions -- a warning is logged above 1000.
    signed : bool, default False
        See above. Only changes behavior for pearson/spearman/partialpearson/
        partialspearman -- a no-op for every other supported method.
    X_reduction, Y_transform, xsea : see ``NiSpace.colocalize()``.
    regress_z, zy_matched : see ``NiSpace.colocalize()``. Must match the colocalize() call
        being explained so the same X/Y data (after ranking/Z-regression) is reproduced.
    pooled : {None, False, True, "mean", "median"}, default None
        Pool (reduce) the per-Y-row result across Y (subjects/maps). None defaults
        to whatever ``pooled_p`` was last set to elsewhere in the pipeline (e.g. by
        permute()); True is treated as "mean". Pools the per-subject delta directly
        (median of deltas, not delta of medians) -- the correct choice for this
        paired quantity.
    force_dict : bool, default False
        For methods that fit one joint model per Y-row (mlr/dominance/pls/pcr),
        the result is a single DataFrame (n_Y x n_regions); force_dict wraps it in a
        length-1 dict for a uniform return type. For per-predictor/per-set methods
        (pearson/spearman/partialpearson/partialspearman/mi/slr, or any XSEA call),
        the result is always a dict of DataFrames keyed by X map / set label, since
        each X-Y pair (or set) has its own region-influence profile.

    Returns
    -------
    pandas.DataFrame or dict of pandas.DataFrame
    """
    verbose = set_log(lgr, nsp._verbose if verbose is None else verbose)
    lgr.info("*** diagnostics.regional_influence() - Estimating regional influence. ***")

    nsp._check_fit()

    n_proc = nsp._n_proc if n_proc is None else n_proc
    dtype = nsp._dtype

    method, X_reduction, Y_transform, xsea = nsp._get_last(
        method=method,
        X_reduction=X_reduction,
        Y_transform=Y_transform,
        xsea=xsea,
    )
    if method is None:
        lgr.critical_raise("No colocalization method defined! Run colocalize() first.",
                           ValueError)
    if method not in nsp._colocs_fun or method not in nsp._coloc_kwargs_by_method:
        lgr.critical_raise(f"No stored colocalize() results for method '{method}'! "
                           "Did you run colocalize() with this method first?",
                           KeyError)

    X, Y, X_arr, Y_arr, X_weights, coloc_kwargs = _resolve_coloc_data(
        nsp, method=method, X=X, Y=Y, Z=Z, X_reduction=X_reduction, Y_transform=Y_transform,
        xsea=xsea, zy_matched=zy_matched, dtype=dtype, verbose=verbose,
    )
    adj_r2 = coloc_kwargs.get("adj_r2", True)
    r_to_z = coloc_kwargs.get("r_to_z", True)
    xsea_aggregation_method = coloc_kwargs.get("xsea_method", "mean")

    ## resolve stat and n_parcels
    if stat is None:
        stat = _get_coloc_stats(method, drop_optional=True)[0]
    n_parcels = X_arr.shape[1] if isinstance(X_arr, np.ndarray) else \
        next(iter(X_arr.values())).shape[1]

    ## build the region-influence function and run it, same Parallel idiom as colocalize()
    _y_colocalize = nsp._colocs_fun[method]
    fun, engine_used = _get_region_influence_fun(
        method=method, engine=engine, n_parcels=n_parcels,
        y_colocalize_fun=_y_colocalize, stat=stat,
        adj_r2=adj_r2, r_to_z=r_to_z, dtype=dtype,
        xsea=xsea, xsea_method=xsea_aggregation_method if xsea else None,
        signed=signed,
    )

    _infl_list = Parallel(n_jobs=n_proc)(
        delayed(fun)(X_arr, Y_arr[i_y, :], X_weights)
        for i_y in tqdm(
            range(Y.shape[0]),
            desc=f"Regional influence ({method}, {engine_used}, {n_proc} proc)",
            disable=not verbose,
        )
    )

    ## sort output
    _infl = _sort_region_influence(
        y_infl_list=_infl_list,
        n_parcels=n_parcels,
        n_Y=Y.shape[0],
        labs_parcels=X.columns,
        labs_Y=Y.index,
        labs_X=X.index if not xsea else list(X_arr.keys()),
        dtype=dtype,
    )

    if pooled is None:
        pooled = nsp._last_settings.get("pooled_p", False)
    if pooled:
        _infl = _pool_region_influence(_infl, "mean" if pooled is True else pooled)

    lgr.info(f"Returning regional_influence results: \n"
             f"{print_arg_pairs(method=method, stat=stat, engine=engine_used, signed=signed, xsea=xsea, X_reduction=X_reduction, Y_transform=Y_transform, pooled=pooled)}")

    if force_dict and not isinstance(_infl, dict):
        return {stat: _infl}
    return _infl


def regional_contribution(nsp, method=None, X_reduction=None, Y_transform=None, xsea=None,
                          regress_z=True, zy_matched=False,
                          X=None, Y=None, Z=None, quadrant=False, pooled=None,
                          n_proc=None, verbose=None):
    """
    Decompose a colocalization result into each region's own additive share of the
    reported correlation: ``contribution_i = zx_i * zy_i`` (population z-scores of
    whatever data is already in the pipeline at this point -- raw values for
    pearson, ranks for spearman/partial*, matching colocalize()'s own convention).
    This is an exact decomposition, not an approximation or a perturbation --
    ``mean(contribution) == rho`` exactly. Requires ``nsp.colocalize()`` to have been run
    first with the same method (reuses its stored settings).

    Also computes a ``quadrant`` label per region -- "high_high", "low_low", or
    "discordant" (sign of zx vs zy) -- retrievable via ``quadrant=True``. This exists
    because regional_influence() (leave-one-out) is structurally symmetric between
    high-high and low-low concordant regions -- both reinforce a positive
    correlation identically, since that symmetry is inherent to what Pearson/
    Spearman measure, not fixable within the LOO framework. ``contribution`` alone
    has the same symmetry (both quadrants give a positive value); ``quadrant`` is
    what actually distinguishes them. Default behavior (``quadrant=False``) returns
    ``contribution`` only (the "whole map").

    This is a standard decomposition of the spatial correlation between two maps,
    similar to what was presented in Faskowitz et al. (2026) :cite:`faskowitz2026` at OHBM 2026.

    Parameters
    ----------
    nsp : NiSpace
        A fitted NiSpace object that has already run ``colocalize()``.
    method : str, optional
        Colocalization method. Defaults to the last method used in ``nsp.colocalize()``.
        Supported: pearson, spearman, partialpearson, partialspearman -- the 4
        methods with a genuinely bidirectional (signed) primary stat. Not
        supported for R^2/MI-based methods (mlr, dominance, pls, pcr, mi, slr),
        which have no "high/low" side to decompose into quadrants.
    X_reduction, Y_transform, xsea : see ``NiSpace.colocalize()``.
    regress_z, zy_matched : see ``NiSpace.colocalize()``. Must match the colocalize() call
        being explained so the same X/Y data (after ranking/Z-regression) is reproduced.
    quadrant : bool, default False
        If False (default), return the ``contribution`` values (the "whole map").
        If True, return the categorical ``quadrant`` labels ("high_high"/"low_low"/
        "discordant") instead.
    pooled : {None, False, True, "mean", "median"}, default None
        Pool (reduce) the per-Y-row result across Y (subjects/maps). None defaults
        to whatever ``pooled_p`` was last set to elsewhere in the pipeline. Only valid
        when ``quadrant=False`` -- pooling isn't meaningful for categorical labels.

    Returns
    -------
    dict of pandas.DataFrame
        Keyed by X-map/set label (always dict-shaped -- all 4 supported methods
        are per-X-pair methods).
    """
    verbose = set_log(lgr, nsp._verbose if verbose is None else verbose)
    lgr.info("*** diagnostics.regional_contribution() - Estimating regional contribution. ***")

    nsp._check_fit()

    n_proc = nsp._n_proc if n_proc is None else n_proc
    dtype = nsp._dtype

    method, X_reduction, Y_transform, xsea = nsp._get_last(
        method=method,
        X_reduction=X_reduction,
        Y_transform=Y_transform,
        xsea=xsea,
    )
    if method is None:
        lgr.critical_raise("No colocalization method defined! Run colocalize() first.",
                           ValueError)
    if method not in _CONTRIBUTION_METHODS:
        lgr.critical_raise(f"regional_contribution() does not support method '{method}'. "
                           f"Supported: {sorted(_CONTRIBUTION_METHODS)} -- methods with a "
                           "genuinely bidirectional primary stat (rho). R^2/MI-based methods "
                           "have no 'high/low' side to decompose into quadrants.",
                           ValueError)
    if method not in nsp._colocs_fun or method not in nsp._coloc_kwargs_by_method:
        lgr.critical_raise(f"No stored colocalize() results for method '{method}'! "
                           "Did you run colocalize() with this method first?",
                           KeyError)

    X, Y, X_arr, Y_arr, X_weights, coloc_kwargs = _resolve_coloc_data(
        nsp, method=method, X=X, Y=Y, Z=Z, X_reduction=X_reduction, Y_transform=Y_transform,
        xsea=xsea, zy_matched=zy_matched, dtype=dtype, verbose=verbose,
    )
    xsea_aggregation_method = coloc_kwargs.get("xsea_method", "mean")

    n_parcels = X_arr.shape[1] if isinstance(X_arr, np.ndarray) else \
        next(iter(X_arr.values())).shape[1]

    ## build the region-contribution function and run it, same Parallel idiom as colocalize()
    fun = _get_region_contribution_fun(
        method=method, dtype=dtype, xsea=xsea,
        xsea_method=xsea_aggregation_method if xsea else None,
    )

    _contrib_list = Parallel(n_jobs=n_proc)(
        delayed(fun)(X_arr, Y_arr[i_y, :], X_weights)
        for i_y in tqdm(
            range(Y.shape[0]),
            desc=f"Regional contribution ({method}, {n_proc} proc)",
            disable=not verbose,
        )
    )

    ## sort output -- always (contribution_dict, quadrant_dict)
    contrib_dict, quadrant_dict = _sort_region_contribution(
        y_list=_contrib_list,
        n_parcels=n_parcels,
        n_Y=Y.shape[0],
        labs_parcels=X.columns,
        labs_Y=Y.index,
        labs_X=X.index if not xsea else list(X_arr.keys()),
        dtype=dtype,
    )

    if quadrant and pooled:
        lgr.critical_raise("'pooled' is not meaningful for categorical quadrant labels "
                           "(quadrant=True). Use quadrant=False for pooling.",
                           ValueError)

    out = quadrant_dict if quadrant else contrib_dict
    if pooled is None:
        pooled = False if quadrant else nsp._last_settings.get("pooled_p", False)
    if pooled:
        out = _pool_region_influence(out, "mean" if pooled is True else pooled)

    lgr.info(f"Returning regional_contribution results: \n"
             f"{print_arg_pairs(method=method, quadrant=quadrant, xsea=xsea, X_reduction=X_reduction, Y_transform=Y_transform, pooled=pooled)}")

    return out


# LOCAL COLOCALIZATION (searchlight) ===============================================================

# FWHM (mm) -> Gaussian sigma: sigma = fwhm / (2*sqrt(2*ln2))
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
# Gaussian-kernel mode is restricted to these 4 methods forever (not planned to be
# extended to mi/slr/mlr/dominance/pls/pcr): they're the only ones with a closed-form
# weighted correlation. Reuses _CONTRIBUTION_METHODS (same set as the k/radius fast
# null path) rather than redefining it.
_LOCAL_GAUSSIAN_METHODS = _CONTRIBUTION_METHODS

_LOCAL_UNSUPPORTED_METHODS = frozenset({"lasso", "ridge", "elasticnet"})
_LOCAL_JOINT_METHODS = frozenset({"mlr", "dominance", "pls", "pcr"})
# fast null path: pearson/spearman/partialpearson/partialspearman all reduce to a plain
# Pearson correlation on (optionally ranked, optionally Z-residualized) data -- literally
# one shared branch in core/colocalize.py's _get_colocalize_fun. Reuses _CONTRIBUTION_METHODS
# (regional_contribution()'s method set) rather than redefining the same 4 methods again.
_LOCAL_NULL_FAST_METHODS = _CONTRIBUTION_METHODS
# slow null path: no batched closed-form exists for these (mi has none at all; mlr/dominance/
# pls/pcr fit a joint model that can't be vectorized across null draws) -- falls back to
# reusing colocalize()'s own per-method closure directly, one call per (window, null draw).
_LOCAL_NULL_SLOW_METHODS = frozenset({"mi", "slr", "mlr", "dominance", "pls", "pcr"})


def _batched_corr(A, B):
    """Pearson correlation along the last axis, batched over any number of leading
    (broadcastable) axes, with pairwise NaN masking (mirrors ``stats/coloc.py``'s
    ``pearson()``: only positions valid in both ``A`` and ``B`` count).

    A vectorized analogue of the scalar, numba-jitted ``pearson()`` -- needed because
    :func:`local_colocalization`'s fast null path recomputes the correlation for every
    (window x null-draw) pair, far too many individual calls for the scalar version to
    be practical. Expects ``A``/``B`` already ranked/Z-regressed upstream (via
    ``_local_null_preprocess``) if the method requires it -- this function only ever
    computes a plain Pearson correlation on whatever it's given.
    """
    A, B = np.broadcast_arrays(A, B)
    mask = ~np.isnan(A) & ~np.isnan(B)
    n_valid = mask.sum(axis=-1)
    A0 = np.where(mask, A, 0.0)
    B0 = np.where(mask, B, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mA = A0.sum(axis=-1) / n_valid
        mB = B0.sum(axis=-1) / n_valid
        Ac = np.where(mask, A - mA[..., np.newaxis], 0.0)
        Bc = np.where(mask, B - mB[..., np.newaxis], 0.0)
        num = (Ac * Bc).sum(axis=-1)
        den = np.sqrt((Ac ** 2).sum(axis=-1) * (Bc ** 2).sum(axis=-1))
        r = num / den
    return np.where(n_valid < 2, np.nan, r)


def _weighted_batched_corr(A, B, w):
    """Weighted Pearson correlation along the last axis, batched over any number of
    leading (broadcastable) axes, for one or more sets of weights ``w`` with shape
    ``(n_seeds, n_parcels)``. Companion to :func:`_batched_corr` for the Gaussian-kernel
    searchlight mode (``local_colocalization(fwhm_mm=...)``): same pairwise NaN-masking
    convention, but each (co)variance sum is weighted by ``w`` (renormalized over the
    valid positions for that particular (A, B) pair -- NaN positions get 0 weight, not
    dropped from ``w`` beforehand) instead of being a uniform average over a hard window.

    Returns an array of shape ``broadcast(A, B).shape[:-1] + (n_seeds,)``. Used with
    ``n_seeds=1`` (one row of ``w``) inside the per-seed null loop, and with
    ``n_seeds=n_parcels`` for the single vectorized observed-statistic call.
    """
    A, B = np.broadcast_arrays(A, B)                        # (..., n_parcels)
    mask = ~np.isnan(A) & ~np.isnan(B)                       # (..., n_parcels)
    A0 = np.where(mask, A, 0.0)[..., np.newaxis, :]           # (..., 1, n_parcels)
    B0 = np.where(mask, B, 0.0)[..., np.newaxis, :]
    w = w * mask[..., np.newaxis, :]                          # (..., n_seeds, n_parcels)
    w_sum = w.sum(axis=-1)                                     # (..., n_seeds)
    with np.errstate(invalid="ignore", divide="ignore"):
        mA = (w * A0).sum(-1) / w_sum
        mB = (w * B0).sum(-1) / w_sum
        Ac = np.where(mask[..., np.newaxis, :], A[..., np.newaxis, :] - mA[..., np.newaxis], 0.0)
        Bc = np.where(mask[..., np.newaxis, :], B[..., np.newaxis, :] - mB[..., np.newaxis], 0.0)
        num = (w * Ac * Bc).sum(-1)
        den = np.sqrt((w * Ac ** 2).sum(-1) * (w * Bc ** 2).sum(-1))
        r = num / den
    return np.where(w_sum <= 0, np.nan, r)


def _local_gaussian_observed(W, X_arr_pre, Y_arr_pre):
    """Observed statistic for the Gaussian-kernel searchlight mode -- one vectorized
    :func:`_weighted_batched_corr` call across every seed at once. Unlike the k/radius
    windowed modes, there is no per-window column subset here (every seed uses all
    ``n_parcels``, that's the whole point of the kernel), so ``X_arr_pre``/``Y_arr_pre``
    only need to be preprocessed once, globally -- exactly the same (rank-/Z-regressed)
    arrays :meth:`NiSpace.colocalize` itself would have used for the whole-brain fit,
    reused verbatim rather than recomputed per seed."""
    return _weighted_batched_corr(
        Y_arr_pre[:, np.newaxis, :],   # (n_Y, 1, n_parcels)
        X_arr_pre[np.newaxis, :, :],   # (1, n_X, n_parcels)
        W,                              # (n_parcels [seeds], n_parcels)
    )  # -> (n_Y, n_X, n_parcels [seeds])


def _local_null_gaussian_seed(w_seed, X_pre, Y_pre, null_pre, null_side):
    """One seed's worth of null correlation for the Gaussian-kernel mode -- companion
    to :func:`_local_null_fast_window`, but with no window slicing: ``X_pre``/``Y_pre``/
    ``null_pre`` are already globally rank-/Z-regressed once outside this loop (there is
    no per-window subset to rank locally when every seed uses all ``n_parcels``), so only
    the per-seed kernel weight row changes what gets summed."""
    w_seed = w_seed[np.newaxis, :]  # (1, n_parcels) -- _weighted_batched_corr's n_seeds axis
    if null_side == "X":
        out = _weighted_batched_corr(
            null_pre[np.newaxis, :, :, :],          # (1, n_X_, n_perm, n_parcels)
            Y_pre[:, np.newaxis, np.newaxis, :],     # (n_Y_, 1, 1, n_parcels)
            w_seed,
        )
    else:
        out = _weighted_batched_corr(
            X_pre[np.newaxis, :, np.newaxis, :],     # (1, n_X_, 1, n_parcels)
            null_pre[:, np.newaxis, :, :],            # (n_Y_, 1, n_perm, n_parcels)
            w_seed,
        )
    return out[..., 0]  # drop the trailing n_seeds=1 axis


def _local_null_gaussian(W, X_pre, Y_pre, null_pre, null_side, n_proc, verbose):
    """Null recomputation for the Gaussian-kernel searchlight mode, parallelized across
    seeds (``n_proc``) exactly like :func:`_local_null_fast` -- the only difference from
    the k/radius fast path is that ``X_pre``/``Y_pre``/``null_pre`` are already globally
    preprocessed once by the caller (no per-window ``_local_null_preprocess`` call)."""
    n_parcels = W.shape[0]
    results = Parallel(n_jobs=n_proc)(
        delayed(_local_null_gaussian_seed)(W[j], X_pre, Y_pre, null_pre, null_side)
        for j in tqdm(range(n_parcels), desc="Local colocalization null (gaussian kernel)",
                     disable=not verbose, mininterval=1.0)
    )
    return np.stack(results, axis=-1)  # (n_Y_, n_X_, n_perm, n_parcels)


def _local_null_preprocess(nb, X_arr, Y_arr, Z_full, null_data, null_side,
                           rank, regress_z, zy_matched, dtype):
    """Per-window rank/Z-regression preprocessing shared by the fast and slow null
    paths -- reuses ``core/colocalize.py``'s own ``_rank_regress()`` (the exact function
    ``colocalize()`` itself calls) on the window-sliced observed and null-drawn arrays,
    so both null paths reproduce exactly what a real ``colocalize()`` call on this
    window's data would have done, matching the observed statistic's own preprocessing.
    Rank/Z-regression are resolved window-locally (using only this window's k parcels),
    consistent with the observed statistic already being computed via a fresh per-window
    ``nsp.colocalize(X=, Y=, Z=)`` call rather than a single whole-brain preprocessing.

    Returns
    -------
    obs_pre : np.ndarray, shape (n_obs_maps, k)
    null_pre : np.ndarray, shape (n_null_maps, n_perm, k)
    """
    regress_x = bool(regress_z) and "x" in regress_z
    regress_y = bool(regress_z) and "y" in regress_z
    Z_sub = None
    if regress_z:
        Z_sub = np.asarray(Z_full.iloc[:, nb], dtype=dtype)
        if rank:
            Z_sub = rank2d(Z_sub.T).T

    null_win = null_data[:, :, nb]  # (n_null_maps, n_perm, k)
    null_list = list(null_win.transpose(1, 0, 2))  # n_perm arrays, each (n_null_maps, k)

    if null_side == "X":
        obs_win = Y_arr[:, nb]
        obs_pre = _rank_regress(arr=obs_win, rank=rank, regress=regress_y, z=Z_sub,
                                zy_matched=zy_matched, verbose=False)
        null_pre_list = _rank_regress(arr=null_list, rank=rank, regress=regress_x, z=Z_sub,
                                      zy_matched=zy_matched, n_proc=1, verbose=False)
    else:
        obs_win = X_arr[:, nb]
        obs_pre = _rank_regress(arr=obs_win, rank=rank, regress=regress_x, z=Z_sub,
                                zy_matched=zy_matched, verbose=False)
        null_pre_list = _rank_regress(arr=null_list, rank=rank, regress=regress_y, z=Z_sub,
                                      zy_matched=zy_matched, n_proc=1, verbose=False)
    null_pre = np.stack(null_pre_list, axis=1)  # (n_null_maps, n_perm, k)
    return obs_pre, null_pre


def _local_null_fast_window(nb, X_arr, Y_arr, Z_full, null_data, null_side,
                            rank, regress_z, zy_matched, dtype):
    """One window's worth of work for :func:`_local_null_fast` -- factored out so it
    can be dispatched as an independent parallel task (windows are fully independent
    of each other, unlike the null draws within a window, which are already handled
    in one vectorized call and don't need their own parallelism)."""
    obs_pre, null_pre = _local_null_preprocess(
        nb, X_arr, Y_arr, Z_full, null_data, null_side, rank, regress_z, zy_matched, dtype,
    )
    if null_side == "X":
        return _batched_corr(
            null_pre[np.newaxis, :, :, :],           # (1, n_X_, n_perm, k)
            obs_pre[:, np.newaxis, np.newaxis, :],   # (n_Y_, 1, 1, k)
        )
    return _batched_corr(
        obs_pre[np.newaxis, :, np.newaxis, :],   # (1, n_X_, 1, k)
        null_pre[:, np.newaxis, :, :],            # (n_Y_, 1, n_perm, k)
    )


def _local_null_fast(windows, X_arr, Y_arr, Z_full, null_data, null_side,
                     rank, regress_z, zy_matched, dtype, n_proc, verbose):
    """Null recomputation for pearson/spearman/partialpearson/partialspearman -- one
    vectorized ``_batched_corr()`` call per window covering all null draws at once,
    parallelized **across windows** (``n_proc``) -- windows are fully independent, the
    embarrassingly-parallel dimension here, unlike the null draws within one window
    (already handled in a single vectorized call, nothing to gain from parallelizing
    further inside it)."""
    n_parcels = X_arr.shape[1]

    results = Parallel(n_jobs=n_proc)(
        delayed(_local_null_fast_window)(
            windows[i], X_arr, Y_arr, Z_full, null_data, null_side,
            rank, regress_z, zy_matched, dtype,
        )
        for i in tqdm(range(n_parcels), desc="Local colocalization null (fast path)",
                     disable=not verbose, mininterval=1.0)
    )
    return np.stack(results, axis=-1)  # (n_Y_, n_X_, n_perm, n_parcels)


def _local_null_slow_window(nb, y_colocalize, stat, X_arr, Y_arr, Z_full, null_data, null_side,
                            rank, regress_z, zy_matched, n_cols, dtype):
    """One window's worth of work for :func:`_local_null_slow` -- serial over
    (Y-row, null draw) *within* the window (``n_proc=1`` in effect here by
    construction: no nested ``Parallel``), since parallelism is applied at the window
    level instead (see :func:`_local_null_slow`'s docstring for why)."""
    obs_pre, null_pre = _local_null_preprocess(
        nb, X_arr, Y_arr, Z_full, null_data, null_side, rank, regress_z, zy_matched, dtype,
    )
    n_Y_ = Y_arr.shape[0]
    n_perm = null_data.shape[1]
    out = np.empty((n_Y_, n_cols, n_perm), dtype=dtype)
    if null_side == "X":
        for iy in range(n_Y_):
            y_row = obs_pre[iy]
            for p in range(n_perm):
                out[iy, :, p] = y_colocalize(null_pre[:, p, :], y_row)[stat]
    else:
        for iy in range(n_Y_):
            for p in range(n_perm):
                out[iy, :, p] = y_colocalize(obs_pre, null_pre[iy, p, :])[stat]
    return out


def _local_null_slow(nsp, method, windows, X_arr, Y_arr, Z_full, null_data, null_side,
                     rank, regress_z, zy_matched, is_joint, dtype, n_proc, verbose):
    """Null recomputation for mi/slr/mlr/dominance/pls/pcr: no batched closed-form
    exists, so this reuses the exact per-Y-row colocalization closure ``colocalize()``
    already built and stored (``nsp._colocs_fun[method]``) -- the same closure
    :func:`regional_influence`'s bruteforce engine reuses for excluded-region data --
    applied directly to each (window x null-draw) pair. No new per-method statistical
    code, but genuinely slower than the fast path: there's no way to batch an actual
    per-draw model refit, so this is one closure call per (window, Y-row, null draw).
    Can be very slow for expensive methods (e.g. ``dominance``, which fits every
    predictor subset) at the default ``n_perm`` -- consider a lower ``n_perm`` in
    ``permute()`` for these methods if this path is too slow.

    Parallelized **across windows** (``n_proc``), each window's (Y-row x null-draw)
    loop running serially inside its own task -- not parallelized per-draw *within* a
    window as an earlier version did, which repeatedly spun up and tore down a joblib
    pool once per (window, Y-row) pair (up to ``n_parcels x n_Y`` times). Windows are
    both more numerous and embarrassingly independent, so parallelizing there instead
    gives the same speedup with far less pool-management overhead.
    """
    n_parcels = X_arr.shape[1]
    n_cols = 1 if is_joint else X_arr.shape[0]
    stat = _get_coloc_stats(method, drop_optional=True)[0]
    y_colocalize = nsp._colocs_fun[method]

    results = Parallel(n_jobs=n_proc)(
        delayed(_local_null_slow_window)(
            windows[i], y_colocalize, stat, X_arr, Y_arr, Z_full, null_data, null_side,
            rank, regress_z, zy_matched, n_cols, dtype,
        )
        for i in tqdm(range(n_parcels), desc=f"Local colocalization null ({method}, slow path)",
                     disable=not verbose, mininterval=1.0)
    )
    return np.stack(results, axis=-1)  # (n_Y_, n_cols, n_perm, n_parcels)


def _local_observed_window(nsp, nb, X, Y, Z_full, method, rank, regress_z, zy_matched, stat):
    """One window's observed statistic for :func:`local_colocalization` -- factored out
    so it can be dispatched as an independent parallel task across windows. ``n_proc=1``
    on the inner ``colocalize()`` call is deliberate: parallelism is applied at the
    window level instead (via the caller's ``Parallel(n_jobs=n_proc)``), so nesting
    another ``n_proc``-way split inside each task would oversubscribe (n_proc workers
    each spawning their own n_proc sub-workers).

    Always requests ``r_to_z=True`` from the inner ``colocalize()`` call (a no-op for
    every method except pearson/spearman/partialpearson/partialspearman, see
    ``core/colocalize.py``'s ``_get_colocalize_fun``) so the observed statistic and the
    null draws (transformed the same way, see :func:`local_colocalization`) are always
    computed and compared/pooled on the variance-stabilized Fisher-z scale internally,
    regardless of the caller-facing ``r_to_z`` display setting -- that setting only
    controls whether :func:`local_colocalization` converts the *returned* ``"stat"``
    back to raw rho at the very end, never the internal null/pooling computation."""
    Z_sub = Z_full.iloc[:, nb] if Z_full is not None else None
    # rank/regress_z/zy_matched passed explicitly (not left to colocalize()'s own
    # defaults/_get_last fallback) -- otherwise this fresh per-window call could
    # silently diverge from the *stored* method's resolved settings, e.g.
    # colocalize(method="pearson", rank=True) would resolve back to rank=False here
    # without this, since colocalize() deliberately never inherits `rank` across
    # calls (see its own staleness-bug comment) and regress_z/zy_matched via
    # _get_last would pick up whatever colocalize() call happened to run last,
    # not necessarily this method's own stored settings.
    r = nsp.colocalize(X=X.iloc[:, nb], Y=Y.iloc[:, nb], Z=Z_sub, method=method,
                       rank=rank, regress_z=regress_z, zy_matched=zy_matched,
                       store=False, verbose=False, r_to_z=True, n_proc=1,
                       force_dict=True)
    r_stat = r[stat]  # (n_Y, n_X_or_1)
    return r_stat.to_numpy(), r_stat.columns


def _local_to_result_dict(arr, x_labels, y_labels, columns, dtype, is_joint):
    """(n_Y, n_X_or_1, n_parcels) array -> dict of per-X-map DataFrames (or a single
    DataFrame for joint methods) -- the shared result-shape convention for
    :func:`local_colocalization`'s ``"stat"``/``"p"``/``"p_corr"`` outputs, used both
    for the initial (unpooled) build and again after pooling collapses the Y axis."""
    out = {
        x_lab: pd.DataFrame(arr[:, j, :], index=y_labels, columns=columns, dtype=dtype)
        for j, x_lab in enumerate(x_labels)
    }
    if is_joint and len(out) == 1:
        return next(iter(out.values()))
    return out


def local_colocalization(nsp, k=None, radius=None, fwhm_mm=None, dist_mat=None,
                         method=None, X_reduction=None, Y_transform=None, xsea=None,
                         X=None, Y=None, Z=None,
                         null=None, pooled=None, r_to_z=None, mc_method="step_maxT", mc_alpha=0.05,
                         n_proc=None, verbose=None, dtype=None):
    """
    "Searchlight" local colocalization: restrict :meth:`NiSpace.colocalize` to each
    region's k-nearest (or radius-based) spatial neighborhood and recompute the same
    correlation there, giving a subject/map x region output instead of
    ``colocalize()``'s usual subject/map x 1. Conceptually just ``colocalize()`` run on
    masked column subsets, looped over all regions.

    Works on **any** fitted ``nsp`` that has already run ``colocalize()`` -- whether that
    used raw Y or a :meth:`NiSpace.transform_y` group-contrast map, exactly the same way
    ``colocalize()`` itself transparently picks up the last ``transform_y()`` result via
    ``Y_transform=None``. There is deliberately no separate "group" variant of this
    function: if :meth:`NiSpace.permute` was also run (``what="maps"`` -- a spatial
    spin/moran null -- or ``what="groups"`` -- a group-label-shuffle null), its stored
    null is auto-detected and reused (re-sliced per window, not regenerated) to produce
    per-region p-values; if no ``permute()`` was run, only the observed local statistic
    is returned. Which pathway is "last used" follows the same ``nsp._last_settings``
    convention :meth:`NiSpace.plot`/``get_p_values()`` already use.

    Parameters
    ----------
    nsp : NiSpace
        A fitted NiSpace object that has already run ``colocalize()``.
    k : int, optional
        Number of nearest neighbors per region (including the region itself). Exactly
        one of ``k``/``radius``/``fwhm_mm`` must be given.
    radius : float, optional
        Distance threshold (same units as ``dist_mat``) for a region's neighborhood,
        as an alternative to a fixed ``k``.
    fwhm_mm : float, optional
        Alternative to ``k``/``radius``: a tapered Gaussian-kernel searchlight instead
        of a hard window. Every region is weighted by ``exp(-d**2 / (2*sigma**2))``
        (``sigma = fwhm_mm / 2.3548...``) based on distance from the seed region and
        contributes to *every* seed's correlation -- there is no hard cutoff, so the
        number of regions used is always the full parcellation (unlike ``k``/``radius``,
        where it shrinks/grows the window). This does **not** mean the *effective*
        degrees of freedom are constant across ``fwhm_mm`` values -- a narrow kernel
        still concentrates weight on few regions, it's just not a hard cutoff; think of
        ``fwhm_mm`` as a smoother, not a df-free, version of ``radius``. Restricted to
        ``method in {"pearson", "spearman", "partialpearson", "partialspearman"}``
        (raises ``ValueError`` otherwise) -- these are the only methods with a
        closed-form weighted correlation; this restriction is permanent, not a
        to-be-extended limitation. Not supported with ``xsea=True`` (raises
        ``NotImplementedError``). No minimum-``k``-style guardrail is needed (unlike
        the joint-model methods under ``k``/``radius``): these 4 methods always fit one
        predictor at a time.
    dist_mat : array, optional
        Precomputed ``(n_parcels, n_parcels)`` distance matrix. Defaults to
        ``nsp._get_dist_mat(dist_mat_type="cv")`` -- the same combined-hemisphere
        euclidean matrix already used by ``colocalize()``'s lasso/ridge/elasticnet
        spatial cross-validation.
    method, X_reduction, Y_transform, xsea, X, Y, Z : see ``NiSpace.colocalize()``.
        ``lasso``/``ridge``/``elasticnet`` are not supported -- their spatial CV-fold
        split is sized to the full parcellation and would misalign against a
        region-window subset (same reason :func:`regional_influence` excludes them).
        ``fwhm_mm`` further restricts this to 4 methods, see above.
    null : bool, optional
        Compute per-region p-values from the last ``permute()`` call's stored null.
        Defaults to auto-detect: True if a ``permute()`` result exists, False otherwise.
        Explicitly requesting ``True`` without a prior ``permute()`` call raises. Not
        supported with XSEA (raises ``NotImplementedError``); use ``null=False`` for the
        observed local statistic alone in that case. Supported for every other method
        NiSpace supports (except lasso/ridge/elasticnet, excluded above) via one of two
        paths, chosen automatically:

        - **Fast path** (``pearson``/``spearman``/``partialpearson``/``partialspearman``):
          one vectorized correlation call per window covering every null draw at once.
        - **Slow path** (``mi``/``slr``/``mlr``/``dominance``/``pls``/``pcr``): no batched
          closed-form exists, so this reuses ``colocalize()``'s own per-method closure
          directly on each null draw -- correct, but can be slow for expensive methods
          (e.g. ``dominance``, which fits every predictor subset) at large ``n_perm``.

        Both paths handle NaN values (e.g. background/dropped parcels) and an active
        Z-regression (``regress_z``) correctly, resolved the same way ``colocalize()``
        itself resolved them for the stored result (rank/Z-regression applied per window,
        matching the window-local preprocessing the observed statistic already gets).

        Picked up (via ``nsp._last_settings["perm"]``, i.e. whichever `permute()` call
        was run most recently) only for single-sided, non-combined permutations:
        ``permute(what="maps")`` (``maps_which="X"`` or ``"Y"``, not both at once) and
        ``permute(what="groups")``. Every other mode/combination (``"sets"``,
        ``"pairs"``, ``maps_which=["X","Y"]``, or any combined ``what=`` like
        ``["groups","maps"]``) raises ``NotImplementedError``.
    pooled : {None, False, True, "mean", "median"}, default None
        Only meaningful when a null is computed (ignored otherwise) and only ever
        affects **p-values**, never the observed statistic itself -- ``"stat"`` always
        keeps its full ``n_Y`` rows regardless of ``pooled`` (matching
        :meth:`NiSpace.get_colocalizations`, which stays unpooled no matter what
        ``pooled_p`` a later ``permute()`` call used -- ``permute()``'s own
        ``_colocs_obs`` is a fresh array copy, never written back to the stored
        result). When active, a *copy* of the observed statistic and every null draw
        are pooled across the Y axis (mean/median) purely for the p comparison --
        not a post-hoc average of already-computed per-Y-row p-values, mirroring
        exactly how :meth:`NiSpace.permute`'s own ``pooled_p`` resolves p (see
        ``core/permute.py``'s ``_resolve_permute_mode_settings``). ``None`` defaults
        to whatever ``pooled_p`` was last resolved to elsewhere in the pipeline
        (``nsp._last_settings["pooled_p"]``); ``True`` is treated as ``"mean"``. A
        no-op when the Y axis only has one row to begin with (e.g. a single
        ``transform_y()`` group-contrast map), same as ``permute()``.

        For ``permute(what="groups")`` specifically, pooling is **not a free choice** --
        it's always forced to ``"mean"`` (a group-label-shuffle null answers one
        aggregate question; per-row nulls under it are either degenerate for paired
        transforms, or meaningless for unpaired multi-row ones -- same rationale
        ``permute()`` itself already enforces). Passing an incompatible ``pooled``
        here with an auto-detected/explicit ``groups`` null logs a warning and is
        overridden, exactly like passing an incompatible ``pooled_p`` to ``permute()``
        would be. For ``permute(what="maps")`` nulls, ``pooled`` remains a free,
        meaningful choice.

        When active, ``"p"``/``"p_corr"`` (only) collapse from ``n_Y`` rows to a
        single row indexed ``["mean"]``/``["median"]`` (see ``Returns`` below).
    r_to_z : bool, optional
        Whether the returned ``"stat"`` is Fisher-z-transformed (``numpy.arctanh``),
        matching :meth:`NiSpace.colocalize`'s own ``r_to_z`` default. Only meaningful
        for ``method in {"pearson", "spearman", "partialpearson", "partialspearman"}``
        (the 4 methods whose stat is ``"rho"``) -- a no-op for every other method, same
        as ``colocalize()`` itself. ``None`` (default) picks up whatever ``r_to_z`` the
        *original* ``colocalize()`` call for this method used (from
        ``nsp._coloc_kwargs_by_method``), so ``"stat"`` matches
        :meth:`NiSpace.get_colocalizations`'s scale by default.

        This setting only affects how ``"stat"`` is *expressed* -- the null comparison
        and any ``pooled`` reduction (mean/median of the observed stat and every null
        draw across the Y axis) always happen internally on the Fisher-z scale for
        these 4 methods regardless of ``r_to_z``, since averaging/medianing raw
        (bounded, non-additive) correlation coefficients is not statistically valid;
        this mirrors how :meth:`NiSpace.permute`'s own ``pooled_p`` pools whatever
        ``colocalize()`` already stored (which is Fisher-z by ``r_to_z=True`` default).
    mc_method : {"step_maxT", "maxT", None}, default "step_maxT"
        Multiple-comparisons correction across regions, applied separately per X map
        (not pooled across X maps) -- same convention as :func:`regional_influence`/
        ``correlate_within_region()``. Raw (uncorrected) p-values are *always* computed
        and returned when a null exists (see ``Returns`` below); ``mc_method`` only
        controls whether a corrected p is *additionally* computed alongside them --
        ``None`` skips the correction step (``"p_corr"`` comes back ``None``).
    mc_alpha : float, default 0.05
        Only affects the (unused-here) reject mask threshold inside the correction
        functions; p-values themselves don't depend on it.
    n_proc : int, optional
        Parallelizes the **null computation** across windows (both the fast and slow
        path) -- this is where the real cost is for a heavy run (large ``n_perm`` and/or
        an expensive method). The observed-statistic loop is deliberately left
        sequential regardless of ``n_proc``: each window's ``colocalize()`` call is
        already fast, and parallelizing it would mean pickling/shipping the whole
        (potentially large) ``nsp`` object to worker processes for every task --
        measured to cost far more than it saves. Defaults to ``nsp._n_proc``.

    Returns
    -------
    dict
        Always the same fixed keys, matching :meth:`NiSpace.get_within_region_correlations`'s
        return shape (the one other place in NiSpace that carries raw and corrected p
        together):

        - ``"stat_type"`` : str -- the colocalization stat name (e.g. ``"rho"``, ``"r2"``).
        - ``"mc_method"`` : str or None -- the correction actually applied to ``"p_corr"``
          (``None`` if ``null=False``, or if ``mc_method=None`` was passed).
        - ``"pooled"`` : {False, "mean", "median"} -- the ``pooled`` setting actually
          applied to ``"p"``/``"p_corr"`` (always ``False`` when ``null=False``, or when
          the Y axis had only one row to begin with). Never affects ``"stat"``.
        - ``"r_to_z"`` : bool -- the ``r_to_z`` setting actually applied to ``"stat"``
          (always ``False`` for methods where it doesn't apply, see ``r_to_z`` above).
        - ``"stat"`` : DataFrame or dict of DataFrame -- the local statistic, shape
          (n_Y x n_parcels) **always**, regardless of ``pooled`` -- the observed
          statistic is never pooled, only compared against a pooled null (see
          ``pooled`` above). A dict keyed by X-map label for per-predictor methods
          (pearson/spearman/partialpearson/partialspearman/mi/slr), a single DataFrame
          for joint-model methods (mlr/dominance/pls/pcr).
        - ``"p"`` : same column/key shape as ``"stat"``, or None -- raw uncorrected
          p-values, shape (n_Y x n_parcels), or (1 x n_parcels) indexed
          ``["mean"]``/``["median"]`` when ``"pooled"`` is active. ``None`` only when
          ``null=False`` (no permutation available).
        - ``"p_corr"`` : same shape as ``"p"``, or None -- ``mc_method``-corrected
          p-values. ``None`` when ``null=False`` or ``mc_method=None``.
        - ``"settings"`` : dict -- ``{"k": k}`` in k-NN mode, ``{"radius": radius,
          "n_neighbors_min": ..., "n_neighbors_median": ..., "n_neighbors_max": ...}``
          in radius mode (window size varies by region there, unlike fixed-k), or
          ``{"fwhm_mm": fwhm_mm, "sigma_mm": sigma_mm}`` in Gaussian-kernel mode.
    """
    verbose = set_log(lgr, nsp._verbose if verbose is None else verbose)
    lgr.info("*** diagnostics.local_colocalization() - Computing local colocalization. ***")

    nsp._check_fit()
    n_proc = nsp._n_proc if n_proc is None else n_proc
    dtype = nsp._dtype if dtype is None else dtype

    if sum(x is not None for x in (k, radius, fwhm_mm)) != 1:
        lgr.critical_raise("Specify exactly one of k, radius, or fwhm_mm.", ValueError)
    gaussian_mode = fwhm_mm is not None

    method, X_reduction, Y_transform, xsea = nsp._get_last(
        method=method,
        X_reduction=X_reduction,
        Y_transform=Y_transform,
        xsea=xsea,
    )
    if method is None:
        lgr.critical_raise("No colocalization method defined! Run colocalize() first.",
                           ValueError)
    if method in _LOCAL_UNSUPPORTED_METHODS:
        lgr.critical_raise(f"local_colocalization() does not support method '{method}' -- its "
                           "spatial CV-fold split is sized to the full parcellation and would "
                           "misalign against a region-window subset.", ValueError)
    if gaussian_mode and method not in _LOCAL_GAUSSIAN_METHODS:
        lgr.critical_raise(f"fwhm_mm is only supported for method in "
                           f"{sorted(_LOCAL_GAUSSIAN_METHODS)} -- these are the only methods "
                           "with a closed-form weighted correlation; this restriction is "
                           "permanent, not planned to be extended. Use k= or radius= for "
                           f"method='{method}'.", ValueError)
    if gaussian_mode and xsea:
        lgr.critical_raise("local_colocalization()'s fwhm_mm (Gaussian-kernel) mode does not "
                           "support xsea=True.", NotImplementedError)
    if method not in nsp._colocs_fun or method not in nsp._coloc_kwargs_by_method:
        lgr.critical_raise(f"No stored colocalize() results for method '{method}'! "
                           "Did you run colocalize() with this method first?", KeyError)

    X, Y, X_arr_pre, Y_arr_pre, _, coloc_kwargs = _resolve_coloc_data(
        nsp, method=method, X=X, Y=Y, Z=Z, X_reduction=X_reduction, Y_transform=Y_transform,
        xsea=xsea, zy_matched=False, dtype=dtype, verbose=verbose,
    )
    Z_full = nsp._Z if Z is None else Z

    n_parcels = X.shape[1]
    n_predictors = X.shape[0]
    is_joint = method in _LOCAL_JOINT_METHODS

    # r_to_z: only meaningful for the 4 correlation-based methods (their stat is "rho",
    # a bounded, non-additive scale that colocalize() Fisher-z-transforms by default --
    # see core/colocalize.py's _get_colocalize_fun). No-op for every other method (mi/
    # slr/mlr/dominance/pls/pcr never go through rho_to_z regardless of this flag, same
    # as colocalize() itself). Default (None) mirrors whatever r_to_z the *original*
    # colocalize() call for this method actually used (coloc_kwargs, same resolution
    # regional_influence() already uses at coloc_kwargs.get("r_to_z", True)) -- so by
    # default "stat" here is on the same scale as get_colocalizations()'s stored result.
    # This only controls what "stat" is expressed as; the null comparison and any
    # pooled_p reduction always happen internally on the Fisher-z scale for these 4
    # methods regardless of this setting (mean/median of raw bounded rho across the Y
    # axis is not a valid operation -- see the pooling comment below), then are
    # converted to raw rho only at the very end if r_to_z resolves to False.
    rho_based = method in _LOCAL_NULL_FAST_METHODS
    if rho_based:
        r_to_z = coloc_kwargs.get("r_to_z", True) if r_to_z is None else bool(r_to_z)
    else:
        r_to_z = False

    ## distance matrix + neighbor windows / Gaussian kernel weights
    if dist_mat is None:
        # _quiet() here isn't just about noise: fetch_template(verbose=False), several
        # calls deep inside _get_dist_mat() -> get_distance_matrix() -> ...
        # (datasets.py/nulls.py), unconditionally calls set_log(lgr, False) -- and
        # set_log() is a no-op ONLY while _quiet_ctx.active, otherwise it directly
        # lowers the shared "nispace" ROOT logger to WARNING with nothing to ever
        # restore it. Outside a _quiet() context (this call site, unguarded, is exactly
        # that case) that permanently silences every lgr.info() call for the rest of the
        # process -- including this function's own closing summary below -- not just a
        # cosmetic nesting-noise concern. Matches _resolve_coloc_data()'s existing
        # convention of wrapping nested NiSpace calls it doesn't own in _quiet().
        with _quiet():
            dist_mat = nsp._get_dist_mat(dist_mat_type="cv", n_proc=n_proc)
    dist_mat = np.asarray(dist_mat)
    if dist_mat.shape[0] != n_parcels or dist_mat.shape[1] != n_parcels:
        lgr.critical_raise(f"dist_mat shape {dist_mat.shape} does not match the "
                           f"n_parcels={n_parcels} of the resolved X/Y data.", ValueError)

    if gaussian_mode:
        sigma_mm = fwhm_mm * _FWHM_TO_SIGMA
        W = np.exp(-(dist_mat ** 2) / (2.0 * sigma_mm ** 2)).astype(dtype)
        settings = {"fwhm_mm": fwhm_mm, "sigma_mm": sigma_mm}
    elif k is not None:
        if is_joint and k <= 2 * n_predictors:
            lgr.critical_raise(f"k={k} is too small for method '{method}' with {n_predictors} "
                               f"predictors -- need k > {2 * n_predictors} (2x predictors) for "
                               "a well-determined per-window fit.", ValueError)
        windows = list(np.argsort(dist_mat, axis=1)[:, :k])
        settings = {"k": k}
    else:
        windows = [np.flatnonzero(dist_mat[i] <= radius) for i in range(n_parcels)]
        if is_joint:
            too_small = [i for i, w in enumerate(windows) if len(w) <= 2 * n_predictors]
            if too_small:
                lgr.critical_raise(f"radius={radius} gives fewer than {2 * n_predictors + 1} "
                                   f"neighbors for {len(too_small)} region(s) -- too small for "
                                   f"method '{method}' with {n_predictors} predictors.", ValueError)
        # radius windows vary in size per region -- report the effective n so the user can
        # judge whether the correlation d.f. is reasonably stable across the brain
        _n_neighbors = np.array([len(w) for w in windows])
        settings = {
            "radius": radius,
            "n_neighbors_min": int(_n_neighbors.min()),
            "n_neighbors_median": float(np.median(_n_neighbors)),
            "n_neighbors_max": int(_n_neighbors.max()),
        }

    ## null setup: auto-detect from the last permute() call, if any
    perm_mode = nsp._last_settings.get("perm")
    if null is None:
        null = perm_mode is not None
    if null:
        if perm_mode is None:
            lgr.critical_raise("null=True requires permute() to have been run first (found no "
                               "prior permutation on this NiSpace object).", ValueError)
        if xsea:
            lgr.critical_raise("local_colocalization()'s null does not yet support xsea=True.",
                               NotImplementedError)
        if perm_mode == "groups":
            null_maps = nsp._nulls.get("groups_null")
            null_side = "Y"
        elif perm_mode in ("Xmaps", "Ymaps"):
            null_maps = nsp._nulls.get("maps_null")
            null_side = "X" if perm_mode == "Xmaps" else "Y"
        elif perm_mode in ("sets", "pairs") or "sets" in perm_mode or "pairs" in perm_mode:
            lgr.critical_raise(f"local_colocalization()'s null is not supported for "
                               f"permute(what='{perm_mode}') -- 'sets' (XSEA) permutes which "
                               "genes populate each set, and 'pairs' (SPICE) permutes subject "
                               "pairings; neither is a per-parcel spatial map that can be "
                               "re-sliced to a region window. Use permute(what='maps') or "
                               "permute(what='groups') for a local_colocalization() null.",
                               NotImplementedError)
        else:
            lgr.critical_raise(f"local_colocalization()'s null does not yet support the combined "
                               f"permutation mode '{perm_mode}' -- only a single-sided "
                               "permute(what='maps') (not maps_which=['X','Y'] together) or "
                               "permute(what='groups') is supported for now. Run a plain "
                               "permute(what='maps') or permute(what='groups') call instead, "
                               "or pass null=False for the observed local statistic alone.",
                               NotImplementedError)
        if null_maps is None:
            _fp_info = nsp._nulls.get("maps_null_fastpath_info")
            if _fp_info is not None:
                lgr.critical_raise(
                    f"local_colocalization()'s null requires the full null-map cube, but the "
                    f"last permute() call used the map-batched fast path for "
                    f"'{_fp_info['XY']}' maps (n={_fp_info['n_rows']} > "
                    f"maps_batch_size={_fp_info['batch_size']}), which does not cache it. "
                    "Re-run permute(..., maps_batch_size=False) (or a size >= n_rows) if you "
                    "need local_colocalization(null=True) for this data.",
                    KeyError)
            lgr.critical_raise(f"Expected a stored null for permutation mode '{perm_mode}' but "
                               "found none on this NiSpace object.", KeyError)
        target_labels = list(X.index if null_side == "X" else Y.index)
        null_maps = null_maps.subset(target_labels)
        n_perm = null_maps.n_perm
        lgr.info(f"local_colocalization(): auto-picked up '{perm_mode}' null ({n_perm} perms, "
                f"nulling {null_side}) from the last permute() call for p-values.")

        # pooled resolution: same source of truth (and same forcing for what="groups") as
        # permute() itself -- default picks up whatever pooled_p was last resolved to
        # (nsp._last_settings["pooled_p"] already reflects permute()'s own forcing, but an
        # explicit `pooled=` override here still needs to be checked against that forcing,
        # exactly like passing an incompatible pooled_p straight to permute() would be)
        if pooled is None:
            pooled = nsp._last_settings.get("pooled_p", False)
        pooled, mode_warning = _resolve_permute_mode_settings([perm_mode], pooled)
        if mode_warning:
            lgr.warning(mode_warning)
        if pooled and pooled not in ("mean", "median"):
            pooled = "mean"
        n_y_rows = Y.shape[0]
        if pooled and n_y_rows <= 1:
            pooled = False

    X_arr = np.asarray(X, dtype=dtype)
    Y_arr = np.asarray(Y, dtype=dtype)
    # resolved verbatim from what colocalize() itself stored -- not re-derived, matching
    # every other diagnostics.py function's convention (see _resolve_coloc_data)
    rank = coloc_kwargs.get("rank", "spearman" in method)
    regress_z = coloc_kwargs.get("regress_z", "")
    zy_matched = coloc_kwargs.get("zy_matched", False)

    ## observed local statistic -- one colocalize() call per window, reusing colocalize()'s
    ## own method dispatch/rank/Z-regression handling rather than duplicating it (validated
    ## approach, see _explore/searchlight_colocalize_test.py). Deliberately NOT parallelized
    ## across windows here, unlike the null paths below: each window's colocalize() call is
    ## already fast (measured ~0.3s total for 200 windows on real 200-parcel data), but
    ## every parallel task would need `nsp` itself pickled and shipped to a worker process
    ## (loky, joblib's default backend) -- for a real, large NiSpace object (~100+MB) that
    ## overhead swamps the actual work by ~40x (measured), making this the one place in
    ## local_colocalization() where parallelizing is a net loss rather than a win.
    ## _silence_repeated_calls() silences the repeated per-window colocalize() logging
    ## (plain _quiet() isn't enough once a prior get_x()/get_y()/get_colocalizations() call
    ## has pinned "nispace.api"'s own level -- see that helper's docstring), leaving a
    ## single clean tqdm bar.
    stat = _get_coloc_stats(method, drop_optional=True)[0]
    if gaussian_mode:
        # single vectorized call across all seeds at once -- no per-window colocalize()
        # calls needed here (nothing to re-rank locally, see _local_gaussian_observed's
        # docstring), so none of the observed-loop pickling-overhead concerns above apply.
        # Gaussian mode is always one of the 4 rho-based methods (restricted above), so
        # the Fisher-z transform is applied unconditionally here, same as the windowed
        # path gets it unconditionally from _local_observed_window's inner colocalize()
        # call -- see r_to_z resolution above for why this is unconditional internally.
        obs_arr = _local_gaussian_observed(W, X_arr_pre, Y_arr_pre)  # (n_Y, n_X, n_parcels)
        obs_arr = rho_to_z(obs_arr)
        x_labels = X.index
    else:
        with _silence_repeated_calls():
            obs_results = [
                _local_observed_window(nsp, windows[i], X, Y, Z_full, method, rank, regress_z,
                                       zy_matched, stat)
                for i in tqdm(range(n_parcels),
                             desc=f"Local colocalization ({method})",
                             disable=not verbose, mininterval=1.0)
            ]
        x_labels = obs_results[0][1]
        obs_arr = np.stack([r[0] for r in obs_results], axis=-1)  # (n_Y, n_X_or_1, n_parcels)
    y_labels = Y.index

    # "stat" (the returned/displayed statistic) honors r_to_z; obs_arr itself stays on
    # the Fisher-z scale (when rho_based) for every computation below it, see r_to_z
    # resolution above.
    stat_arr = z_to_rho(obs_arr) if (rho_based and not r_to_z) else obs_arr
    result = _local_to_result_dict(stat_arr, x_labels, y_labels, X.columns, dtype, is_joint)

    if not null:
        lgr.info(f"Returning local_colocalization results: \n"
                 f"{print_arg_pairs(method=method, **settings, null=False, r_to_z=r_to_z, xsea=xsea, X_reduction=X_reduction, Y_transform=Y_transform)}")
        return {"stat_type": stat, "mc_method": None, "pooled": False, "r_to_z": r_to_z,
               "stat": result, "p": None, "p_corr": None, "settings": settings}

    if mc_method not in ("step_maxT", "maxT", None):
        lgr.critical_raise(f"mc_method='{mc_method}' not supported; use 'step_maxT', 'maxT', "
                           "or None.", ValueError)

    ## null: recompute the same correlation for every precomputed null draw, re-slicing/
    ## reweighting the already-generated null maps rather than regenerating them.
    ## Fast (batched, vectorized) path for pearson/spearman/partialpearson/partialspearman;
    ## slow (per-draw closure reuse) path for mi/slr/mlr/dominance/pls/pcr; Gaussian-kernel
    ## path for fwhm_mm mode (always one of the 4 fast-path methods, see the fwhm_mm
    ## restriction above) -- see _local_null_fast()/_local_null_slow()/_local_null_gaussian()
    ## docstrings. None of the three paths call nsp.colocalize() (they reuse
    ## _rank_regress()/nsp._colocs_fun[method] directly), so plain _quiet() is enough here --
    ## no "nispace.api" logger involved, unlike the observed-stat loop above (windowed modes).
    null_data = null_maps.data  # (n_null_maps, n_perm, n_parcels)
    with _quiet():
        if gaussian_mode:
            # global (not per-window) rank/Z-regression of the null draws -- matches
            # X_arr_pre/Y_arr_pre's own global preprocessing above: there's no per-window
            # subset to rank locally when every seed uses all n_parcels, so this happens
            # once here rather than once per seed (cheaper than the k/radius fast path's
            # equivalent, which repeats it per window)
            regress_x = bool(regress_z) and "x" in regress_z
            regress_y = bool(regress_z) and "y" in regress_z
            Z_arr = None
            if regress_z:
                Z_arr = np.asarray(Z_full, dtype=dtype)
                if rank:
                    Z_arr = rank2d(Z_arr.T).T
            null_list = list(null_data.transpose(1, 0, 2))  # n_perm arrays, (n_null_maps, n_parcels)
            null_pre_list = _rank_regress(
                arr=null_list, rank=rank, regress=(regress_x if null_side == "X" else regress_y),
                z=Z_arr, zy_matched=zy_matched, n_proc=n_proc, verbose=False,
            )
            null_pre = np.stack(null_pre_list, axis=1)  # (n_null_maps, n_perm, n_parcels)
            null_arr = _local_null_gaussian(
                W, X_arr_pre, Y_arr_pre, null_pre, null_side, n_proc, verbose,
            )
            null_arr = rho_to_z(null_arr)  # gaussian mode is always rho-based, see above
        elif method in _LOCAL_NULL_FAST_METHODS:
            null_arr = _local_null_fast(
                windows, X_arr, Y_arr, Z_full, null_data, null_side,
                rank, regress_z, zy_matched, dtype, n_proc, verbose,
            )
            null_arr = rho_to_z(null_arr)  # fast path only ever runs for rho-based methods
        else:
            null_arr = _local_null_slow(
                nsp, method, windows, X_arr, Y_arr, Z_full, null_data, null_side,
                rank, regress_z, zy_matched, is_joint, dtype, n_proc, verbose,
            )

    # pooled_p: pools the OBSERVED stat AND every null draw across the Y axis *before*
    # comparing them for the p computation ONLY -- mirrors permute()'s own par_fun exactly
    # (api.py:2622 shows _colocs_obs there is already a fresh numpy-array copy from
    # get_colocalizations(), so pooling it in-place never touches what get_colocalizations()
    # itself reports afterward). obs_arr/null_arr are already on the Fisher-z scale here
    # for rho-based methods (unconditionally, see r_to_z resolution above) -- averaging/
    # medianing raw bounded correlations across the Y axis would be invalid, so this
    # pooling step is never done on raw rho regardless of the caller-facing r_to_z
    # setting, which only affects "stat"/display (already applied above, decoupled from
    # obs_arr itself). "stat" (result, built above from a separate stat_arr) is therefore
    # *never* touched here and always keeps its full n_Y rows, regardless of pooled --
    # only "p"/"p_corr" collapse to the pooled 1-row shape. Averaging already-computed
    # per-Y-row p-values afterward would not be equivalent to this and would silently
    # diverge from what get_p_values() reports for the same permute() call.
    p_labels = y_labels
    p_obs_arr = obs_arr
    p_null_arr = null_arr
    if pooled:
        reducer = np.nanmedian if pooled == "median" else np.nanmean
        p_obs_arr = reducer(obs_arr, axis=0, keepdims=True)
        p_null_arr = reducer(null_arr, axis=0, keepdims=True)
        p_labels = [pooled]

    # raw (uncorrected) p is always computed and returned -- mc_method only controls
    # whether a *second*, corrected p is additionally computed alongside it
    p_floor = max(np.finfo(float).eps, 1.0 / n_perm)
    p_raw_arr = np.mean(np.abs(p_null_arr) >= np.abs(p_obs_arr[:, :, np.newaxis, :]), axis=2)
    p_raw_arr = np.clip(p_raw_arr, p_floor, 1.0 - p_floor)

    p_raw_result = _local_to_result_dict(p_raw_arr, x_labels, p_labels, X.columns, dtype, is_joint)

    p_corr_result = None
    if mc_method is not None:
        corr_fun = step_maxT_correction if mc_method == "step_maxT" else maxT_correction
        p_corr_result = {}
        p_obs_result = _local_to_result_dict(p_obs_arr, x_labels, p_labels, X.columns, dtype, is_joint)
        for j, x_lab in enumerate(x_labels):
            obs_df = p_obs_result[x_lab] if isinstance(p_obs_result, dict) else p_obs_result
            null_colocs = [{"rho": p_null_arr[:, j, p, :]} for p in range(n_perm)]
            p_corr, _ = corr_fun(obs_df, null_colocs, stat="rho", tail="two",
                                 how="r", alpha=mc_alpha, dtype=dtype)
            p_corr_result[x_lab] = p_corr
        if is_joint and len(p_corr_result) == 1:
            p_corr_result = next(iter(p_corr_result.values()))

    lgr.info(f"Returning local_colocalization results: \n"
             f"{print_arg_pairs(method=method, **settings, null=perm_mode, mc_method=mc_method, pooled=pooled or False, r_to_z=r_to_z, xsea=xsea, X_reduction=X_reduction, Y_transform=Y_transform)}")

    return {"stat_type": stat, "mc_method": mc_method, "pooled": pooled or False, "r_to_z": r_to_z,
           "stat": result, "p": p_raw_result, "p_corr": p_corr_result, "settings": settings}


# SET COMPARISON ====================================================================================

def _rowwise_corr(A, B):
    """Pearson r between matching rows of A and B, both (n, n_parcels)."""
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    return (A * B).sum(axis=1) / np.sqrt((A * A).sum(axis=1) * (B * B).sum(axis=1))


def _remove_ref_map(M, ref_unit):
    """Residuals of each row of M after OLS on [intercept, ref map]. ``ref_unit`` is the
    centered, unit-norm reference map, so this is just centering + removing the projection."""
    Mc = M - M.mean(axis=1, keepdims=True)
    return Mc - np.outer(Mc @ ref_unit, ref_unit)


def _compare_sets_pair_null(bg, w_a, w_b, n_shared, n_perm, ref_unit, seed):
    """Random-pair null for one pair of sets. Each draw takes ``n_a + n_b - n_shared``
    distinct background rows: the first ``n_shared`` go into both random sets, the rest
    are split into the A-only and B-only parts. ``w_a``/``w_b`` are the observed weights
    ordered [shared genes, set-only genes], so shared genes keep their weight pairing."""
    rng = np.random.default_rng(seed)
    n_bg = bg.shape[0]
    n_a, n_b = len(w_a), len(w_b)
    n_draw = n_a + n_b - n_shared
    idx = np.stack([rng.choice(n_bg, size=n_draw, replace=False) for _ in range(n_perm)])
    idx_a = idx[:, :n_a]
    idx_b = np.concatenate([idx[:, :n_shared], idx[:, n_a:]], axis=1)

    # (weighted) set means as one sparse (n_perm, n_bg) @ (n_bg, n_parcels) product
    def _set_maps(idc, w):
        rows = np.repeat(np.arange(n_perm), idc.shape[1])
        S = csr_matrix((np.tile(w / w.sum(), n_perm), (rows, idc.ravel())), shape=(n_perm, n_bg))
        return np.asarray(S @ bg)

    M_a, M_b = _set_maps(idx_a, w_a), _set_maps(idx_b, w_b)
    r = _rowwise_corr(M_a, M_b)
    r_resid = None
    if ref_unit is not None:
        r_resid = _rowwise_corr(_remove_ref_map(M_a, ref_unit), _remove_ref_map(M_b, ref_unit))
    return r, r_resid


def compare_sets(X, background, pairs=None, null="overlap", n_perm=1000,
                 weighted=None, standardize=None, rank=False, remove_background_mean=True,
                 p_tails="upper", z_method="robust", return_nulls=False, n_proc=None, seed=None,
                 verbose=None):
    """
    Compare X sets with each other: how many maps (genes) they share, and how similar
    their set-mean maps are compared to random sets of the same sizes.

    Intended for questions like "are my gene sets really different?". In
    transcriptomic data, the mean maps of unrelated gene sets are often highly
    correlated, because every set-mean map carries the dataset's dominant spatial
    pattern. A raw correlation between two set-mean maps is therefore hard to
    interpret on its own; this function compares it to a null of random set pairs
    drawn from ``background`` with the same sizes (and, by default, the same number
    of shared maps). Optionally, it repeats the comparison after regressing the
    background's mean map out of every set-mean map, which removes the shared
    pattern.

    The set-mean map is what mean-aggregated XSEA "sees": for z-scored X rows, the
    mean Pearson ``rho`` of a set's maps with any Y map equals exactly
    ``r(set-mean map, Y) * sd(set-mean map)``. XSEA averages on the Fisher-z scale,
    which stays close to this for the small per-map correlations typical of gene
    data. Two sets with near-identical mean maps therefore give near-identical XSEA
    results for every Y. Use ``rank=True`` for the Spearman
    equivalent. This does not hold for multivariate methods (e.g., ``pcr``), which
    represent a set by its main components rather than its mean map.

    Does not need ``colocalize()`` and does not use Y.

    Parameters
    ----------
    X : NiSpace or pandas.DataFrame
        A fitted NiSpace object with set-structured X (its fitted X is used), or a
        DataFrame with a ``"set"`` index level, a map-name level (``"map"``, else the
        first other level that is not ``"weight"``), and optionally ``"weight"``, e.g.
        from ``fetch_reference(..., collection=...)``.
    background : pandas.DataFrame
        Background pool of maps (rows, indexed by map name) for the random sets, e.g.
        the full dataset from ``fetch_reference(<dataset>, parcellation=...)``.
        Columns must include all X parcels. Required: the union of the sets in X is
        a biased background. Also defines the universe for the hypergeometric test.
    pairs : list of tuple, optional
        ``(set_a, set_b)`` pairs to compare. Default: all unordered pairs of sets in X.
    null : {"overlap", "disjoint"}, default "overlap"
        ``"overlap"``: random pairs share as many maps as the observed pair, so the
        null asks whether the two sets are more similar than their overlap alone
        explains. ``"disjoint"``: random pairs share no maps.
    n_perm : int, default 1000
        Number of random pairs per observed pair.
    weighted : bool, optional
        Weighted set means using the ``"weight"`` index level of X. Defaults to True
        if X has a ``"weight"`` level, else False. Random sets reuse the observed
        weights.
    standardize : bool, optional
        Z-standardize each X and background map across parcels before averaging.
        Defaults to the NiSpace object's X standardization if ``X`` is a NiSpace
        object, else True (the NiSpace default).
    rank : bool, default False
        Rank each map across parcels before standardizing and averaging, i.e., the
        set maps that Spearman-based XSEA aggregates.
    remove_background_mean : bool, default True
        Also compare set-mean maps after regressing out (with intercept) the mean map
        of all background maps.
    p_tails : {"upper", "two", "lower"}, default "upper"
        Tail for the map-similarity p-values. ``"upper"``: are the sets more similar
        than random sets?
    z_method : {"robust", "standard"}, default "robust"
        How to z-score the observed r against its null, as in
        :meth:`NiSpace.normalize_colocalizations`: ``"robust"`` (median/MAD) or
        ``"standard"`` (mean/SD). Both r and null are Fisher-z transformed first.
    return_nulls : bool, default False
        Also return the null distributions.
    n_proc : int, optional
        Number of parallel jobs (over pairs). Defaults to the NiSpace object's setting,
        else 1.
    seed : int, optional
        Random seed. Results do not depend on ``n_proc``.
    verbose : bool, optional

    Returns
    -------
    pandas.DataFrame
        One row per pair, indexed by ``(set_a, set_b)``, with columns:

        - ``n_a``, ``n_b``, ``n_shared``: set sizes and number of shared maps;
        - ``jaccard``: ``n_shared / |A ∪ B|``; ``overlap_coef``: ``n_shared / min(n_a, n_b)``;
        - ``p_overlap``: hypergeometric p for at least ``n_shared`` shared maps, given the
          background (plus any set maps missing from it) as universe;
        - ``r``: Pearson r between the two set-mean maps; ``r_null_median``,
          ``r_null_lo``, ``r_null_hi``: median and 2.5/97.5 percentiles of the
          random-pair null; ``p_r``: permutation p-value; ``z_r``: r z-scored against
          the null (see ``z_method``);
        - ``r_resid``, ``r_resid_null_median``, ``r_resid_null_lo``, ``r_resid_null_hi``,
          ``p_r_resid``, ``z_r_resid``: the same after removing the background mean map
          (only if ``remove_background_mean=True``).

        p-values are not corrected for multiple comparisons.
    dict, optional
        Only if ``return_nulls=True``: ``{(set_a, set_b): {"r": array, "r_resid": array}}``.
    """
    from .api import NiSpace

    ## resolve inputs
    if isinstance(X, NiSpace):
        nsp = X
        verbose = set_log(lgr, nsp._verbose if verbose is None else verbose)
        nsp._check_fit()
        X = nsp._X
        if standardize is None:
            standardize = bool(nsp._zscore) and "x" in nsp._zscore
        if n_proc is None:
            n_proc = nsp._n_proc
    else:
        verbose = set_log(lgr, True if verbose is None else verbose)
        if standardize is None:
            standardize = True
        if n_proc is None:
            n_proc = 1
    lgr.info("*** diagnostics.compare_sets() - Comparing X sets. ***")

    if (not isinstance(X, pd.DataFrame) or not isinstance(X.index, pd.MultiIndex)
            or "set" not in X.index.names):
        lgr.critical_raise("compare_sets() requires X with a MultiIndex containing a 'set' "
                           "level!", ValueError)
    # map names: the "map" level, else the first level that is neither "set" nor "weight"
    map_level_name = "map" if "map" in X.index.names else \
        next((n for n in X.index.names if n not in ("set", "weight")), None)
    if map_level_name is None:
        lgr.critical_raise("compare_sets() requires X with a map-name index level next to "
                           "'set'!", ValueError)
    if not isinstance(background, pd.DataFrame):
        lgr.critical_raise("'background' must be a pandas DataFrame indexed by map name, not "
                           f"{type(background)}!", TypeError)
    if null not in ("overlap", "disjoint"):
        lgr.critical_raise(f"'null' must be 'overlap' or 'disjoint', not '{null}'!", ValueError)
    if weighted is None:
        weighted = "weight" in X.index.names
    elif weighted and "weight" not in X.index.names:
        lgr.warning("weighted=True but X has no 'weight' index level. Using unweighted means.")
        weighted = False

    missing_cols = X.columns.difference(background.columns)
    if len(missing_cols) > 0:
        lgr.critical_raise(f"{len(missing_cols)} X parcels are missing from the background "
                           f"columns, e.g. {list(missing_cols[:3])}!", ValueError)
    background = background.loc[:, X.columns]

    ## NaN handling: drop parcels with NaN in X, then background maps with NaN in the rest
    nan_parcels = X.isnull().any(axis=0)
    if nan_parcels.any():
        lgr.warning(f"Dropping {nan_parcels.sum()} parcels with NaN values in X.")
        X = X.loc[:, ~nan_parcels]
        background = background.loc[:, ~nan_parcels.values]
    nan_bg = background.isnull().any(axis=1)
    if nan_bg.any():
        lgr.warning(f"Dropping {nan_bg.sum()} background maps with NaN values.")
        background = background.loc[~nan_bg]

    ## preprocess maps (identical for X and background)
    def _prep(arr):
        arr = np.array(arr, dtype=np.float64)
        if rank:
            arr = rank2d(arr.T).T
        if standardize:
            arr = zscore_df(arr, along="rows", force_df=False)
        return arr

    X_arr = _prep(X)
    bg_arr = _prep(background)
    bg_names = pd.Index(background.index.get_level_values(-1) if isinstance(background.index, pd.MultiIndex)
                        else background.index)
    if bg_names.duplicated().any():
        lgr.warning("Background contains duplicated map names. Keeping the first of each.")
        keep = ~bg_names.duplicated()
        bg_arr, bg_names = bg_arr[keep], bg_names[keep]
    n_bg = bg_arr.shape[0]

    ref_unit = None
    if remove_background_mean:
        ref = bg_arr.mean(axis=0)
        ref = ref - ref.mean()
        ref_unit = ref / np.linalg.norm(ref)

    ## observed sets: {set: (map names, weights, set-mean map)}
    set_level = X.index.get_level_values("set")
    map_level = X.index.get_level_values(map_level_name)
    weight_level = X.index.get_level_values("weight").astype(float) if weighted else None
    sets = {}
    for set_name in pd.unique(set_level):
        idc = np.flatnonzero(set_level == set_name)
        names = pd.Index(map_level[idc])
        keep = ~names.duplicated()
        idc, names = idc[keep], names[keep]
        w = np.asarray(weight_level[idc]) if weighted else np.ones(len(idc))
        sets[set_name] = {"maps": names, "w": pd.Series(w, index=names),
                          "map": (w @ X_arr[idc]) / w.sum()}
    if weighted and any((s["w"] <= 0).any() for s in sets.values()):
        lgr.critical_raise("compare_sets() needs positive weights for weighted set means!",
                           ValueError)
    all_set_maps = pd.Index(np.unique(np.concatenate([s["maps"] for s in sets.values()])))
    n_missing = len(all_set_maps.difference(bg_names))
    if n_missing > 0:
        lgr.warning(f"{n_missing} set maps are not in the background. They are used for the "
                    "observed set maps and added to the hypergeometric universe, but random "
                    "sets can't draw them.")
    n_universe = len(bg_names.union(all_set_maps))

    ## pairs
    if pairs is None:
        pairs = list(combinations(sets.keys(), 2))
    else:
        pairs = [tuple(p) for p in pairs]
        unknown = {s for p in pairs for s in p} - set(sets.keys())
        if unknown:
            lgr.critical_raise(f"Unknown sets in 'pairs': {sorted(unknown)}", ValueError)

    ## observed statistics + inputs for the null
    rows, null_args = [], []
    for a, b in pairs:
        A, B = sets[a], sets[b]
        shared = A["maps"].intersection(B["maps"])
        n_a, n_b, n_sh = len(A["maps"]), len(B["maps"]), len(shared)
        if n_a + n_b - (n_sh if null == "overlap" else 0) > n_bg:
            lgr.critical_raise(f"Sets '{a}' and '{b}' together are larger than the background "
                               f"({n_bg} maps)!", ValueError)
        row = {
            "set_a": a, "set_b": b, "n_a": n_a, "n_b": n_b, "n_shared": n_sh,
            "jaccard": n_sh / (n_a + n_b - n_sh),
            "overlap_coef": n_sh / min(n_a, n_b),
            "p_overlap": hypergeom.sf(n_sh - 1, n_universe, n_a, n_b),
            "r": _rowwise_corr(A["map"][None], B["map"][None])[0],
        }
        if remove_background_mean:
            row["r_resid"] = _rowwise_corr(_remove_ref_map(A["map"][None], ref_unit),
                                           _remove_ref_map(B["map"][None], ref_unit))[0]
        rows.append(row)
        if null == "overlap":
            w_a = np.r_[A["w"][shared].values, A["w"].drop(shared).values]
            w_b = np.r_[B["w"][shared].values, B["w"].drop(shared).values]
            null_args.append((w_a, w_b, n_sh))
        else:
            null_args.append((A["w"].values, B["w"].values, 0))

    ## random-pair nulls, one independent seed per pair
    # TODO (postponed 2026-10-19): stratified/matched random sets (shared sampler with the
    # permute(what="sets") TODO in api.py) and the PCA position of sets across a large pool of
    # set-mean maps. See _explore/compare_gene_sets/HANDOFF.md.
    seeds = np.random.SeedSequence(seed).spawn(len(pairs))
    nulls = Parallel(n_jobs=n_proc)(
        delayed(_compare_sets_pair_null)(bg_arr, w_a, w_b, n_sh, n_perm, ref_unit, s)
        for (w_a, w_b, n_sh), s in tqdm(
            zip(null_args, seeds), total=len(pairs),
            desc=f"Random set pairs ({n_proc} proc)", disable=not verbose,
        )
    )

    ## summarize
    score_fn = rzscore_nan if z_method == "robust" else zscore_nan
    for row, (r_null, r_resid_null) in zip(rows, nulls):
        for stat, null_arr in (("r", r_null), ("r_resid", r_resid_null)):
            if null_arr is None:
                continue
            row[f"{stat}_null_median"] = np.median(null_arr)
            row[f"{stat}_null_lo"], row[f"{stat}_null_hi"] = np.percentile(null_arr, [2.5, 97.5])
            row[f"p_{stat}"] = null_to_p(row[stat], null_arr, tail=p_tails)
            # same as NiSpace.normalize_colocalizations(): z-score of Fisher-z rho vs. null
            row[f"z_{stat}"] = score_fn(rho_to_z(np.atleast_2d(row[stat])),
                                        rho_to_z(null_arr)[:, None])[0, 0]
    cols = ["n_a", "n_b", "n_shared", "jaccard", "overlap_coef", "p_overlap",
            "r", "r_null_median", "r_null_lo", "r_null_hi", "p_r", "z_r"]
    if remove_background_mean:
        cols += ["r_resid", "r_resid_null_median", "r_resid_null_lo", "r_resid_null_hi",
                 "p_r_resid", "z_r_resid"]
    out = pd.DataFrame(rows).set_index(["set_a", "set_b"])[cols]

    lgr.info(f"Returning compare_sets results: \n"
             f"{print_arg_pairs(n_pairs=len(pairs), null=null, n_perm=n_perm, weighted=weighted, standardize=standardize, rank=rank, remove_background_mean=remove_background_mean, n_background=n_bg)}")

    if return_nulls:
        return out, {p: {"r": r_null, "r_resid": r_resid_null}
                     for p, (r_null, r_resid_null) in zip(pairs, nulls)}
    return out


# SHARED AXES =======================================================================================

_MORAN_NULL_METHODS = frozenset({"moran", "variomoran"})
_MAX_FLAG_LOG_LINES = 20


def _corr_rows(A, B):
    """Pearson r between every row of A and every row of B -> (n_A, n_B); NaN for constant rows."""
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        A = A / np.linalg.norm(A, axis=1, keepdims=True)
        B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A @ B.T


def _resid_rows(M, A):
    """OLS residuals of every row of M on the rows of A plus an intercept."""
    D = np.column_stack([np.ones(M.shape[1]), A.T])
    beta, *_ = np.linalg.lstsq(D, M.T, rcond=None)
    return (M.T - D @ beta).T


def _r2_rows(M, A):
    """R^2 of every row of M regressed on the rows of A (plus intercept)."""
    res = _resid_rows(M, A)
    tot = ((M - M.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1 - (res ** 2).sum(axis=1) / tot


def _axes_attribution(X, Y, A, spearman):
    """Observed r, partial r (all rows of A partialled jointly), loadings and R^2 on one parcel set.

    Spearman = rank X, Y and A, then Pearson on residuals -- the same convention as
    ``colocalize("partialspearman")`` (ranks Z too, see _resolve_coloc_data).
    """
    if spearman:
        X, Y, A = (rankdata(M, axis=1) for M in (X, Y, A))
    rho = _corr_rows(Y, X)
    rho_partial = _corr_rows(_resid_rows(Y, A), _resid_rows(X, A))
    return dict(rho=rho, rho_partial=rho_partial, load_x=_corr_rows(X, A), load_y=_corr_rows(Y, A),
                r2_x=_r2_rows(X, A), r2_y=_r2_rows(Y, A))


def _pool_rows(values, how, axis=0, fisher=True):
    """Pool correlations across Y rows on the Fisher-z scale (never average raw r)."""
    v = np.arctanh(np.clip(values, -0.9999999, 0.9999999)) if fisher else values
    v = np.nanmedian(v, axis=axis) if how == "median" else np.nanmean(v, axis=axis)
    return np.tanh(v) if fisher else v


def _moran_null_blocks(nsp, keep):
    """Blocks the (stored or default) maps null randomizes independently, mirroring
    generate_null_maps(): split cx/sc for a tuple method on a combined parcellation, per
    hemisphere for surface (tuple) distance matrices, else the whole map.

    Returns (null_method, source, blocks, notes); blocks = list of (name, parcel_idc, dist_mat).
    """
    from .nulls import _parse_null_method
    parc = nsp._parc
    notes = []
    stored = nsp._nulls.get("maps_null") if isinstance(getattr(nsp, "_nulls", None), dict) else None
    if stored is not None and getattr(stored, "null_method", None) is not None:
        null_method, source = stored.null_method, "stored"
    else:
        null_method, source = parc.default_null_method, "default"
    cx_method, sc_method = _parse_null_method(null_method)
    n = len(keep)
    lh_prefix, rh_prefix = parc._lh_prefix, parc._rh_prefix

    def _hemi_split(name, idc, dist):
        labels = np.asarray(parc._labels)[idc]
        out = []
        for h, prefix, d in zip(("L", "R"), (lh_prefix, rh_prefix), dist):
            sel = np.array([str(l).startswith(prefix) for l in labels])
            if sel.any() and d is not None:
                out.append((f"{name}-{h}" if name else h, idc[sel], np.asarray(d, dtype=float)))
        return out

    def _block(name, idc, method, dist):
        if method in _MORAN_NULL_METHODS:
            if method == "variomoran":
                notes.append(f"{name or 'map'}: 'variomoran' fits a map-specific weight matrix; "
                             "checked with the 1/d Moran basis.")
            if dist is None:
                notes.append(f"{name or 'map'}: no distance matrix available, skipped.")
                return []
            if isinstance(dist, tuple):
                return _hemi_split(name, idc, dist)
            return [(name or "map", idc, np.asarray(dist, dtype=float))]
        notes.append(f"{name or 'map'}: null method '{method}' is not a Moran null, skipped.")
        return []

    with _quiet():
        if parc._is_combined and sc_method is not None:
            sc_idc = np.sort(np.asarray(parc.get_sc_idc(), dtype=int))
            cx_idc = np.setdiff1d(np.arange(n), sc_idc)
            dist_cx, dist_sc = parc.get_cx_dist_mat()[0], parc.get_sc_dist_mat()[0]
            if dist_cx is None or dist_sc is None:
                # same fallback as generate_null_maps(): sub-blocks of the full matrix
                full = nsp._get_dist_mat("null_maps", store=False)
                if not isinstance(full, tuple):
                    full = np.asarray(full, dtype=float)
                    dist_cx = full[np.ix_(cx_idc, cx_idc)] if dist_cx is None else dist_cx
                    dist_sc = full[np.ix_(sc_idc, sc_idc)] if dist_sc is None else dist_sc
            blocks = _block("cortex", cx_idc, cx_method, dist_cx) + \
                _block("subcortex", sc_idc, sc_method, dist_sc)
        else:
            blocks = _block("", np.arange(n), cx_method, nsp._get_dist_mat("null_maps", store=False))
    for name, idc, dist in blocks:
        if dist.shape != (len(idc), len(idc)):
            lgr.critical_raise(f"Null block '{name}': distance matrix {dist.shape} does not match "
                               f"{len(idc)} parcels.", RuntimeError)
    return null_method, source, blocks, notes


def _null_degeneracy(P, F, blocks, keep, n_mem, spearman):
    """MEM spectra per map and per (fixed, permuted) pair; see shared_axes() Notes.

    P : permuted maps (n_P, n_parcels); F : fixed maps (n_F, n_parcels); keep : finite parcels.
    Returns per-map spectra for P and F and pair weights a (n_F, n_P, n_terms) + term labels.
    """
    from .nulls import _moran_fit_1_over_d
    # fixed side enters the correlation over the whole (finite) map: rank there, then center
    F_full = F[:, keep]
    if spearman:
        F_full = rankdata(F_full, axis=1)
    F_full = F_full - F_full.mean(axis=1, keepdims=True)
    F_full = F_full / np.linalg.norm(F_full, axis=1, keepdims=True)
    F_c = np.full(F.shape, np.nan)
    F_c[:, keep] = F_full

    F_c0 = np.nan_to_num(F_c)   # parcels outside `keep` don't enter the correlation

    # generate_null_maps() seeds every block (hemispheres, cortex/subcortex) identically, so MSR
    # draws the same sign sequence for MEM i of all blocks with the same number of MEMs: those
    # terms flip together and act as one term, key (k, i). Verified against stored nulls in
    # tests/integration/test_shared_axes.py (test_msr_signs_shared_across_blocks).
    spectra, contrib = [], {}   # contrib[(k, i)] -> (n_F, n_P) summed pair weights
    for name, idc, dist in blocks:
        # MEMs are fit per map on its own finite parcels of the block, like the null generator
        # (one fit per unique NaN mask)
        fits = {}

        def _basis(local):
            key = local.tobytes()
            if key not in fits:
                n_l = int(local.sum())
                if n_l < 4:
                    fits[key] = None
                else:
                    mem, _ = _moran_fit_1_over_d(dist[np.ix_(local, local)].copy(),
                                                 n_components=min(n_mem, n_l - 1))
                    fits[key] = np.asarray(mem, dtype=float)
            return fits[key]

        for role, M in (("P", P), ("F", F)):
            Mb = M[:, idc]
            n_rows = len(Mb)
            finite = np.isfinite(Mb)
            n_p = finite.sum(axis=1)
            top_mem = np.zeros(n_rows, dtype=int)
            top_share, recon_r2 = np.full(n_rows, np.nan), np.full(n_rows, np.nan)
            masks, group = np.unique(finite, axis=0, return_inverse=True)
            for g, local in enumerate(masks):
                rows = np.flatnonzero(group.ravel() == g)
                mem = _basis(local)
                if mem is None:
                    continue
                V = Mb[np.ix_(rows, np.flatnonzero(local))]
                sd = V.std(axis=1, ddof=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    c = ((V - V.mean(axis=1, keepdims=True)) / sd[:, None]) @ mem   # MSR coefs
                c2 = c ** 2
                top_mem[rows] = np.argmax(c2, axis=1) + 1
                top_share[rows] = c2.max(axis=1) / c2.sum(axis=1)
                recon_r2[rows] = c2.sum(axis=1) / (local.sum() - 1)
                if role == "P":
                    # MSR null in this block: mean + sd * mem @ (+-c) -> its correlation with the
                    # (centred, unit-norm) fixed map is sum_i s_i * (c_i sd) * <F, mem_i>
                    proj_F = F_c0[:, idc[local]] @ mem   # (n_F, k)
                    cP = c * sd[:, None]                 # (n_rows_g, k)
                    k = mem.shape[1]
                    for i in range(k):
                        w = contrib.setdefault((k, i), np.zeros((len(F), len(P))))
                        w[:, rows] += np.outer(proj_F[:, i], cP[:, i])
            spectra.append(dict(role=role, block=name, n_parcels=n_p, top_mem=top_mem,
                                top_share=top_share, recon_r2=recon_r2))
    if not contrib:
        return spectra, None, []
    uniq = list(contrib)
    a = np.stack([contrib[u] for u in uniq], axis=-1)   # (n_F, n_P, n_terms)
    one_group = len({kk[0] for kk in uniq}) == 1
    terms = [f"mem{i + 1}" if one_group else f"mem{i + 1} (k={k})" for k, i in uniq]
    return spectra, a, terms


def _pair_metrics(a):
    """top share / effective number of terms of r_null = sum_k s_k a_k (random signs s_k)."""
    a2 = a ** 2
    tot = a2.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.argmax(a2, axis=-1), a2.max(axis=-1) / tot, tot ** 2 / (a2 ** 2).sum(axis=-1)


def shared_axes(nsp, axes="default", method=None, X_reduction=None, Y_transform=None,
                pooled=None, null_check=True, n_mem=15, delta_threshold=0.2,
                loading_threshold=0.3, mem_share_threshold=0.75, verbose=None):
    """Check whether Y and X share known spatial axes, and whether the Moran null is degenerate.

    Descriptive only (no p-values). Needs a fitted ``nsp``; ``colocalize()``/``permute()`` are
    optional (their settings are picked up if present). Design notes:
    ``_explore/shared_axes_design.md``.

    1. **Loadings**: correlation of every Y row and X map with every axis (tissue, geometry,
       compartment, cortical hierarchy; see :func:`nispace.datasets.fetch_axes`), plus the axis
       x axis correlations.
    2. **Attribution**: every X-Y correlation before and after partialling each axis alone and
       all axes jointly -- the same computation as ``colocalize("partialspearman"/
       "partialpearson")`` with that axis as Z. A row is flagged when the correlation changes by
       ``>= delta_threshold`` *and* both X and Y load on the axis (``|rho| >= loading_threshold``;
       joint row: R^2 of X and of Y on all axes ``>= loading_threshold**2``). Without the loading
       condition, an axis that only Y follows would flag every X map: partialling it inflates the
       correlation through the denominator, not because the axis is shared.
    3. **Null degeneracy** (``null_check=True``): Moran spectral randomization flips the signs of a
       map's coefficients on the leading Moran eigenvectors (MEMs). For one X-Y pair, a null
       correlation is then a sum of random-sign terms ``a_k``, each the product of the permuted
       map's and the fixed map's MEM-k coefficient. If one term carries a share s of
       ``sum a_k**2``, the null is approximately a mixture of two normals at +-a_1 (Gaussian
       approximation of the remaining terms): it has two modes for s > 0.5 and is clearly
       bimodal (Sarle's coefficient > 0.555) for s > 0.775. Pairs with s above
       ``mem_share_threshold`` (default 0.75) are flagged: their null mainly tests the sign of one
       gradient.

    Parameters
    ----------
    nsp : NiSpace
        Fitted NiSpace object. If X has a ``set`` level (collections, XSEA), every map is checked
        individually, once even if it is listed in several sets.
    axes : str or list of str, default "default"
        Axis names, ``"default"`` or ``"all"`` (see :func:`nispace.datasets.fetch_axes`).
        Unavailable axes are skipped for ``"default"``/``"all"`` and raise otherwise. Axes that
        cover only some parcels (e.g. ``saaxis`` on a combined parcellation) are evaluated on
        those parcels (``n_parcels`` column).
    method : {"spearman", "pearson"}, optional
        Default: Spearman if the last ``colocalize()`` ranked the data (spearman,
        partialspearman, ``rank=True``), Pearson for pearson/partialpearson, else Spearman.
        The diagnostic is pairwise for every colocalization method; for joint models
        (mlr/dominance/pls/...) it shows which X maps share an axis with Y, not how the joint
        fit changes.
    X_reduction, Y_transform : str, optional
        Use reduced X / transformed Y, as in ``colocalize()``.
    pooled : {False, "mean", "median"}, optional
        For several Y rows (e.g. subject-level transforms): add a row pooled across Y on the
        Fisher-z scale, and flag that row (single-subject correlations are noisy). Default: as
        the last ``permute()`` (forced to "mean" for group permutation), else False.
    null_check : bool, default True
        Run the null-degeneracy check (needs a parcellation and a Moran null: the stored maps
        null if ``permute("maps")`` ran, else the parcellation's default null). Skipped after
        ``permute("groups")``.
    n_mem : int, default 15
        Number of MEMs per block (NiSpace's Moran null default).
    delta_threshold, loading_threshold, mem_share_threshold : float
        Flag thresholds, see above.
    verbose : bool, optional

    Returns
    -------
    dict
        ``"axes"`` (axes x parcels), ``"loadings"`` (rows: role/map, cols: axes), ``"axis_corr"``,
        ``"attribution"`` (y, x, axis [name or "joint"], n_parcels, rho, rho_partial, delta,
        flagged), ``"null_degeneracy"`` (y, x, top_term, top_share, n_eff, flagged) and
        ``"null_spectrum"`` (role, map, block, n_parcels, top_mem, top_share, recon_r2) -- both
        None when the check is skipped -- and ``"settings"``. Correlations are raw rho (not
        Fisher-z).
    """
    from .core.axes import build_axes
    verbose = set_log(lgr, nsp._verbose if verbose is None else verbose)

    # --- method / data --------------------------------------------------------------------------
    last_method = nsp._last_settings.get("method")
    if method is None:
        last_kwargs = getattr(nsp, "_coloc_kwargs_by_method", {}).get(last_method, {}) \
            if last_method else {}
        if last_method in ("pearson", "partialpearson") and not last_kwargs.get("rank", False):
            method = "pearson"
        else:
            method = "spearman"
    if method not in ("spearman", "pearson"):
        lgr.critical_raise(f"method must be 'spearman' or 'pearson', got '{method}'.", ValueError)
    spearman = method == "spearman"

    with _quiet():
        X = nsp._X if not X_reduction else nsp.get_x(X_reduction=X_reduction)
        if Y_transform:
            if not nsp._check_transform(ytrans=Y_transform, raise_error=True):
                lgr.critical_raise(f"Y transform '{Y_transform}' was not run.", KeyError)
            Y = nsp.get_y(Y_transform=Y_transform)
        else:
            Y = nsp._Y
    def _labels(index):
        return [str(i[-1] if isinstance(i, tuple) else i) for i in index]

    if isinstance(X.index, pd.MultiIndex) and "set" in X.index.names:
        # collections come with a (set, map) index even without XSEA: check every map on its
        # own (as colocalize() does without xsea=True), once if it is listed in several sets
        X = X[~pd.Index(_labels(X.index)).duplicated()]
        lgr.info(f"X has a 'set' level: {len(X)} unique map(s) checked individually, "
                 "set membership ignored.")
    x_lab, y_lab = _labels(X.index), _labels(Y.index)
    X_arr, Y_arr = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    keep = np.isfinite(X_arr).all(axis=0) & np.isfinite(Y_arr).all(axis=0)
    n_y = len(y_lab)

    # --- pooling --------------------------------------------------------------------------------
    perm_mode = nsp._last_settings.get("perm")
    if pooled is None:
        pooled = nsp._last_settings.get("pooled_p", False)
    if perm_mode:
        pooled, mode_warning = _resolve_permute_mode_settings([perm_mode], pooled)
        if mode_warning:
            lgr.warning(mode_warning)
    if pooled and pooled not in ("mean", "median"):
        pooled = "mean"
    if n_y <= 1:
        pooled = False
    pooled_lab = f"pooled ({pooled})" if pooled else None

    # --- axes -----------------------------------------------------------------------------------
    if nsp._parc is None:
        lgr.critical_raise("shared_axes() needs a parcellation (NiSpace(parcellation=...)).",
                           ValueError)
    if nsp._Z is not None:
        lgr.info("Z stored in nsp is not used; list the axes to check in 'axes'.")
    # X's columns are parcel labels -- all of them, or a subset with NiSpace(drop_nan=True):
    # axes and null blocks are defined on the full parcellation and selected by label
    lab_pos = {str(l): i for i, l in enumerate(nsp._parc._labels)}
    not_parcels = [str(c) for c in X.columns if str(c) not in lab_pos]
    if not_parcels:
        lgr.critical_raise(
            f"shared_axes(): {len(not_parcels)} of {X.shape[1]} data column(s) are not parcel "
            f"labels of the parcellation (e.g. '{not_parcels[0]}'); axes are defined per parcel.",
            ValueError)
    pos = np.array([lab_pos[str(c)] for c in X.columns])
    n_parc = len(lab_pos)
    with _quiet():
        A = build_axes(nsp._parc, axes, verbose=False)
    A = A.iloc[:, pos]
    A.columns = X.columns
    if len(A) == 0:
        lgr.critical_raise(f"No axis available for axes={axes!r}.", ValueError)
    ax_names = list(A.index)
    A_arr = A.to_numpy(dtype=float)
    lgr.info(f"shared_axes(): {len(x_lab)} X map(s), {n_y} Y row(s), axes {ax_names}, {method}.")

    axis_corr = A.T.corr(method=method)

    # --- loadings + attribution: one parcel set per axis, plus all axes jointly ------------------
    lt, dt = loading_threshold, delta_threshold
    load_x = np.full((len(x_lab), len(ax_names)), np.nan)
    load_y = np.full((n_y, len(ax_names)), np.nan)
    rows = []

    def _add_rows(axis, mask, res, crit_x, crit_y):
        rho, rho_p = res["rho"], res["rho_partial"]
        y_names = list(y_lab)
        if pooled:
            rho = np.vstack([rho, _pool_rows(rho, pooled)])
            rho_p = np.vstack([rho_p, _pool_rows(rho_p, pooled)])
            crit_y = np.append(crit_y, _pool_rows(crit_y, pooled, fisher=axis != "joint"))
            y_names.append(pooled_lab)
        delta = rho_p - rho
        thr = lt ** 2 if axis == "joint" else lt
        shared = (np.abs(crit_y)[:, None] >= thr) & (np.abs(crit_x)[None, :] >= thr)
        flagged = (np.abs(delta) >= dt) & shared
        for i, y in enumerate(y_names):
            for j, x in enumerate(x_lab):
                rows.append((y, x, axis, int(mask.sum()), rho[i, j], rho_p[i, j], delta[i, j],
                             bool(flagged[i, j])))

    for k, a in enumerate(ax_names):
        mask = keep & np.isfinite(A_arr[k])
        res = _axes_attribution(X_arr[:, mask], Y_arr[:, mask], A_arr[[k]][:, mask], spearman)
        load_x[:, k], load_y[:, k] = res["load_x"][:, 0], res["load_y"][:, 0]
        _add_rows(a, mask, res, res["load_x"][:, 0], res["load_y"][:, 0])
    if len(ax_names) > 1:
        mask = keep & np.isfinite(A_arr).all(axis=0)
        res = _axes_attribution(X_arr[:, mask], Y_arr[:, mask], A_arr[:, mask], spearman)
        _add_rows("joint", mask, res, res["r2_x"], res["r2_y"])

    attribution = pd.DataFrame(rows, columns=["y", "x", "axis", "n_parcels", "rho", "rho_partial",
                                              "delta", "flagged"])
    load_rows = [("Y", y) for y in y_lab] + [("X", x) for x in x_lab]
    load_vals = [load_y, load_x]
    if pooled:
        load_rows.insert(n_y, ("Y", pooled_lab))
        load_vals = [load_y, _pool_rows(load_y, pooled)[None, :], load_x]
    loadings = pd.DataFrame(np.vstack(load_vals), columns=ax_names,
                            index=pd.MultiIndex.from_tuples(load_rows, names=["role", "map"]))

    # log flagged rows: one line per (Y, axis), strongest changes first (pooled row only when
    # pooling, with the per-row flag count)
    flag_rows = attribution[attribution["flagged"]]
    if pooled:
        n_rows_flagged = flag_rows[flag_rows["y"] != pooled_lab].groupby(["x", "axis"]).size()
        flag_rows = flag_rows[flag_rows["y"] == pooled_lab]
    n_lines = 0
    for (y_name, axis), grp in flag_rows.groupby(["y", "axis"], sort=False):
        if n_lines == _MAX_FLAG_LOG_LINES:
            lgr.warning("... more flagged rows, see 'attribution'.")
            break
        grp = grp.reindex(grp["delta"].abs().sort_values(ascending=False).index)
        shown = [f"'{r.x}' {r.rho:+.2f}->{r.rho_partial:+.2f}"
                 + (f" [{n_rows_flagged.get((r.x, axis), 0)}/{n_y} rows]" if pooled else "")
                 for r in grp.head(5).itertuples()]
        more = f" (+{len(grp) - 5} more)" if len(grp) > 5 else ""
        z_arg = repr(axis) if axis != "joint" else repr(ax_names)
        lgr.warning(f"'{y_name}' shares axis '{axis}' with {len(grp)} X map(s): "
                    f"{', '.join(shown)}{more}. Sensitivity: NiSpace(..., z={z_arg}).")
        n_lines += 1

    # --- null degeneracy ------------------------------------------------------------------------
    null_deg, null_spec, null_settings = None, None, {}
    if null_check and perm_mode and "groups" in perm_mode and "maps" not in perm_mode:
        lgr.info("Null check skipped: permute('groups') uses no spatial (Moran) null.")
    elif null_check:
        side = "Y" if perm_mode == "Ymaps" else "X"
        if perm_mode == "XYmaps":
            lgr.info("Null check: both sides were permuted; reporting X as the permuted side.")
        # null blocks index the full parcellation: place the data there (dropped parcels NaN)
        keep_full = np.zeros(n_parc, dtype=bool)
        keep_full[pos] = keep

        def _full(M):
            out = np.full((len(M), n_parc), np.nan)
            out[:, pos] = M
            return out

        null_method, source, blocks, notes = _moran_null_blocks(nsp, keep_full)
        for note in notes:
            lgr.info(f"Null check: {note}")
        null_settings = dict(null_method=null_method, null_source=source, permuted=side,
                             null_blocks={b[0]: int(keep_full[b[1]].sum()) for b in blocks})
        if blocks:
            P, F = (X_arr, Y_arr) if side == "X" else (Y_arr, X_arr)
            p_lab, f_lab = (x_lab, y_lab) if side == "X" else (y_lab, x_lab)
            spectra, a, terms = _null_degeneracy(_full(P), _full(F), blocks, keep_full, n_mem,
                                                 spearman)
            spec_rows = []
            for s in spectra:
                labs, role = (p_lab, side) if s["role"] == "P" else \
                    (f_lab, "Y" if side == "X" else "X")
                for i, lab in enumerate(labs):
                    spec_rows.append((role, lab, s["block"], int(s["n_parcels"][i]),
                                      int(s["top_mem"][i]), s["top_share"][i], s["recon_r2"][i]))
            null_spec = pd.DataFrame(spec_rows, columns=["role", "map", "block", "n_parcels",
                                                         "top_mem", "top_share", "recon_r2"])
            # a: (n_F, n_P, n_terms) -> orient as (n_y, n_x, n_terms)
            a_yx = a if side == "X" else a.transpose(1, 0, 2)
            y_names = list(y_lab)
            if pooled and side == "X":
                # X nulls are shared by all Y rows: the pooled null sums the averaged terms
                a_yx = np.concatenate([a_yx, a_yx.mean(axis=0, keepdims=True)], axis=0)
                y_names.append(pooled_lab)
            top, share, n_eff = _pair_metrics(a_yx)
            flag = share > mem_share_threshold
            if pooled:
                # flags only on what is tested: the pooled row (X permuted) or none (Y permuted:
                # independent nulls per row average out, the pooled null is not bimodal)
                flag[: n_y] = False
            null_deg = pd.DataFrame(
                [(y, x, terms[top[i, j]], share[i, j], n_eff[i, j], bool(flag[i, j]))
                 for i, y in enumerate(y_names) for j, x in enumerate(x_lab)],
                columns=["y", "x", "top_term", "top_share", "n_eff", "flagged"])
            n_flag = int(null_deg["flagged"].sum())
            if n_flag:
                worst = null_deg[null_deg["flagged"]].sort_values("top_share").iloc[-1]
                lgr.warning(
                    f"Moran null degenerate for {n_flag} X-Y pair(s) (e.g. '{worst.x}' ~ "
                    f"'{worst.y}': {worst.top_term} carries {worst.top_share:.0%} of the null "
                    "variance). These nulls mainly test the sign of one gradient (bimodal null "
                    "distribution); consider a spin null (cortex) or label permutation.")

    settings = dict(method=method, axes=ax_names,
                    n_parcels={a: int((keep & np.isfinite(A_arr[k])).sum())
                               for k, a in enumerate(ax_names)},
                    pooled=pooled, delta_threshold=dt, loading_threshold=lt,
                    mem_share_threshold=mem_share_threshold, n_mem=n_mem,
                    X_reduction=X_reduction, Y_transform=Y_transform, **null_settings)
    lgr.info(f"Returning shared_axes results: \n"
             f"{print_arg_pairs(method=method, axes=ax_names, pooled=pooled, n_flagged=int(attribution['flagged'].sum()), null_check=null_deg is not None, X_reduction=X_reduction, Y_transform=Y_transform)}")
    return dict(axes=A, loadings=loadings, axis_corr=axis_corr, attribution=attribution,
                null_degeneracy=null_deg, null_spectrum=null_spec, settings=settings)


# SPATIAL CROSS-VALIDATION ==========================================================================

_SPATIAL_CV_METHODS = frozenset({"slr", "mlr", "pls", "pcr", "lasso", "ridge", "elasticnet"})


def spatial_cross_validate(nsp, method=None, X_reduction=None, Y_transform=None,
                           train_pct=0.75, dist_mat=None,
                           X=None, Y=None, Z=None,
                           null=False, mc_method="step_maxT", mc_alpha=0.05,
                           n_proc=None, seed=None, dtype=None, verbose=None):
    """
    Hansen-et-al.-style distance-dependent cross-validation (GitHub issue #74): for
    each region as an anchor, fit ``method`` on the spatially nearest ``train_pct``
    fraction of regions and project the held-out farthest fraction through those same
    fitted weights, giving one train/test correlation pair per anchor region (one fold
    per region, `nsp._get_dist_mat(dist_mat_type="cv")` by default) -- a spatial
    generalization check for a global colocalization result, complementary to
    :func:`local_colocalization`'s opposite goal of finding local heterogeneity.

    Scoped to methods with projectable linear weights (``method in
    {"slr", "mlr", "pls", "pcr", "lasso", "ridge", "elasticnet"}``) -- undefined for a
    bare pearson/spearman correlation, which has no fitted weight to reapply to
    held-out data.

    **Not implemented yet.** Blocked on checking whether `nispace/stats/coloc.py`'s
    regression functions expose the fitted estimator/coefficients needed for the
    train-to-test projection step, or whether this needs its own lightweight
    sklearn-based fit+project implementation; see the implementation plan for details.
    """
    lgr.critical_raise("diagnostics.spatial_cross_validate() is not implemented yet "
                       "(GitHub issue #74). See the docstring/implementation plan for "
                       "the design and the open blocker.", NotImplementedError)
