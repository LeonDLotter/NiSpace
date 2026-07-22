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

``regional_influence``/``regional_contribution`` moved here from :class:`NiSpace` (formerly
``NiSpace.regional_influence()``/``.regional_contribution()``) -- this is a clean break, no
backward-compat wrapper was kept on the class.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from joblib import Parallel, delayed
from tqdm import tqdm
from contextlib import contextmanager

import logging
lgr = logging.getLogger(__name__)

from .core.colocalize import _get_coloc_stats, _rank_regress
from .core.region_influence import _get_region_influence_fun, _sort_region_influence, _pool_region_influence
from .core.region_contribution import (_get_region_contribution_fun, _sort_region_contribution,
                                       _CONTRIBUTION_METHODS)
from .stats.coloc import rank2d
from .stats.misc import maxT_correction, step_maxT_correction
from .utils.utils import set_log, _quiet


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

    return out


# LOCAL COLOCALIZATION (searchlight) ===============================================================

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
    each spawning their own n_proc sub-workers)."""
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
                       store=False, verbose=False, r_to_z=False, n_proc=1,
                       force_dict=True)
    r_stat = r[stat]  # (n_Y, n_X_or_1)
    return r_stat.to_numpy(), r_stat.columns


def local_colocalization(nsp, k=None, radius=None, dist_mat=None,
                         method=None, X_reduction=None, Y_transform=None, xsea=None,
                         X=None, Y=None, Z=None,
                         null=None, mc_method="step_maxT", mc_alpha=0.05,
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
        one of ``k``/``radius`` must be given.
    radius : float, optional
        Distance threshold (same units as ``dist_mat``) for a region's neighborhood,
        as an alternative to a fixed ``k``.
    dist_mat : array, optional
        Precomputed ``(n_parcels, n_parcels)`` distance matrix. Defaults to
        ``nsp._get_dist_mat(dist_mat_type="cv")`` -- the same combined-hemisphere
        euclidean matrix already used by ``colocalize()``'s lasso/ridge/elasticnet
        spatial cross-validation.
    method, X_reduction, Y_transform, xsea, X, Y, Z : see ``NiSpace.colocalize()``.
        ``lasso``/``ridge``/``elasticnet`` are not supported -- their spatial CV-fold
        split is sized to the full parcellation and would misalign against a
        region-window subset (same reason :func:`regional_influence` excludes them).
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
        - ``"stat"`` : DataFrame or dict of DataFrame -- the local statistic, shape
          (n_Y x n_parcels). A dict keyed by X-map label for per-predictor methods
          (pearson/spearman/partialpearson/partialspearman/mi/slr), a single DataFrame
          for joint-model methods (mlr/dominance/pls/pcr).
        - ``"p"`` : same shape as ``"stat"``, or None -- raw uncorrected p-values.
          ``None`` only when ``null=False`` (no permutation available).
        - ``"p_corr"`` : same shape as ``"stat"``, or None -- ``mc_method``-corrected
          p-values. ``None`` when ``null=False`` or ``mc_method=None``.
        - ``"settings"`` : dict -- ``{"k": k}`` in k-NN mode, or ``{"radius": radius,
          "n_neighbors_min": ..., "n_neighbors_median": ..., "n_neighbors_max": ...}``
          in radius mode (window size varies by region there, unlike fixed-k).
    """
    verbose = set_log(lgr, nsp._verbose if verbose is None else verbose)
    lgr.info("*** diagnostics.local_colocalization() - Computing local colocalization. ***")

    nsp._check_fit()
    n_proc = nsp._n_proc if n_proc is None else n_proc
    dtype = nsp._dtype if dtype is None else dtype

    if (k is None) == (radius is None):
        lgr.critical_raise("Specify exactly one of k or radius.", ValueError)

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
    if method not in nsp._colocs_fun or method not in nsp._coloc_kwargs_by_method:
        lgr.critical_raise(f"No stored colocalize() results for method '{method}'! "
                           "Did you run colocalize() with this method first?", KeyError)

    X, Y, _, _, _, coloc_kwargs = _resolve_coloc_data(
        nsp, method=method, X=X, Y=Y, Z=Z, X_reduction=X_reduction, Y_transform=Y_transform,
        xsea=xsea, zy_matched=False, dtype=dtype, verbose=verbose,
    )
    Z_full = nsp._Z if Z is None else Z

    n_parcels = X.shape[1]
    n_predictors = X.shape[0]
    is_joint = method in _LOCAL_JOINT_METHODS

    ## distance matrix + neighbor windows
    if dist_mat is None:
        dist_mat = nsp._get_dist_mat(dist_mat_type="cv", n_proc=n_proc)
    dist_mat = np.asarray(dist_mat)
    if dist_mat.shape[0] != n_parcels or dist_mat.shape[1] != n_parcels:
        lgr.critical_raise(f"dist_mat shape {dist_mat.shape} does not match the "
                           f"n_parcels={n_parcels} of the resolved X/Y data.", ValueError)

    if k is not None:
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
            lgr.critical_raise(f"Expected a stored null for permutation mode '{perm_mode}' but "
                               "found none on this NiSpace object.", KeyError)
        target_labels = list(X.index if null_side == "X" else Y.index)
        null_maps = null_maps.subset(target_labels)
        n_perm = null_maps.n_perm
        lgr.info(f"local_colocalization(): auto-picked up '{perm_mode}' null ({n_perm} perms, "
                f"nulling {null_side}) from the last permute() call for p-values.")

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
    with _silence_repeated_calls():
        obs_results = [
            _local_observed_window(nsp, windows[i], X, Y, Z_full, method, rank, regress_z,
                                   zy_matched, stat)
            for i in tqdm(range(n_parcels),
                         desc=f"Local colocalization ({method})",
                         disable=not verbose, mininterval=1.0)
        ]
    x_labels = obs_results[0][1]
    y_labels = Y.index
    obs_arr = np.stack([r[0] for r in obs_results], axis=-1)  # (n_Y, n_X_or_1, n_parcels)

    result = {
        x_lab: pd.DataFrame(obs_arr[:, j, :], index=y_labels, columns=X.columns, dtype=dtype)
        for j, x_lab in enumerate(x_labels)
    }
    if is_joint and len(result) == 1:
        result = next(iter(result.values()))

    if not null:
        return {"stat_type": stat, "mc_method": None, "stat": result, "p": None,
               "p_corr": None, "settings": settings}

    if mc_method not in ("step_maxT", "maxT", None):
        lgr.critical_raise(f"mc_method='{mc_method}' not supported; use 'step_maxT', 'maxT', "
                           "or None.", ValueError)

    ## null: recompute the same windowed correlation for every precomputed null draw,
    ## re-slicing the already-generated null maps rather than regenerating them per window.
    ## Fast (batched, vectorized) path for pearson/spearman/partialpearson/partialspearman;
    ## slow (per-draw closure reuse) path for mi/slr/mlr/dominance/pls/pcr -- see
    ## _local_null_fast()/_local_null_slow() docstrings. Neither path calls nsp.colocalize()
    ## (they reuse _rank_regress()/nsp._colocs_fun[method] directly), so plain _quiet() is
    ## enough here -- no "nispace.api" logger involved, unlike the observed-stat loop above.
    null_data = null_maps.data  # (n_null_maps, n_perm, n_parcels)
    with _quiet():
        if method in _LOCAL_NULL_FAST_METHODS:
            null_arr = _local_null_fast(
                windows, X_arr, Y_arr, Z_full, null_data, null_side,
                rank, regress_z, zy_matched, dtype, n_proc, verbose,
            )
        else:
            null_arr = _local_null_slow(
                nsp, method, windows, X_arr, Y_arr, Z_full, null_data, null_side,
                rank, regress_z, zy_matched, is_joint, dtype, n_proc, verbose,
            )

    # raw (uncorrected) p is always computed and returned -- mc_method only controls
    # whether a *second*, corrected p is additionally computed alongside it
    p_floor = max(np.finfo(float).eps, 1.0 / n_perm)
    p_raw_arr = np.mean(np.abs(null_arr) >= np.abs(obs_arr[:, :, np.newaxis, :]), axis=2)
    p_raw_arr = np.clip(p_raw_arr, p_floor, 1.0 - p_floor)

    p_raw_result = {
        x_lab: pd.DataFrame(p_raw_arr[:, j, :], index=y_labels, columns=X.columns, dtype=dtype)
        for j, x_lab in enumerate(x_labels)
    }
    if is_joint and len(p_raw_result) == 1:
        p_raw_result = next(iter(p_raw_result.values()))

    p_corr_result = None
    if mc_method is not None:
        corr_fun = step_maxT_correction if mc_method == "step_maxT" else maxT_correction
        p_corr_result = {}
        for j, x_lab in enumerate(x_labels):
            obs_df = result[x_lab] if isinstance(result, dict) else result
            null_colocs = [{"rho": null_arr[:, j, p, :]} for p in range(n_perm)]
            p_corr, _ = corr_fun(obs_df, null_colocs, stat="rho", tail="two",
                                 how="r", alpha=mc_alpha, dtype=dtype)
            p_corr_result[x_lab] = p_corr
        if is_joint and len(p_corr_result) == 1:
            p_corr_result = next(iter(p_corr_result.values()))

    return {"stat_type": stat, "mc_method": mc_method, "stat": result,
           "p": p_raw_result, "p_corr": p_corr_result, "settings": settings}


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
