"""Regional influence -- true leave-one-region-out sensitivity of a
colocalization result, computed via two interchangeable engines.

Definition (see NiSpace.regional_influence() docstring for the short
rationale): ``influence[y, i] = stat_full[y] - stat_loo_excluding_i[y]`` for
the method's primary stat, unless ``signed=False`` (the default), in which
case it's ``|stat_full[y]| - |stat_loo_excluding_i[y]|``.

Why default to the absolute-value version: mlr/dominance/pls/pcr/mi/slr use
an inherently unsigned stat (R^2, MI -- never negative), so their delta is
already "does this region help or hurt the fit/dependency strength," with
no notion of direction. pearson/spearman/partialpearson/partialspearman use
a signed stat (rho), so their *raw* (signed=True) delta additionally encodes
direction -- positive delta means the region pulls the correlation toward
+1, negative means it pulls toward -1, regardless of the sign of rho itself
(a point can be "discordant" and pull toward -1 even when the overall
relationship is positive). Taking the absolute value before differencing
(the default) makes all methods report the same kind of quantity --
"does this region strengthen or weaken the association" -- homogeneously;
``signed=True`` recovers the original directional version for the 4
correlation methods (a no-op for the unsigned-stat methods, since
|x| == x when x >= 0 already).

Two engines:
- "analytic": closed-form case-deletion identities (stats/loo.py) -- exact,
  O(n_parcels) total. Only available for pearson/spearman/partialpearson/
  partialspearman/mlr (the methods with a closed-form fit).
- "bruteforce": reruns the actual colocalization closure with each region
  excluded -- exact by construction, works for any method whose internal
  masks are recomputed fresh from the passed-in X/y (i.e. every supported
  method except lasso/ridge/elasticnet, whose closures capture a
  fixed-size CV split/regularization mask that would misalign against
  region-excluded data).

Output shape per Y-row depends on whether the method fits one joint model
per Y-row (mlr/dominance/pls/pcr -> one (n_parcels,) profile) or one model
per X-predictor/gene (pearson/spearman/partialpearson/partialspearman/mi/
slr, and any XSEA call, which aggregates per set instead of per X-map ->
one (n_x_or_n_sets, n_parcels) profile). Both engines determine this
generically from the shape of the underlying colocalization stat, no
per-method branching needed in _sort_region_influence.
"""

import numpy as np
import pandas as pd

import logging
lgr = logging.getLogger(__name__)

from ..stats.loo import pearson_loo, mlr_loo
from ..stats.coloc import pearson, mlr
from ..stats.misc import rho_to_z
from ..utils.utils import nan_detector


_ANALYTIC_METHODS = {"pearson", "spearman", "partialpearson", "partialspearman", "mlr"}
_BRUTEFORCE_UNSUPPORTED = {"lasso", "ridge", "elasticnet"}
_BRUTEFORCE_N_WARN_THRESHOLD = 1000

# Methods whose primary stat is genuinely bidirectional (rho: -1..+1) -- these are
# the only methods `signed=False` (the default) changes anything for. NOTE: R^2-based
# stats (mlr/dominance/pls/pcr/slr) are NOT reliably >= 0 -- *adjusted* R^2 can go
# negative (worse fit than the intercept-only baseline, more likely with fewer
# observations after one is excluded) -- so `abs()` must be gated explicitly on
# method identity here, not relied on as a coincidental no-op for "unsigned" stats.
_SIGNED_METHODS = {"pearson", "spearman", "partialpearson", "partialspearman"}


def _get_aggr_fun_axis0(xsea_method):
    """Aggregation functions for XSEA, along axis 0 of a 2D (n_genes, n_parcels)
    array. Same semantics as the ``aggr()`` closure inside
    core.colocalize._get_colocalize_fun -- kept manually in sync with it,
    not imported, because that closure operates on a 1D per-gene array
    with no axis argument and isn't exposed at module level, so it can't
    aggregate a per-parcel LOO array directly.
    """
    if xsea_method == "mean":
        return lambda arr, weights=None: np.nanmean(arr, axis=0)
    elif xsea_method == "median":
        return lambda arr, weights=None: np.nanmedian(arr, axis=0)
    elif xsea_method == "absmean":
        return lambda arr, weights=None: np.nanmean(np.abs(arr), axis=0)
    elif xsea_method == "absmedian":
        return lambda arr, weights=None: np.nanmedian(np.abs(arr), axis=0)
    elif xsea_method == "weightedmean":
        def f(arr, weights):
            m = np.ma.array(arr, mask=np.isnan(arr))
            return np.ma.average(m, weights=weights, axis=0).filled(np.nan)
        return f
    elif xsea_method == "weightedabsmean":
        def f(arr, weights):
            m = np.ma.array(np.abs(arr), mask=np.isnan(arr))
            return np.ma.average(m, weights=weights, axis=0).filled(np.nan)
        return f
    else:
        lgr.critical_raise(f"XSEA aggregation method '{xsea_method}' not defined!", ValueError)


def _get_region_influence_fun_analytic(method, adj_r2=True, r_to_z=True, dtype=np.float32,
                                       xsea=False, xsea_method=None, signed=False):
    """Build the per-Y-row analytic regional-influence closure for one method.

    ``signed=False`` (default) takes the absolute value of the full-data and
    LOO stat before differencing -- a no-op for mlr (R^2 is already >= 0),
    but for the correlation methods it converts the raw signed delta (which
    also encodes direction, see module docstring) into the same
    "strengthens vs. weakens" quantity mlr already reports by construction.
    """

    is_corr = method in _SIGNED_METHODS

    def _maybe_abs(arr):
        # gate on is_corr, not just `signed`: R^2 (mlr) is not reliably >= 0 under
        # adj_r2=True, so abs() must never be applied to it regardless of `signed`
        return arr if (signed or not is_corr) else np.abs(arr)

    if is_corr and not xsea:
        def _y_region_influence(X, y, weights=None):
            n_x, n_parcels = X.shape
            out = np.empty((n_x, n_parcels), dtype=dtype)
            for i_x in range(n_x):
                x = X[i_x]
                mask = ~np.isnan(y) & ~np.isnan(x)
                rho_full = pearson(x[mask], y[mask])
                loo = pearson_loo(x, y).astype(np.float64)
                if r_to_z:
                    rho_full = rho_to_z(np.array([rho_full], dtype=np.float64))[0]
                    loo = rho_to_z(loo)
                out[i_x] = (_maybe_abs(rho_full) - _maybe_abs(loo)).astype(dtype)
            return out

    elif is_corr and xsea:
        aggr_fn = _get_aggr_fun_axis0(xsea_method)
        weighted = "weighted" in xsea_method

        def _y_region_influence(X_dict, y, weights=None):
            n_parcels = y.shape[0]
            n_sets = len(X_dict)
            out = np.empty((n_sets, n_parcels), dtype=dtype)
            for i_s, (set_name, set_X) in enumerate(X_dict.items()):
                n_genes = set_X.shape[0]
                loo_per_gene = np.empty((n_genes, n_parcels), dtype=np.float64)
                rho_full_per_gene = np.empty(n_genes, dtype=np.float64)
                for i_g in range(n_genes):
                    x = set_X[i_g]
                    mask = ~np.isnan(y) & ~np.isnan(x)
                    rho_full_per_gene[i_g] = pearson(x[mask], y[mask])
                    loo_per_gene[i_g] = pearson_loo(x, y)
                if r_to_z:
                    rho_full_per_gene = rho_to_z(rho_full_per_gene)
                    loo_per_gene = rho_to_z(loo_per_gene)
                if weighted:
                    w = weights[set_name]
                    full_aggr = aggr_fn(rho_full_per_gene[:, np.newaxis], w)[0]
                    loo_aggr = aggr_fn(loo_per_gene, w)
                else:
                    full_aggr = aggr_fn(rho_full_per_gene[:, np.newaxis])[0]
                    loo_aggr = aggr_fn(loo_per_gene)
                out[i_s] = (_maybe_abs(full_aggr) - _maybe_abs(loo_aggr)).astype(dtype)
            return out

    elif method == "mlr" and not xsea:
        def _y_region_influence(X, y, weights=None):
            X_T = X.T
            mask = ~nan_detector(X_T, y)
            r2_full, _ = mlr(X_T[mask], y[mask], adj_r2=adj_r2, intercept=True)
            loo = mlr_loo(X_T, y, adj_r2=adj_r2)
            return (_maybe_abs(r2_full) - _maybe_abs(loo)).astype(dtype)

    elif method == "mlr" and xsea:
        def _y_region_influence(X_dict, y, weights=None):
            n_parcels = y.shape[0]
            n_sets = len(X_dict)
            out = np.empty((n_sets, n_parcels), dtype=dtype)
            for i_s, (set_name, set_X) in enumerate(X_dict.items()):
                X_T = set_X.T
                mask = ~nan_detector(X_T, y)
                r2_full, _ = mlr(X_T[mask], y[mask], adj_r2=adj_r2, intercept=True)
                loo = mlr_loo(X_T, y, adj_r2=adj_r2)
                out[i_s] = (_maybe_abs(r2_full) - _maybe_abs(loo)).astype(dtype)
            return out

    else:
        lgr.critical_raise(f"Analytic regional-influence engine does not support method "
                           f"'{method}'. Supported: {sorted(_ANALYTIC_METHODS)}.",
                           ValueError)

    return _y_region_influence


def _get_region_influence_fun_bruteforce(y_colocalize_fun, stat, dtype=np.float32, xsea=False,
                                         signed=False):
    """Build the per-Y-row brute-force regional-influence closure.

    Reuses the already-built colocalization closure (self._colocs_fun[method]
    in NiSpace.regional_influence()), rerunning it once per excluded region.
    Output shape is inferred generically from the shape of `stat` in the
    full-data result: scalar (joint-model methods: mlr/dominance/pls/pcr)
    -> (n_parcels,); vector (per-X/per-set methods) -> (n_x_or_sets, n_parcels).

    ``signed=False`` (default) takes the absolute value of the full-data and
    per-exclusion stat before differencing, but only when `stat == "rho"` --
    the only stat name any of the 4 bidirectional (pearson/spearman/
    partialpearson/partialspearman) methods produce (see _COLOC_METHODS).
    Every other stat (r2/mi/sum/individual/...) is left untouched regardless
    of `signed`: R^2-based stats are not reliably >= 0 under adj_r2=True
    (adjusted R^2 can go negative), so abs() must be gated on the stat being
    genuinely bidirectional, not relied on as a coincidental no-op.
    """

    def _maybe_abs(arr):
        return arr if (signed or stat != "rho") else np.abs(arr)

    if not xsea:
        def _y_region_influence(X, y, weights=None):
            n_parcels = X.shape[1]
            full = _maybe_abs(np.asarray(y_colocalize_fun(X, y, weights)[stat], dtype=np.float64))
            out = np.empty((n_parcels,) + full.shape, dtype=np.float64)
            for i in range(n_parcels):
                X_i = np.delete(X, i, axis=1)
                y_i = np.delete(y, i)
                res_i = _maybe_abs(np.asarray(y_colocalize_fun(X_i, y_i, weights)[stat], dtype=np.float64))
                out[i] = full - res_i
            return np.moveaxis(out, 0, -1).astype(dtype)
    else:
        def _y_region_influence(X_dict, y, weights=None):
            n_parcels = y.shape[0]
            full = _maybe_abs(np.asarray(y_colocalize_fun(X_dict, y, weights)[stat], dtype=np.float64))
            out = np.empty((n_parcels,) + full.shape, dtype=np.float64)
            for i in range(n_parcels):
                X_dict_i = {k: np.delete(v, i, axis=1) for k, v in X_dict.items()}
                y_i = np.delete(y, i)
                res_i = _maybe_abs(np.asarray(y_colocalize_fun(X_dict_i, y_i, weights)[stat], dtype=np.float64))
                out[i] = full - res_i
            return np.moveaxis(out, 0, -1).astype(dtype)

    return _y_region_influence


def _get_region_influence_fun(method, engine, n_parcels, y_colocalize_fun=None, stat=None,
                              adj_r2=True, r_to_z=True, dtype=np.float32,
                              xsea=False, xsea_method=None, signed=False):
    """Top-level dispatcher. Returns (closure, engine_used)."""

    if method in _BRUTEFORCE_UNSUPPORTED:
        lgr.critical_raise(
            f"regional_influence() does not support method '{method}': its colocalization "
            "closure captures a fixed-size CV train/test split and regularization mask sized "
            "to the original number of parcels, which would misalign against region-excluded "
            "data. Not supported in this version.",
            NotImplementedError
        )

    if engine == "auto":
        engine = "analytic" if method in _ANALYTIC_METHODS else "bruteforce"
    elif engine == "analytic" and method not in _ANALYTIC_METHODS:
        lgr.critical_raise(
            f"Analytic regional-influence engine not available for method '{method}'. "
            f"Supported: {sorted(_ANALYTIC_METHODS)}. Use engine='bruteforce' or engine='auto'.",
            ValueError
        )
    elif engine not in {"analytic", "bruteforce"}:
        lgr.critical_raise(f"Unknown engine '{engine}'. Use 'auto', 'analytic', or 'bruteforce'.",
                           ValueError)

    if engine == "bruteforce":
        if n_parcels > _BRUTEFORCE_N_WARN_THRESHOLD:
            lgr.warning(
                f"Brute-force regional_influence() reruns colocalize() once per excluded "
                f"region ({n_parcels} regions here) -- this can be very slow. Consider "
                f"engine='analytic' if '{method}' supports it (pearson/spearman/"
                "partialpearson/partialspearman/mlr)."
            )
        fun = _get_region_influence_fun_bruteforce(y_colocalize_fun, stat, dtype=dtype, xsea=xsea,
                                                   signed=signed)
    else:
        fun = _get_region_influence_fun_analytic(
            method, adj_r2=adj_r2, r_to_z=r_to_z, dtype=dtype, xsea=xsea, xsea_method=xsea_method,
            signed=signed,
        )

    return fun, engine


def _sort_region_influence(y_infl_list, n_parcels, n_Y, labs_parcels, labs_Y, labs_X=None,
                           dtype=np.float32):
    """Sort per-Y-row regional-influence results into a DataFrame (joint-model
    methods) or a dict of DataFrames keyed by X-map/set label (per-X/per-set
    methods) -- decided generically from the ndim of the per-Y-row result.
    """
    first = np.asarray(y_infl_list[0])

    if first.ndim == 1:
        arr = np.zeros((n_Y, n_parcels), dtype=dtype)
        for y, infl in enumerate(y_infl_list):
            arr[y] = infl
        return pd.DataFrame(arr, index=labs_Y, columns=labs_parcels)

    elif first.ndim == 2:
        n_x = first.shape[0]
        labs_X = list(labs_X) if labs_X is not None else list(range(n_x))
        out = {}
        for i_x, x_lab in enumerate(labs_X):
            arr = np.zeros((n_Y, n_parcels), dtype=dtype)
            for y, infl in enumerate(y_infl_list):
                arr[y] = infl[i_x]
            out[x_lab] = pd.DataFrame(arr, index=labs_Y, columns=labs_parcels)
        return out

    else:
        lgr.critical_raise(f"Unexpected regional-influence result ndim={first.ndim}!", ValueError)


def _pool_region_influence(result, pooled):
    """Reduce a regional-influence result across the Y-axis (subjects/maps).

    ``pooled`` is "mean" or "median" (True is treated as "mean"). Pools the
    per-subject delta directly (median of deltas, not delta of medians) --
    the correct choice for a paired quantity: each subject already
    contributes one paired (stat_full, stat_loo) observation for the same
    excluded region, so the per-subject effect should be summarized across
    subjects, not recomputed from two independently-pooled marginals.
    """
    reducer = np.nanmedian if pooled == "median" else np.nanmean

    def _pool_one(df):
        pooled_row = reducer(df.to_numpy(), axis=0)
        return pd.DataFrame([pooled_row], index=["pooled"], columns=df.columns)

    if isinstance(result, dict):
        return {k: _pool_one(v) for k, v in result.items()}
    return _pool_one(result)
