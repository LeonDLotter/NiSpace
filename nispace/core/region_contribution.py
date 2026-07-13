"""Regional contribution -- per-region decomposition of a correlation into
each region's own additive share, plus a quadrant label distinguishing
co-elevation ("high_high") from co-depression ("low_low") from discordant
regions.

Not a leave-one-out method (see core/region_influence.py for that) -- this
is a much simpler, O(n_parcels) decomposition, since no exclusion/refit is
involved: ``contribution_i = zx_i * zy_i`` is literally region i's own
additive term in ``rho = mean(zx*zy)`` (zx/zy are population-standardized,
ddof=0, matching pearson()'s own centered-sums convention).

Exists specifically because regional_influence() is structurally symmetric
between high-high and low-low concordant regions (both reinforce a positive
correlation identically) -- that symmetry is inherent to what Pearson/
Spearman correlation measures, not fixable within the LOO framework. The
quadrant label is the actual point of this feature: it lets callers filter
to just the co-elevation quadrant instead of conflating it with
co-depression under one signed number.

Scoped to the 4 bidirectional correlation methods only (pearson/spearman/
partialpearson/partialspearman) -- quadrants aren't a meaningful concept for
R^2-based methods, and those methods' primary stat has no "high/low" side to
begin with.
"""

import numpy as np

import logging
lgr = logging.getLogger(__name__)

from .region_influence import _get_aggr_fun_axis0


_CONTRIBUTION_METHODS = {"pearson", "spearman", "partialpearson", "partialspearman"}


def _zscore_nan_pair(x, y):
    """Paired NaN-safe zscore: x and y share ONE mask, and both are
    standardized using mean/std computed only over that shared mask.

    This is not the same as zscoring x and y independently via nanmean/
    nanstd on each array separately (a real bug caught by live verification
    against a real dataset, not just synthetic tests -- worth remembering
    why): pearson(x[mask], y[mask]) -- what the underlying colocalization
    closure actually computes -- uses y's mean/std over the *same* mask as
    x, not over all of y. If x has NaNs at positions where y doesn't (a real
    case: a reference map missing a couple of parcels a phenotype map has
    values for), zscoring y independently uses a different effective sample
    than the correlation itself did, giving a `contribution` whose mean
    silently drifts from the true rho by a small but real amount.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    zx = np.full(x.shape, np.nan, dtype=np.float64)
    zy = np.full(y.shape, np.nan, dtype=np.float64)

    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2:
        return zx, zy

    mx, sx = x[mask].mean(), x[mask].std()
    my, sy = y[mask].mean(), y[mask].std()
    if sx == 0 or sy == 0:
        return zx, zy

    zx[mask] = (x[mask] - mx) / sx
    zy[mask] = (y[mask] - my) / sy
    return zx, zy


def _quadrant_from_z(zx, zy):
    """Elementwise quadrant label from the sign of two zscore arrays.

    "high_high" (zx>0 & zy>0) and "low_low" (zx<0 & zy<0) are both
    concordant with a positive correlation -- the distinction `contribution`
    alone can't make, since both give the same positive value. "discordant"
    covers the other two sign combinations. NaN in either input propagates
    to None (not a string) so it doesn't silently get treated as a category.
    """
    q = np.where((zx > 0) & (zy > 0), "high_high",
        np.where((zx < 0) & (zy < 0), "low_low", "discordant"))
    return np.where(np.isnan(zx) | np.isnan(zy), None, q)


def _get_region_contribution_fun(method, dtype=np.float32, xsea=False, xsea_method=None):
    """Build the per-Y-row regional-contribution closure for one method.

    Returns a closure producing (contribution, quadrant) -- both shape
    (n_x, n_parcels) for non-XSEA, (n_sets, n_parcels) for XSEA -- since
    all 4 supported methods are per-X-predictor/per-set methods (there is no
    joint-model case here, unlike regional_influence()).
    """

    if method not in _CONTRIBUTION_METHODS:
        lgr.critical_raise(f"regional_contribution() does not support method '{method}'. "
                           f"Supported: {sorted(_CONTRIBUTION_METHODS)}.",
                           ValueError)

    if not xsea:
        def _y_region_contribution(X, y, weights=None):
            n_x, n_parcels = X.shape
            contrib = np.empty((n_x, n_parcels), dtype=dtype)
            quadrant = np.empty((n_x, n_parcels), dtype=object)
            for i_x in range(n_x):
                zx, zy = _zscore_nan_pair(X[i_x], y)
                contrib[i_x] = (zx * zy).astype(dtype)
                quadrant[i_x] = _quadrant_from_z(zx, zy)
            return contrib, quadrant

    else:
        aggr_fn = _get_aggr_fun_axis0(xsea_method)
        weighted = "weighted" in xsea_method

        def _y_region_contribution(X_dict, y, weights=None):
            n_parcels = y.shape[0]
            n_sets = len(X_dict)
            contrib = np.empty((n_sets, n_parcels), dtype=dtype)
            quadrant = np.empty((n_sets, n_parcels), dtype=object)
            for i_s, (set_name, set_X) in enumerate(X_dict.items()):
                n_genes = set_X.shape[0]
                zx_per_gene = np.empty((n_genes, n_parcels), dtype=np.float64)
                zy_per_gene = np.empty((n_genes, n_parcels), dtype=np.float64)
                contrib_per_gene = np.empty((n_genes, n_parcels), dtype=np.float64)
                for i_g in range(n_genes):
                    zx_g, zy_g = _zscore_nan_pair(set_X[i_g], y)
                    zx_per_gene[i_g] = zx_g
                    zy_per_gene[i_g] = zy_g
                    contrib_per_gene[i_g] = zx_g * zy_g
                if weighted:
                    w = weights[set_name]
                    contrib[i_s] = aggr_fn(contrib_per_gene, w).astype(dtype)
                else:
                    contrib[i_s] = aggr_fn(contrib_per_gene).astype(dtype)
                # quadrant derived from the plain (non-abs) mean zx/zy across genes --
                # the magnitude aggregation (which may use abs, e.g. "absmean") is for
                # `contrib`; the quadrant label always reflects unsigned-mean direction,
                # since "high_high"/"low_low" is inherently directional.
                zx_mean = np.nanmean(zx_per_gene, axis=0)
                zy_mean = np.nanmean(zy_per_gene, axis=0)
                quadrant[i_s] = _quadrant_from_z(zx_mean, zy_mean)
            return contrib, quadrant

    return _y_region_contribution


def _sort_region_contribution(y_list, n_parcels, n_Y, labs_parcels, labs_Y, labs_X=None,
                              dtype=np.float32):
    """Sort per-Y-row (contribution, quadrant) tuples into two dicts of
    DataFrames, both keyed by X-map/set label -- always dict-shaped, since
    all 4 supported methods are per-X-predictor/per-set methods.
    """
    import pandas as pd

    n_x = y_list[0][0].shape[0]
    labs_X = list(labs_X) if labs_X is not None else list(range(n_x))

    contrib_out, quadrant_out = {}, {}
    for i_x, x_lab in enumerate(labs_X):
        contrib_arr = np.zeros((n_Y, n_parcels), dtype=dtype)
        quadrant_arr = np.empty((n_Y, n_parcels), dtype=object)
        for y, (contrib, quadrant) in enumerate(y_list):
            contrib_arr[y] = contrib[i_x]
            quadrant_arr[y] = quadrant[i_x]
        contrib_out[x_lab] = pd.DataFrame(contrib_arr, index=labs_Y, columns=labs_parcels)
        quadrant_out[x_lab] = pd.DataFrame(quadrant_arr, index=labs_Y, columns=labs_parcels)

    return contrib_out, quadrant_out
