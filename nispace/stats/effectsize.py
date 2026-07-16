import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _welford_1d(arr):
    n = 0
    mean = 0.0
    M2 = 0.0
    for x in arr:
        if not np.isnan(x):
            n += 1
            delta = x - mean
            mean += delta / n
            M2 += delta * (x - mean)
    var = M2 / (n - 1) if n >= 2 else np.nan
    return n, mean, var

# ---------------------------------------------------
# Cohen's d for independent groups
# ---------------------------------------------------
def cohen(a, b):
    """Cohen's d for independent groups: pooled-SD standardized mean difference.

    Parameters
    ----------
    a, b : array_like, shape (n_obs_a, n_features) / (n_obs_b, n_features)
        Independent-group samples; `a`/`b` need not have the same `n_obs`.
        Reduced along `axis=0` (rows = observations, columns = features/maps).

    Returns
    -------
    d : np.ndarray, shape (n_features,)

    Notes
    -----
    Uses plain `np.mean`/`np.var` (ddof=1) -- NaN in `a`/`b` propagates into
    the corresponding output column rather than being skipped; use
    `cohen_nan` for NaN-aware columns. Not called anywhere in NiSpace's
    internal pipeline: `Y_transform="cohen(a,b)"` dispatches to the
    numba-jitted `cohen_nan_fast` (`core/transform_y.py`), not this function.

    References
    ----------
    Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
    Sciences* (2nd ed.). Routledge.
    """
    a = np.array(a)
    b = np.array(b)

    # Number of elements in each column
    na = a.shape[0]
    nb = b.shape[0]
    dof = na + nb - 2

    # Calculate the pooled standard deviation for each column
    pooled_std = np.sqrt(((na - 1) * np.var(a, ddof=1, axis=0) + (nb - 1) * np.var(b, ddof=1, axis=0)) / dof)

    # Calculate Cohen's d for each column
    d = (np.mean(a, axis=0) - np.mean(b, axis=0)) / pooled_std

    return d

def cohen_nan(a, b):
    """NaN-aware Cohen's d for independent groups (see `cohen`).

    Parameters
    ----------
    a, b : array_like, shape (n_obs_a, n_features) / (n_obs_b, n_features)
        Same convention as `cohen`. NaN entries are excluded per-column via
        `nanmean`/`nanvar` (column-specific `n`/dof).

    Returns
    -------
    d : np.ndarray, shape (n_features,)

    Notes
    -----
    Not called anywhere in NiSpace's internal pipeline: `Y_transform=
    "cohen(a,b)"` dispatches to the numba-jitted `cohen_nan_fast`
    (`core/transform_y.py`), which computes the same statistic faster via a
    single-pass Welford algorithm, not this function.

    References
    ----------
    Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
    Sciences* (2nd ed.). Routledge.
    """
    a = np.array(a)
    b = np.array(b)

    # Number of elements in each column
    na = np.sum(~np.isnan(a), axis=0)
    nb = np.sum(~np.isnan(b), axis=0)
    dof = na + nb - 2

    # Calculate the pooled standard deviation for each column
    pooled_std = np.sqrt(((na - 1) * np.nanvar(a, ddof=1, axis=0) + (nb - 1) * np.nanvar(b, ddof=1, axis=0)) / dof)

    # Calculate Cohen's d for each column
    d = (np.nanmean(a, axis=0) - np.nanmean(b, axis=0)) / pooled_std

    return d

@njit(cache=True, nogil=True)
def cohen_nan_fast(a, b):
    n_cols = a.shape[1]
    d = np.empty(n_cols, dtype=np.float64)
    for j in range(n_cols):
        na, mean_a, var_a = _welford_1d(a[:, j])
        nb, mean_b, var_b = _welford_1d(b[:, j])
        dof = na + nb - 2
        if dof <= 0:
            d[j] = np.nan
        else:
            pooled_std = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / dof)
            d[j] = (mean_a - mean_b) / pooled_std
    return d


# ---------------------------------------------------
# Cohen's d for dependent groups
# ---------------------------------------------------
def cohen_paired(a, b):
    """Cohen's d for paired/dependent samples: mean difference over SD of differences.

    Parameters
    ----------
    a, b : array_like, shape (n_obs, n_features)
        Matched-pair samples; must have identical shape (row `i` in `a` is
        paired with row `i` in `b`).

    Returns
    -------
    d : np.ndarray, shape (n_features,)

    Raises
    ------
    ValueError
        If `a.shape != b.shape`.

    Notes
    -----
    Uses plain `np.mean`/`np.std` (ddof=1) on `a - b` -- NaN in either array
    propagates. Not called anywhere in NiSpace's internal pipeline (no
    `pairedcohen` entry currently uses this function; the paired-cohen
    formula path in `core/transform_y.py` dispatches to the numba-jitted
    `cohen_paired_nan_fast`).

    This is Cohen's *d_z* (mean difference over the SD of the difference
    scores itself), not *d_avg* (mean difference over the average of each
    condition's own, unpaired variance -- the formula pingouin's
    `compute_effsize(paired=True)` uses by default). The two aren't
    interchangeable: `d_z`'s denominator folds in the correlation between
    the paired conditions (`var(diff) = var_a + var_b - 2*cov(a,b)`), while
    `d_avg` discards it entirely. `d_z` is the internally-consistent choice
    here because it's the standardized member of the same family as
    NiSpace's other paired `Y_transform` options (`prc`/`logfc`/plain
    difference) -- all of them, `d_z` included, are pure functions of the
    paired observations alone. `d_avg` would break that by pulling in each
    condition's *unpaired* variance, a quantity none of the others touch.

    References
    ----------
    Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
    Sciences* (2nd ed.). Routledge.
    """
    a = np.array(a)
    b = np.array(b)
    if a.shape != b.shape:
        raise ValueError("Arrays 'a' and 'b' must have the same shape.")

    # Calculate the difference between pairs for each column
    diff = a - b

    # Calculate Cohen's d for each column
    d = np.mean(diff, axis=0) / np.std(diff, ddof=1, axis=0)

    return d

def cohen_paired_nan(a, b):
    """NaN-aware Cohen's d for paired/dependent samples (see `cohen_paired`).

    Parameters
    ----------
    a, b : array_like, shape (n_obs, n_features)
        Same convention as `cohen_paired`; must have identical shape. NaN
        pairs are excluded per-column via `nanmean`/`nanstd` on `a - b`.

    Returns
    -------
    d : np.ndarray, shape (n_features,)

    Raises
    ------
    ValueError
        If `a.shape != b.shape`.

    Notes
    -----
    Not called anywhere in NiSpace's internal pipeline: `Y_transform=
    "pairedcohen(a,b)"` dispatches to the numba-jitted
    `cohen_paired_nan_fast` (`core/transform_y.py`), not this function.

    References
    ----------
    Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
    Sciences* (2nd ed.). Routledge.
    """
    a = np.array(a)
    b = np.array(b)
    if a.shape != b.shape:
        raise ValueError("Arrays 'a' and 'b' must have the same shape.")
    
    # Calculate the difference between pairs for each column
    diff = a - b

    # Calculate Cohen's d for each column
    d = np.nanmean(diff, axis=0) / np.nanstd(diff, ddof=1, axis=0)

    return d

@njit(cache=True, nogil=True)
def cohen_paired_nan_fast(a, b):
    """Numba-jitted NaN-aware Cohen's *d_z* for paired/dependent samples (see `cohen_paired`).

    Parameters
    ----------
    a, b : np.ndarray, shape (n_obs, n_features), dtype float
        Numba-jitted: must be plain 2D `np.ndarray` instances of identical
        shape (row `i` in `a` paired with row `i` in `b`). NaN pairs are
        excluded per-column via a single-pass Welford update over `a - b`
        (mirrors `cohen_nan_fast`'s approach, applied to the difference
        scores rather than to `a`/`b` separately).

    Returns
    -------
    d : np.ndarray, shape (n_features,)
        NaN where fewer than 2 valid (non-NaN in both `a` and `b`) pairs
        remain in a column.

    Notes
    -----
    This is the function `core/transform_y.py`'s `Y_transform=
    "pairedcohen(a,b)"` actually dispatches to -- see `cohen_paired`'s Notes
    for why *d_z* (not *d_avg*) is the correct formula for this use: it's
    the standardized member of the same family as NiSpace's other paired
    `Y_transform` options (`prc`/`logfc`/plain difference), all of which are
    pure functions of the paired observations alone.

    References
    ----------
    Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
    Sciences* (2nd ed.). Routledge.
    """
    n_rows, n_cols = a.shape
    d = np.empty(n_cols, dtype=np.float64)
    for j in range(n_cols):
        n = 0
        mean_d = 0.0
        M2 = 0.0
        for i in range(n_rows):
            ai, bi = a[i, j], b[i, j]
            if not np.isnan(ai) and not np.isnan(bi):
                diff = ai - bi
                n += 1
                delta = diff - mean_d
                mean_d += delta / n
                M2 += delta * (diff - mean_d)
        if n < 2:
            d[j] = np.nan
        else:
            d[j] = mean_d / np.sqrt(M2 / (n - 1))
    return d


# ---------------------------------------------------
# Hedges g 
# ---------------------------------------------------
def hedges(a, b):
    """Hedges' g: small-sample bias-corrected Cohen's d for independent groups.

    Applies the exact correction factor `1 - 3/(4*dof - 1)` to `cohen(a, b)`.

    Parameters
    ----------
    a, b : array_like, shape (n_obs_a, n_features) / (n_obs_b, n_features)
        Same convention as `cohen`.

    Returns
    -------
    g : np.ndarray, shape (n_features,)

    Notes
    -----
    Uses plain `cohen` internally -- NaN in `a`/`b` propagates. Not called
    anywhere in NiSpace's internal pipeline: `Y_transform="hedges(a,b)"`
    dispatches to the numba-jitted `hedges_nan_fast` (`core/transform_y.py`),
    not this function.

    References
    ----------
    Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
    Sciences* (2nd ed.). Routledge.
    Hedges, L. V., & Olkin, I. (1985). *Statistical Methods for
    Meta-Analysis*. Academic Press.
    """
    a = np.array(a)
    b = np.array(b)

    # Calculate Cohen's d for each column
    d = cohen(a, b)

    # Calculate the correction factor for Hedges' g for each column
    na = a.shape[0]
    nb = b.shape[0]
    dof = na + nb - 2
    correction = 1 - (3 / (4 * dof - 1))

    # Calculate Hedges' g for each column
    g = d * correction

    return g

def hedges_nan(a, b):
    """NaN-aware Hedges' g for independent groups (see `hedges`).

    Parameters
    ----------
    a, b : array_like, shape (n_obs_a, n_features) / (n_obs_b, n_features)
        Same convention as `cohen_nan`. NaN entries excluded per-column.

    Returns
    -------
    g : np.ndarray, shape (n_features,)

    Notes
    -----
    Not called anywhere in NiSpace's internal pipeline: `Y_transform=
    "hedges(a,b)"` dispatches to the numba-jitted `hedges_nan_fast`
    (`core/transform_y.py`), not this function.

    References
    ----------
    Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
    Sciences* (2nd ed.). Routledge.
    Hedges, L. V., & Olkin, I. (1985). *Statistical Methods for
    Meta-Analysis*. Academic Press.
    """
    a = np.array(a)
    b = np.array(b)

    # Calculate Cohen's d for each column
    d = cohen_nan(a, b)

    # Calculate the correction factor for Hedges' g for each column
    na = np.sum(~np.isnan(a), axis=0)
    nb = np.sum(~np.isnan(b), axis=0)
    dof = na + nb - 2
    correction = 1 - (3 / (4 * dof - 1))

    # Calculate Hedges' g for each column
    g = d * correction

    return g

@njit(cache=True, nogil=True)
def hedges_nan_fast(a, b):
    n_cols = a.shape[1]
    g = np.empty(n_cols, dtype=np.float64)
    for j in range(n_cols):
        na, mean_a, var_a = _welford_1d(a[:, j])
        nb, mean_b, var_b = _welford_1d(b[:, j])
        dof = na + nb - 2
        if dof <= 0:
            g[j] = np.nan
        else:
            pooled_std = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / dof)
            d = (mean_a - mean_b) / pooled_std
            g[j] = d * (1.0 - 3.0 / (4.0 * dof - 1.0))
    return g


# def hedges_paired(a, b):
#     a = np.array(a)
#     b = np.array(b)
#     if a.shape != b.shape:
#         raise ValueError("Arrays 'a' and 'b' must have the same shape.")

#     # Calculate Cohen's d for each column
#     d = cohen_paired_nan(a, b)

#     # Calculate the correction factor for Hedges' g for each column
#     n = np.sum(~np.isnan(a), axis=0)
#     correction = 1 - (3 / (4 * n - 1))

#     # Calculate Hedges' g for each column
#     g = d * correction

#     return g


# ---------------------------------------------------
# Zscores 
# ---------------------------------------------------
def zscore(a, b=None):
    """Standard (mean/SD) z-score of `a`, optionally against a reference `b`.

    Parameters
    ----------
    a : array_like, shape (n_obs_a, n_features)
        Values to standardize.
    b : array_like, shape (n_obs_b, n_features), optional
        Reference sample supplying the mean/SD (e.g. a null distribution);
        `n_obs_b` need not match `n_obs_a` since `mean`/`std` are computed
        along `axis=0` before broadcasting against `a`. If None (default),
        `a` is standardized against its own mean/SD.

    Returns
    -------
    z : np.ndarray, shape (n_obs_a, n_features)

    Notes
    -----
    Uses plain `np.mean`/`np.std` (ddof=1) -- NaN in `a` or `b` propagates;
    use `zscore_nan` for NaN-aware columns. Not called by the
    `Y_transform="zscore(...)"` formula path (which dispatches to the
    numba-jitted `zscore_nan_fast`), but `zscore_nan` (the NaN-aware sibling)
    is called directly by `api.NiSpace.normalize_colocalizations()` to
    z-score observed colocalizations against permutation null distributions.
    """
    a = np.array(a)

    if b is not None:
        b = np.array(b)
        z = (a - np.mean(b, axis=0)) / np.std(b, ddof=1, axis=0)
    else:
        z = (a - np.mean(a, axis=0)) / np.std(a, ddof=1, axis=0)

    return z

def zscore_nan(a, b=None):
    """NaN-aware standard z-score of `a`, optionally against a reference `b`.

    Parameters
    ----------
    a : array_like, shape (n_obs_a, n_features)
        Values to standardize.
    b : array_like, shape (n_obs_b, n_features), optional
        Reference sample supplying the mean/SD, reduced along `axis=0` via
        `nanmean`/`nanstd` (NaN entries excluded per-column). If None
        (default), `a` is standardized against its own NaN-aware mean/SD.

    Returns
    -------
    z : np.ndarray, shape (n_obs_a, n_features)

    Notes
    -----
    Called directly (not via a numba `_fast` twin) by
    `api.NiSpace.normalize_colocalizations(z_method="standard")` to z-score
    observed colocalization values against the corresponding null
    distribution array.
    """
    a = np.array(a)

    if b is not None:
        b = np.array(b)
        z = (a - np.nanmean(b, axis=0)) / np.nanstd(b, ddof=1, axis=0)
    else:
        z = (a - np.nanmean(a, axis=0)) / np.nanstd(a, ddof=1, axis=0)

    return z

@njit(cache=True, nogil=True)
def _col_stats(arr):
    """Per-column Welford mean and std for a 2D array."""
    n_cols = arr.shape[1]
    means = np.empty(n_cols, dtype=np.float64)
    stds  = np.empty(n_cols, dtype=np.float64)
    for j in range(n_cols):
        _, mean, var = _welford_1d(arr[:, j])
        means[j] = mean
        stds[j]  = np.sqrt(var)   # nan propagates when n < 2
    return means, stds

def zscore_nan_fast(a, b=None):
    ref = a if b is None else b
    means, stds = _col_stats(ref)
    return (a - means) / stds    # numpy broadcast: row-major, SIMD-friendly


# ---------------------------------------------------------------------------
# Robust Zscores
# ---------------------------------------------------------------------------
def rzscore_nan(a, b=None):
    """NaN-aware robust z-score of `a` (median/MAD), optionally against a reference `b`.

    Parameters
    ----------
    a : array_like, shape (n_obs_a, n_features)
        Values to standardize.
    b : array_like, shape (n_obs_b, n_features), optional
        Reference sample supplying the median/MAD, reduced along `axis=0`
        via `nanmedian` (NaN entries excluded per-column). If None
        (default), `a` is standardized against its own median/MAD.

    Returns
    -------
    z : np.ndarray, shape (n_obs_a, n_features)
        Columns where the reference MAD is exactly 0 are set to NaN (a
        constant reference has no meaningful robust scale).

    Notes
    -----
    Called directly (not via a numba `_fast` twin) by
    `api.NiSpace.normalize_colocalizations(z_method="robust")` (the default)
    to z-score observed colocalization values against the corresponding null
    distribution array. The `Y_transform="rzscore(...)"` formula path uses
    the separate numba-jitted `rzscore_nan_fast` instead.
    """
    a = np.array(a)
    if b is not None:
        b = np.array(b)
        med = np.nanmedian(b, axis=0)
        mad = np.nanmedian(np.abs(b - med), axis=0)
    else:
        med = np.nanmedian(a, axis=0)
        mad = np.nanmedian(np.abs(a - med), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (a - med) / (1.4826 * mad)
    zero_mad = np.atleast_1d(mad == 0)
    if np.any(zero_mad):
        result = np.where(zero_mad, np.nan, result)
    return result

@njit(cache=True, nogil=True)
def _nanmedian_1d(arr):
    """NaN-safe median of a 1D array via sort."""
    n = 0
    for x in arr:
        if not np.isnan(x):
            n += 1
    if n == 0:
        return np.nan
    valid = np.empty(n, dtype=np.float64)
    k = 0
    for x in arr:
        if not np.isnan(x):
            valid[k] = x
            k += 1
    valid = np.sort(valid)
    mid = n // 2
    if n % 2 == 0:
        return (valid[mid - 1] + valid[mid]) / 2.0
    else:
        return valid[mid]

@njit(cache=True, nogil=True)
def _col_robust_stats(arr):
    """Per-column NaN-safe median and MAD for a 2D array."""
    n_cols = arr.shape[1]
    medians = np.empty(n_cols, dtype=np.float64)
    mads    = np.empty(n_cols, dtype=np.float64)
    for j in range(n_cols):
        col = arr[:, j]
        med = _nanmedian_1d(col)
        medians[j] = med
        mads[j] = _nanmedian_1d(np.abs(col - med))   # NaN propagates, skipped in median
    return medians, mads

def rzscore_nan_fast(a, b=None):
    ref = a if b is None else b
    medians, mads = _col_robust_stats(ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (a - medians) / (1.4826 * mads)
    zero_mad = mads == 0
    if np.any(zero_mad):
        result[:, zero_mad] = np.nan
    return result


# ---------------------------------------------------
# Percent change 
# ---------------------------------------------------
def prc(a, b):
    """Percent change `(a - b) / a * 100`, element-wise.

    Parameters
    ----------
    a, b : array_like, shape (n_obs, n_features)
        Paired values (`a` is the reference/denominator); must have
        identical shape.

    Returns
    -------
    p : np.ndarray, shape (n_obs, n_features)
        NaN where `a == 0` (avoids division by zero).

    Raises
    ------
    ValueError
        If `a.shape != b.shape`.

    Notes
    -----
    NaN in `a`/`b` (other than the `a == 0` guard) is not explicitly masked
    -- it propagates through the arithmetic like any other float NaN; there
    is no separate `prc_nan` variant. Not called anywhere in NiSpace's
    internal pipeline: `Y_transform="prc(a,b)"` dispatches to the
    numba-jitted `prc_fast` (`core/transform_y.py`), not this function.
    """
    a = np.array(a, dtype=float)  # Ensure input is a numpy array and convert to float for safe division
    b = np.array(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Arrays 'a' and 'b' must have the same shape.")

    # Calculate percentage change
    # Use np.where to avoid division by zero
    p = np.where(a != 0, (a - b) / a * 100, np.nan)

    return p

@njit(cache=True, nogil=True)
def prc_fast(a, b):
    n_rows, n_cols = a.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    for i in range(n_rows):
        for j in range(n_cols):
            ai = a[i, j]
            result[i, j] = np.nan if ai == 0.0 else (ai - b[i, j]) / ai * 100.0
    return result


# ---------------------------------------------------
# Log fold change: log((a+c) / (b+c))
# c = shift to ensure all values are positive.
# For raw positive data (e.g. CT in mm): c = 0.
# For z-scored / residual data with negatives: c is
# auto-computed as |global_min| + eps.
# Symmetric under permutation regardless of c:
#   swap(a,b) -> log((b+c)/(a+c)) = -logfc(a,b)  ✓
# -> null distribution is always exactly 0-centered.
# ---------------------------------------------------
def centile_fast(a, b=None):
    """NaN-aware percentile rank of `a` within a reference distribution `b`.

    For each column, ranks every value of `a` against the sorted, NaN-excluded
    values of the reference (`b`, or `a` itself if `b` is None) via
    `np.searchsorted(..., side="right")`, i.e. the percentage of reference
    values that are `<=` the query value.

    Parameters
    ----------
    a : array_like, shape (n_obs_a, n_features)
        Values to rank.
    b : array_like, shape (n_obs_b, n_features), optional
        Reference sample supplying the distribution to rank against
        (e.g. a null distribution); `n_obs_b` need not match `n_obs_a`. If
        None (default), `a` is ranked against its own values.

    Returns
    -------
    p : np.ndarray, shape (n_obs_a, n_features)
        Percentile ranks in `[0, 100]`. NaN wherever `a` is NaN, or where the
        reference column has zero non-NaN values.

    Notes
    -----
    Despite the `_fast` name, this is **not** numba-jitted (plain Python with
    a per-column/per-row loop over `np.searchsorted`) -- unlike every other
    `_fast` function in this module. No plain (non-`_fast`) sibling exists.
    Backs `Y_transform="centile(a)"`/`"centile(a,b)"` (`core/transform_y.py`).
    """
    a = np.array(a, dtype=float)
    ref = a if b is None else np.array(b, dtype=float)
    n_cols = ref.shape[1]
    result = np.empty_like(a, dtype=float)
    for j in range(n_cols):
        col_ref = ref[:, j]
        col_ref_valid = np.sort(col_ref[~np.isnan(col_ref)])
        n_valid = len(col_ref_valid)
        for i in range(a.shape[0]):
            v = a[i, j]
            if np.isnan(v) or n_valid == 0:
                result[i, j] = np.nan
            else:
                result[i, j] = np.searchsorted(col_ref_valid, v, side="right") / n_valid * 100
    return result


def logfc_nan(a, b):
    """NaN-aware log fold-change `log((a + c) / (b + c))`, with an auto-computed shift `c`.

    `c` is 0 for all-non-negative input (e.g. raw cortical thickness), or
    `|global_min(a, b)| + 1e-6` when either array has negative values (e.g.
    z-scored/residualized data) -- just enough to make every shifted value
    positive for the log. The shift makes the statistic symmetric under
    permutation: `logfc(b, a) == -logfc(a, b)` exactly, so its null
    distribution is always centered on 0.

    Parameters
    ----------
    a, b : array_like, shape (n_obs, n_features)
        Paired values; must have identical shape.

    Returns
    -------
    lfc : np.ndarray, shape (n_obs, n_features)
        NaN wherever `a` or `b` is NaN (explicitly masked).

    Notes
    -----
    Not called anywhere in NiSpace's internal pipeline: `Y_transform=
    "logfc(a,b)"` dispatches to the numba-jitted `logfc_fast`
    (`core/transform_y.py`), not this function.
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Arrays 'a' and 'b' must have the same shape.")
    global_min = min(np.nanmin(a), np.nanmin(b))
    shift = max(0.0, -global_min) + 1e-6
    return np.where(np.isnan(a) | np.isnan(b), np.nan, np.log((a + shift) / (b + shift)))

@njit(cache=True, nogil=True)
def logfc_fast(a, b):
    # compute global min over both arrays (ignoring NaN) to derive shift
    global_min = np.inf
    n_rows, n_cols = a.shape
    for i in range(n_rows):
        for j in range(n_cols):
            v = a[i, j]
            if not np.isnan(v) and v < global_min:
                global_min = v
            v = b[i, j]
            if not np.isnan(v) and v < global_min:
                global_min = v
    shift = -global_min + 1e-6 if global_min < 0.0 else 0.0

    result = np.empty((n_rows, n_cols), dtype=np.float64)
    for i in range(n_rows):
        for j in range(n_cols):
            ai = a[i, j]
            bi = b[i, j]
            if np.isnan(ai) or np.isnan(bi):
                result[i, j] = np.nan
            else:
                result[i, j] = np.log((ai + shift) / (bi + shift))
    return result
