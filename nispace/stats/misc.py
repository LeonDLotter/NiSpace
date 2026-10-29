import logging
import warnings
import numpy as np
import pandas as pd
from numba import njit
from itertools import permutations
from ..utils.utils import tqdm
from joblib import Parallel, delayed

from scipy.stats import zscore, norm, rankdata
from statsmodels.stats.multitest import multipletests

from .coloc import _ols_beta_via_eigh, _pca_via_eigh

lgr = logging.getLogger(__name__)


@njit(cache=True)
def np_any_axis1(x):
    """Numba-compatible ``np.any(x, axis=1)``.

    Numba does not support the ``axis`` argument of ``np.any``; this is a
    manual row-wise OR reduction over a 2D ``np.ndarray``, used internally by
    :func:`residuals_nan`/:func:`partial_residuals_nan` to build a combined
    NaN mask across their stacked input columns.
    """
    out = np.zeros(x.shape[0], dtype=np.bool_)
    for i in range(x.shape[1]):
        out = np.logical_or(out, x[:, i])
    return out 


@njit(cache=True, nogil=True)
def residuals(x, y, decenter=False):
    """OLS residuals of ``y`` on ``x`` (plus intercept).

    Numba-compiled; requires plain ``np.ndarray`` inputs (no pandas) and does
    **not** handle NaN — any NaN in ``x``/``y`` propagates silently through
    ``np.linalg.pinv``. Dead code on the live pipeline: nothing in NiSpace
    actually calls this function — every real caller uses the NaN-tolerant
    :func:`residuals_nan` instead.

    Args:
        x (numpy.ndarray): shape (n_values, n_predictors)
        y (numpy.ndarray): shape (n_values, 1) or (n_values,)
        decenter (bool, optional): add mean of y to residuals before return

    Returns:
        numpy.ndarray: 1D array of residuals w or w/o added mean of y
    """

    X = np.column_stack((x, np.ones(x.shape[0], dtype=x.dtype)))
    beta = np.linalg.pinv((X.T).dot(X)).dot(X.T.dot(y))
    y_hat = np.dot(X, beta)
    resid = y - y_hat

    if decenter:
        resid += y.mean()

    return resid


@njit(cache=True, nogil=True)
def residuals_nan(x, y, decenter=False):
    """OLS residuals of ``y`` on ``x`` (plus intercept), NaN-tolerant.

    NaN-tolerant sibling of :func:`residuals`: rows where any of ``x``/``y``
    is NaN are dropped before fitting (via :func:`np_any_axis1`), and NaN is
    restored at those positions in the output. Numba-compiled; requires
    plain ``np.ndarray`` inputs (no pandas). This is the version actually
    used throughout NiSpace: by ``core/clean_y.py`` for covariate regression
    and by ``core/colocalize.py``'s ``_rank_regress`` to residualize X/Y
    against Z before correlating (the real mechanism behind
    ``method="partialpearson"/"partialspearman"`` in ``colocalize()`` — see
    :func:`nispace.stats.coloc.partialcorr` for the standalone version
    without NaN handling that is *not* on this code path).

    Coefficients come from :func:`nispace.stats.coloc._ols_beta_via_eigh` (``eigh`` on the
    Gram matrix, not ``np.linalg.pinv`` -- see its docstring for why).

    Args:
        x (numpy.ndarray): shape (n_values, n_predictors)
        y (numpy.ndarray): shape (n_values, 1) or (n_values,)
        decenter (bool, optional): add mean of y to residuals before return

    Returns:
        numpy.ndarray: 1D array of residuals w or w/o added mean of y, NaN
        where input had NaN
    """
    nan_mask = np_any_axis1(np.isnan(np.column_stack((x, y))))
    x_ = x[~nan_mask]
    y_ = y[~nan_mask]

    X = np.column_stack((x_, np.ones(x_.shape[0], dtype=x_.dtype)))
    beta = _ols_beta_via_eigh(X, y_)
    y_hat = np.dot(X, beta)
    resid_ = y_ - y_hat

    if decenter:
        resid_ += y_.mean()

    resid = np.full(y.shape, np.nan, dtype=y.dtype)
    resid[~nan_mask] = resid_

    return resid


@njit(cache=True, nogil=True)
def partial_residuals_nan(x_nuisance, x_protect, y):
    """Partial residuals: regress x_nuisance from y while controlling for x_protect.

    Fits a joint model [x_nuisance | x_protect | intercept] so that nuisance
    coefficients are estimated controlling for the protected variables, then
    removes only the nuisance component. The protected effects (e.g. group
    differences) are preserved in the returned values.

    Numba-compiled; requires plain ``np.ndarray`` inputs (no pandas).
    NaN-tolerant: rows where any input is NaN are dropped before fitting
    (via :func:`np_any_axis1`), and NaN is restored at those positions in the
    output. Used by ``core/clean_y.py``'s "protect" covariate-regression mode
    (protected OLS), where ``x_protect`` holds variables whose effect should
    be kept in ``y`` while ``x_nuisance`` is removed.

    Coefficients come from :func:`nispace.stats.coloc._ols_beta_via_eigh` (``eigh`` on the
    Gram matrix, not ``np.linalg.pinv`` -- see its docstring for why).

    Args:
        x_nuisance (numpy.ndarray): shape (n, p) — confounds to remove
        x_protect  (numpy.ndarray): shape (n, q) — variables to control for but keep
        y          (numpy.ndarray): shape (n,)

    Returns:
        numpy.ndarray: shape (n,), NaN where input had NaN
    """
    nan_mask = np_any_axis1(np.isnan(np.column_stack((x_nuisance, x_protect, y))))
    xn = x_nuisance[~nan_mask]
    xp = x_protect[~nan_mask]
    y_ = y[~nan_mask]

    X_full = np.column_stack((xn, xp, np.ones(xn.shape[0], dtype=xn.dtype)))
    n_nuisance = x_nuisance.shape[1]
    beta = _ols_beta_via_eigh(X_full, y_)

    resid = np.full(y.shape, np.nan, dtype=y.dtype)
    resid[~nan_mask] = y_ - xn.dot(beta[:n_nuisance])
    return resid


def rho_to_z(array, replace_1=1 - np.finfo(float).eps):
    """Fisher's z-transformation of correlation coefficients.

    Plain numpy (no numba). Values isclose to 1 are clipped to ``replace_1``
    first, since ``arctanh(1) == inf``. Used in ``core/region_influence.py``
    and ``core/colocalize.py`` to average/compare correlation coefficients
    on a variance-stabilized scale.
    """
    array = np.array(array)
    array[np.isclose(array, 1)] = replace_1
    array_z = np.arctanh(array)
    return array_z


def z_to_rho(array):
    """Inverse Fisher's z-transformation of correlation coefficients.

    Plain numpy (no numba). Used in ``diagnostics.py``'s ``local_colocalization()``
    to convert its internally Fisher-z-scale statistic back to raw rho for display
    when ``r_to_z=False`` is requested, and in ``api.py``'s
    ``_cwr_omnibus_aggregate()`` to report ``omnibus_stat="rho"`` as a correlation
    after Fisher-z averaging; exercised directly by round-trip tests in
    ``tests/test_stats_misc.py``.
    """
    array = np.array(array)
    array_rho = np.tanh(array)
    return array_rho


def zscore_df(df, along="cols", force_df=True):
    """Z-standardize a DataFrame/Series/ndarray, preserving its type.

    Plain numpy/pandas (via ``scipy.stats.zscore``, ``ddof=0``, NaN omitted
    per column/row). Widely used across NiSpace (``api.py``, ``core/permute.py``,
    ``core/clean_y.py``, ``core/nullmaps.py``, ``datasets.py``) for
    self-normalization of maps/vectors (population, not sample, convention).

    Args:
        df (pandas dataframe, Series, or ndarray): input data
        along (str, optional): Either "cols" or "rows". Defaults to "cols".
        force_df (bool, optional): for Series/ndarray input, wrap the output
            in a DataFrame instead of returning a Series/ndarray. Defaults
            to True.

    Returns:
        pd.DataFrame or pd.Series: standardized dataframe/series

    Raises:
        ValueError: if `along` is not "cols" or "rows".
        TypeError: if `df` is not a DataFrame, Series, or ndarray.
    """

    if along=="cols":
        axis = 0
    elif along=="rows":
        axis = 1
    else:
        raise ValueError(f"Option along=={along} not defined!")
        
    # ddof=0: standardizes the given map/vector to itself, not estimating a
    # population parameter from a reference sample -- matches scipy/sklearn
    # convention, used throughout for X/Y/Z standardization
    arr_stand = zscore(df, axis=axis, ddof=0, nan_policy="omit")
    
    # DataFrame
    if isinstance(df, pd.DataFrame):
        df_stand = pd.DataFrame(
            data=arr_stand,
            columns=df.columns,
            index=df.index
        )
        
    # Series
    elif isinstance(df, pd.Series):
        if force_df:
            df_stand = pd.DataFrame(
                data=arr_stand,
                columns=[df.name],
                index=df.index
            )
        else:
            df_stand = pd.Series(
                data=arr_stand,
                index=df.index,
                name=df.name
            )
    
    # ndarray
    elif isinstance(df, np.ndarray):
        if force_df:
            df_stand = pd.DataFrame(
                data=arr_stand
            )
        else:
            df_stand = arr_stand
            
    # not defined
    else:
        raise TypeError(f"Input data type {type(df)} not defined!")

    return df_stand


def pca(data, n_components=None, standardize=False, return_ev=False):
    """Principal component analysis (PCA) of a data matrix.

    Reduces `data`'s columns to `n_components` orthogonal principal components, computed via
    eigendecomposition of the (small) column covariance matrix
    (:func:`nispace.stats.coloc._pca_via_eigh`) rather than SVD -- more numerically robust
    (avoids a known LAPACK flakiness in SVD-based PCA under some BLAS backends) and, for the
    typical "few columns, many rows" shape, faster. Component 1's sign is fixed to be
    non-negatively correlated with the mean of the (centered) input columns -- a reproducible,
    interpretable convention ("this component increases when the average input column
    increases") instead of an arbitrary, LAPACK-dependent one; components beyond the first
    keep their raw (arbitrary but deterministic-per-run) sign, since they are orthogonal to
    component 1 by construction and have no equivalent "overall mean" to anchor to.

    Args:
        data (pandas.DataFrame or numpy.ndarray): shape (n_obs, n_features). Must be 2D, with
            at least 2 rows and 1 column, and free of NaN.
        n_components (int, optional): Number of components to retain. Defaults to
            ``min(n_obs, n_features)`` (all components).
        standardize (bool, optional): Z-score each column (population convention, ddof=0)
            before running PCA -- i.e. run PCA on the correlation rather than covariance
            structure, appropriate when columns are on different scales. Defaults to False.
        return_ev (bool, optional): Also return each retained component's fraction of total
            variance explained. Defaults to False.

    Returns:
        pandas.DataFrame or numpy.ndarray: shape (n_obs, n_components), matching `data`'s
        input type. DataFrame output keeps `data`'s row index and uses "PC1", "PC2", ... as
        column names; ndarray input gets an ndarray output.
        numpy.ndarray: shape (n_components,), each retained component's fraction of total
        variance explained. Only returned (as a second element, in a tuple) if
        ``return_ev=True``.

    Raises:
        TypeError: if `data` is not a DataFrame or ndarray.
        ValueError: if `data` is not 2D, has fewer than 2 rows or 1 column, contains NaN, or
            `n_components` is out of range.
    """
    if isinstance(data, pd.DataFrame):
        index = data.index
        arr = data.to_numpy(dtype=float)
    elif isinstance(data, np.ndarray):
        arr = np.asarray(data, dtype=float)
        index = None
    else:
        raise TypeError(f"Input data type {type(data)} not supported -- pass a pandas "
                        "DataFrame or a numpy ndarray.")

    if arr.ndim != 2:
        raise ValueError(f"Input data must be 2D (n_obs, n_features), got shape {arr.shape}.")

    n_obs, n_features = arr.shape
    if n_obs < 2 or n_features < 1:
        raise ValueError(f"Input data must have at least 2 rows and 1 column, got shape "
                        f"{arr.shape}.")

    if np.isnan(arr).any():
        raise ValueError("Input data contains NaN -- pca() does not handle missing data; "
                        "drop or impute NaN rows/columns first.")

    max_components = min(n_obs, n_features)
    if n_components is None:
        n_components = max_components
    elif not (1 <= n_components <= max_components):
        raise ValueError(f"n_components={n_components} out of range -- must be between 1 "
                        f"and min(n_obs={n_obs}, n_features={n_features})={max_components}.")

    if standardize:
        arr = zscore(arr, axis=0, ddof=0)

    x_pcs = _pca_via_eigh(arr, n_components)

    if return_ev:
        ev = x_pcs.var(axis=0, ddof=0) / arr.var(axis=0, ddof=0).sum()

    if index is not None:
        out = pd.DataFrame(x_pcs, index=index,
                           columns=[f"PC{i + 1}" for i in range(n_components)])
    else:
        out = x_pcs

    return (out, ev) if return_ev else out


def permute_groups(groups, strategy="shuffle", paired=False, subjects=None, n_perm=1,
                   n_proc=1, seed=None, verbose=False):
    """Permute group-membership labels for group-comparison null distributions.

    Plain numpy/pandas + joblib (no numba); operates on discrete group labels,
    not continuous data, so NaN handling doesn't apply. Used internally by
    ``api.py``'s group-comparison null generation (``groups_*`` kwargs of
    :meth:`~nispace.api.NiSpace.permute`).

    Parameters
    ----------
    groups : array-like
        Group/session labels, length n_samples.
    strategy : str, default="shuffle"
        Must contain one of ``"shuff"``, ``"draw"`` (unpaired only), or
        ``"prop"``. What each does depends on `paired`:

        * ``"shuff"`` -- **unpaired**: one free random permutation of the
          whole `groups` vector (``rng.permutation``) -- the classic
          permutation-test null, exchanging labels with no constraint beyond
          the overall label counts staying fixed. **paired**: an independent
          random permutation of each subject's own labels across their
          sessions (a free per-subject sign-flip) -- every one of the
          ``n_labels!`` orderings is equally likely for every subject,
          independently of every other subject.
        * ``"prop"`` (proportional) -- **unpaired**: iteratively assigns each
          new group the same *proportion* of samples from every original
          group as that original group's overall share of the sample, so the
          permuted groups reproduce the original composition as closely as
          floor-rounding allows. **paired**: splits subjects into exactly
          ``floor(n_subjects / n_labels!)`` blocks, one per possible session-label
          ordering (for 2 sessions: exactly half the subjects keep the
          original order, half get the fully swapped order) -- a *fixed*,
          balanced split, not a free per-subject choice. This mirrors
          JuSpace's own exact-permutation scheme :cite:`dukart2021` (``compute_exact_pvalue.m``):
          the unpaired "es between" case (``options(1)==1``) draws a fixed,
          ``round()``-determined count from each original group's pool into
          each new group (not a free permutation), and the paired "es
          within" case (``options(1)==2``) uses a fixed vector
          ``v = [ones(floor(N/2),1); 2*ones(floor(N/2)+1,1)]`` to force an
          exact half-swap/half-keep split, randomized only in *which*
          subjects land in which half -- the same restriction as NiSpace's
          paired ``"prop"`` above.
        * ``"draw"`` -- **unpaired only**: draws `groups` with replacement
          (``rng.choice(..., replace=True)``), a bootstrap-style resample
          rather than a strict relabeling (label counts are not preserved
          exactly, and repeats are possible).

        **Default is ``"shuffle"``, not ``"prop"``, because ``"prop"``'s fixed
        (unpaired) / exactly-balanced (paired) restriction shrinks the space
        of achievable permutations relative to the free ``"shuffle"`` scheme,
        which narrows the resulting null distribution and inflates the false
        positive rate.** This was empirically confirmed via
        ``docs/nb_benchmarks/bench03_group_permutation_fpr.ipynb`` (GRF-based
        FPR benchmark for :meth:`~nispace.api.NiSpace.permute`'s
        ``what="groups"`` mode): with ``strategy="proportional"``, FPR was
        inflated for both paired and unpaired designs, worst for **paired**
        at small `n_subjects` (e.g. FPR ~0.09 at n_subjects=10 vs. nominal
        0.05, shrinking towards ~0.06 by n_subjects=30 but never fully
        resolving) -- consistent with the paired case's exact 50/50 split
        being the more restrictive of the two ``"prop"`` variants. Switching
        both paired and unpaired to ``"shuffle"`` resolved this completely
        (FPR ~0.05 across all tested `n_subjects`, with no residual trend).
        ``"prop"``/``"draw"`` remain available for cases that specifically
        need a fixed-composition or bootstrap-style null.
    paired : bool, default=False
        If True, permute within each subject (across that subject's own
        group/session labels) instead of across the whole sample; requires
        `subjects`.
    subjects : array-like, optional
        Subject identifier per sample, same length as `groups`. Required if
        `paired=True`.
    n_perm : int, default=1
        Number of permuted label vectors to generate.
    n_proc : int, default=1
        Number of parallel jobs (joblib).
    seed : int, optional
        Base random seed; each permutation draws from ``seed + i``.
    verbose : bool or "debug", optional
        Show a progress bar; "debug" additionally prints per-group sample
        counts for the "prop" strategy (permutation 0 only).

    Returns
    -------
    ndarray or list of ndarray
        Permuted group-label vector(s), same dtype as `groups`. A single
        array if `n_perm` == 1, otherwise a list of length `n_perm`.

    Raises
    ------
    ValueError
        If `paired=True` without `subjects`, if group sizes are unequal
        across sessions in the paired case, or if `strategy` doesn't match
        a known mode.

    References
    ----------
    :cite:`dukart2021`.
    """
    groups = np.array(groups)
    n = len(groups)
    group_labels, group_sizes = np.unique(groups, return_counts=True)
    n_labels = len(group_labels)
    lgr.debug(f"{n} samples with labels: {group_labels} with sizes {group_sizes}")
    
    if paired:
        # get subjects
        if subjects is None:
            raise ValueError("Paired permutation requires argument 'subjects'!")
        subjects = np.array(subjects)
        unique_subjects = np.unique(subjects)
        n_subjects = len(unique_subjects)
        
        # check input
        if len(subjects) != len(groups):
            raise ValueError("Number of subjects must equal length of group/sessions vector!")
        if not (np.all(group_sizes == group_sizes[0]) and n_subjects == group_sizes[0]):
            raise ValueError("All sessions must have the same number of samples")
        
    groups_perm = []

    # unpaired: permute across the whole set
    if not paired:
        
        # proportional: permuted groups have the same size & proportion of subjects as the original groups
        if "prop" in strategy:
            
            group_idc = {label: np.where(groups==label)[0] for label in group_labels}
                
            def perm_fun(i):
                rng = np.random.default_rng(None if seed is None else seed + i)
                # position in group_labels per sample, -1 = not yet assigned (works for any
                # label dtype, e.g. strings)
                groups_tmp = np.full(groups.shape, -1)
                group_idc_tmp = group_idc.copy()
                for k, (label, size) in enumerate(zip(group_labels, group_sizes)):
                    if verbose=="debug" and i == 0:
                        print(f"Permuted group '{label}' of size {size}:")

                    for l, s in zip(group_labels, group_sizes):
                        idc = group_idc_tmp[l]
                        fraction = s / n
                        n_sample = np.floor(size * fraction).astype(int)
                        if verbose=="debug" and i == 0:
                            print(f"{n_sample} samples ({n_sample / size * 100:.02f} %) from group '{l}'.")
                        idc_perm = rng.choice(idc, n_sample, replace=False)
                        group_idc_tmp[l] = idc[~np.isin(idc, idc_perm)]
                        groups_tmp[idc_perm] = k

                # samples left over by the floor-rounding: give each group its missing
                # count, so group sizes are preserved (for 2 groups, at most one each)
                unassigned = groups_tmp < 0
                if unassigned.any():
                    n_missing = group_sizes - np.bincount(groups_tmp[~unassigned], minlength=n_labels)
                    groups_tmp[unassigned] = rng.permutation(np.repeat(np.arange(n_labels), n_missing))
                return group_labels[groups_tmp]
        
        # shuffle: random permutation across the whole set
        elif "shuff" in strategy:
            def perm_fun(i):
                rng = np.random.default_rng(None if seed is None else seed + i)
                return rng.permutation(groups)
            
        # draw: random permutation across the whole set, with replacement
        elif "draw" in strategy:
            def perm_fun(i):
                rng = np.random.default_rng(None if seed is None else seed + i)
                return rng.choice(groups, len(groups), replace=True)
        
        # not defined
        else:
            raise ValueError(f"Unknown unpaired permutation strategy: {strategy}!")
        
    # paired: permute within each subject
    else:
        
        # proportional: permuted sessions are equally sized and consist of 50% of each original session
        # (given two groups, should also work with more)
        if "prop" in strategy:
            # get a df with indices of each subject in the original vector by session/"group label"
            subs_by_session_idc = pd.DataFrame(
                np.zeros((n_subjects, n_labels), dtype=int),
                index=unique_subjects,
                columns=group_labels,
                dtype=int
            )
            for ses in group_labels:
                subs_by_session_idc.loc[unique_subjects, ses] = np.where((groups==ses) & np.isin(subjects, unique_subjects))[0]
            
            # orderings of positions in group_labels (works for any label dtype, e.g. strings)
            session_combinations = list(permutations(range(n_labels)))
            n_samples = np.floor(n_subjects / len(session_combinations)).astype(int)

            def perm_fun(i):
                rng = np.random.default_rng(None if seed is None else seed + i)
                sessions_perm = np.full(n, -1)
                subs_tmp = unique_subjects.copy()
                for session_labels_perm in session_combinations:
                    subs = rng.choice(subs_tmp, n_samples, replace=False)
                    subs_tmp = subs_tmp[~np.isin(subs_tmp, subs)]
                    sessions_perm[subs_by_session_idc.loc[subs, :]] = session_labels_perm

                if len(subs_tmp) > 0:
                    sessions_perm[subs_by_session_idc.loc[subs_tmp, :]] = rng.permutation(n_labels)

                if (sessions_perm < 0).any():
                    raise ValueError("Problem with group assignment")
                return group_labels[sessions_perm]
                    
        
        # shuffle: random permutation within subjects
        elif "shuff" in strategy:

            def perm_fun(i):
                rng = np.random.default_rng(None if seed is None else seed + i)
                sessions_perm = groups.copy()
                for sub in np.unique(subjects):
                    sessions_perm[subjects==sub] = rng.permutation(groups[subjects==sub])
                return sessions_perm
                
        # not defined
        else:
            raise ValueError(f"Unknown paired permutation strategy: {strategy}!")
        
    # Run
    groups_perm = Parallel(n_jobs=n_proc)(
        [delayed(perm_fun)(i) 
         for i in tqdm(range(n_perm), 
                       desc=f"Permuting groups ({n_proc} proc)", 
                       disable=not verbose)]
    )
    
    # return permuted group vector(s), as 1d array or list thereof, if n_perm > 1
    if len(groups_perm) == 1:
        groups_perm = groups_perm[0]
    return groups_perm


def null_to_p(test_value, null_array, tail="two", fit_norm=False):
    """Return p-value for test value(s) against null array.
    
    Adopted from NiMARE v0.0.12: https://zenodo.org/record/6600700
    (NiMARE/nimare/stats.py)
    
    Parameters
    ----------
    test_value : 1D array_like
        Values for which to determine p-value.
    null_array : 1D array_like
        Null distribution against which test_value is compared.
    tail : {'two', 'upper', 'lower'}, optional
        Whether to compare value against null distribution in a two-sided
        ('two') or one-sided ('upper' or 'lower') manner.
        If 'upper', then higher values for the test_value are more significant.
        If 'lower', then lower values for the test_value are more significant.
        Default is 'two'.
    fit_norm : boolean
        Whether to fit a normal distribution to null_array data and compute p values from that.
        Might be useful if the relative order of multiple highly significant p values is of
        interest, but the number of null values cannot be increased sufficiently.
    
    Returns
    -------
    p_value : :obj:`float`
        P-value(s) associated with the test value when compared against the null
        distribution. Return type matches input type (i.e., a float if
        test_value is a single float, and an array if test_value is an array).
    
    Notes
    -----
    Empirical p-values (``fit_norm=False``) use the augmented convention: the test value
    counts as one of ``n_null + 1`` exchangeable draws, so p = (c + 1) / (n_null + 1) for
    one tail (c = number of null values at least as extreme) and
    ``min(1, 2 * min(c_upper + 1, c_lower + 1) / (n_null + 1))`` for two tails. These
    p-values are exact (valid) under exchangeability and never 0; the smallest possible
    value is ``1 / (n_null + 1)`` (one-tailed) or ``2 / (n_null + 1)`` (two-tailed)
    :cite:`phipson2010`. NaN values in the null are ignored; a NaN test value gives NaN.
    The same convention is used by the min-P corrections (:func:`minP_correction`).
    """
    
    if tail not in {"two", "upper", "lower"}:
        raise ValueError('Argument "tail" must be one of ["two", "upper", "lower"]')

    return_first = isinstance(test_value, (float, int, np.floating, np.integer))
    test_value = np.atleast_1d(np.asarray(test_value, dtype=float))
    null_array = np.asarray(null_array, dtype=float).ravel()

    ## empirical p value: the observed value counts as one of n_null + 1 exchangeable draws,
    ## as in the minP machinery (_counts_to_p); rank-based, so no centering is needed
    if not fit_norm:
        null = np.sort(null_array[~np.isnan(null_array)])
        n = null.shape[0] + 1
        le = 1 + np.searchsorted(null, test_value, side="right")      # 1 + #{null <= t}
        ge = n - np.searchsorted(null, test_value, side="left")       # 1 + #{null >= t}
        result = np.asarray(_counts_to_p(ge, le, n, tail), dtype=float)
        result[np.isnan(test_value) | (n == 1)] = np.nan
        return result[0] if return_first else result

    ## fit a normal distribution and get p value
    def compute_p(t, null):
        mu, sd = norm.fit(null)
        return 1 - norm.cdf(t, mu, sd)

    null_array_mean = np.mean(null_array)
    test_value_centered = test_value - null_array_mean
    null_array_centered = null_array - null_array_mean
    if tail == "two":
        p_l = compute_p(test_value_centered, null_array_centered)
        p_r = compute_p(-test_value_centered, -null_array_centered)
        p = 2 * np.minimum(p_l, p_r)
    elif tail == "lower":
        p = compute_p(-test_value_centered, -null_array_centered)
    elif tail == "upper":
        p = compute_p(test_value_centered, null_array_centered)
    result = np.clip(p, 0.0, 1.0)

    return result[0] if return_first else result


def compute_meff(X, method="galwey"):
    """Effective number of independent tests from data matrix via eigendecomposition.

    Parameters
    ----------
    X : array-like, shape (n_maps, n_features)
    method : "galwey" (default) or "li_ji"
        :cite:`galwey2009`; :cite:`liji2005`.

    Returns
    -------
    float
    """
    X = np.atleast_2d(np.array(X, dtype=float))
    # drop parcel columns that contain any NaN (NaN propagates through corrcoef → eigvalsh fails)
    X = X[:, ~np.isnan(X).any(axis=0)]
    if X.shape[1] < 2:
        raise ValueError(f"compute_meff: fewer than 2 non-NaN parcels after dropping NaNs "
                         f"(got {X.shape[1]}).")
    # single map: nothing to decorrelate, no effective reduction
    if X.shape[0] < 2:
        return 1.0
    corr = np.corrcoef(X)
    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = eigenvalues[eigenvalues > 0]
    if method == "galwey":
        meff = np.sum(np.sqrt(eigenvalues)) ** 2 / np.sum(eigenvalues)
    elif method == "li_ji":
        meff = np.sum((eigenvalues >= 1).astype(float) + (eigenvalues - np.floor(eigenvalues)))
    else:
        raise ValueError(f"method must be 'galwey' or 'li_ji', not '{method}'")
    return float(meff)


def meff_sidak_correction(p_array, meff, alpha=0.05, dtype=None):
    """Sidak correction using Meff effective number of tests.

    p_corr = 1 - (1 - p)^meff

    Plain numpy/pandas (no numba). `meff` is typically obtained from
    :func:`compute_meff` (eigenvalue-based effective test count from a data
    matrix). Used in ``api.py`` as one of the available `mc_method` options.

    Parameters
    ----------
    p_array : array-like, DataFrame, or Series
        Uncorrected p-values.
    meff : float
        Effective number of independent tests.
    alpha : float, default=0.05
        Significance threshold for the `reject` output.
    dtype : optional
        Cast the DataFrame/Series outputs to this dtype.

    Returns
    -------
    p_corr, reject : same type as `p_array`
        Corrected p-values (clipped to [0, 1]) and boolean rejection mask.
    """
    p = np.array(p_array, dtype=float)
    p_corr = np.clip(1.0 - (1.0 - p) ** meff, 0.0, 1.0)
    reject = p_corr <= alpha
    if isinstance(p_array, pd.DataFrame):
        p_corr = pd.DataFrame(p_corr, index=p_array.index, columns=p_array.columns, dtype=dtype)
        reject = pd.DataFrame(reject, index=p_array.index, columns=p_array.columns, dtype=dtype)
    elif isinstance(p_array, pd.Series):
        p_corr = pd.Series(p_corr, index=p_array.index, name=p_array.name, dtype=dtype)
        reject = pd.Series(reject, index=p_array.index, name=p_array.name, dtype=dtype)
    return p_corr, reject


def _null_stats_to_array(null_colocs, stat):
    """Stack per-permutation null statistic dicts into (n_perm, n_Y, n_X) array."""
    return np.stack([null_colocs[i][stat] for i in range(len(null_colocs))], axis=0)


def _signed_stats(obs, null_arr, tail):
    """Apply tail direction: return (obs_cmp, null_cmp) ready for >= comparison."""
    if tail == "two":
        return np.abs(obs), np.abs(null_arr)
    elif tail == "upper":
        return obs, null_arr
    elif tail == "lower":
        return -obs, -null_arr
    raise ValueError(f"tail must be 'two', 'upper', or 'lower', not '{tail}'")


def _df_like(array, template, dtype):
    """Wrap array in DataFrame/Series matching template's index/columns."""
    if isinstance(template, pd.DataFrame):
        return pd.DataFrame(array, index=template.index, columns=template.columns, dtype=dtype)
    elif isinstance(template, pd.Series):
        return pd.Series(array, index=template.index, name=template.name, dtype=dtype)
    return array


# ── family-wise (max-statistic / min-p) corrections ──────────────────────────────
# Shared machinery for maxT/step_maxT/minP/step_minP. A "family" is the set of tests
# corrected jointly: one Y row across its X columns (how="r") or all Y x X cells
# (how="a"). Each 1D family routine takes finite observed stats `obs_1d` (m,) and the
# matching null `null_2d` (n_perm, m), which may contain sporadic NaN.

# max elements of the (n_perm + 1, chunk) column-p block minP materializes at once --
# bounds memory for large families (e.g. thousands of gene sets x 10k permutations)
_MINP_CHUNK_ELEMENTS = 1_000_000


def _check_tail(tail):
    if tail not in ("two", "upper", "lower"):
        raise ValueError(f"tail must be 'two', 'upper', or 'lower', not '{tail}'")


def _prep_family(obs_stats, null_colocs, stat):
    """Float copies of observed stats (n_Y, n_X) and stacked null (n_perm, n_Y, n_X).

    Raises on shape mismatch instead of letting numpy broadcast e.g. a pooled
    (1, n_X) null against unpooled (n_Y, n_X) observations into a silently wrong
    result.
    """
    obs = np.array(obs_stats, dtype=float)
    null_arr = _null_stats_to_array(null_colocs, stat).astype(float)  # (n_perm, n_Y, n_X)
    if obs.ndim != 2 or null_arr.ndim != 3 or null_arr.shape[1:] != obs.shape:
        raise ValueError(
            f"Null statistics (per permutation: {null_arr.shape[1:]}) do not match the observed "
            f"statistics ({obs.shape}) for stat='{stat}'."
        )
    return obs, null_arr


def _apply_family(obs, null_arr, how, family_fn, method_name, p_raw=None):
    """Apply a 1D family routine per Y row (how="r") or across all cells (how="a").

    A cell with NaN observed statistic has no defined test: it is excluded from the
    family (so it can't distort the other tests' null) and returned as NaN, instead
    of a fake p from 'NaN >= x' silently evaluating to False everywhere.

    `family_fn(obs_1d, null_2d, p_raw_1d)` gets the family's slice of `p_raw` (or None),
    so step-down routines can apply the p_raw floor before enforcing monotonicity.
    """
    n_Y, n_X = obs.shape
    if p_raw is not None:
        p_raw = np.array(p_raw, dtype=float)
        if p_raw.shape != obs.shape:
            raise ValueError(f"p_raw shape {p_raw.shape} does not match the observed statistics "
                             f"shape {obs.shape}.")
    if how == "r":
        p = np.full(obs.shape, np.nan)
        for y in range(n_Y):
            valid = ~np.isnan(obs[y])
            pr = None if p_raw is None else p_raw[y]
            if valid.all():  # view, not a boolean-mask copy of the whole null
                p[y] = family_fn(obs[y], null_arr[:, y, :], pr)
            elif valid.any():
                p[y, valid] = family_fn(obs[y, valid], null_arr[:, y, valid],
                                        None if pr is None else pr[valid])
        return p
    elif how == "a":
        obs_flat = obs.ravel()
        null_flat = null_arr.reshape(null_arr.shape[0], -1)
        pr = None if p_raw is None else p_raw.ravel()
        p_flat = np.full(obs_flat.shape, np.nan)
        valid = ~np.isnan(obs_flat)
        if valid.all():
            p_flat = family_fn(obs_flat, null_flat, pr)
        elif valid.any():
            p_flat[valid] = family_fn(obs_flat[valid], null_flat[:, valid],
                                      None if pr is None else pr[valid])
        return p_flat.reshape(n_Y, n_X)
    raise ValueError(f"how='{how}' not supported for {method_name}. Use 'r' or 'a'.")


def _finalize_family_p(p, p_raw, floor):
    """Clip to [0, 1]; if `p_raw` is given, enforce p >= max(p_raw, floor).

    A corrected p-value below its own uncorrected p-value (or exactly 0 from a finite
    permutation null) is never defensible; `p_raw` is the stored uncorrected p.
    """
    p = np.clip(p, 0.0, 1.0)
    if p_raw is not None:
        p_raw = np.array(p_raw, dtype=float)
        if p_raw.shape != p.shape:
            raise ValueError(f"p_raw shape {p_raw.shape} does not match the corrected p-values "
                             f"shape {p.shape}.")
        # fmax: a NaN uncorrected p never overrides a defined corrected one
        p = np.where(np.isnan(p), np.nan, np.maximum(np.fmax(p, p_raw), floor))
    return p


def _maxt_family(obs_1d, null_2d, tail, stepdown, p_raw_1d=None):
    """Max-T over one family: compare each test to the null distribution of the family max.

    Augmented convention as in :func:`_minp_family`: p = (1 + #{null max >= obs}) / (n_perm + 1).

    With `stepdown`, `p_raw_1d` (uncorrected p) is applied as a floor before enforcing
    monotonicity; flooring afterwards could break the step-down order whenever p_raw
    orders the tests differently than the raw statistic does.
    """
    obs_cmp, null_cmp = _signed_stats(obs_1d, null_2d, tail)
    # a sporadic per-permutation NaN can never win the max
    null_cmp = np.where(np.isnan(null_cmp), -np.inf, null_cmp)
    # augmented convention: the observed family counts as one of n_perm + 1 draws
    n_rep = null_cmp.shape[0] + 1
    if not stepdown:
        null_max = null_cmp.max(axis=1)  # (n_perm,)
        return (1 + np.sum(null_max[:, np.newaxis] >= obs_cmp[np.newaxis, :], axis=0)) / n_rep
    order = np.argsort(obs_cmp)[::-1]
    obs_s = obs_cmp[order]
    null_s = null_cmp[:, order]  # (n_perm, m)
    # step-down max: null_max_j[perm] = max(null_s[perm, j:]), via reverse cumulative max
    null_rev_cummax = np.maximum.accumulate(null_s[:, ::-1], axis=1)[:, ::-1]
    p_s = (1 + np.sum(null_rev_cummax >= obs_s[np.newaxis, :], axis=0)) / n_rep
    if p_raw_1d is not None:
        p_s = np.fmax(p_s, p_raw_1d[order])  # fmax: a NaN p_raw never overrides
    # enforce monotonicity (step-down)
    p_s = np.maximum.accumulate(p_s)
    p_out = np.empty_like(p_s)
    p_out[order] = p_s
    return p_out


def _counts_to_p(ge, le, n, tail):
    """Permutation p from counts of sample values >= / <= the test value.

    Two-tailed uses the equal-tail definition 2 * min(upper, lower), same as
    :func:`null_to_p` -- invariant to the null's location, so no centering is needed.
    """
    if tail == "upper":
        return ge / n
    if tail == "lower":
        return le / n
    return np.minimum(1.0, 2.0 * np.minimum(ge, le) / n)


def _augmented_column_p(obs_c, null_c, tail):
    """Column-wise permutation p of every value in the augmented sample [obs; null].

    Returns (n_perm + 1, c): row 0 is the observed value, rows 1.. the permutations.
    NaN null values are dropped from their column's sample (and returned as NaN).
    """
    S = np.vstack([obs_c[np.newaxis, :], null_c])
    P = np.full(S.shape, np.nan)
    has_nan = np.isnan(S).any(axis=0)
    full = ~has_nan
    if full.any():
        S_full = S[:, full]
        n = S_full.shape[0]
        le = rankdata(S_full, method="max", axis=0)         # #{sample <= v}
        ge = n - rankdata(S_full, method="min", axis=0) + 1  # #{sample >= v}
        P[:, full] = _counts_to_p(ge, le, n, tail)
    for j in np.flatnonzero(has_nan):
        ok = ~np.isnan(S[:, j])
        v = S[ok, j]
        srt = np.sort(v)
        n = v.shape[0]
        le = np.searchsorted(srt, v, side="right")
        ge = n - np.searchsorted(srt, v, side="left")
        P[ok, j] = _counts_to_p(ge, le, n, tail)
    return P


def _minp_family(obs_1d, null_2d, tail, stepdown, p_raw_1d=None, k=1):
    """Min-P over one family (augmented convention).

    The observed value is treated as one of n_perm + 1 exchangeable draws, for its own
    column p AND for every permutation's column p. This keeps the observed and the
    permutation p-values on the same grid -- with the observed excluded from its own
    reference (as in :func:`null_to_p`), an observation beyond the whole two-tailed null
    gets p=1/n_perm while no permutation can go below 2/n_perm, which inflates the
    two-tailed FWER far past alpha for large families.

    With `stepdown`, `p_raw_1d` is applied as a floor before enforcing monotonicity
    (see :func:`_maxt_family`).

    With ``k > 1`` (single-step only), each replicate's k-th smallest column p replaces
    its minimum (generalized family-wise error, see :func:`kminP_correction`).

    Returns (p_adj, attainable) where `attainable` is the smallest corrected p any test
    in this family could reach (an observation at the column-p floor).
    """
    if stepdown and k > 1:
        raise ValueError("k > 1 is only implemented for the single-step variant.")
    m = obs_1d.shape[0]
    n_rep = null_2d.shape[0] + 1  # row 0 = observed, rows 1.. = permutations
    chunk = max(1, _MINP_CHUNK_ELEMENTS // n_rep)

    # pass 1: observed column p-values (same counts/formula as row 0 of pass 2)
    p_obs = np.empty(m)
    for start in range(0, m, chunk):
        cols = slice(start, start + chunk)
        o, nl = obs_1d[cols], null_2d[:, cols]
        n_col = 1 + (~np.isnan(nl)).sum(axis=0)
        ge = 1 + (nl >= o[np.newaxis, :]).sum(axis=0)
        le = 1 + (nl <= o[np.newaxis, :]).sum(axis=0)
        p_obs[cols] = _counts_to_p(ge, le, n_col, tail)

    floor_col = (2.0 if tail == "two" else 1.0) / n_rep
    if k > 1:
        # fewer tests than k: k false rejections are impossible, nothing to correct
        if m < k:
            return p_obs, floor_col
        # k smallest column p per replicate, merged chunk by chunk (memory-bounded)
        running_k = np.full((n_rep, 0), np.inf)
        for start in range(0, m, chunk):
            cols = slice(start, start + chunk)
            P = _augmented_column_p(obs_1d[cols], null_2d[:, cols], tail)
            P[np.isnan(P)] = np.inf
            stack = np.hstack([running_k, P])
            if stack.shape[1] > k:
                stack = np.partition(stack, k - 1, axis=1)[:, :k]
            running_k = stack
        kth = running_k.max(axis=1)  # k-th smallest column p per replicate
        # the observed draw always counts itself (for k=1, row 0's min does so implicitly)
        p = (1 + np.searchsorted(np.sort(kth[1:]), p_obs, side="right")) / n_rep
        attainable = (1 + np.sum(kth[1:] <= floor_col)) / n_rep
        return p, attainable

    # pass 2: walk tests from least to most significant, carrying the running min over
    # the tests seen so far -- after test j that is min(p) over the step-down family of
    # j (j and everything less significant); after the last test, the global family min
    order = np.argsort(p_obs, kind="stable")  # most significant first
    rev = order[::-1]
    running = np.full(n_rep, np.inf)
    counts = np.zeros(m)
    for start in range(0, m, chunk):
        cols = rev[start:start + chunk]
        P = _augmented_column_p(obs_1d[cols], null_2d[:, cols], tail)
        P[np.isnan(P)] = np.inf  # a dropped NaN permutation can never win the min
        cummin = np.minimum(np.minimum.accumulate(P, axis=1), running[:, np.newaxis])
        running = cummin[:, -1].copy()
        if stepdown:
            counts[cols] = (cummin <= p_obs[cols][np.newaxis, :]).sum(axis=0)
    if not stepdown:
        counts = np.searchsorted(np.sort(running), p_obs, side="right").astype(float)
    p = counts / n_rep  # >= 1/n_rep: row 0 always counts itself
    if stepdown:
        if p_raw_1d is not None:
            p = np.fmax(p, p_raw_1d)
        p[order] = np.maximum.accumulate(p[order])  # enforce monotonicity

    attainable = (1 + np.sum(running[1:] <= floor_col)) / n_rep
    return p, attainable


def _family_scale_ratio(null_arr, tail, how="r"):
    """Worst ratio of per-test null tail widths within a max-statistic family.

    Width is q95 - q50 of each test's null on maxT's own comparison scale (|x| for
    two-tailed, x / -x for upper / lower). maxT compares these raw values across tests,
    so a large ratio means the widest-null tests produce nearly every family maximum
    and the narrowest are effectively unrejectable (FWER still valid, power unbalanced).

    Parameters
    ----------
    null_arr : array, shape (n_perm, n_Y, n_X)
    tail : {"two", "upper", "lower"}
    how : {"r", "a"}

    Returns
    -------
    ratio : float
        NaN if no family has two tests with a defined width.
    widest, narrowest : tuple of (y, x) indices, or None
    """
    _, e = _signed_stats(np.zeros(null_arr.shape[1:]), null_arr, tail)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN tests
        width = np.nanpercentile(e, 95, axis=0) - np.nanpercentile(e, 50, axis=0)
    n_Y, n_X = width.shape
    idx = np.arange(n_Y * n_X).reshape(n_Y, n_X)
    families = [idx.ravel()] if how == "a" else [idx[y] for y in range(n_Y)]
    w = width.ravel()
    worst_ratio, worst_pair = np.nan, None
    for fam in families:
        fam = fam[np.isfinite(w[fam])]
        if fam.size < 2:
            continue
        hi, lo = fam[np.argmax(w[fam])], fam[np.argmin(w[fam])]
        if w[hi] <= 0:
            ratio = 1.0
        elif w[lo] <= 0:
            ratio = np.inf
        else:
            ratio = w[hi] / w[lo]
        if worst_pair is None or ratio > worst_ratio:
            worst_ratio, worst_pair = ratio, (hi, lo)
    if worst_pair is None:
        return np.nan, None, None
    return (worst_ratio,
            tuple(int(i) for i in np.unravel_index(worst_pair[0], (n_Y, n_X))),
            tuple(int(i) for i in np.unravel_index(worst_pair[1], (n_Y, n_X))))


def maxT_correction(obs_stats, null_colocs, stat, tail="two", how="r", alpha=0.05, dtype=None,
                    p_raw=None):
    """Max-T FWER correction (:cite:`westfall1993`).

    Each test's corrected p is the fraction of permutations whose family maximum (of
    the raw statistic) reaches the observed statistic, with the observed family counted
    as one of ``n_perm + 1`` draws (as in :func:`null_to_p`). This assumes the tests' null
    distributions are on a comparable scale: otherwise the widest-null tests produce
    nearly every family maximum and the narrowest become effectively unrejectable.
    FWER control remains valid, but power is unbalanced. That is the case e.g. for XSEA
    set-level statistics, whose null spread shrinks with set size; use
    :func:`minP_correction` there.

    Plain numpy/pandas (no numba). Used in ``api.py`` as the ``mc_method="maxT"``
    option of ``NiSpace.correct_p()``.

    Parameters
    ----------
    obs_stats : DataFrame or array, shape (n_Y, n_X)
        Observed test statistics.
    null_colocs : list[n_perm] of {stat: (n_Y, n_X) array}
        Per-permutation null statistic dicts, as produced by the permutation
        machinery in ``core/permute.py``.
    stat : str
        Key to extract from each `null_colocs` dict.
    tail : {"two", "upper", "lower"}, default="two"
        Sidedness of the comparison.
    how : {"r", "a"}, default="r"
        "r": family = each Y row, max across X. "a": family = all Y x X cells.
    alpha : float, default=0.05
        Significance threshold for the `reject` output.
    dtype : optional
        Cast the DataFrame/Series outputs to this dtype.
    p_raw : DataFrame or array, shape (n_Y, n_X), optional
        Uncorrected p-values of the same tests. If given, the output is floored at
        ``max(p_raw, 1 / (n_perm + 1))``, so a corrected p-value is never below its own
        uncorrected one nor exactly 0.

    Returns
    -------
    p, reject : same type as `obs_stats`
        Corrected p-values and boolean rejection mask. Tests with a NaN observed
        statistic are excluded from their family and get NaN (not rejected).

    Raises
    ------
    ValueError
        If `tail`/`how` are invalid, or the null and observed shapes don't match.
    """
    _check_tail(tail)
    obs, null_arr = _prep_family(obs_stats, null_colocs, stat)
    p = _apply_family(obs, null_arr, how,
                      lambda o, n, pr: _maxt_family(o, n, tail, stepdown=False), "maxT")
    p = _finalize_family_p(p, p_raw, floor=1.0 / (null_arr.shape[0] + 1))
    reject = p <= alpha
    return _df_like(p, obs_stats, dtype), _df_like(reject, obs_stats, dtype)


def step_maxT_correction(obs_stats, null_colocs, stat, tail="two", how="r", alpha=0.05, dtype=None,
                         p_raw=None):
    """Step-down Max-T FWER correction (:cite:`westfall1993`).

    Tests are ordered by observed statistic; each is compared to the null maximum over
    itself and all less extreme tests, with monotonicity enforced. At least as powerful
    as :func:`maxT_correction` while preserving FWER control. Shares maxT's
    comparable-scale assumption, and additionally orders tests by *raw* statistic --
    with heterogeneous null scales (e.g. XSEA sets of different sizes) that ordering
    tracks scale rather than evidence; use :func:`step_minP_correction` there.

    Plain numpy/pandas (no numba). Used in ``api.py`` as the ``mc_method="step_maxT"``
    option of ``NiSpace.correct_p()``.

    Parameters
    ----------
    obs_stats : DataFrame or array, shape (n_Y, n_X)
        Observed test statistics.
    null_colocs : list[n_perm] of {stat: (n_Y, n_X) array}
        Per-permutation null statistic dicts, as produced by the permutation
        machinery in ``core/permute.py``.
    stat : str
        Key to extract from each `null_colocs` dict.
    tail : {"two", "upper", "lower"}, default="two"
        Sidedness of the comparison.
    how : {"r", "a"}, default="r"
        "r": family = each Y row, across X. "a": family = all Y x X cells.
    alpha : float, default=0.05
        Significance threshold for the `reject` output.
    dtype : optional
        Cast the DataFrame/Series outputs to this dtype.
    p_raw : DataFrame or array, shape (n_Y, n_X), optional
        Uncorrected p-values of the same tests. If given, the output is floored at
        ``max(p_raw, 1 / (n_perm + 1))``.

    Returns
    -------
    p, reject : same type as `obs_stats`
        Step-down max-T p-values and boolean rejection mask. Tests with a NaN
        observed statistic are excluded from their family and get NaN.

    Raises
    ------
    ValueError
        If `tail`/`how` are invalid, or the null and observed shapes don't match.
    """
    _check_tail(tail)
    obs, null_arr = _prep_family(obs_stats, null_colocs, stat)
    p = _apply_family(obs, null_arr, how,
                      lambda o, n, pr: _maxt_family(o, n, tail, stepdown=True, p_raw_1d=pr),
                      "step_maxT", p_raw=p_raw)
    p = _finalize_family_p(p, p_raw, floor=1.0 / (null_arr.shape[0] + 1))
    reject = p <= alpha
    return _df_like(p, obs_stats, dtype), _df_like(reject, obs_stats, dtype)


def _minp_correction(obs_stats, null_colocs, stat, tail, how, alpha, dtype, p_raw, stepdown,
                     method_name, k=1):
    """Shared driver of :func:`minP_correction` / :func:`step_minP_correction` /
    :func:`kminP_correction`."""
    _check_tail(tail)
    obs, null_arr = _prep_family(obs_stats, null_colocs, stat)
    n_perm = null_arr.shape[0]
    attainable = []

    def _family(o, n, pr):
        p, att = _minp_family(o, n, tail, stepdown, p_raw_1d=pr, k=k)
        attainable.append(att)
        return p

    p = _apply_family(obs, null_arr, how, _family, method_name,
                      p_raw=p_raw if stepdown else None)
    if attainable and max(attainable) > alpha:
        worst = max(attainable)
        lgr.warning(
            f"{method_name}: with n_perm={n_perm}, the smallest corrected p-value any test in this "
            f"family can reach is {worst:.3g} > alpha={alpha} -- no test can be declared "
            f"significant. Per-test permutation p-values (resolution 1/{n_perm + 1}) are too "
            f"coarse for this many tests. Increase n_perm to at least "
            f"~{int(np.ceil((n_perm + 1) * worst / alpha))}, or use an FDR correction (e.g. "
            f"'fdr_bh') for large families."
        )
    p = _finalize_family_p(p, p_raw, floor=1.0 / (n_perm + 1))
    reject = p <= alpha
    return _df_like(p, obs_stats, dtype), _df_like(reject, obs_stats, dtype)


def minP_correction(obs_stats, null_colocs, stat, tail="two", how="r", alpha=0.05, dtype=None,
                    p_raw=None):
    """Min-P FWER correction (:cite:`westfall1993`).

    Like :func:`maxT_correction`, but the family extreme is taken over per-test
    permutation *p-values* instead of raw statistics: every statistic is first
    converted to a p-value against its own test's null, then each test's corrected p
    is the fraction of permutations whose family-minimum p reaches the observed p.
    Because a permutation p-value is uniform under the null regardless of the
    statistic's scale, location, or tail shape, every test gets an equal share of the
    error budget ("balance"). Use this instead of maxT whenever tests are not on a
    comparable scale -- e.g. XSEA set-level statistics, whose null spread shrinks with
    set size and within-set coherence. For identically distributed tests it agrees
    with maxT.

    Per-test p-values use the augmented permutation convention: the observed value
    counts as one of ``n_perm + 1`` exchangeable draws, for its own p-value and for
    every permutation's. Two-tailed p-values use the equal-tail definition
    ``2 * min(upper, lower)``, as in :func:`null_to_p` (rank-based, hence invariant to
    the null's location).

    **Resolution**: the smallest attainable corrected p-value is roughly
    ``(k * m_eff + 1) / (n_perm + 1)`` for ``m_eff`` effectively independent tests
    (k=2 two-tailed, 1 one-tailed), so ``n_perm`` must grow with the family size --
    about ``k * m / alpha`` for independent tests. A warning is logged when no test in
    a family can reach `alpha`; for very large families (thousands of gene sets) an
    FDR correction is usually the practical choice.

    Plain numpy/pandas (no numba). Used in ``api.py`` as the ``mc_method="minP"``
    option of ``NiSpace.correct_p()``. Memory-bounded: per-test p-values are computed
    in column chunks and only the running family minimum is kept.

    Parameters
    ----------
    obs_stats : DataFrame or array, shape (n_Y, n_X)
        Observed test statistics.
    null_colocs : list[n_perm] of {stat: (n_Y, n_X) array}
        Per-permutation null statistic dicts, as produced by the permutation
        machinery in ``core/permute.py``.
    stat : str
        Key to extract from each `null_colocs` dict.
    tail : {"two", "upper", "lower"}, default="two"
        Sidedness of the per-test p-values.
    how : {"r", "a"}, default="r"
        "r": family = each Y row, across X. "a": family = all Y x X cells.
    alpha : float, default=0.05
        Significance threshold for the `reject` output and the resolution warning.
    dtype : optional
        Cast the DataFrame/Series outputs to this dtype.
    p_raw : DataFrame or array, shape (n_Y, n_X), optional
        Uncorrected p-values of the same tests. If given, the output is floored at
        ``max(p_raw, 1 / (n_perm + 1))``, so a corrected p-value is never below its own
        uncorrected one.

    Returns
    -------
    p, reject : same type as `obs_stats`
        Corrected p-values and boolean rejection mask. Tests with a NaN observed
        statistic are excluded from their family and get NaN; sporadic NaN null
        values are dropped from their own test's reference sample.

    Raises
    ------
    ValueError
        If `tail`/`how` are invalid, or the null and observed shapes don't match.
    """
    return _minp_correction(obs_stats, null_colocs, stat, tail, how, alpha, dtype, p_raw,
                            stepdown=False, method_name="minP")


def step_minP_correction(obs_stats, null_colocs, stat, tail="two", how="r", alpha=0.05,
                         dtype=None, p_raw=None):
    """Step-down Min-P FWER correction (:cite:`westfall1993`).

    Tests are ordered by their per-test permutation p-value; each is compared to the
    null minimum p over itself and all less significant tests, with monotonicity
    enforced. At least as powerful as :func:`minP_correction` while preserving FWER
    control. Same per-test p-value convention, balance property and resolution
    requirement as :func:`minP_correction`.

    Plain numpy/pandas (no numba). Used in ``api.py`` as the
    ``mc_method="step_minP"`` option of ``NiSpace.correct_p()``.

    Parameters
    ----------
    obs_stats : DataFrame or array, shape (n_Y, n_X)
        Observed test statistics.
    null_colocs : list[n_perm] of {stat: (n_Y, n_X) array}
        Per-permutation null statistic dicts.
    stat : str
        Key to extract from each `null_colocs` dict.
    tail : {"two", "upper", "lower"}, default="two"
        Sidedness of the per-test p-values.
    how : {"r", "a"}, default="r"
        "r": family = each Y row, across X. "a": family = all Y x X cells.
    alpha : float, default=0.05
        Significance threshold for the `reject` output and the resolution warning.
    dtype : optional
        Cast the DataFrame/Series outputs to this dtype.
    p_raw : DataFrame or array, shape (n_Y, n_X), optional
        Uncorrected p-values of the same tests. If given, the output is floored at
        ``max(p_raw, 1 / (n_perm + 1))``.

    Returns
    -------
    p, reject : same type as `obs_stats`
        Step-down min-P p-values and boolean rejection mask (NaN handling as in
        :func:`minP_correction`).

    Raises
    ------
    ValueError
        If `tail`/`how` are invalid, or the null and observed shapes don't match.
    """
    return _minp_correction(obs_stats, null_colocs, stat, tail, how, alpha, dtype, p_raw,
                            stepdown=True, method_name="step_minP")


def kminP_correction(obs_stats, null_colocs, stat, tail="two", how="r", alpha=0.05, dtype=None,
                     p_raw=None, n_false=1):
    """Single-step k-min-P correction of the generalized family-wise error rate (k-FWER).

    Controls the k-FWER with k = ``n_false + 1``, the probability of rejecting at least
    k true null hypotheses (:cite:`lehmann2005`): ``P(V >= n_false + 1) <= alpha``. With
    probability at least ``1 - alpha``, at most `n_false` of the significant tests are
    false positives. The statement is about the set of significant tests, not about any
    single one of them. ``n_false=0`` is :func:`minP_correction` (the usual FWER).

    This is the single-step common-quantile P(n_false + 1) procedure of :cite:`dudoit2008`
    (Procedure 3.19; adjusted p-values in Corollary 4.8, resampling estimate in
    Procedure 4.20, where gFWER(n_false) = P(V > n_false)), the p-value form of the
    single-step k-max procedure of :cite:`romano2007` (Section 2.1, with T = -p; their
    Remark 3.1 recommends p-values for balance): the critical value is the
    alpha-quantile of the k-th smallest per-test p-value across the family, from the
    joint permutation null. A test's corrected p is
    ``(1 + #{permutations whose k-th smallest p <= the test's p}) / (n_perm + 1)``,
    i.e., the estimate of :cite:`dudoit2008` (eq. 4.42) with the observed draw counted
    as one of the ``n_perm + 1`` draws; per-test p-values use the same augmented
    convention (:func:`minP_correction`).
    Counting the observed draw in its own reference keeps the procedure valid for a
    finite number of permutations: the k-th smallest p over all tests is never larger
    than over the true nulls alone (the monotonicity condition of :cite:`romano2007`,
    eq. 14), so k or more false rejections imply that the true nulls' own k-th smallest
    p ranks among the lowest ``alpha * (n_perm + 1)`` of ``n_perm + 1`` exchangeable
    draws. As with min-P, this needs a joint null that is valid for any subset of true
    nulls (e.g. group permutations, null maps of Y).

    Each test's decision depends only on its own p-value and the joint null, not on the
    other tests' observed values, and every test gets an equal share of the error
    budget. The step-down version (:cite:`romano2007`, Algorithm 2.1) is more powerful
    but not implemented. The gain over :func:`minP_correction` shrinks as the tests
    become more correlated (:cite:`romano2007`, Section 6).

    Plain numpy/pandas (no numba). Used in ``api.py`` as the ``mc_method="kminP"``
    option of ``NiSpace.correct_p()`` (``"kminP(<n_false>)"``, default n_false 1).

    Parameters
    ----------
    obs_stats : DataFrame or array, shape (n_Y, n_X)
        Observed test statistics.
    null_colocs : list[n_perm] of {stat: (n_Y, n_X) array}
        Per-permutation null statistic dicts.
    stat : str
        Key to extract from each `null_colocs` dict.
    tail : {"two", "upper", "lower"}, default="two"
        Sidedness of the per-test p-values.
    how : {"r", "a"}, default="r"
        "r": family = each Y row, across X. "a": family = all Y x X cells.
    alpha : float, default=0.05
        Significance threshold for the `reject` output and the resolution warning.
    dtype : optional
        Cast the DataFrame/Series outputs to this dtype.
    p_raw : DataFrame or array, shape (n_Y, n_X), optional
        Uncorrected p-values of the same tests. If given, the output is floored at
        ``max(p_raw, 1 / (n_perm + 1))``.
    n_false : int, default=1
        Number of false positives tolerated among the significant tests (k = n_false + 1).
        A family with at most `n_false` tests is returned uncorrected.

    Returns
    -------
    p, reject : same type as `obs_stats`
        Corrected p-values and boolean rejection mask (NaN handling as in
        :func:`minP_correction`).

    Raises
    ------
    ValueError
        If `tail`/`how`/`n_false` are invalid, or the null and observed shapes don't match.
    """
    if isinstance(n_false, bool) or not isinstance(n_false, (int, np.integer)) or n_false < 0:
        raise ValueError(f"n_false must be a non-negative integer, not {n_false!r}.")
    return _minp_correction(obs_stats, null_colocs, stat, tail, how, alpha, dtype, p_raw,
                            stepdown=False, method_name=f"kminP({n_false})", k=n_false + 1)


def mc_correction(p_array, alpha=0.05, method="fdr_bh", how="array", dtype=None):
    """Multiple-comparison correction via ``statsmodels.stats.multitest.multipletests``.

    Plain numpy/pandas (no numba). Used in ``api.py`` as the default
    (non-permutation-based) `mc_method` option.

    Parameters
    ----------
    p_array : array-like, DataFrame, or Series
        Uncorrected p-values, 1D or 2D.
    alpha : float, default=0.05
        Significance threshold for the `reject` output.
    method : str, default="fdr_bh"
        Correction method name forwarded to `multipletests` (e.g.
        ``"fdr_bh"``, ``"bonferroni"``, ``"holm"``, ...).
    how : str, default="array"
        Correction scope for 2D input: ``"array"``/``"a"``/``"arr"``
        (flatten and correct across the whole array), ``"cols"``/``"c"``/...
        (correct each column independently), or ``"rows"``/``"r"``/... (each
        row independently). Ignored (treated as whole-array) for 1D input.
    dtype : optional
        Cast the DataFrame/Series outputs to this dtype.

    Returns
    -------
    pcor, reject : same type as `p_array`
        Corrected p-values and boolean rejection mask.
    """
    # prepare data
    p = np.array(p_array)
    p_shape = p.shape

    ## correct across whole input array -> flattern & reshape
    if how in ["a", "arr", "array"]:
        # flatten row-wise
        p_1d = p.flatten("C") 
        # get corrected p-values
        res = multipletests(p_1d, alpha=alpha, method=method)
        pcor_1d = res[1]
        reject_1d = res[0]
        # reshape to original form
        pcor = np.reshape(pcor_1d, p_shape, "C")
        reject = np.reshape(reject_1d, p_shape, "C")
    
    ## correct across each column/row
    else:
        pcor, reject = np.zeros_like(p, dtype=dtype), np.zeros_like(p, dtype=dtype)
        if how in ["c", "col", "cols", "column", "columns"]:
            for col in range(p.shape[1]):
                res = multipletests(p[:,col], alpha=alpha, method=method)
                pcor[:,col], reject[:,col] = res[1], res[0]
        elif how in ["r", "row", "rows"]:
            for row in range(p.shape[0]):
                res = multipletests(p[row,:], alpha=alpha, method=method)
                pcor[row,:], reject[row,:] = res[1], res[0]
        else:
            print(f"Input how='{how}' not defined!")
          
    ## return as input dtype
    if isinstance(p_array, pd.DataFrame):
        pcor = pd.DataFrame(
            pcor, 
            index=p_array.index, 
            columns=p_array.columns, 
            dtype=dtype)
        reject = pd.DataFrame(
            reject, 
            index=p_array.index, 
            columns=p_array.columns, 
            dtype=dtype)
    elif isinstance(p_array, pd.Series):
        pcor = pd.Series(
            pcor, 
            index=p_array.index, 
            name=p_array.name, 
            dtype=dtype)
        reject = pd.Series(
            reject, 
            index=p_array.index, 
            name=p_array.name, 
            dtype=dtype)     
    return pcor, reject

