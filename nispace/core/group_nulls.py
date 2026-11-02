"""Group-permutation null maps: label shuffling and Freedman-Lane permutation.

Used by :meth:`nispace.api.NiSpace.permute` (``what="groups"``) and
:meth:`nispace.api.NiSpace.compare_within_region`. Both share one cached null cube of shape
``(n_contrasts, n_perm, n_parcels)``: every slice is the Y transform (e.g. Hedges' g)
recomputed under one permutation.

Without between-subject covariates, group labels are shuffled (or swapped within subjects
for paired designs) and the transform is recomputed on the data. With covariates regressed
out by :meth:`~nispace.api.NiSpace.clean_y`, shuffling labels on the cleaned data gives too
narrow a null whenever the covariates are associated with group: shuffling breaks that
association, and with it the extra variance of the group effect that it causes. The
Freedman-Lane procedure :cite:`freedman1983,anderson1999,winkler2014` keeps the design fixed
instead: it permutes the residuals of the covariates-only model, adds the fitted values back,
repeats the observed covariate regression and recomputes the transform with the original
labels.
"""

import hashlib
import logging

import numpy as np
from joblib import Parallel, delayed

from ..utils.utils import tqdm
from .nullmaps import open_null_memmap

lgr = logging.getLogger(__name__)


def _hash_arrays(*arrays):
    """Short hex digest of the bytes, shapes and dtypes of the given arrays (None allowed)."""
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        if a is None:
            h.update(b"none")
            continue
        a = np.ascontiguousarray(np.asarray(a))
        if a.dtype == object:
            a = a.astype(str)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _between_steps_hash(steps):
    """Digest of recorded clean_y() between steps (None/[] -> 'none')."""
    if not steps:
        return "none"
    return _hash_arrays(steps[0]["Y_pre"],
                        *[a for s in steps for a in (s["nuisance"], s["protect"])])


# ------------------------------------------------------------------------------------------
# permutations
# ------------------------------------------------------------------------------------------

def _row_permutations(groups, subjects=None, paired=False, n_perm=1, seed=None):
    """Row permutations matching :func:`~nispace.stats.misc.permute_groups` ("shuffle").

    Consumes each permutation's random generator (``default_rng(seed + i)``) exactly like
    ``permute_groups(strategy="shuffle")`` and returns, per permutation, the index array
    ``pi`` such that the data rows ``Y[pi]`` with the ORIGINAL labels equal the original rows
    with that permutation's shuffled labels. Freedman-Lane with only an intercept therefore
    reproduces the label-shuffle nulls exactly.

    Parameters
    ----------
    groups : array-like, shape (n,)
        Group labels of the rows taking part in the permutation.
    subjects : array-like, shape (n,), optional
        Subject IDs; required if `paired`.
    paired : bool
        Swap rows only within subjects.
    n_perm : int
    seed : int, optional

    Returns
    -------
    list of ndarray of int, each of shape (n,)
    """
    groups = np.asarray(groups)
    n = len(groups)
    if paired:
        if subjects is None:
            raise ValueError("Paired permutation requires argument 'subjects'!")
        subjects = np.asarray(subjects)
        sub_rows = [np.flatnonzero(subjects == sub) for sub in np.unique(subjects)]
    perms = []
    for i in range(n_perm):
        rng = np.random.default_rng(None if seed is None else seed + i)
        if not paired:
            # rng.permutation(groups) == groups[idx] for idx = rng.permutation(n); the rows
            # that carry the original labels in that order are argsort(idx)
            idx = rng.permutation(n)
            perms.append(np.argsort(idx))
        else:
            pi = np.arange(n)
            for rows in sub_rows:
                idx = rng.permutation(len(rows))
                pi[rows] = rows[np.argsort(idx)]
            perms.append(pi)
    return perms


# ------------------------------------------------------------------------------------------
# Freedman-Lane
# ------------------------------------------------------------------------------------------

class _FreedmanLane:
    """Freedman-Lane permutation data for one design.

    Parameters
    ----------
    Y_pre : ndarray (n_all, n_parcels)
        Y before the first recorded between-subject regression (after ComBat, if any).
    steps : list of dict
        Recorded between steps (``"nuisance"`` (n_all, k), ``"protect"`` (n_all, q)), in the
        order clean_y() applied them.
    keep : ndarray of bool (n_all,)
        Rows that take part in the group comparison (group label not NaN). Other rows enter
        the regressions (as in clean_y) but are never permuted.
    subjects : ndarray (n_keep,), optional
        Subject IDs of the kept rows; adds subject indicators to the reduced model (paired).

    Notes
    -----
    Rows with a missing covariate are dropped from every regression and come out as missing
    rows, as in :meth:`~nispace.api.NiSpace.clean_y`; they are never permuted
    (:attr:`perm_pos`). Parcels with missing values are fitted one by one on their observed
    rows; their missing values move with the permuted residuals.
    """

    def __init__(self, Y_pre, steps, keep, subjects=None, dtype=np.float32):
        self.dtype = dtype
        Y_pre = np.asarray(Y_pre, dtype=np.float64)
        n_all = Y_pre.shape[0]
        self.keep_idx = np.flatnonzero(np.asarray(keep, dtype=bool))
        self.steps = [(np.asarray(s["nuisance"], dtype=np.float64),
                       np.asarray(s["protect"], dtype=np.float64)) for s in steps]
        self.steps = [(N, P) for N, P in self.steps if N.shape[1]]

        # rows with complete covariates (clean_y drops the others from its fits)
        row_ok = np.ones(n_all, dtype=bool)
        for N, P in self.steps:
            row_ok &= ~np.isnan(N).any(axis=1) & ~np.isnan(P).any(axis=1)
        self.row_ok = row_ok
        # positions among the kept rows that take part in the permutation
        self.perm_pos = np.flatnonzero(row_ok[self.keep_idx])

        # reduced model: intercept + all nuisance columns (+ subject indicators, paired)
        R = [np.ones((n_all, 1))] + [N for N, _ in self.steps]
        if subjects is not None:
            subjects = np.asarray(subjects)
            uniq = np.unique(subjects)
            D = np.zeros((n_all, len(uniq) - 1))
            for j, sub in enumerate(uniq[1:]):
                D[self.keep_idx[subjects == sub], j] = 1.0
            R.append(D)
        R = np.column_stack(R)
        R[~row_ok] = np.nan
        self.R = R
        # per step: full model [nuisance | protect | 1] and the rows of its pseudo-inverse that
        # give the nuisance coefficients, as clean_y's partial_residuals_nan fits them
        self.X_full = [np.column_stack([N, P, np.ones(n_all)]) for N, P in self.steps]

        # parcels without missing values share one design
        self.nan_cols = np.isnan(Y_pre[row_ok]).any(axis=0)
        self.full_cols = ~self.nan_cols
        self.fitted = np.full_like(Y_pre, np.nan)
        self.res = np.full_like(Y_pre, np.nan)
        if self.full_cols.any():
            Yf = Y_pre[row_ok][:, self.full_cols]
            fitted = R[row_ok] @ (np.linalg.pinv(R[row_ok]) @ Yf)
            self.fitted[np.ix_(row_ok, self.full_cols)] = fitted
            self.res[np.ix_(row_ok, self.full_cols)] = Yf - fitted
        self.step_ops = [
            (N[row_ok], np.linalg.pinv(X[row_ok])[:N.shape[1]])
            for (N, _), X in zip(self.steps, self.X_full)
        ]
        # parcels with missing values: per-parcel fits on the observed rows; fitted values for
        # all rows with covariates, so a permuted row is missing only where it receives a
        # missing residual
        for j in np.flatnonzero(self.nan_cols):
            ok = row_ok & ~np.isnan(Y_pre[:, j])
            b = np.linalg.pinv(R[ok]) @ Y_pre[ok, j]
            self.fitted[row_ok, j] = R[row_ok] @ b
            self.res[ok, j] = Y_pre[ok, j] - self.fitted[ok, j]

    def clean(self, Y):
        """Replay the recorded between-subject regressions on Y (n_all, n_parcels)."""
        Y = np.array(Y, dtype=np.float64)
        out = np.full_like(Y, np.nan)
        row_ok = self.row_ok
        if self.full_cols.any():
            Yf = Y[row_ok][:, self.full_cols]
            for N, B in self.step_ops:
                Yf = Yf - N @ (B @ Yf)
            out[np.ix_(row_ok, self.full_cols)] = Yf
        for j in np.flatnonzero(self.nan_cols):
            y = Y[:, j].copy()
            for (N, _), X in zip(self.steps, self.X_full):
                ok = row_ok & ~np.isnan(y)
                beta = np.linalg.pinv(X[ok]) @ y[ok]
                y_new = np.full_like(y, np.nan)
                y_new[ok] = y[ok] - N[ok] @ beta[:N.shape[1]]
                y = y_new
            out[:, j] = y
        return out

    def permuted(self, pi):
        """Fitted values + residuals permuted among the kept rows (`pi`, positions within the
        kept rows), cleaned as observed; returns the kept rows (n_keep, n_parcels)."""
        res = self.res.copy()
        res[self.keep_idx] = self.res[self.keep_idx[pi]]
        return self.clean(self.fitted + res)[self.keep_idx].astype(self.dtype)


def generate_group_nulls(Y_arr, groups, subjects, apply_transform, n_perm, seed, n_proc=1,
                         paired=False, strategy="shuffle", fl=None, memmap_path=None,
                         desc="Null transformations", verbose=True):
    """Null cube (n_contrasts, n_perm, n_parcels) of a Y transform under group permutation.

    Parameters
    ----------
    Y_arr : ndarray (n_keep, n_parcels)
        Observed (cleaned) Y rows taking part in the comparison. Used for label shuffling.
    groups : ndarray (n_keep,)
        Dummy-coded group labels (0 = a, 1 = b).
    subjects : ndarray (n_keep,) or None
        Numeric subject codes; required for paired designs (rows are matched by subject).
    apply_transform : callable
        ``apply_transform(y=..., groups=..., subjects=...)`` -> (n_contrasts, n_parcels).
    fl : _FreedmanLane, optional
        If given, Freedman-Lane permutation (requires ``strategy="shuffle"``); otherwise
        labels are shuffled via :func:`~nispace.stats.misc.permute_groups`.
    memmap_path : str, Path, True, or None
        Write the cube to disk as it is generated.

    Returns
    -------
    ndarray or np.memmap
    """
    from ..stats.misc import permute_groups

    if fl is None:
        group_perms = permute_groups(groups=groups, subjects=subjects, n_perm=n_perm,
                                     n_proc=n_proc, seed=seed, verbose=verbose,
                                     paired=paired, strategy=strategy)
        if n_perm == 1:
            group_perms = [group_perms]

        def par_fun(g):
            return apply_transform(y=Y_arr, groups=g, subjects=subjects)
        items = group_perms
    else:
        if "shuff" not in strategy:
            raise ValueError(f"Group permutation strategy '{strategy}' is not supported with "
                             "between-subject covariates: the Freedman-Lane permutation needs "
                             "free shuffling (strategy='shuffle').")
        pos = fl.perm_pos
        sub_perms = _row_permutations(groups[pos], subjects[pos] if paired else None,
                                      paired=paired, n_perm=n_perm, seed=seed)
        row_perms = []
        for sp in sub_perms:
            pi = np.arange(len(groups))
            pi[pos] = pos[sp]
            row_perms.append(pi)

        def par_fun(pi):
            return apply_transform(y=fl.permuted(pi), groups=groups, subjects=subjects)
        items = row_perms

    # each result (n_contrasts, n_parcels) goes straight into one (n_contrasts, n_perm,
    # n_parcels) cube (on disk with memmap_path), so no list of n_perm arrays is held next to it
    cube = None
    for i, Y_null_i in enumerate(Parallel(n_jobs=n_proc, return_as="generator")(
        delayed(par_fun)(item) for item in tqdm(items, desc=f"{desc} ({n_proc} proc)",
                                                disable=not verbose)
    )):
        if cube is None:
            cube = open_null_memmap(memmap_path, (Y_null_i.shape[0], n_perm, Y_null_i.shape[1]),
                                    Y_null_i.dtype)
        cube[:, i, :] = Y_null_i
    return cube
