"""Tests for NiSpace.clean_y() -- specifically the "between" covariate regression.

Regression tests for a bug where clean_y(how="between", covariates_between=...)
without an explicit `protect` argument fully demeaned every parcel across subjects
(removing the intercept, not just the covariate-attributable component). That's
harmless for group-difference transforms (a shared per-parcel constant cancels out
of any a-vs-b contrast, e.g. group_colocalization()'s hedges/cohen/zscore/meandiff/
elemdiff/pairedcohen), but it silently destroyed real signal for any pipeline that
colocalizes the cleaned maps directly -- colocalization()/xsea(), which pass
covariates_between with no protect (nispace/workflows.py). On real data (see
_explore/mem_covariate_regression_prototype.ipynb), cleaning against a covariate
with *zero* true relationship to anything collapsed mean per-subject
colocalization from ~0.81 to ~0.002.
"""

import numpy as np
import pandas as pd

from nispace import NiSpace


def test_between_regression_preserves_intercept(rng):
    """clean_y(how='between') without `protect` must remove only the
    covariate-attributable component, never the per-parcel intercept."""
    n_subj, n_parcels = 25, 40
    Y_true = rng.normal(size=n_parcels)
    Y = Y_true[np.newaxis, :] + rng.normal(scale=0.3, size=(n_subj, n_parcels))
    cov = rng.normal(size=n_subj)  # covariate uncorrelated with Y_true by construction

    parcel_labels = [f"p{i}" for i in range(n_parcels)]
    y_df = pd.DataFrame(Y, index=[f"y{i}" for i in range(n_subj)], columns=parcel_labels)
    x_df = pd.DataFrame(Y_true[np.newaxis, :], index=["x0"], columns=parcel_labels)

    nsp = NiSpace(x=x_df, y=y_df, parcellation=None, standardize=False,
                 n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    nsp.clean_y(how="between", covariates_between=pd.Series(cov, name="cov"), verbose=False)
    Y_clean = nsp.get_y().to_numpy()

    # reference: OLS on [cov | intercept], remove only the cov term (partial_residuals_nan's
    # definition) -- this is the invariant that a full-residual (residuals_nan) regression breaks
    X_design = np.column_stack([cov, np.ones(n_subj)])
    beta = np.linalg.pinv(X_design.T @ X_design) @ (X_design.T @ Y)
    Y_expected = Y - np.outer(cov, beta[0])

    np.testing.assert_allclose(Y_clean, Y_expected, atol=1e-4)


def test_irrelevant_covariate_does_not_destroy_colocalization(rng):
    """An irrelevant (pure-noise) between-covariate must not measurably change
    per-subject colocalization with X after clean_y(how='between'). Direct
    regression test for the colocalization()/xsea()-facing symptom: previously,
    cleaning against a covariate with zero real relationship to anything
    collapsed mean per-subject colocalization from ~0.8 to ~0."""
    n_subj, n_parcels = 30, 100
    Y_true = rng.normal(size=n_parcels)
    Y = Y_true[np.newaxis, :] + rng.normal(scale=0.5, size=(n_subj, n_parcels))
    irrelevant_cov = rng.normal(size=n_subj)

    parcel_labels = [f"p{i}" for i in range(n_parcels)]
    y_df = pd.DataFrame(Y, index=[f"y{i}" for i in range(n_subj)], columns=parcel_labels)
    x_df = pd.DataFrame(Y_true[np.newaxis, :], index=["x0"], columns=parcel_labels)

    def mean_coloc(Y_arr):
        return np.mean([np.corrcoef(Y_arr[i], Y_true)[0, 1] for i in range(n_subj)])

    coloc_before = mean_coloc(Y)
    assert coloc_before > 0.5  # sanity: the synthetic signal is actually detectable pre-cleaning

    nsp = NiSpace(x=x_df, y=y_df, parcellation=None, standardize=False,
                 n_proc=1, verbose=False, return_self=False)
    nsp.fit()
    nsp.clean_y(how="between", covariates_between=pd.Series(irrelevant_cov, name="cov"),
                verbose=False)
    coloc_after = mean_coloc(nsp.get_y().to_numpy())

    # previously this collapsed to near 0 (>95% drop); an irrelevant covariate should
    # leave colocalization essentially unchanged
    assert coloc_after > 0.9 * coloc_before
