import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from nilearn.plotting import plot_design_matrix
from tqdm.auto import tqdm

from .. import lgr
from ..stats.misc import residuals_nan, partial_residuals_nan, zscore_df
from ..utils.utils import set_log


def _normalize_cov_df(cov, n_subjects, name="covariates"):
    """Normalize covariate input to a lowercase-column DataFrame."""
    if isinstance(cov, np.ndarray):
        if cov.ndim == 1:
            cov = cov[:, np.newaxis]
        df = pd.DataFrame(cov, columns=[f"cov_{i}" for i in range(cov.shape[1])])
        lgr.warning(f"'{name}' provided as array without column names. "
                    f"Using generic names: {list(df.columns)}")
    elif isinstance(cov, pd.Series):
        df = cov.to_frame()
    elif isinstance(cov, pd.DataFrame):
        df = cov.copy()
    else:
        lgr.critical_raise(f"'{name}' of type {type(cov)} not supported!", TypeError)
    df.columns = [str(c).lower() for c in df.columns]
    df = df.reset_index(drop=True)
    if df.shape[0] != n_subjects:
        lgr.critical_raise(f"'{name}' has {df.shape[0]} rows but Y has "
                           f"{n_subjects}. They must match.", ValueError)
    return df


def _detect_categoricals(df):
    """Return (cat_cols, cont_cols); 'site' is always categorical."""
    cat_cols = [
        c for c in df.columns
        if (pd.api.types.is_object_dtype(df[c]) or
            pd.api.types.is_string_dtype(df[c]) or
            isinstance(df[c].dtype, pd.CategoricalDtype) or
            c == "site")
    ]
    cont_cols = [c for c in df.columns if c not in cat_cols]
    return cat_cols, cont_cols


def _encode(df, cont_cols, cols_to_encode, dtype=np.float32):
    """One-hot encode cols_to_encode (drop_first=True), prepend continuous cols."""
    parts = [df[cont_cols]] if cont_cols else []
    for col in cols_to_encode:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=dtype)
        parts.append(dummies)
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)


def _clean_y_between(Y_arr, covariates_between, n_subjects,
                     protect,
                     combat, combat_protect, combat_train, combat_model, combat_kwargs,
                     plot_design_between, n_proc, dtype, verbose):
    """
    Between-subjects covariate regression (and optional ComBat harmonization).

    Returns
    -------
    Y_arr : np.ndarray
        Cleaned data array.
    combat_model : object or None
        Fitted ComBat model (None if not used).
    combat_covariates : pd.DataFrame or None
        Covariates passed to ComBat (None if not used).
    """
    # --- normalize input ---
    bcov_df = _normalize_cov_df(covariates_between, n_subjects, "covariates_between")
    cat_cols, cont_cols = _detect_categoricals(bcov_df)
    non_site_cat_cols = [c for c in cat_cols if c != "site"]
    has_site = "site" in bcov_df.columns
    lgr.info(f"Detected categorical covariates: {cat_cols or 'none'}; "
             f"continuous: {cont_cols or 'none'}.")

    # --- check site for combat ---
    if combat and not has_site:
        lgr.warning("ComBat requested but no 'site' column found in covariates_between. "
                    "Disabling ComBat.")
        combat = False

    # --- encode ---
    if combat:
        # regression matrix: continuous + non-site categoricals; site stays raw for ComBat
        bcov_encoded = _encode(bcov_df, cont_cols, non_site_cat_cols, dtype=dtype)
        # combat_protect: protect-only columns (not regressed)
        protect_encoded = None
        if combat_protect is not None:
            cp_df = _normalize_cov_df(combat_protect, n_subjects, "combat_protect")
            cp_cat, cp_cont = _detect_categoricals(cp_df)
            protect_encoded = _encode(cp_df, cp_cont, cp_cat, dtype=dtype)
            lgr.info(f"ComBat protection-only covariates: {list(cp_df.columns)}.")
        # ComBat covariates: SITE + regression covariates + protect-only covariates; site always first
        combat_covariates = pd.concat(
            [p for p in [
                bcov_df[["site"]].rename(columns={"site": "SITE"}),
                bcov_encoded if not bcov_encoded.empty else None,
                protect_encoded,
            ] if p is not None],
            axis=1
        )
    else:
        # encode everything; site one-hot last
        bcov_encoded = _encode(bcov_df, cont_cols,
                               non_site_cat_cols + (["site"] if has_site else []),
                               dtype=dtype)
        combat_covariates = None

    # --- plot design matrix ---
    if plot_design_between:
        if combat:
            site_int = pd.Categorical(bcov_df["site"]).codes
            plot_df = pd.concat(
                [bcov_encoded, pd.Series(site_int, name="ComBat: site")],
                axis=1
            )
        else:
            plot_df = bcov_encoded  # site already last
        plot_design_matrix(plot_df.astype(float))
        plt.title("$Between$ design matrix")
        plt.ylabel("Y maps / subjects")
        plt.show()

    # --- ComBat first ---
    if combat:
        n_prot = protect_encoded.shape[1] if protect_encoded is not None else 0
        lgr.info(f"Performing ComBat harmonization with {bcov_encoded.shape[1]} protected "
                 f"covariate(s) for regression and {n_prot} protect-only covariate(s).")
        Y_arr_isnan = np.isnan(Y_arr)
        if Y_arr_isnan.any():
            lgr.warning("Detected missing values in Y data, which is not supported with "
                        "ComBat harmonization. Missing values will be imputed with "
                        "map-wise medians and replaced by nan after harmonization. "
                        "CAVE: experimental feature!")
            Y_arr = np.apply_along_axis(
                lambda x: np.where(np.isnan(x), np.nanmedian(x), x),
                axis=1, arr=Y_arr,
            )
        # validate combat_train
        idx_train = None
        if combat_train is not None:
            if (isinstance(combat_train, (list, tuple, np.ndarray, pd.Series))
                    and len(combat_train) == n_subjects
                    and all(i in {True, False, 0, 1} for i in combat_train)):
                idx_train = np.array(combat_train).astype(bool)
            else:
                lgr.warning(f"'combat_train' must be a boolean vector of length {n_subjects}. "
                            "Ignoring.")
        # apply
        from neuroHarmonize import harmonizationLearn, harmonizationApply
        if combat_model is None:
            if idx_train is None:
                combat_model, Y_arr = harmonizationLearn(
                    data=Y_arr, covars=combat_covariates, **combat_kwargs)
            else:
                lgr.info(f"Training ComBat on {idx_train.sum()} subjects, "
                         f"applying to {(~idx_train).sum()}.")
                temp = np.zeros(Y_arr.shape)
                combat_model, temp[idx_train, :] = harmonizationLearn(
                    data=Y_arr[idx_train, :],
                    covars=combat_covariates.iloc[idx_train],
                    **combat_kwargs)
                temp[~idx_train, :] = harmonizationApply(
                    data=Y_arr[~idx_train, :],
                    covars=combat_covariates.iloc[~idx_train],
                    model=combat_model)
                Y_arr = temp
        else:
            Y_arr = harmonizationApply(
                data=Y_arr, covars=combat_covariates, model=combat_model)
        Y_arr[Y_arr_isnan] = np.nan
        Y_arr = Y_arr.astype(dtype)

    # --- encode protect for OLS (analogous to combat_protect for ComBat) ---
    protect_arr = None
    if protect is not None:
        prot_df = _normalize_cov_df(protect, n_subjects, "protect")
        prot_cat, prot_cont = _detect_categoricals(prot_df)
        prot_encoded = _encode(prot_df, prot_cont, prot_cat, dtype=dtype)
        if not prot_encoded.empty:
            protect_arr = prot_encoded.values.astype(dtype)
            lgr.info(f"Protecting {protect_arr.shape[1]} variable(s) during regression: "
                     f"{list(prot_encoded.columns)}.")

    # --- regression (after ComBat if both requested) ---
    if not bcov_encoded.empty:
        reg_arr = bcov_encoded.values.astype(dtype)
        lgr.info(f"Regressing {reg_arr.shape[1]} between covariate(s) from Y.")
        if protect_arr is not None:
            Y_partial = Parallel(n_jobs=n_proc)(
                delayed(partial_residuals_nan)(reg_arr, protect_arr, Y_arr[:, i_p]) for i_p in tqdm(
                    range(Y_arr.shape[1]),
                    desc=f"Regressing {reg_arr.shape[1]} between covariate(s) from Y, "
                         f"protecting {protect_arr.shape[1]} variable(s) ({n_proc} proc)",
                    disable=not verbose
                ))
        else:
            Y_partial = Parallel(n_jobs=n_proc)(
                delayed(residuals_nan)(reg_arr, Y_arr[:, i_p]) for i_p in tqdm(
                    range(Y_arr.shape[1]),
                    desc=f"Regressing {reg_arr.shape[1]} between covariate(s) from Y ({n_proc} proc)",
                    disable=not verbose
                ))
        Y_arr = np.array(Y_partial, dtype=dtype).T

    return Y_arr, combat_model, combat_covariates


def _clean_y_within(Y_arr, covariates_within, Z, n_maps, n_parcels,
                    within_y_specific, n_proc, dtype, verbose):
    """
    Within-subjects covariate regression (across parcels per map).

    Returns
    -------
    Y_arr : np.ndarray
    used_z : bool
    """
    used_z = False
    wcov_arr = None

    if isinstance(covariates_within, str):
        if covariates_within in ["z", "Z"]:
            lgr.info("Using Z data for 'within' covariate regression.")
            if Z is not None:
                wcov_arr = np.array(Z)
                used_z = True
            else:
                lgr.critical_raise("Provide Z data at initialization for Z regression!", ValueError)
        else:
            lgr.error(f"'within' covariate '{covariates_within}' not defined! "
                      "Pass 'Z' if you want to regress Z data, or provide a custom array.")
    elif isinstance(covariates_within, (np.ndarray, pd.Series, pd.DataFrame)):
        wcov_arr = np.array(covariates_within, dtype=dtype)
        if wcov_arr.ndim == 1:
            wcov_arr = wcov_arr[np.newaxis, :]
        lgr.info(f"Assuming {wcov_arr.shape[0]} 'within' covariate map(s) for {wcov_arr.shape[1]} parcels.")
        if wcov_arr.shape[1] != n_parcels:
            lgr.error(f"Covariate number of parcels {wcov_arr.shape[1]} does not match Y data!")
            wcov_arr = None
        elif within_y_specific and wcov_arr.shape[0] != n_maps:
            lgr.error(f"If 'within_y_specific' is True, the number of covariate maps "
                      f"({wcov_arr.shape[0]}) must match the number of Y maps ({n_maps})!")
            wcov_arr = None
    else:
        lgr.critical_raise(f"'covariates_within' of type {type(covariates_within)} not supported!",
                           TypeError)

    if wcov_arr is not None:
        if wcov_arr.shape[0] == 1:
            lgr.info("Got one covariate map. Using this for each Y map.")
            wcov_arr = np.row_stack([wcov_arr] * n_maps)
        elif within_y_specific and wcov_arr.shape[0] == n_maps:
            lgr.info("Got as many covariate maps as Y maps. Running y-specific regression.")
        else:
            lgr.info(f"Got {wcov_arr.shape[0]} covariate maps. Using these for each Y map.")
            wcov_arr = np.stack([wcov_arr.T] * n_maps, axis=0)
        Y_partial = Parallel(n_jobs=n_proc)(
            delayed(residuals_nan)(wcov_arr[i_y], Y_arr[i_y, :]) for i_y in tqdm(
                range(n_maps),
                desc=f"Regressing within covariate(s) from Y ({n_proc} proc)",
                disable=not verbose
            ))
        Y_arr = np.array(Y_partial, dtype=dtype)

    return Y_arr, used_z, wcov_arr is not None
