"""NiMARE integration helpers."""

import logging

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import label as nd_label
from tqdm.auto import tqdm

from ..utils.utils import vol_to_vect_arr

lgr = logging.getLogger(__name__)

# NiMARE's default brain mask is FSL MNI152 (= MNI152NLin6Asym) at 2 mm.
_NIMARE_SPACE = "MNI152NLin6Asym"


# Module-level so joblib can pickle it.
def _run_one_perm(
    iter_xyz, iter_df, est, parc_arr, parc_idc, bg_values,
    voxel_thresh_z, min_cluster_extent, cluster_stat,
    return_cluster_stat=False,
):
    """Run one coordinate-sampling permutation and return a parcellated map.

    The parcellation array (parc_arr/parc_idc/bg_values) is pre-resampled to
    ALE space once before the loop, so this function only calls vol_to_vect_arr
    on the raw 3-D array — no neuromaps resampling per iteration.

    When return_cluster_stat=True and voxel_thresh_z is not None, returns
    (parc_map, max_cluster_stat) where max_cluster_stat is the size or mass of
    the largest cluster BEFORE FWE filtering — matching what NiMARE stores in
    null_distributions_ for FWE threshold derivation.
    """
    iter_xyz = np.squeeze(iter_xyz)   # (n_coords, 1, 3) → (n_coords, 3)
    iter_df = iter_df.copy()
    iter_df[["x", "y", "z"]] = iter_xyz
    iter_ma = est.kernel_transformer.transform(
        iter_df, masker=est.masker, return_type="sparse"
    )
    iter_ss = est._compute_summarystat(iter_ma)  # ALE values in [0, 1], NOT z-scores

    if voxel_thresh_z is not None:
        # Convert ALE stat → z before comparing to voxel_thresh_z,
        # mirroring NiMARE's own cluster correction pipeline.
        if hasattr(est, "_summarystat_to_p"):
            _result = est._summarystat_to_p(iter_ss, null_method="approximate")
            if isinstance(_result, tuple):
                # NiMARE >= 0.6: returns (p_values, z_values) — use z directly.
                iter_z = _result[1]
            else:
                # older NiMARE: returns p_values only.
                from nimare.transforms import p_to_z
                iter_z = p_to_z(_result, tail="one")
        elif hasattr(est, "_ale_to_p"):
            from nimare.transforms import p_to_z
            iter_z = p_to_z(est._ale_to_p(iter_ss), tail="one")
        else:
            raise AttributeError(
                "Estimator has neither _summarystat_to_p nor _ale_to_p. "
                "Cluster-map null generation requires NiMARE >= 0.2.0."
            )
        arr3d = est.masker.inverse_transform(
            (iter_z >= voxel_thresh_z).astype(np.float32)
        ).get_fdata()

        # Label connected components once; compute per-cluster stats for both
        # filtering and validation (max stat before filtering = NiMARE semantics).
        max_stat = 0
        need_labels = (
            (min_cluster_extent is not None and min_cluster_extent > 0)
            or return_cluster_stat
        )
        if need_labels:
            labeled, n_comps = nd_label(arr3d > 0)
            if cluster_stat == "mass":
                z_arr3d = est.masker.inverse_transform(iter_z).get_fdata()
            ci_stats = []
            for ci in range(1, n_comps + 1):
                s = (
                    float(z_arr3d[labeled == ci].sum())
                    if cluster_stat == "mass"
                    else int((labeled == ci).sum())
                )
                ci_stats.append(s)
            max_stat = max(ci_stats) if ci_stats else 0
            if min_cluster_extent is not None and min_cluster_extent > 0:
                for ci, s in enumerate(ci_stats, 1):
                    if s < min_cluster_extent:
                        arr3d[labeled == ci] = 0

        arr3d = arr3d.astype(np.float32)
    else:
        arr3d = est.masker.inverse_transform(iter_ss).get_fdata().astype(np.float32)
        max_stat = None

    parc_map = vol_to_vect_arr(arr3d, parc_arr, parc_idc, bg_values)
    if return_cluster_stat:
        return parc_map, max_stat
    return parc_map


def _validate_null_maps(
    null_arr, max_stats, result, outcome_map,
    parc_arr, parc_idc, voxel_thresh_z, nimare_cluster_extent,
    cluster_stat, alpha, observed_map,
):
    """Log validation metrics comparing our null distribution to NiMARE's own outputs.

    Cluster mode: compares the (1-alpha) percentile of our max-cluster-stat null
    distribution against the threshold stored in corrected_result.estimator.null_distributions_.
    Optionally re-derives the cluster map with our threshold and compares to
    observed_map (Pearson r).

    Continuous mode: computes empirical parcel-level p-values from null_arr and
    correlates (Spearman) against NiMARE's parametric voxelwise p-map parcellated
    with the same scheme.
    """
    _bg = np.array([], dtype=np.float64)

    if max_stats is not None:
        # ---- cluster mode ----
        max_stats_arr = np.asarray(max_stats, dtype=float)
        our_thresh = np.percentile(max_stats_arr, 100.0 * (1.0 - alpha))
        lgr.info("=== Validation (cluster mode) ===")
        lgr.info(
            f"  NiMARE FWE {cluster_stat} threshold (from corrector): "
            f"{nimare_cluster_extent}"
        )
        lgr.info(
            f"  Our null-derived {cluster_stat} threshold "
            f"({100*(1-alpha):.0f}th pctile of {len(max_stats)} perms): "
            f"{our_thresh:.1f}"
        )
        if nimare_cluster_extent is not None and nimare_cluster_extent > 0:
            ratio = our_thresh / nimare_cluster_extent
            lgr.info(
                f"  Ratio (ours / NiMARE): {ratio:.3f}  "
                "(should be ~1.0 for large n_perm; deviations are sampling noise)"
            )

        # Re-derive cluster map with our threshold and compare to observed_map.
        if observed_map is not None:
            obs_z_img = result.get_map("z")
            obs_z_arr = obs_z_img.get_fdata()
            binary_arr = (obs_z_arr >= voxel_thresh_z).astype(np.float32)
            labeled, n_comps = nd_label(binary_arr > 0)
            for ci in range(1, n_comps + 1):
                s = (
                    float(obs_z_arr[labeled == ci].sum())
                    if cluster_stat == "mass"
                    else int((labeled == ci).sum())
                )
                if s < our_thresh:
                    binary_arr[labeled == ci] = 0
            our_cluster_parc = vol_to_vect_arr(
                binary_arr, parc_arr, parc_idc, _bg
            ).astype(float)
            if isinstance(observed_map, nib.spatialimages.SpatialImage):
                obs_user_parc = vol_to_vect_arr(
                    observed_map.get_fdata().astype(np.float32),
                    parc_arr, parc_idc, _bg,
                ).astype(float)
            else:
                obs_user_parc = np.asarray(observed_map, dtype=float)
            mask = np.isfinite(our_cluster_parc) & np.isfinite(obs_user_parc)
            if mask.sum() > 1:
                from scipy.stats import pearsonr
                r, _ = pearsonr(our_cluster_parc[mask], obs_user_parc[mask])
                lgr.info(
                    f"  Re-derived cluster map vs observed_map: r={r:.4f}  "
                    "(should be ~1.0; low r → threshold mismatch or wrong cluster_stat)"
                )

    else:
        # ---- continuous mode ----
        lgr.info("=== Validation (continuous mode) ===")
        obs_parc = vol_to_vect_arr(
            result.get_map(outcome_map).get_fdata().astype(np.float32),
            parc_arr, parc_idc, _bg,
        ).astype(float)
        valid = np.isfinite(obs_parc)
        if valid.sum() < 2:
            lgr.warning("  Too few valid parcels for continuous-mode validation.")
            return
        # Empirical p: proportion of null values >= observed at each parcel.
        emp_p = (null_arr[:, valid] >= obs_parc[valid]).mean(axis=0)
        nimare_p_parc = vol_to_vect_arr(
            result.get_map("p").get_fdata().astype(np.float32),
            parc_arr, parc_idc, _bg,
        ).astype(float)
        nimare_p_valid = nimare_p_parc[valid]
        finite = np.isfinite(emp_p) & np.isfinite(nimare_p_valid)
        if finite.sum() > 1:
            from scipy.stats import spearmanr
            r, pval = spearmanr(emp_p[finite], nimare_p_valid[finite])
            lgr.info(
                f"  Empirical p (our null) vs NiMARE parametric p-map: "
                f"Spearman r={r:.3f}  (p={pval:.4f})"
            )
            lgr.info(
                "  Note: NiMARE p is voxelwise-approximate; ours is parcel-level "
                "Monte Carlo — perfect agreement is not expected, but r should be high."
            )


def null_maps_from_nimare(
    result,
    parcellation,
    corrected_result=None,
    voxel_thresh=0.001,
    outcome_map="stat",
    cluster_stat="size",
    alpha=0.05,
    n_perm=1000,
    map_label=None,
    observed_map=None,
    validate=True,
    seed=None,
    n_proc=1,
    dtype=np.float32,
    verbose=True,
):
    """Generate parcellated null maps from a fitted NiMARE MetaResult.

    Re-implements NiMARE's MonteCarlo FWE coordinate-sampling loop: for each
    permutation, study coordinates are drawn uniformly at random from the brain
    mask, a full ALE map is computed, and the result is parcellated. This
    produces null maps under the same generative model NiMARE uses for its own
    FWE correction, making them the principled null for colocalization of
    meta-analytic maps.

    Two modes are supported:

    * **Continuous** (default, ``corrected_result=None``): each null map
      contains parcellated ALE stat values, matching an unthresholded
      ``result.get_map("stat")`` as Y.
    * **Binary cluster** (``corrected_result`` provided): each null ALE map is
      converted to z-scores via the estimator's null distribution, thresholded
      at the voxel-level CDT, and parcellated to a [0, 1] cluster-coverage map.
      The FWE cluster extent filter is **not** applied to null maps — applying it
      would make ~95% of them empty by construction (that is what FWE alpha=0.05
      means). Null maps represent CDT-thresholded random ALE; the FWE correction
      is already reflected in the observed Y image from :func:`get_binary_cluster_map`.

    Parameters
    ----------
    result : nimare.results.MetaResult
        Fitted MetaResult (e.g. from ``ALE().fit(dataset)``). Must embed a
        CBMA estimator with ``inputs_["coordinates"]`` and
        ``kernel_transformer``. IBMA estimators are not supported.
    parcellation : str or :class:`~nispace.core.parcellation.Parcellation`
        Parcellation to use. Either a NiSpace parcellation name (e.g.
        ``"Schaefer200"``) or a
        :class:`~nispace.core.parcellation.Parcellation` object (e.g.
        ``nsp._parc`` from a fitted :class:`~nispace.NiSpace` instance). Must
        have at least one volumetric/MNI space — surface-only parcellations are
        not supported.
    corrected_result : nimare.results.MetaResult or None, optional
        The corrected MetaResult returned by ``FWECorrector.transform(result)``.
        If provided, switches to binary cluster mode: the FWE cluster extent
        threshold is extracted from
        ``corrected_result.estimator.null_distributions_`` via
        :func:`nimare_fwe_thresholds` and applied to each null ALE map. Only
        ``method="montecarlo"`` is supported. Default: ``None`` (continuous
        mode).
    voxel_thresh : float, optional
        Uncorrected voxel-level p-value for the cluster-defining threshold
        (CDT) used when *corrected_result* is provided. Must match the
        threshold actually applied during FWE correction. NiMARE's default
        (and the default here) is ``0.001`` (z ≈ 3.09). NiMARE's
        ``FWECorrector`` silently ignores ``voxel_thresh`` passed to its
        constructor, so always specify it here explicitly if you used a
        non-default CDT. Ignored when *corrected_result* is ``None``.
        Default: ``0.001``.
    outcome_map : str, optional
        Key in ``result.maps`` used as the default *map_label* and for the
        continuous-mode sanity check. For ALE, available keys are ``"stat"``
        (ALE values), ``"z"``, and ``"p"``. The permutation loop always starts
        from ALE stat values (``_compute_summarystat``) regardless of this
        setting. Default: ``"stat"``.
    cluster_stat : {"size", "mass"}, optional
        Cluster-level statistic used for FWE correction. Only relevant when
        *corrected_result* is provided.

        * ``"size"`` (default): cluster extent in voxels. Each null cluster is
          removed if its voxel count is below the FWE threshold.
        * ``"mass"`` : cluster mass (sum of z-scores within the cluster). Each
          null cluster is removed if its mass is below the FWE threshold.

        Must match the statistic used to produce the observed Y image (i.e. the
        ``map_key`` passed to :func:`get_binary_cluster_map`).
    alpha : float, optional
        FWE significance level passed to :func:`nimare_fwe_thresholds` when
        *corrected_result* is provided, to derive the cluster extent threshold.
        Default: 0.05.
    n_perm : int, optional
        Number of null permutations. Default: 1000.
    map_label : str or None, optional
        Key in the returned dict. Defaults to *outcome_map*. Must match the
        Y-map label in your NiSpace instance so that
        ``permute(maps_nulls=...)`` routes to the right null distribution.
    observed_map : NIfTI image or array-like of shape (n_parcels,), optional
        Y map for a sanity check (Pearson r against the expected observed map,
        warned if < 0.99). Accepts either a NIfTI image in NiMARE's native
        space (parcellated internally) or a pre-parcellated array.
        In **continuous mode**: compared against the parcellated
        ``result.get_map(outcome_map)``.
        In **binary cluster mode**: compared against the observed ALE z-map
        thresholded and cluster-filtered with the same parameters — i.e. the
        exact procedure applied to each null map. Pass the output of
        :func:`get_binary_cluster_map` here (image or parcellated array).
    validate : bool, optional
        If True (default), log validation metrics after the permutation loop.

        * **Cluster mode**: compares the ``(1-alpha)`` percentile of the
          max-cluster-stat null distribution collected across permutations
          against the threshold from
          ``corrected_result.estimator.null_distributions_`` (these should
          agree within sampling noise). If *observed_map* is provided, also
          re-derives the cluster map using our threshold and reports Pearson r
          against *observed_map*.
        * **Continuous mode**: computes empirical parcel-level p-values from
          the null array and correlates (Spearman) against NiMARE's parametric
          voxelwise p-map parcellated with the same scheme.
    seed : int or None, optional
        Random seed for reproducibility. Default: None.
    n_proc : int, optional
        Number of parallel processes (``joblib.Parallel`` n_jobs). Default: 1.
    dtype : numpy dtype, optional
        Output array dtype. Default: ``np.float32``.
    verbose : bool, optional
        Show progress bar and log messages. Default: True.

    Returns
    -------
    dict
        ``{map_label: ndarray(n_perm, n_parcels)}``, directly passable as
        ``maps_nulls=`` to :meth:`~nispace.NiSpace.permute`.

    Notes
    -----
    **Binary cluster mode — value distributions and correlation method.**
    After parcellation, each parcel in the binary cluster map receives the
    fraction of its voxels that fall inside the significant cluster (values in
    [0, 1], typically 0–0.4 for the observed map). Use ``method="pearson"``
    in :meth:`~nispace.NiSpace.colocalize` — Pearson on a fractional
    cluster-coverage map is equivalent to point-biserial correlation and is
    more appropriate than Spearman, which degrades due to heavy ties at zero.

    Binary null maps are thresholded at the voxel-level CDT only; the FWE
    cluster extent filter is **not** applied to null maps (applying it would
    make ~95% of them empty by construction — that is what FWE alpha=0.05
    means). Null maps therefore have a lower value range (typically 0–0.05–0.1)
    than the observed map: random coordinates produce many small scattered
    clusters while real coordinates converge into one focused cluster. This
    contrast is the signal the permutation test exploits.

    Relies on the private NiMARE methods ``_compute_summarystat`` and
    ``_summarystat_to_p`` (or ``_ale_to_p`` for older versions). Requires
    NiMARE >= 0.2.0; informative errors are raised for missing attributes.

    NiMARE's default brain mask is MNI152NLin6Asym at 2 mm. The
    MNI152NLin6Asym image for the parcellation is preferred to minimise
    resampling error. The parcellation is resampled to ALE space once before
    the permutation loop; each null map is parcellated with a direct array
    operation rather than a per-iteration neuromaps call.

    **Continuous workflow** (unthresholded ALE stat map as Y)::

        import nimare
        from nimare.meta.cbma import ALE
        from nimare.tests.utils import get_test_data_path
        from nispace import NiSpace, null_maps_from_nimare
        from nispace.datasets import fetch_reference

        dset = nimare.dataset.Dataset(get_test_data_path() + "/test_pain_dataset.json")
        result = ALE().fit(dset)
        ale_img = result.get_map("stat")

        ref = fetch_reference("pet", maps="VAChT", parcellation="Schaefer200")
        nsp = NiSpace(x=ref, y=ale_img, y_labels=["pain_ale"],
                      parcellation="Schaefer200")
        nsp.fit()
        nsp.colocalize()

        null_maps = null_maps_from_nimare(
            result, nsp._parc,
            map_label="pain_ale",
            observed_map=ale_img,
            n_perm=1000,
        )
        nsp.permute("maps", maps_which="Y", maps_nulls=null_maps, n_perm=1000)
        p = nsp.get_p_values()

    **Binary cluster workflow** (FWE cluster map binarized as Y)::

        from nimare.correct import FWECorrector
        from nispace import get_binary_cluster_map

        fwe = FWECorrector(method="montecarlo", n_iters=1000)
        corrected = fwe.transform(result)    # null_distributions_ live here
        binary_img = get_binary_cluster_map(corrected)

        nsp = NiSpace(x=ref, y=binary_img, y_labels=["pain_cluster"],
                      parcellation="Schaefer200")
        nsp.fit()
        nsp.colocalize()

        null_maps = null_maps_from_nimare(
            result, nsp._parc,
            corrected_result=corrected,      # pass the corrected MetaResult
            voxel_thresh=0.001,
            map_label="pain_cluster",
            observed_map=binary_img,
            n_perm=1000,
        )
        nsp.permute("maps", maps_which="Y", maps_nulls=null_maps, n_perm=1000)
        p = nsp.get_p_values()
    """
    lgr.warning(
        "null_maps_from_nimare() is experimental. The API and behaviour may "
        "change in future releases without notice."
    )

    # --- lazy import ---
    try:
        from nimare.utils import vox2mm
    except ImportError as exc:
        raise ImportError(
            "null_maps_from_nimare requires 'nimare'. "
            "Install with: pip install nimare"
        ) from exc

    # --- validate result ---
    if not (hasattr(result, "estimator") and hasattr(result, "maps")):
        raise TypeError(
            "result must be a NiMARE MetaResult (returned by estimator.fit(dataset)). "
            f"Got: {type(result)}"
        )
    est = result.estimator

    # --- validate estimator ---
    if not (
        hasattr(est, "inputs_")
        and "coordinates" in est.inputs_
        and hasattr(est, "kernel_transformer")
    ):
        raise TypeError(
            "The estimator in result does not appear to be a CBMA estimator "
            "(missing inputs_['coordinates'] or kernel_transformer). "
            "IBMA and non-standard estimators are not supported."
        )
    if not hasattr(est, "_compute_summarystat"):
        raise AttributeError(
            "estimator._compute_summarystat not found. "
            "This NiMARE version may be incompatible (requires >= 0.2.0)."
        )

    # --- validate outcome_map ---
    available = [k for k, v in result.maps.items() if v is not None]
    if outcome_map not in result.maps or result.maps[outcome_map] is None:
        raise ValueError(
            f"outcome_map='{outcome_map}' not found in result.maps. "
            f"Available: {available}"
        )

    # --- derive cluster thresholds from corrected result (binary cluster mode) ---
    if corrected_result is not None:
        if cluster_stat not in ("size", "mass"):
            raise ValueError(f"cluster_stat must be 'size' or 'mass', got {cluster_stat!r}.")
        _voxel_thresh_z, _min_cluster_extent = nimare_fwe_thresholds(
            corrected_result, alpha=alpha, cluster_stat=cluster_stat,
            voxel_thresh=voxel_thresh,
        )
    else:
        _voxel_thresh_z, _min_cluster_extent = None, None

    # --- resolve Parcellation object ---
    from ..core.parcellation import Parcellation as _Parcellation
    if isinstance(parcellation, str):
        from ..datasets import fetch_parcellation
        lgr.info(f"Fetching parcellation '{parcellation}' for null map generation.")
        parc_obj = fetch_parcellation(
            parcellation, return_parcellation_only=True, verbose=verbose
        )
    elif isinstance(parcellation, _Parcellation):
        parc_obj = parcellation
    else:
        raise TypeError(
            f"parcellation must be a NiSpace parcellation name (str) or Parcellation "
            f"object, got {type(parcellation)}."
        )

    mni_spaces = [s for s in parc_obj.spaces if "mni" in s.lower()]
    if not mni_spaces:
        raise ValueError(
            "Parcellation has no volumetric/MNI space. "
            "Surface-only parcellations are not supported for NiMARE null maps."
        )
    # Prefer NiMARE's native space (MNI152NLin6Asym) to minimise resampling error.
    nimare_mni = [s for s in mni_spaces if "lin6" in s.lower()]
    parc_space = nimare_mni[0] if nimare_mni else mni_spaces[0]
    parc_img = parc_obj.get_image(parc_space)

    # --- pre-resample parcellation to ALE space (once, avoids per-iteration neuromaps call) ---
    from nilearn.image import resample_img as _nilearn_resample_img
    lgr.info("Pre-resampling parcellation to ALE native space.")
    _parc_resampled = _nilearn_resample_img(
        parc_img,
        target_affine=est.masker.mask_img.affine,
        target_shape=est.masker.mask_img.shape,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    parc_arr_precomp = _parc_resampled.get_fdata().astype(np.float32)
    parc_idc_precomp = np.trim_zeros(np.unique(parc_arr_precomp)).astype(np.float32)
    # bg_values for vol_to_vect_arr: NaN is always excluded automatically.
    # Both modes pass an empty array so that 0-valued voxels are included in
    # each parcel's mean:
    #   binary mode  → parcel value = fraction of voxels in cluster ∈ [0, 1]
    #   continuous   → parcel value = mean ALE stat (ALE=0 inside brain is valid)
    # Using [0.0] in binary mode would exclude non-cluster voxels from the mean,
    # making every parcel 1.0 or NaN (fully binary) instead of fractional.
    bg_values_precomp = np.array([], dtype=np.float64)

    # --- optional observed-map comparison ---
    if observed_map is not None:
        if _voxel_thresh_z is not None:
            # Binary mode: apply the same voxel threshold + cluster extent to the
            # observed z-map, parcellate, then compare to the user's binary Y.
            obs_z_img = result.get_map("z")
            obs_z_arr = obs_z_img.get_fdata()
            binary_arr = (obs_z_arr >= _voxel_thresh_z).astype(np.float32)
            if _min_cluster_extent is not None and _min_cluster_extent > 0:
                labeled, n_comps = nd_label(binary_arr > 0)
                for ci in range(1, n_comps + 1):
                    if (labeled == ci).sum() < _min_cluster_extent:
                        binary_arr[labeled == ci] = 0
            obs_img = nib.Nifti1Image(binary_arr, obs_z_img.affine, obs_z_img.header)
            check_desc = (
                f"binary cluster map (z≥{_voxel_thresh_z:.3f}, "
                f"cluster_stat={cluster_stat}, min_extent={_min_cluster_extent})"
            )
        else:
            obs_img = result.get_map(outcome_map)
            check_desc = f"'{outcome_map}' map"

        # obs_img and observed_map (if NIfTI) are both in NiMARE's native ALE space,
        # so we can call vol_to_vect_arr directly with the pre-resampled parcellation.
        _bg_none = np.array([], dtype=np.float64)
        obs_parc = vol_to_vect_arr(
            obs_img.get_fdata().astype(np.float32),
            parc_arr_precomp, parc_idc_precomp, _bg_none,
        ).astype(float)
        if isinstance(observed_map, nib.spatialimages.SpatialImage):
            obs_user = vol_to_vect_arr(
                observed_map.get_fdata().astype(np.float32),
                parc_arr_precomp, parc_idc_precomp, _bg_none,
            ).astype(float)
        else:
            obs_user = np.asarray(observed_map, dtype=float)
        mask = np.isfinite(obs_parc) & np.isfinite(obs_user)
        if mask.sum() > 1:
            from scipy.stats import pearsonr
            r, _ = pearsonr(obs_parc[mask], obs_user[mask])
            if r < 0.99:
                lgr.warning(
                    f"Parcellated {check_desc} and observed_map have r={r:.3f} (<0.99). "
                    "Verify that the Y map passed to NiSpace was produced with the "
                    "same corrected_result (use get_binary_cluster_map)."
                )
            else:
                lgr.info(f"Observed-map sanity check passed (r={r:.3f}).")

    # --- coordinate sampling ---
    mask_arr = est.masker.mask_img.get_fdata()
    null_ijk = np.vstack(np.where(mask_arr)).T              # (n_mask_voxels, 3)
    n_coords = est.inputs_["coordinates"].shape[0]
    rng = np.random.default_rng(seed)
    rand_idx = rng.integers(null_ijk.shape[0], size=(n_coords, n_perm))
    rand_ijk = null_ijk[rand_idx, :]                        # (n_coords, n_perm, 3)
    rand_xyz = vox2mm(rand_ijk, est.masker.mask_img.affine)  # (n_coords, n_perm, 3)
    iter_xyzs = np.split(rand_xyz, rand_xyz.shape[1], axis=1)  # list of (n_coords, 1, 3)

    iter_df = est.inputs_["coordinates"].copy()

    mode = "binary cluster" if _voxel_thresh_z is not None else "continuous"
    lgr.info(
        f"Generating {n_perm} {mode} null maps from NiMARE {type(est).__name__} "
        f"estimator (n_proc={n_proc})."
    )
    if _voxel_thresh_z is not None:
        lgr.info(
            f"  voxel_thresh_z={_voxel_thresh_z:.3f}, "
            f"min_cluster_voxels={_min_cluster_extent}"
        )

    # In cluster mode with validate=True, _run_one_perm returns (parc_map, max_stat)
    # so we can compare our null-derived FWE threshold to NiMARE's stored threshold.
    # Note: min_cluster_extent is NOT passed to _run_one_perm — the FWE cluster
    # extent filter is applied only to the observed map (by get_binary_cluster_map),
    # not to null maps. Applying it to nulls makes ~95% of them empty by design
    # (that is what FWE alpha=0.05 means). return_cluster_stat still collects the
    # pre-filter max stat per perm so the validation comparison is meaningful.
    _return_cluster_stat = validate and _voxel_thresh_z is not None

    # --- permutation loop (NiMARE-style: generator + tqdm for progress) ---
    raw_results = [
        r for r in tqdm(
            Parallel(return_as="generator", n_jobs=n_proc)(
                delayed(_run_one_perm)(
                    iter_xyzs[i], iter_df, est,
                    parc_arr_precomp, parc_idc_precomp, bg_values_precomp,
                    _voxel_thresh_z, None, cluster_stat,   # None: no FWE extent filter on nulls
                    _return_cluster_stat,
                )
                for i in range(n_perm)
            ),
            total=n_perm,
            disable=not verbose,
            desc="NiMARE null maps",
        )
    ]

    if _return_cluster_stat:
        null_maps_list = [r[0] for r in raw_results]
        max_stats = [r[1] for r in raw_results]
    else:
        null_maps_list = raw_results
        max_stats = None

    # --- stack ---
    null_arr = np.stack(null_maps_list, axis=0).astype(dtype)  # (n_perm, n_parcels)
    label = map_label or outcome_map
    lgr.info(f"Done. null_maps['{label}'] shape: {null_arr.shape}")

    # --- empty-map check ---
    empty_frac = float((null_arr == 0).all(axis=1).mean())
    if empty_frac > 0.5:
        lgr.warning(
            f"{100 * empty_frac:.0f}% of null maps are entirely zero. "
            "Colocalization against all-zero maps is undefined. "
            "Consider switching to continuous mode (corrected_result=None)."
        )
    elif empty_frac > 0.1:
        lgr.warning(
            f"{100 * empty_frac:.0f}% of null maps are entirely zero "
            "(no CDT-surviving cluster in the permutation). "
            "This is expected for datasets with few coordinates."
        )

    # --- validation ---
    if validate:
        _validate_null_maps(
            null_arr, max_stats, result, outcome_map,
            parc_arr_precomp, parc_idc_precomp,
            _voxel_thresh_z, _min_cluster_extent, cluster_stat, alpha,
            observed_map,
        )

    return {label: null_arr}


def nimare_fwe_thresholds(corrected_result, alpha=0.05, cluster_stat="size", voxel_thresh=0.001):
    """Extract voxel z-threshold and cluster extent threshold from a NiMARE FWE-corrected result.

    Reads ``corrected_result.estimator.null_distributions_`` (populated during
    ``FWECorrector.transform()``) to derive the FWE cluster extent threshold.
    The voxel-level CDT z-score is derived from *voxel_thresh* directly —
    NiMARE's ``FWECorrector`` silently ignores any ``voxel_thresh`` passed to
    its constructor, so it must be provided explicitly here.

    Parameters
    ----------
    corrected_result : nimare.results.MetaResult
        The corrected MetaResult returned by ``FWECorrector.transform(result)``.
        The cluster-size null distributions are stored on its estimator
        (``corrected_result.estimator.null_distributions_``).
        Only ``method="montecarlo"`` is supported.
    alpha : float, optional
        FWE significance level used to derive the cluster extent threshold from
        the null distribution. Default: 0.05.
    cluster_stat : {"size", "mass"}, optional
        Which cluster-level null distribution to read. ``"size"`` (default)
        uses the max-cluster-voxel-count distribution; ``"mass"`` uses the
        max-cluster-z-sum distribution. Must match the statistic used to
        produce the observed cluster image. Default: ``"size"``.
    voxel_thresh : float, optional
        Uncorrected voxel-level p-value for the cluster-defining threshold
        (CDT). Must match the threshold actually used during FWE correction.
        NiMARE's default (and the default here) is ``0.001`` (z ≈ 3.09).

    Returns
    -------
    voxel_thresh_z : float
        Z-score corresponding to *voxel_thresh*
        (``scipy.stats.norm.isf(voxel_thresh)``).
        E.g. ``voxel_thresh=0.001`` → z ≈ 3.09.
    min_cluster_extent : int or float or None
        FWE cluster extent threshold: the ``(1 - alpha)`` percentile of the
        max-cluster-stat null distribution. Voxel count (``int``) for
        ``cluster_stat="size"``; z-score sum (``float``) for
        ``cluster_stat="mass"``. ``None`` if the relevant distribution is not
        found in ``corrected_result.estimator.null_distributions_``.

    Examples
    --------
    >>> fwe = FWECorrector(method="montecarlo", n_iters=1000)
    >>> corrected = fwe.transform(result)
    >>> z_thresh, min_ext = nimare_fwe_thresholds(corrected, cluster_stat="size", voxel_thresh=0.001)
    >>> binary_img = get_binary_cluster_map(corrected, cluster_stat="size")
    >>> null_maps = null_maps_from_nimare(result, nsp._parc,
    ...                                   corrected_result=corrected, voxel_thresh=0.001)
    """
    from scipy.stats import norm

    voxel_thresh_z = float(norm.isf(voxel_thresh))
    lgr.info(f"Voxel z-threshold (p<{voxel_thresh}): {voxel_thresh_z:.4f}")

    null_dists = getattr(
        getattr(corrected_result, "estimator", None), "null_distributions_", {}
    )
    cluster_keys = [
        k for k in null_dists
        if f"desc-{cluster_stat}" in k and "values" in k
    ]
    min_cluster_extent = None
    if cluster_keys:
        null_vals = np.asarray(null_dists[cluster_keys[0]])
        raw = np.percentile(null_vals, 100 * (1 - alpha))
        min_cluster_extent = int(np.ceil(raw)) if cluster_stat == "size" else float(raw)
        lgr.info(
            f"Cluster {cluster_stat} threshold (FWE alpha={alpha}): {min_cluster_extent}."
        )
    else:
        lgr.warning(
            f"No cluster-{cluster_stat} null distribution found in "
            "corrected_result.estimator.null_distributions_. "
            "min_cluster_extent will be None."
        )

    return voxel_thresh_z, min_cluster_extent


def get_binary_cluster_map(corrected_result, cluster_stat="size", alpha=0.05, map_key=None):
    """Extract a binary (0/1) cluster mask from a NiMARE FWE-corrected MetaResult.

    NiMARE's cluster-corrected logp maps (e.g.
    ``logp_desc-size_level-cluster_corr-FWE_method-montecarlo``) store a
    cluster-level ``-log10(p_FWE)`` value at every voxel that survived the
    voxel-level CDT, **including non-significant clusters**. Voxels in
    non-significant clusters have ``0 < logp < -log10(alpha)``. This function
    binarizes correctly by thresholding at ``logp >= -log10(alpha)``.

    Pass the returned image as ``y=`` to :class:`~nispace.NiSpace` and use
    :func:`null_maps_from_nimare` with the same *corrected_result*, *cluster_stat*,
    and *alpha* to generate matching binary null maps for permutation testing.

    Parameters
    ----------
    corrected_result : nimare.results.MetaResult
        Result of ``FWECorrector.transform(result)``.
    cluster_stat : {"size", "mass"}, optional
        Which cluster-corrected map to binarize. ``"size"`` (default) uses
        ``logp_desc-size_level-cluster_corr-*``; ``"mass"`` uses
        ``logp_desc-mass_level-cluster_corr-*``. Must match the *cluster_stat*
        passed to :func:`null_maps_from_nimare`.
    alpha : float, optional
        FWE significance threshold. Voxels with ``logp >= -log10(alpha)`` are
        kept. Default: 0.05. Must match the *alpha* passed to
        :func:`null_maps_from_nimare`.
    map_key : str or None, optional
        Override: explicit key in ``corrected_result.maps`` to binarize,
        bypassing the *cluster_stat* search. Pass when NiMARE uses a
        non-standard naming scheme.

    Returns
    -------
    nibabel.Nifti1Image
        Binary NIfTI image (float32, values 0 or 1) in NiMARE's native space
        (MNI152NLin6Asym at 2 mm).

    Examples
    --------
    >>> fwe = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=1000)
    >>> corrected = fwe.transform(result)
    >>> binary_img = get_binary_cluster_map(corrected, alpha=0.05)
    >>> nsp = NiSpace(x=ref_maps, y=binary_img, y_labels=["pain_cluster"],
    ...               parcellation="Schaefer200")
    >>> nsp.fit(); nsp.colocalize()
    >>> null_maps = null_maps_from_nimare(
    ...     result, "Schaefer200", corrector=fwe, alpha=0.05,
    ...     map_label="pain_cluster", observed_map=binary_img,
    ... )
    >>> nsp.permute("maps", maps_which="Y", maps_nulls=null_maps)
    """
    if map_key is None:
        available = [k for k, v in corrected_result.maps.items() if v is not None]
        candidates = [
            k for k in available
            if f"logp_desc-{cluster_stat}" in k and "cluster_corr" in k
        ]
        if not candidates and cluster_stat == "size":
            candidates = [k for k in available if "logp_desc-mass" in k and "cluster_corr" in k]
        if not candidates:
            raise ValueError(
                "Could not find a cluster-corrected logp map in corrected_result.maps. "
                f"Available keys: {available}. Pass map_key= explicitly."
            )
        if len(candidates) > 1:
            lgr.warning(
                f"Multiple cluster logp maps found: {candidates}. Using '{candidates[0]}'. "
                "Pass map_key= to select explicitly."
            )
        map_key = candidates[0]

    img = corrected_result.get_map(map_key)
    arr = img.get_fdata()
    logp_thresh = -np.log10(alpha)
    binary = (arr >= logp_thresh).astype(np.float32)
    lgr.info(
        f"Binary cluster map from '{map_key}' "
        f"(logp >= {logp_thresh:.4f}, alpha={alpha}): "
        f"{int(binary.sum())} significant voxels."
    )
    return nib.Nifti1Image(binary, img.affine, img.header)
