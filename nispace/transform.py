"""Transforms between MNI volume spaces and fsLR / fsaverage surface spaces.

All transforms run in pure Python (no Connectome Workbench) on files NiSpace hosts and
hash-checks (EasyReg fields, template spheres, vertex areas, medial-wall masks, registration
fusion meshes). :class:`SpaceTransformer` fits a transform once and applies it to any number of
maps; :func:`mni_to_mni`, :func:`surf_to_surf`, :func:`mni_to_surf` and :func:`transform` are
convenience wrappers.
"""

import logging
from pathlib import Path
import warnings

import numpy as np
import nibabel as nib

from .io import read_json
from .utils.utils import set_log
from .utils.utils_datasets import get_file
from .utils.utils_transforms import (adap_bary_area_weights, resample_popular, sample_field,
                                     volume_sampling_matrix)
from .datasets import _resolve_nispace_data_dir, fetch_template

lgr = logging.getLogger("nispace")

_DATALIB = Path(__file__).parent / "datalib"

_KNOWN_SPACES = [
    "MNI152NLin2009cAsym",
    "MNI152NLin2009cSym",
    "MNI152NLin6Asym",
    "MNI152NLin6Sym",
    "MNI152Lin",
    "MNI305",
    "MNIColin27",
    "CAT12IXI555",
    "CVSavg35inMNI152",
]

# Exact-match aliases resolved before pattern matching. Every alias contains "MNI", so that space names
# are recognised as volumetric wherever NiSpace distinguishes volume from surface spaces.
_SPACE_ALIASES = {
    # SPM space depends on the normalisation method, not the version:
    # templates/T1.nii (old normalise, SPM5/SPM8 TPMs) -> MNI152Lin; SPM12 segment/normalise
    # (tpm/TPM.nii) -> TPM frame, represented by CAT12IXI555. "MNI-SPM"/"MNI-SPM12" are ambiguous, see below.
    "MNI-SPM-T1":       "MNI152Lin",
    "MNI-SPM-TPM":      "CAT12IXI555",
    "MNI-SPM5":         "MNI152Lin",
    "MNI-SPM8":         "MNI152Lin",
    "MNI-FSL":          "MNI152NLin6Asym",
    "MNI-HCP":          "MNI152NLin6Asym",
    # ICBM 2009a/b templates: same template space as 2009c, no separate fields (noted once when used)
    "MNI152NLin2009a":      "MNI152NLin2009cAsym",
    "MNI152NLin2009b":      "MNI152NLin2009cAsym",
    "MNI152NLin2009aAsym":  "MNI152NLin2009cAsym",
    "MNI152NLin2009bAsym":  "MNI152NLin2009cAsym",
    "MNI152NLin2009Asym":   "MNI152NLin2009cAsym",   # neuromaps name
    "MNI152NLin2009aSym":   "MNI152NLin2009cSym",
    "MNI152NLin2009bSym":   "MNI152NLin2009cSym",
    # CAT12 >= 12.8 uses MNI152NLin2009cAsym templates; the old IXI555 template
    # (CAT12 <= 12.7) must be requested explicitly as CAT12IXI555
    "MNI-CAT":          "MNI152NLin2009cAsym",
    "MNI-CAT12":        "MNI152NLin2009cAsym",
}

# Names that are refused with an explanation instead of being resolved (with and without "MNI-")
_AMBIGUOUS_ALIASES = {
    "mni-spm":   "SPM output space depends on the normalisation method",
    "mni-spm12": "SPM12 output space depends on the normalisation method",
    "spm":       "SPM output space depends on the normalisation method",
    "spm12":     "SPM12 output space depends on the normalisation method",
}
_AMBIGUOUS_HINT = (
    "Use 'MNI-SPM-TPM' (SPM12 segment/normalise with tpm/TPM.nii -> CAT12IXI555) or "
    "'MNI-SPM-T1' (old normalise with templates/T1.nii, SPM5/SPM8 -> MNI152Lin)."
)


# Generic names for "some MNI space, template unspecified": resolve to None without a warning
_GENERIC_MNI_NAMES = {"mni", "mni152", "mnioriginal"}

_CIVET_MSG = ("The CIVET surface space is not supported by NiSpace. Transform the data to fsLR or fsaverage "
              "first (e.g. with neuromaps).")

_LOGGED_ONCE = set()


def _info_once(message):
    """Log `message` at INFO level once per Python session."""
    if message not in _LOGGED_ONCE:
        _LOGGED_ONCE.add(message)
        lgr.info(message)


def _is_mni_space_name(space):
    """True for MNI volume space names: containing "mni" (all aliases and accepted partial names do) or a
    canonical MNI space name (CAT12IXI555 is the only one without "mni")."""
    return isinstance(space, str) and (
        "mni" in space.lower() or space.lower() in {k.lower() for k in _KNOWN_SPACES}
    )

_AVAILABLE_RES = [1, 2, 3, 4]

# String aliases for nitransforms interpolation order
_ORDER_ALIASES = {"nearest": 0, "linear": 1, "cubic": 3, "continuous": 3}

_EASYREG_DOI = "10.1038/s41598-023-33781-0"

# Surface spaces (case-insensitive names) and their densities
_SURF_SPACES = {"fslr": "fsLR", "fsaverage": "fsaverage"}
_SURF_DENSITIES = {"fsLR": ["4k", "8k", "32k", "164k"], "fsaverage": ["3k", "10k", "41k", "164k"]}
_SURF_DEFAULT_DEN = {"fsLR": "32k", "fsaverage": "41k"}
_N_VERT_TO_DEN = {4002: "4k", 7842: "8k", 32492: "32k", 2562: "3k", 10242: "10k", 40962: "41k",
                  163842: "164k"}

# Registration fusion (Wu et al. 2018) vertex positions are defined in this MNI space
_REGFUSION_SPACE = "MNI152NLin6Asym"
_REGFUSION_DOI = "10.1002/hbm.24213"

# Fitted surface weights / composite vertex coordinates, reused across SpaceTransformer instances
_OPERATOR_CACHE = {}
_OPERATOR_CACHE_MAX = 32


def _resolve_space(space, strict=True):
    """Resolve a (possibly partial or aliased) space name to a canonical name.

    With ``strict=False``, return None (without logging) instead of raising when `space` does not
    identify exactly one MNI space, e.g. the generic "MNI152".
    """
    def _fail(message):
        if strict:
            lgr.critical_raise(message, ValueError)
        return None

    # 1. Exact canonical name (case-insensitive)
    s = space.lower()
    canonical = {k.lower(): k for k in _KNOWN_SPACES}
    if s in canonical:
        return canonical[s]
    # 2. Exact alias: exact case first (neuromaps' 'MNI152NLin2009Asym' and templateflow's
    # 'MNI152NLin2009aSym' differ only in case), then case-insensitive
    if space in _SPACE_ALIASES:
        target = _SPACE_ALIASES[space]
    else:
        if s in _AMBIGUOUS_ALIASES:
            return _fail(f"'{space}' is ambiguous: {_AMBIGUOUS_ALIASES[s]}. {_AMBIGUOUS_HINT}")
        if s == "mni152nlin2009asym":
            return _fail(f"'{space}' is ambiguous: write 'MNI152NLin2009Asym' (asymmetric, neuromaps) or "
                         "'MNI152NLin2009aSym' (symmetric, templateflow) with exact capitalisation, or use "
                         "'MNI152NLin2009cAsym' / 'MNI152NLin2009cSym'.")
        target = {k.lower(): v for k, v in _SPACE_ALIASES.items()}.get(s)
    if target is not None:
        if s.startswith(("mni152nlin2009a", "mni152nlin2009b")):
            _info_once(f"MNI space '{space}' is treated as '{target}' (same ICBM 2009 template space).")
        return target
    # Partial names must contain "MNI" (like all aliases), so that every accepted name is recognised as
    # volumetric by `_is_mni_space_name`; e.g. "Colin" or "cat12" are not matched
    if "mni" not in s:
        return _fail(f"Unknown MNI space '{space}'. Partial space names must contain 'MNI' "
                     f"(e.g. 'MNIColin'). Supported spaces: {_KNOWN_SPACES}. "
                     f"Known aliases: {list(_SPACE_ALIASES)}.")
    # 3. Case-insensitive prefix matching
    prefix_matches = [k for k in _KNOWN_SPACES if k.lower().startswith(s)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        return _fail(f"Space name '{space}' is ambiguous — matches: {prefix_matches}. "
                     f"Please be more specific.")
    # 4. Case-insensitive substring matching
    substr_matches = [k for k in _KNOWN_SPACES if s in k.lower()]
    if len(substr_matches) == 1:
        return substr_matches[0]
    if len(substr_matches) > 1:
        return _fail(f"Space name '{space}' is ambiguous — matches: {substr_matches}. "
                     f"Please be more specific.")
    return _fail(f"Unknown MNI space '{space}'. Supported spaces: {_KNOWN_SPACES}. "
                 f"Known aliases: {list(_SPACE_ALIASES)}.")


def _resolve_field(mni_from, mni_to, nispace_data_dir, hash_check=True):
    """Return the local path to the EasyReg field for mni_from → mni_to."""
    transform_json = read_json(_DATALIB / "transform.json")
    if mni_to in transform_json and mni_from in transform_json[mni_to]:
        # fwd_field: on ref (mni_to) grid, stores flo (mni_from) coords — correct for pull resampling
        direction = "forward"
        entry = transform_json[mni_to][mni_from][direction]
    elif mni_from in transform_json and mni_to in transform_json[mni_from]:
        # bak_field: on flo (mni_to here) grid, stores ref (mni_from) coords — correct for pull into flo
        direction = "backward"
        entry = transform_json[mni_from][mni_to][direction]
    else:
        all_spaces = sorted(set(transform_json) | {s for v in transform_json.values() for s in v})
        lgr.critical_raise(
            f"No transform available between '{mni_from}' and '{mni_to}'. "
            f"Supported spaces: {all_spaces}.",
            ValueError,
        )
    lgr.info(
        f"Using {direction} warp field estimated with EasyReg "
        f"(DOI: {_EASYREG_DOI})."
    )
    nispace_data_dir = _resolve_nispace_data_dir(nispace_data_dir)
    return get_file(
        Path(nispace_data_dir) / entry["remote"],
        **entry,
        hash_check=hash_check,
    )


def mni_to_mni(img, mni_from, mni_to, order=3, res=None,
               nispace_data_dir=None, hash_check=True, verbose=True):
    """Transform a NIfTI image between MNI template spaces using EasyReg fields.

    Deformation fields were estimated with EasyReg :cite:`iglesias2023` (Freesurfer), which
    builds on the SynthMorph :cite:`hoffmann2022` registration and SynthSeg :cite:`billot2023` segmentation
    networks, between each supported source space and both hub spaces
    (``MNI152NLin2009cAsym`` and ``MNI152NLin6Asym``). Transforms between two
    non-hub spaces (e.g. ``MNI305`` → ``MNIColin27``) are not directly
    supported.

    Parameters
    ----------
    img : str, Path, or nibabel.SpatialImage
        Input image to resample.
    mni_from : str
        Source MNI space. Accepts canonical names, unambiguous prefixes or
        substrings (case-insensitive), and named aliases:

        .. list-table::
           :header-rows: 1

           * - Canonical name
             - Notes
           * - ``MNI152NLin2009cAsym``
             - fMRIPrep / templateflow default
           * - ``MNI152NLin6Asym``
             - FSL / HCP standard
           * - ``MNI152Lin``
             - Linearly registered ICBM 152
           * - ``MNI305``
             - Original MNI template
           * - ``MNIColin27``
             - Colin Holmes single-subject average
           * - ``MNI152NLin2009cSym``
             -
           * - ``MNI152NLin6Sym``
             -
           * - ``CAT12IXI555``
             - CAT12 (≤ 12.7) IXI555 Geodesic Shooting T1w average; stand-in
               for the SPM12 TPM frame (SPM12 ships no T1w template)
           * - ``CVSavg35inMNI152``
             - FreeSurfer ``cvs_avg35_inMNI152`` subject

        Named aliases (all start with ``MNI-``): ``MNI-SPM-TPM`` →
        ``CAT12IXI555`` (SPM12 segment/normalise with ``tpm/TPM.nii``);
        ``MNI-SPM-T1`` / ``MNI-SPM5`` / ``MNI-SPM8`` → ``MNI152Lin`` (old
        normalise with ``templates/T1.nii``, SPM5/SPM8 tissue priors);
        ``MNI-FSL`` / ``MNI-HCP`` → ``MNI152NLin6Asym``;
        ``MNI152NLin2009a`` / ``MNI152NLin2009b`` / ``MNI152NLin2009aAsym`` /
        ``MNI152NLin2009bAsym`` / ``MNI152NLin2009Asym`` → ``MNI152NLin2009cAsym``
        and ``MNI152NLin2009aSym`` / ``MNI152NLin2009bSym`` →
        ``MNI152NLin2009cSym`` (same ICBM 2009 template space; ``MNI152NLin2009Asym``
        and ``MNI152NLin2009aSym`` differ only in case and must be written exactly);
        ``MNI-CAT`` / ``MNI-CAT12`` → ``MNI152NLin2009cAsym`` (CAT12 ≥ 12.8
        templates; use ``CAT12IXI555`` for CAT12 ≤ 12.7). ``MNI-SPM`` and
        ``MNI-SPM12`` (also without ``MNI-``) raise an error because their
        space depends on the normalisation method. Aliases are
        case-insensitive.

        Unambiguous partial names containing ``"MNI"`` are also accepted,
        e.g. ``"MNIColin"`` → ``MNIColin27``, ``"MNI152Li"`` →
        ``MNI152Lin``, ``"CVSavg35inMNI"`` → ``CVSavg35inMNI152``.
        ``CAT12IXI555`` (no ``"MNI"`` in its name) is only matched exactly.
    mni_to : str
        Target MNI space (same options and aliases as ``mni_from``).
    order : int or str
        Interpolation order for nitransforms. Accepts integers ``0``–``5``
        or the strings ``"nearest"`` (→ 0), ``"linear"`` (→ 1), or
        ``"cubic"``/``"continuous"`` (→ 3). Default is ``3`` (cubic spline), 
        which gives the best quality for continuous data. Use ``0`` / ``"nearest"`` 
        for label or parcellation images to avoid interpolation artefacts.

        Comparison with other tools: neuromaps ``"linear"`` = this
        ``order=1``; nilearn ``"continuous"`` ≈ this ``order=3``.
    res : int, str, or None
        Output voxel size in mm. Accepted forms: ``1``, ``2``, ``3``, ``4``
        or ``"1mm"``, ``"2mm"``, ``"3mm"``, ``"4mm"``. If ``None`` (default),
        the nearest supported resolution to the input voxel size is used.
        A ``UserWarning`` is raised when ``res`` is finer than the input
        voxel size (the output grid is denser but effective resolution is
        still limited by the input).
    nispace_data_dir : str or Path, optional
        Override for the NiSpace data directory. Defaults to the
        ``NISPACE_DATA_DIR`` environment variable or the standard download
        cache.
    hash_check : bool
        Verify the downloaded field file against the expected hash.
        Default ``True``. Set to ``False`` to skip integrity checks (e.g.
        when working with custom or locally estimated fields).
    verbose : bool
        Emit log messages. Default ``True``.

    Returns
    -------
    nibabel.Nifti1Image
        Input image resampled into ``mni_to`` space at the requested
        resolution.

    Notes
    -----
    **EasyReg field convention — why the forward field is used for resampling**

    EasyReg produces two coordinate-map fields (each voxel stores absolute
    RAS coordinates, not displacements):

    * ``fwd_field``: defined on the **reference (target) grid**, stores the
      corresponding **floating (source) RAS coordinates**. This is the field
      needed for pull resampling: "for each output voxel, sample from the
      input at these coordinates."
    * ``bak_field``: defined on the **floating (source) grid**, stores the
      corresponding **reference (target) RAS coordinates**. This is a forward
      map used by EasyReg's own ``mri_easywarp`` tool (push resampling), but
      it is not directly usable for pull resampling.

    NiSpace uses nitransforms for pull resampling, so it loads the
    ``fwd_field`` — despite "forward" sounding counter-intuitive for a
    source→target transform. Using ``bak_field`` instead produces a
    systematic ~1–2 mm shift (visible as a slight left-hemisphere offset
    in standard radiological view) because the field is defined on the wrong
    grid.

    References
    ----------
    :cite:`iglesias2023` (EasyReg); :cite:`hoffmann2022` (SynthMorph);
    :cite:`billot2023` (SynthSeg).
    """
    _resolve_nispace_data_dir(nispace_data_dir)  # deprecated parameter: sets NISPACE_DATA_DIR
    return SpaceTransformer(
        mni_from, mni_to, res=res, method=order, hash_check=hash_check, verbose=verbose,
    ).fit().transform(img)


def compute_transform_displacement(mni_from, mni_to,
                                   nispace_data_dir=None, hash_check=True,
                                   verbose=True):
    """Compute a voxel-wise displacement magnitude map for a MNI space transform.

    For each voxel in the ``fwd_field`` deformation field (estimated with
    EasyReg :cite:`iglesias2023`, built on SynthMorph :cite:`hoffmann2022` and SynthSeg :cite:`billot2023`; the field used
    for pull resampling — see Notes in :func:`mni_to_mni`), computes the
    Euclidean distance in mm between the voxel's RAS position in ``mni_to``
    space and the stored ``mni_from`` RAS coordinates. The result visualises
    how much each location is displaced between the two spaces and is useful
    for registration quality control.

    Parameters
    ----------
    mni_from : str
        Source MNI space. Accepts the same canonical names, partial names,
        and aliases as :func:`mni_to_mni`.
    mni_to : str
        Target MNI space. Same options as ``mni_from``.
    nispace_data_dir : str or Path, optional
        Override for the NiSpace data directory. Defaults to the
        ``NISPACE_DATA_DIR`` environment variable or the standard download
        cache.
    hash_check : bool
        Verify the downloaded field file against the expected hash.
        Default ``True``.
    verbose : bool
        Emit log messages. Default ``True``.

    Returns
    -------
    nibabel.Nifti1Image
        Float32 scalar image in ``mni_to`` space at 1 mm isotropic resolution
        (the native resolution of the deformation field). Each voxel value is
        the displacement magnitude in mm. Near-zero values indicate little
        warping between the two spaces; large values indicate strong local
        deformation. Expected range for closely related MNI variants is
        roughly 0–6 mm; values > 20 mm outside the brain mask are normal
        field extrapolation artefacts.

    Notes
    -----
    The displacement at voxel **v** is computed as::

        ||field[v] - ras(v)||₂

    where ``field[v]`` is the stored ``mni_from`` RAS coordinate and
    ``ras(v)`` is the ``mni_to`` RAS coordinate of voxel **v** derived from
    the field affine. The output image is in ``mni_to`` space because the
    ``fwd_field`` is defined on the reference (target) grid.

    References
    ----------
    :cite:`iglesias2023` (EasyReg); :cite:`hoffmann2022` (SynthMorph);
    :cite:`billot2023` (SynthSeg).
    """
    verbose = set_log(lgr, verbose)

    mni_from = _resolve_space(mni_from)
    mni_to   = _resolve_space(mni_to)

    lgr.info(f"Computing displacement map '{mni_from}' → '{mni_to}'.")
    field_path = _resolve_field(mni_from, mni_to, nispace_data_dir, hash_check=hash_check)

    field_img = nib.load(str(field_path))
    field_data = field_img.get_fdata()          # (x, y, z, 3) absolute RAS coords
    affine = field_img.affine
    sx, sy, sz = field_data.shape[:3]

    # RAS coordinates of every voxel in the field grid
    i, j, k = np.mgrid[0:sx, 0:sy, 0:sz]
    vox = np.stack([i, j, k, np.ones_like(i)], axis=-1)          # (x,y,z,4)
    ras = (affine @ vox.reshape(-1, 4).T).T[:, :3]               # (N,3)
    ras = ras.reshape(sx, sy, sz, 3)

    magnitude = np.linalg.norm(field_data - ras, axis=-1).astype(np.float32)
    return nib.Nifti1Image(magnitude, affine)


# ==================================================================================================
# SPACE / DATA HELPERS
# ==================================================================================================

_SURF_TO_MNI_MSG = (
    "Surface-to-MNI transforms are not implemented yet. NiSpace will soon provide them based on "
    "registration fusion (Wu et al., 2018, doi:10.1002/hbm.24213), mapped to all supported MNI spaces "
    "via the EasyReg fields."
)


def _resolve_any_space(space):
    """Return ('surface', name) for fsLR / fsaverage (case-insensitive), else ('mni', canonical name)."""
    if not isinstance(space, str):
        lgr.critical_raise(f"Space must be a string, got {type(space).__name__}.", TypeError)
    if space.lower() in _SURF_SPACES:
        return "surface", _SURF_SPACES[space.lower()]
    if space.lower() == "civet":
        lgr.critical_raise(_CIVET_MSG, ValueError)
    return "mni", _resolve_space(space)


def _check_density(space, den):
    if den not in _SURF_DENSITIES[space]:
        lgr.critical_raise(f"Density '{den}' not available for {space}. "
                           f"Choose from {_SURF_DENSITIES[space]}.", ValueError)
    return den


def _density_of(n_vert, space):
    den = _N_VERT_TO_DEN.get(int(n_vert))
    if den is None or den not in _SURF_DENSITIES[space]:
        lgr.critical_raise(f"{n_vert} vertices per hemisphere do not match a standard {space} density "
                           f"({_SURF_DENSITIES[space]}).", ValueError)
    return den


def _resolve_hemi(hemi):
    if hemi is None:
        return ("L", "R")
    hemi = (hemi,) if isinstance(hemi, str) else tuple(hemi)
    if hemi not in (("L",), ("R",), ("L", "R")):
        lgr.critical_raise(f"hemi={hemi!r} not understood. Use 'L', 'R', or None (both).", ValueError)
    return hemi


def _resolve_order(method, default=3):
    """Interpolation order for volume-to-volume resampling (int 0-5 or a string alias)."""
    if method is None:
        return default
    if isinstance(method, str):
        if method not in _ORDER_ALIASES:
            lgr.critical_raise(f"order/method={method!r} not recognised. "
                               f"Use one of {list(_ORDER_ALIASES)} or an integer 0–5.", ValueError)
        return _ORDER_ALIASES[method]
    return int(method)


def _resolve_surf_method(method):
    method = "linear" if method is None else method
    if method not in ("linear", "nearest"):
        lgr.critical_raise(f"method={method!r} not supported for surface targets. "
                           "Use 'linear' (continuous data) or 'nearest' (labels / parcellations).", ValueError)
    return method


def _gii_arrays(path):
    return [d.data for d in nib.load(str(path)).darrays]


def _gifti_data(img):
    """(n_maps, n_vert) array and label table (or None) of a GIFTI image or path."""
    img = nib.load(str(img)) if isinstance(img, (str, Path)) else img
    arr = np.vstack([np.asarray(d.data).ravel() for d in img.darrays])
    lt = img.labeltable if img.labeltable is not None and len(img.labeltable.labels) > 0 else None
    return arr, lt


def _split_surface(data, hemis):
    """Surface input -> ({hemi: (n_maps, n_vert)}, output format, {hemi: label table or None}).

    Accepted: an (L, R) tuple of GIFTI images / paths / arrays; a single image / path / array with one
    `hemi`; or a numpy array (n_vert_total,) or (n_maps, n_vert_total) with hemispheres concatenated L, R.
    """
    if isinstance(data, np.ndarray):
        arr = np.atleast_2d(data)
        if len(hemis) == 1:
            return {hemis[0]: arr}, ("array", data.ndim), {hemis[0]: None}
        if arr.shape[1] % 2:
            lgr.critical_raise(f"Array with {arr.shape[1]} vertices cannot be split into two hemispheres.",
                               ValueError)
        n = arr.shape[1] // 2
        return {"L": arr[:, :n], "R": arr[:, n:]}, ("array", data.ndim), {"L": None, "R": None}
    imgs = tuple(data) if isinstance(data, tuple) else (data,)
    if len(imgs) != len(hemis):
        lgr.critical_raise(f"Got {len(imgs)} surface image(s) for hemi={hemis}. Pass an (L, R) tuple, or a "
                           "single hemisphere together with hemi='L' / 'R'.", ValueError)
    arrs, lts = {}, {}
    for h, img in zip(hemis, imgs):
        if isinstance(img, np.ndarray):
            arrs[h], lts[h] = np.atleast_2d(img), None
        else:
            arrs[h], lts[h] = _gifti_data(img)
    return arrs, ("gifti", len(imgs)), lts


def _is_hemi_pair_list(data, hemis):
    """True for a 2-element list of single-hemisphere GIFTI images / paths with both hemispheres requested.

    One map needs both hemispheres, so such a list can only be one (L, R) map, not two maps (neuromaps
    accepts this form too).
    """
    return (isinstance(data, list) and len(data) == 2 and tuple(hemis) == ("L", "R")
            and all(isinstance(d, (nib.GiftiImage, str, Path)) for d in data))


def _label_table(arr):
    """Minimal GIFTI label table for the integer values in `arr` (key 0 = unlabelled, transparent)."""
    lt = nib.gifti.GiftiLabelTable()
    for k in np.unique(arr):
        lab = nib.gifti.GiftiLabel(key=int(k), red=0.5, green=0.5, blue=0.5, alpha=float(k != 0))
        lab.label = "???" if k == 0 else str(int(k))
        lt.labels.append(lab)
    return lt


def _to_gifti(arr, labeltable=None):
    """GIFTI image with one data array per row of `arr` (labels if a label table is given)."""
    if labeltable is not None:
        darrays = [nib.gifti.GiftiDataArray(row.astype(np.int32), intent="NIFTI_INTENT_LABEL",
                                            datatype="NIFTI_TYPE_INT32") for row in arr]
        img = nib.GiftiImage(darrays=darrays)
        img.labeltable = labeltable
        return img
    darrays = [nib.gifti.GiftiDataArray(row.astype(np.float32), intent="NIFTI_INTENT_SHAPE",
                                        datatype="NIFTI_TYPE_FLOAT32") for row in arr]
    return nib.GiftiImage(darrays=darrays)


def _merge_surface(out, hemis, fmt, lts=None):
    """Inverse of `_split_surface` for transformed {hemi: (n_maps, n_vert)} arrays."""
    if fmt[0] == "array":
        arr = np.concatenate([out[h] for h in hemis], axis=1)
        return arr.ravel() if fmt[1] == 1 else arr
    imgs = tuple(_to_gifti(out[h], (lts or {}).get(h)) for h in hemis)
    return imgs[0] if len(imgs) == 1 else imgs


def _cached(key, build):
    if key not in _OPERATOR_CACHE:
        if len(_OPERATOR_CACHE) >= _OPERATOR_CACHE_MAX:
            _OPERATOR_CACHE.pop(next(iter(_OPERATOR_CACHE)))
        _OPERATOR_CACHE[key] = build()
    return _OPERATOR_CACHE[key]


def _surface_weights(space_from, den_from, space_to, den_to, hemi, hash_check=True):
    """(weights (n_to, n_from), target non-medial-wall mask) for one hemisphere, cached.

    Workbench-equivalent ``ADAP_BARY_AREA`` resampling with the source medial wall as ROI; fsLR <-> fsaverage
    uses the fsLR sphere registered to fsaverage (``desc-sphere_space-fsaverage``).
    """
    def build():
        cross = space_from != space_to
        tpl = lambda space, den, desc: fetch_template(space, res=den, desc=desc, hemi=hemi,
                                                      check_file_hash=hash_check, verbose=False)
        sph = lambda space, den: "sphere_space-fsaverage" if (cross and space == "fsLR") else "sphere"
        cur = _gii_arrays(tpl(space_from, den_from, sph(space_from, den_from)))[:2]
        new = _gii_arrays(tpl(space_to, den_to, sph(space_to, den_to)))[:2]
        weights = adap_bary_area_weights(
            cur, new,
            cur_area=_gii_arrays(tpl(space_from, den_from, "vaavg"))[0],
            new_area=_gii_arrays(tpl(space_to, den_to, "vaavg"))[0],
            cur_roi=_gii_arrays(tpl(space_from, den_from, "medial"))[0],
        )
        return weights, _gii_arrays(tpl(space_to, den_to, "medial"))[0] > 0
    return _cached(("surf", space_from, den_from, space_to, den_to, hemi, hash_check), build)


def _surface_coords_in_mni(mni_from, space_to, den_to, hemi, hash_check=True):
    """Positions of the `space_to` vertices in `mni_from` (n_vert, 3), cached.

    Registration-fusion vertex positions are defined in MNI152NLin6Asym. For any other MNI space they are
    mapped a priori through the EasyReg field from `mni_from` to MNI152NLin6Asym (defined on the
    MNI152NLin6Asym grid, storing `mni_from` coordinates), so data are interpolated only once.
    """
    def build():
        mesh = fetch_template(space_to, res=den_to, desc=f"regfusion_space-{_REGFUSION_SPACE}", hemi=hemi,
                              check_file_hash=hash_check, verbose=False)
        coords = _gii_arrays(mesh)[0].astype(np.float64)
        if mni_from != _REGFUSION_SPACE:
            field = _resolve_field(mni_from, _REGFUSION_SPACE, None, hash_check=hash_check)
            coords = sample_field(nib.load(str(field)), coords)
        return coords
    return _cached(("regfusion", mni_from, space_to, den_to, hemi, hash_check), build)


# ==================================================================================================
# SPACE TRANSFORMER
# ==================================================================================================

class SpaceTransformer:
    """Transform maps between MNI volume spaces and fsLR / fsaverage surface spaces.

    Fit once, apply to any number of maps: :meth:`fit` builds everything that does not depend on the
    data (the EasyReg field for MNI → MNI, the sparse surface resampling weights for surface →
    surface, the vertex positions in the source MNI space for MNI → surface), :meth:`transform` applies
    it. For MNI → surface, the sampling matrix for an input grid is built on the first map with that
    grid and reused for all further maps on it.

    Supported pairs:

    - **MNI → MNI**: EasyReg deformation fields :cite:`iglesias2023`, see :func:`mni_to_mni`.
    - **surface → surface** (fsLR / fsaverage, any standard density): equivalent to Connectome
      Workbench ``-metric-resample`` / ``-label-resample`` with ``ADAP_BARY_AREA``, the source medial
      wall as ROI and the target medial wall set to 0 (as in neuromaps :cite:`markello2022`), in pure
      Python.
    - **MNI → surface**: registration fusion :cite:`wu2018` vertex positions (defined in
      MNI152NLin6Asym), composed a priori with the EasyReg field for any other MNI source space, so the
      data are interpolated only once. As in neuromaps, all vertices (including the medial wall) are
      sampled.
    - **surface → MNI**: not implemented yet (raises ``NotImplementedError``).

    Parameters
    ----------
    space_from, space_to : str
        Source and target space: ``"fsLR"`` or ``"fsaverage"`` (case-insensitive), or any MNI space name
        or alias accepted by :func:`mni_to_mni`.
    den_from : str, optional
        Density of surface source data (e.g. ``"32k"``). Inferred from the data passed to :meth:`fit`
        (or :meth:`fit_transform`) if not given.
    den_to : str, optional
        Density of a surface target. Default: ``"32k"`` (fsLR) / ``"41k"`` (fsaverage).
    res : int or str, optional
        Output resolution of an MNI target (MNI → MNI only), see :func:`mni_to_mni`. Default: nearest to
        the input voxel size.
    hemi : {'L', 'R'} or None, optional
        Hemisphere(s) of surface data (source or target). Default: both.
    method : str or int, optional
        Interpolation. Surface targets: ``"linear"`` (default) or ``"nearest"`` (labels / parcellations;
        surface → surface uses the label with the largest summed weight). MNI → MNI: interpolation order
        as in :func:`mni_to_mni` (default ``3``).
    hash_check : bool
        Verify NiSpace-hosted fields and templates against their known hashes. Default ``True``.
    verbose : bool
        Emit log messages. Default ``True``.

    Notes
    -----
    **Performance:** fitting takes a moment, applying is fast — fit once, then transform many maps.
    Rough timings on a laptop:

    - surface → surface: :meth:`fit` ~1–3 s (both hemispheres, e.g. ~2.5 s for fsLR 32k → fsaverage
      41k; ~6 s with a 164k mesh), then a few milliseconds per map.
    - MNI → surface: :meth:`fit` well below 1 s (~0.5 s when an EasyReg field is composed in); the first
      map on a new grid adds ~50 ms, every further map takes milliseconds.
    - MNI → MNI: the field is loaded once in :meth:`fit`; each map is then resampled by nitransforms.

    Fitted surface weights, composed vertex positions and sampling matrices are also cached within the
    Python session, so repeated calls of :func:`surf_to_surf` / :func:`mni_to_surf` (or new instances)
    for the same spaces skip the setup cost.

    Examples
    --------
    >>> st = SpaceTransformer("MNI152NLin2009cAsym", "fsLR", den_to="32k").fit()
    >>> surf_maps = st.transform(["map1.nii.gz", "map2.nii.gz"])  # list of (L, R) GIFTI tuples
    """

    def __init__(self, space_from, space_to, den_from=None, den_to=None, res=None, hemi=None,
                 method=None, hash_check=True, verbose=True):
        self.verbose = set_log(lgr, verbose)
        self._kind_from, self.space_from = _resolve_any_space(space_from)
        self._kind_to, self.space_to = _resolve_any_space(space_to)
        if self._kind_from == "surface" and self._kind_to == "mni":
            raise NotImplementedError(_SURF_TO_MNI_MSG)
        self.den_from = None if den_from is None else _check_density(self.space_from, den_from) \
            if self._kind_from == "surface" else None
        self.den_to = _check_density(self.space_to, den_to or _SURF_DEFAULT_DEN[self.space_to]) \
            if self._kind_to == "surface" else None
        self.res = res
        self.hemi = _resolve_hemi(hemi)
        if self._kind_to == "surface":
            self.method = _resolve_surf_method(method)
        else:
            self.method = _resolve_order(method, default=3)
        self.hash_check = hash_check
        self._fitted = False

    # ----------------------------------------------------------------------------------------------
    def fit(self, data=None):
        """Build the data-independent part of the transform.

        Parameters
        ----------
        data : optional
            Example data, only used to infer `den_from` for surface sources.

        Returns
        -------
        self : SpaceTransformer
        """
        pair = (self._kind_from, self._kind_to)
        if pair == ("mni", "mni"):
            self._fit_mni_to_mni()
        elif pair == ("surface", "surface"):
            if _is_hemi_pair_list(data, self.hemi):
                data = tuple(data)
            if self.den_from is None:
                if data is None:
                    lgr.critical_raise("Surface source density unknown: pass den_from, or data to fit().",
                                       ValueError)
                arrs, _, _ = _split_surface(data[0] if isinstance(data, list) else data, self.hemi)
                self.den_from = _density_of(next(iter(arrs.values())).shape[1], self.space_from)
            self._identity = (self.space_from, self.den_from) == (self.space_to, self.den_to)
            if not self._identity:
                lgr.info(f"Fitting surface transform {self.space_from} {self.den_from} → "
                         f"{self.space_to} {self.den_to} ({'/'.join(self.hemi)}).")
                self._weights = {h: _surface_weights(self.space_from, self.den_from, self.space_to,
                                                     self.den_to, h, self.hash_check) for h in self.hemi}
        elif pair == ("mni", "surface"):
            lgr.info(f"Fitting volume-to-surface transform {self.space_from} → {self.space_to} "
                     f"{self.den_to} ({'/'.join(self.hemi)}), registration fusion (DOI: {_REGFUSION_DOI})"
                     + (f" + EasyReg (DOI: {_EASYREG_DOI})." if self.space_from != _REGFUSION_SPACE else "."))
            self._coords = {h: _surface_coords_in_mni(self.space_from, self.space_to, self.den_to, h,
                                                      self.hash_check) for h in self.hemi}
            self._sampling = {}
        self._fitted = True
        return self

    def transform(self, data):
        """Apply the fitted transform to one map or a list of maps.

        Parameters
        ----------
        data
            MNI source: NIfTI image or path (3D, or 4D = several maps), or a list of them. Surface source:
            an (L, R) tuple of GIFTI images / paths / arrays, a single hemisphere (with `hemi`), a numpy
            array (n_vert_total,) or (n_maps, n_vert_total) with hemispheres concatenated L, R, or a list of
            these. A list of two single GIFTI images / paths (both hemispheres requested) is read as one
            (L, R) map.

        Returns
        -------
        Transformed data in the same structure: NIfTI for MNI targets; for surface targets GIFTI images
        (an (L, R) tuple, or a single image for one hemisphere; integer label volumes projected with
        ``method="nearest"`` give label GIFTIs) or numpy arrays if arrays were passed; a list for list
        input.
        """
        if not self._fitted:
            lgr.critical_raise("SpaceTransformer is not fitted yet; call fit() first.", RuntimeError)
        if self._kind_from == "surface" and _is_hemi_pair_list(data, self.hemi):
            data = tuple(data)
        if isinstance(data, list):
            return [self._transform_one(d) for d in data]
        return self._transform_one(data)

    def fit_transform(self, data):
        """Fit (inferring `den_from` from `data` if needed) and transform `data`."""
        return self.fit(data).transform(data)

    # ----------------------------------------------------------------------------------------------
    def _transform_one(self, data):
        pair = (self._kind_from, self._kind_to)
        if pair == ("mni", "mni"):
            return self._mni_to_mni(data)
        if pair == ("surface", "surface"):
            return self._surf_to_surf(data)
        return self._mni_to_surf(data)

    def _fit_mni_to_mni(self):
        self._identity = self.space_from == self.space_to
        if self._identity:
            return
        try:
            from nitransforms import DenseFieldTransform
        except ImportError as exc:
            raise ImportError("MNI-to-MNI transforms require 'nitransforms'. "
                              "Install with: pip install nitransforms") from exc
        field_path = _resolve_field(self.space_from, self.space_to, None, hash_check=self.hash_check)
        # EasyReg coordinate-map field (is_deltas=False: absolute RAS coords, not displacements)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="nitransforms")
            self._xfm = DenseFieldTransform(nib.load(str(field_path)), is_deltas=False)
        self._references = {}

    def _mni_to_mni(self, img):
        if self._identity:
            return img
        from nitransforms.resampling import apply as _nt_apply
        if isinstance(img, (str, Path)):
            img = nib.load(str(img))

        # Resolution handling
        res = self.res
        vox_size = float(np.min(np.abs(img.header.get_zooms()[:3])))
        if res is None:
            res = min(_AVAILABLE_RES, key=lambda r: abs(r - vox_size))
        else:
            if isinstance(res, str):
                res = int(res.lower().replace("mm", ""))
            if res not in _AVAILABLE_RES:
                lgr.critical_raise(f"res={res!r} not supported. Choose from {_AVAILABLE_RES}.", ValueError)
            if res < round(vox_size):
                lgr.warning(
                    f"Requested output resolution ({res}mm) is finer than input voxel size "
                    f"({vox_size:.1f}mm). Effective resolution remains limited by the input."
                )
        lgr.info(f"Transforming '{self.space_from}' → '{self.space_to}' at {res}mm")

        # Reference grid from affines.json — no template download needed
        if res not in self._references:
            affines_json = read_json(_DATALIB / "affines.json")
            if self.space_to not in affines_json or f"{res}mm" not in affines_json[self.space_to]:
                lgr.critical_raise(f"No {res}mm affine entry for '{self.space_to}' in affines.json.",
                                   ValueError)
            aff = affines_json[self.space_to][f"{res}mm"]
            self._references[res] = nib.Nifti1Image(np.zeros(aff["shape"], dtype=np.uint8),
                                                    np.array(aff["affine"]))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="nitransforms")
            return _nt_apply(self._xfm, img, reference=self._references[res], order=self.method)

    def _surf_to_surf(self, data):
        arrs, fmt, lts = _split_surface(data, self.hemi)
        for h, arr in arrs.items():
            if _density_of(arr.shape[1], self.space_from) != self.den_from:
                lgr.critical_raise(f"Data ({h}: {arr.shape[1]} vertices) do not match the fitted source "
                                   f"density {self.den_from}.", ValueError)
        lts = lts if self.method == "nearest" else None
        if self._identity:
            # same structure as a real transform (e.g. GIFTI images for path input)
            return _merge_surface(arrs, self.hemi, fmt, lts)
        out = {}
        for h, arr in arrs.items():
            weights, trg_mask = self._weights[h]
            if self.method == "nearest":
                res = resample_popular(weights, arr)
            else:
                res = np.asarray(weights @ arr.T.astype(np.float64)).T
            res[:, ~trg_mask] = 0
            out[h] = res
        return _merge_surface(out, self.hemi, fmt, lts)

    def _mni_to_surf(self, img):
        if isinstance(img, (str, Path)):
            img = nib.load(str(img))
        if not isinstance(img, nib.spatialimages.SpatialImage):
            lgr.critical_raise("MNI-to-surface transforms need a NIfTI image or path (the grid is required).",
                               TypeError)
        labels = self.method == "nearest" and np.issubdtype(img.get_data_dtype(), np.integer)
        arr = img.get_fdata(dtype=np.float32)                         # cached by nibabel across calls
        flat = arr.reshape(int(np.prod(img.shape[:3])), -1)          # C-order voxels x maps
        grid = (img.affine.tobytes(), tuple(img.shape[:3]))
        out = {}
        for h in self.hemi:
            if (grid, h) not in self._sampling:
                key = ("sampling", self.space_from, self.space_to, self.den_to, h, self.method, grid,
                       self.hash_check)
                self._sampling[(grid, h)] = _cached(key, lambda: volume_sampling_matrix(
                    self._coords[h], img.affine, img.shape, method=self.method))
            mat, inside = self._sampling[(grid, h)]
            res = np.asarray(mat @ flat).T
            if labels:
                res = np.rint(res).astype(np.int32)                   # outside the grid: empty row -> 0
            else:
                res = res.astype(np.float32)
                res[:, ~inside] = np.nan if self.method == "linear" else 0
            out[h] = res
        lts = {h: _label_table(out[h]) for h in self.hemi} if labels else None
        return _merge_surface(out, self.hemi, ("gifti", len(self.hemi)), lts)


# ==================================================================================================
# PUBLIC WRAPPERS
# ==================================================================================================

def surf_to_surf(data, surf_from, surf_to, den_to=None, den_from=None, hemi=None, method="linear",
                 hash_check=True, verbose=True):
    """Transform surface data between fsLR / fsaverage spaces and densities.

    Pure-Python equivalent of Connectome Workbench ``-metric-resample`` (``method="linear"``) and
    ``-label-resample`` (``method="nearest"``) with ``ADAP_BARY_AREA``, the source medial wall as ROI
    and the target medial wall set to 0 (the neuromaps :cite:`markello2022` defaults). The first call for
    a space/density pair takes ~1–3 s (building the resampling weights); later calls in the same session
    take milliseconds. See :class:`SpaceTransformer` to transform many maps with one fitted transform.

    Parameters
    ----------
    data
        (L, R) tuple of GIFTI images / paths / arrays, a single hemisphere (with `hemi`), a numpy array
        (n_vert_total,) or (n_maps, n_vert_total) with hemispheres concatenated L, R, or a list of these.
    surf_from, surf_to : {'fsLR', 'fsaverage'}
        Source and target surface space (case-insensitive).
    den_to : str, optional
        Target density. Default: ``"32k"`` (fsLR) / ``"41k"`` (fsaverage).
    den_from : str, optional
        Source density; inferred from the number of vertices if not given.
    hemi : {'L', 'R'} or None
        Hemisphere(s) in `data`. Default: both.
    method : {'linear', 'nearest'}
        ``"linear"`` for continuous data, ``"nearest"`` for labels / parcellations (label with the
        largest summed resampling weight).
    hash_check : bool
        Verify the NiSpace-hosted templates against their known hashes. Default ``True``.
    verbose : bool
        Emit log messages. Default ``True``.

    Returns
    -------
    Transformed data in the structure of `data` (GIFTI images, or numpy arrays for array input).
    """
    return SpaceTransformer(surf_from, surf_to, den_from=den_from, den_to=den_to, hemi=hemi,
                            method=method, hash_check=hash_check, verbose=verbose).fit_transform(data)


def mni_to_surf(img, mni_from, surf_to, den_to=None, hemi=None, method="linear",
                hash_check=True, verbose=True):
    """Project volumetric MNI data onto the fsLR / fsaverage surface.

    Uses registration fusion :cite:`wu2018` vertex positions (defined in MNI152NLin6Asym). For any other
    source MNI space, these positions are first mapped through the EasyReg field :cite:`iglesias2023`
    into `mni_from`, so the data are interpolated only once (no intermediate volume resampling). The first
    call takes well below a second; later calls in the same session (same spaces and image grid) take
    milliseconds. See :class:`SpaceTransformer` to project many maps with one fitted transform.

    Parameters
    ----------
    img : str, Path, nibabel image, or list
        Volume(s) in `mni_from` space (4D = several maps).
    mni_from : str
        Source MNI space; names and aliases as in :func:`mni_to_mni`.
    surf_to : {'fsLR', 'fsaverage'}
        Target surface space (case-insensitive).
    den_to : str, optional
        Target density. Default: ``"32k"`` (fsLR) / ``"41k"`` (fsaverage).
    hemi : {'L', 'R'} or None
        Hemisphere(s) to project. Default: both.
    method : {'linear', 'nearest'}
        Trilinear interpolation, or nearest voxel (labels / parcellations). Vertices outside the image
        grid get NaN (linear) or 0 (nearest). Integer images with ``"nearest"`` give label GIFTIs.
    hash_check : bool
        Verify the NiSpace-hosted files against their known hashes. Default ``True``.
    verbose : bool
        Emit log messages. Default ``True``.

    Returns
    -------
    (L, R) tuple of GIFTI images (a single image for one hemisphere), or a list for list input.
    """
    return SpaceTransformer(mni_from, surf_to, den_to=den_to, hemi=hemi, method=method,
                            hash_check=hash_check, verbose=verbose).fit().transform(img)


def surf_to_mni(data, surf_from, mni_to, res=None, hemi=None, method="linear", hash_check=True,
                verbose=True):
    """Project surface data into an MNI volume space. Not implemented yet.

    Will provide registration-fusion-based :cite:`wu2018` surface-to-volume transforms for all supported
    MNI spaces (via the EasyReg fields).

    Raises
    ------
    NotImplementedError
        Always, for now.
    """
    raise NotImplementedError(_SURF_TO_MNI_MSG)


def transform(data, space_from, space_to, order=None, **kwargs):
    """Transform `data` from `space_from` to `space_to` (MNI or fsLR / fsaverage).

    Dispatches to the appropriate transform: :func:`mni_to_mni`, :func:`surf_to_surf`,
    :func:`mni_to_surf` (and, once available, :func:`surf_to_mni`). Keyword arguments are passed to
    :class:`SpaceTransformer` (`den_from`, `den_to`, `res`, `hemi`, `method`, `hash_check`, `verbose`).
    `order` is accepted as an alias of `method` (as in :func:`mni_to_mni`).
    """
    if order is not None:
        if kwargs.get("method") is not None:
            lgr.critical_raise("Pass either 'order' or 'method', not both.", ValueError)
        kwargs["method"] = order
    return SpaceTransformer(space_from, space_to, **kwargs).fit_transform(data)
