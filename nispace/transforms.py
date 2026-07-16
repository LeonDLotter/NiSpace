import logging
from pathlib import Path
import warnings

import numpy as np
import nibabel as nib

from .io import read_json
from .utils.utils import set_log
from .utils.utils_datasets import get_file
from .datasets import _resolve_nispace_data_dir

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
]

# Exact-match aliases resolved before pattern matching
_SPACE_ALIASES = {
    "SPM":              "MNI152Lin",
    "SPM5":             "MNI152Lin",
    "SPM8":             "MNI152Lin",
    "SPM12":            "MNI152Lin",
    "FSL":              "MNI152NLin6Asym",
    "HCP":              "MNI152NLin6Asym",
    "MNI152NLin2009a":  "MNI152NLin2009cAsym",
    "MNI152NLin2009b":  "MNI152NLin2009cAsym",
}

_AVAILABLE_RES = [1, 2, 3, 4]

# String aliases for nitransforms interpolation order
_ORDER_ALIASES = {"nearest": 0, "linear": 1, "cubic": 3, "continuous": 3}

_EASYREG_DOI = "10.1038/s41598-023-33781-0"


def _resolve_space(space):
    """Resolve a (possibly partial or aliased) space name to a canonical name."""
    # 1. Exact canonical name
    if space in _KNOWN_SPACES:
        return space
    # 2. Exact alias
    if space in _SPACE_ALIASES:
        return _SPACE_ALIASES[space]
    # 3. Case-insensitive prefix matching
    s = space.lower()
    prefix_matches = [k for k in _KNOWN_SPACES if k.lower().startswith(s)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        lgr.critical_raise(
            f"Space name '{space}' is ambiguous — matches: {prefix_matches}. "
            f"Please be more specific.",
            ValueError,
        )
    # 4. Case-insensitive substring matching
    substr_matches = [k for k in _KNOWN_SPACES if s in k.lower()]
    if len(substr_matches) == 1:
        return substr_matches[0]
    if len(substr_matches) > 1:
        lgr.critical_raise(
            f"Space name '{space}' is ambiguous — matches: {substr_matches}. "
            f"Please be more specific.",
            ValueError,
        )
    lgr.critical_raise(
        f"Unknown MNI space '{space}'. Supported spaces: {_KNOWN_SPACES}. "
        f"Known aliases: {list(_SPACE_ALIASES)}.",
        ValueError,
    )


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

    Deformation fields were estimated with EasyReg [1]_ (Freesurfer), which
    builds on the SynthMorph [2]_ registration and SynthSeg [3]_ segmentation
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

        Named aliases: ``SPM`` / ``SPM5`` / ``SPM8`` / ``SPM12`` →
        ``MNI152Lin`` (all SPM versions normalise to the same linearly-
        registered ICBM 152 space); ``FSL`` / ``HCP`` →
        ``MNI152NLin6Asym``; ``MNI152NLin2009a`` / ``MNI152NLin2009b`` →
        ``MNI152NLin2009cAsym``.

        Unambiguous partial names are also accepted, e.g. ``"Colin"`` →
        ``MNIColin27``, ``"MNI152Li"`` → ``MNI152Lin``.
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
    .. [1] Iglesias et al. (2023). Ready-to-use, open-source, deformable
           registration of brain MRI. *Scientific Reports*.
           https://doi.org/10.1038/s41598-023-33781-0
    .. [2] Hoffmann et al. (2022). SynthMorph: learning contrast-invariant
           registration without acquired images. *IEEE Transactions on
           Medical Imaging*. https://doi.org/10.1109/TMI.2021.3116879
    .. [3] Billot et al. (2023). SynthSeg: Segmentation of brain MRI scans
           of any contrast and resolution without retraining. *Medical
           Image Analysis*. https://doi.org/10.1016/j.media.2023.102789
    """
    try:
        from nitransforms import DenseFieldTransform
        from nitransforms.resampling import apply as _nt_apply
    except ImportError as exc:
        raise ImportError(
            "mni_to_mni requires 'nitransforms'. "
            "Install with: pip install nitransforms"
        ) from exc

    verbose = set_log(lgr, verbose)

    # Resolve space names (aliases + pattern matching) and short-circuit identity
    mni_from = _resolve_space(mni_from)
    mni_to   = _resolve_space(mni_to)
    if mni_from == mni_to:
        return img

    if isinstance(img, (str, Path)):
        img = nib.load(str(img))

    # Interpolation order — accept string aliases
    if isinstance(order, str):
        if order not in _ORDER_ALIASES:
            lgr.critical_raise(
                f"order={order!r} not recognised. "
                f"Use one of {list(_ORDER_ALIASES)} or an integer 0–5.",
                ValueError,
            )
        order = _ORDER_ALIASES[order]

    # Resolution handling
    vox_size = float(np.min(np.abs(img.header.get_zooms()[:3])))
    if res is None:
        res = min(_AVAILABLE_RES, key=lambda r: abs(r - vox_size))
    else:
        if isinstance(res, str):
            res = int(res.lower().replace("mm", ""))
        if res not in _AVAILABLE_RES:
            lgr.critical_raise(
                f"res={res!r} not supported. Choose from {_AVAILABLE_RES}.",
                ValueError,
            )
        if res < round(vox_size):
            lgr.warning(
                f"Requested output resolution ({res}mm) is finer than input voxel size "
                f"({vox_size:.1f}mm). Effective resolution remains limited by the input."
            )

    lgr.info(f"Transforming '{mni_from}' → '{mni_to}' at {res}mm")
    field_path = _resolve_field(mni_from, mni_to, nispace_data_dir, hash_check=hash_check)

    # Build reference grid from affines.json — no template download needed
    affines_json = read_json(_DATALIB / "affines.json")
    if mni_to not in affines_json or f"{res}mm" not in affines_json[mni_to]:
        lgr.critical_raise(
            f"No {res}mm affine entry for '{mni_to}' in affines.json.",
            ValueError,
        )
    aff = affines_json[mni_to][f"{res}mm"]
    reference = nib.Nifti1Image(
        np.zeros(aff["shape"], dtype=np.uint8),
        np.array(aff["affine"]),
    )

    # Apply EasyReg coordinate-map field (is_deltas=False: absolute RAS coords, not displacements)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="nitransforms")
        xfm = DenseFieldTransform(nib.load(str(field_path)), is_deltas=False)
        return _nt_apply(xfm, img, reference=reference, order=order)


def compute_transform_displacement(mni_from, mni_to,
                                   nispace_data_dir=None, hash_check=True,
                                   verbose=True):
    """Compute a voxel-wise displacement magnitude map for a MNI space transform.

    For each voxel in the ``fwd_field`` deformation field (estimated with
    EasyReg [1]_, built on SynthMorph [2]_ and SynthSeg [3]_; the field used
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
    .. [1] Iglesias et al. (2023). Ready-to-use, open-source, deformable
           registration of brain MRI. *Scientific Reports*.
           https://doi.org/10.1038/s41598-023-33781-0
    .. [2] Hoffmann et al. (2022). SynthMorph: learning contrast-invariant
           registration without acquired images. *IEEE Transactions on
           Medical Imaging*. https://doi.org/10.1109/TMI.2021.3116879
    .. [3] Billot et al. (2023). SynthSeg: Segmentation of brain MRI scans
           of any contrast and resolution without retraining. *Medical
           Image Analysis*. https://doi.org/10.1016/j.media.2023.102789
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
