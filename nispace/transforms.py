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

_SPACE_ALIASES = {
    "SPM":   "MNI152Lin",
    "SPM5":  "MNI152Lin",
    "SPM8":  "MNI152Lin",
    "SPM12": "MNI152Lin",
}

_AVAILABLE_RES = [1, 2, 3, 4]


def mni_to_mni(img, mni_from, mni_to, order=3, res=None,
               nispace_data_dir=None, verbose=True):
    """Transform a NIfTI image between MNI template spaces using EasyReg fields.

    Parameters
    ----------
    img : str, Path, or nibabel.SpatialImage
        Input image to resample.
    mni_from : str
        Source MNI space. Supported: ``MNI152NLin2009cAsym``,
        ``MNI152NLin6Asym``, ``MNI152Lin``, ``MNI305``, ``MNIColin27``,
        ``MNI152NLin2009cSym``, ``MNI152NLin6Sym``. Aliases: ``SPM``,
        ``SPM5``, ``SPM8``, ``SPM12`` → ``MNI152Lin``.
    mni_to : str
        Target MNI space (same options as ``mni_from``).
    order : int
        Spline interpolation order (0 = nearest-neighbour, 1 = trilinear,
        3 = cubic spline). Default 3.
    res : int, str, or None
        Output voxel resolution in mm. Accepted: ``1``, ``2``, ``3``, ``4``
        or ``"1mm"``, ``"2mm"``, ``"3mm"``, ``"4mm"``. If ``None``, snapped
        from the input image voxel size.
    nispace_data_dir : str or Path, optional
        Override for the NiSpace data directory. Default uses the configured
        data directory (``NISPACE_DATA_DIR`` env var or download cache).
    verbose : bool
        Whether to emit log messages. Default ``True``.

    Returns
    -------
    nibabel.Nifti1Image
        Resampled image in the target MNI space.
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

    # Resolve aliases and short-circuit identity transforms
    mni_from = _SPACE_ALIASES.get(mni_from, mni_from)
    mni_to   = _SPACE_ALIASES.get(mni_to,   mni_to)
    if mni_from == mni_to:
        return img

    if isinstance(img, (str, Path)):
        img = nib.load(str(img))

    # Resolution handling
    vox_size = float(np.min(np.abs(img.header.get_zooms()[:3])))
    if res is None:
        res = min(_AVAILABLE_RES, key=lambda r: abs(r - vox_size))
    else:
        if isinstance(res, str):
            res = int(res.lower().replace("mm", ""))
        if res not in _AVAILABLE_RES:
            raise ValueError(f"res={res!r} not supported. Choose from {_AVAILABLE_RES}.")
        if res < round(vox_size):
            warnings.warn(
                f"Requested output resolution ({res}mm) is finer than input voxel size "
                f"({vox_size:.1f}mm). Effective resolution remains limited by the input.",
                UserWarning, stacklevel=2,
            )

    # Look up the deformation field in transform.json
    transform_json = read_json(_DATALIB / "transform.json")

    if mni_to in transform_json and mni_from in transform_json[mni_to]:
        # Direct: stored as transform_json[target][source] — backward field resamples source→target
        entry = transform_json[mni_to][mni_from]["backward"]
    elif mni_from in transform_json and mni_to in transform_json[mni_from]:
        # Reverse: pair stored with roles swapped — forward field resamples target→source, i.e. our source→target
        entry = transform_json[mni_from][mni_to]["forward"]
    else:
        all_spaces = sorted(set(transform_json) | {s for v in transform_json.values() for s in v})
        raise ValueError(
            f"No transform available between '{mni_from}' and '{mni_to}'. "
            f"Supported spaces: {all_spaces}."
        )

    lgr.info(f"Transforming '{mni_from}' → '{mni_to}' at {res}mm.")

    # Fetch the deformation field (download if needed)
    nispace_data_dir = _resolve_nispace_data_dir(nispace_data_dir)
    field_path = get_file(
        Path(nispace_data_dir) / entry["remote"],
        **entry,
    )

    # Build reference grid from affines.json — no template download needed
    affines_json = read_json(_DATALIB / "affines.json")
    if mni_to not in affines_json or f"{res}mm" not in affines_json[mni_to]:
        raise ValueError(f"No {res}mm affine entry for '{mni_to}' in affines.json.")
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
