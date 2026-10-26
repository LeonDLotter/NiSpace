"""Deprecated alias of :mod:`nispace.transform` (renamed to match :mod:`nispace.parcellate`).

TODO (first non-dev release): remove this module.
"""

import logging

from .transform import *  # noqa: F401,F403
from .transform import (mni_to_mni, compute_transform_displacement,  # noqa: F401
                        _resolve_space, _resolve_field)

_DEPR_TRANSFORMS_MODULE = (
    "The 'nispace.transforms' module is deprecated and will be removed in the first non-dev release. "
    "Please import from 'nispace.transform' instead."
)
logging.getLogger("nispace").warning(_DEPR_TRANSFORMS_MODULE)
