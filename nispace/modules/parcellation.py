import logging
lgr = logging.getLogger(__name__)
lgr.warning(
    "'nispace.modules.parcellation' has moved to 'nispace._core.parcellation' and will be "
    "removed in a future version. Update your imports accordingly."
)
from nispace._core.parcellation import *  # noqa: F401, F403
from nispace._core.parcellation import _bilateral_labels_match, _norm_space, _spaces_match  # noqa: F401
