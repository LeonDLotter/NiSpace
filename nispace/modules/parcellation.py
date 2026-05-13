import logging
lgr = logging.getLogger(__name__)
lgr.warning(
    "'nispace.modules.parcellation' has moved to 'nispace.core.parcellation' and will be "
    "removed in a future version. Update your imports accordingly."
)
from nispace.core.parcellation import *  # noqa: F401, F403
from nispace.core.parcellation import _bilateral_labels_match, _norm_space, _spaces_match  # noqa: F401
