import logging
lgr = logging.getLogger(__name__)
lgr.warning(
    "'nispace.modules.brainspace_moran' has moved to 'nispace._brainspace_moran' and will be "
    "removed in a future version. Update your imports accordingly."
)
from nispace._brainspace_moran import *  # noqa: F401, F403
