import logging
lgr = logging.getLogger(__name__)
lgr.warning(
    "The 'nispace.modules' namespace has been renamed to 'nispace.core' and will be "
    "removed in a future version. Update your imports accordingly."
)
