
# initialize logger to make it available for all modules
from .utils import _init_lgr
lgr = _init_lgr(__name__)

# make NiSpace available from package level
from .api import NiSpace
from . import _version
__version__ = _version.get_versions()['version']
