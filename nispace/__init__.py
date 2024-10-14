
# initialize logger to make it available for all modules
from .utils.utils import _init_lgr
lgr = _init_lgr(__name__)

# get version
from . import _version
__version__ = _version.get_versions()['version']

# Public API
from .api import NiSpace
from .workflows import simple_colocalization, simple_xsea, group_comparison
from .datasets import fetch_reference, fetch_template, fetch_parcellation, fetch_metadata
