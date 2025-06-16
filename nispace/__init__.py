
# initialize logger to make it available for all modules
from .utils.utils import _init_lgr
lgr = _init_lgr(__name__)

# get version
from . import _version
__version__ = _version.get_versions()['version']

# git commit
def get_commit(get_version=True):
    if not get_version:
        try:
            from subprocess import check_output
            commit = check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        except:
            commit = f"version:{__version__}"
    else:
        commit = f"{__version__}"
    return commit
__commit__ = get_commit()

# Public API
from .api import NiSpace
from .workflows import simple_colocalization, simple_xsea, group_comparison

