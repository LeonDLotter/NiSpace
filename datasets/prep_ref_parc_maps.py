# %% Init

import pathlib
import numpy as np
import pandas as pd

from utils import parcellate_reference_dataset 

# add nispace to path
from nispace.datasets import fetch_reference
from nispace.utils.utils import _rm_ext
from nispace.io import write_json
from nispace.modules.constants import _PARCS_NICE, _DSETS, _DSETS_TAB_ONLY

# nispace data path
nispace_source_data_path = pathlib.Path.cwd() / "nispace-data"

# datasets with maps
_DSETS_MAP = [ds for ds in _DSETS if ds not in _DSETS_TAB_ONLY]

# %% Parcellate map-based image data ---------------------------------------------------------------

# iterate datasets
for dataset in _DSETS_MAP:
    print("-------- " + dataset.upper() + " --------")
    
    # get files
    files = fetch_reference(dataset)
    
    # parcellate
    parcellate_reference_dataset(
        reference_name=dataset,
        reference_files=files,
        reference_path=nispace_source_data_path / "reference" / dataset,
        parcs=_PARCS_NICE,
    )


# %%
