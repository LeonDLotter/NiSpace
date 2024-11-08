# %% Init

import sys
from pathlib import Path
import numpy as np
import pandas as pd

wd = Path.cwd().parent
print(f"Working dir: {wd}")
sys.path.append(wd.as_posix())

# import NiSpace functions
from nispace.datasets import fetch_parcellation, parcellation_lib
from nispace.nulls import get_distance_matrix

# nispace data path 
nispace_source_data_path = wd / "datasets" / "nispace-data_source"


# %% Generate distance matrices

for parc in parcellation_lib:
    if "alias" in parcellation_lib[parc]:
        continue
    for space in parcellation_lib[parc]:
        print(parc, space)
        
        # load
        parc_loaded, labels = fetch_parcellation(parc, space, return_loaded=True) 
            
        # distance matrix 
        dist_mat = get_distance_matrix(
            parc_loaded, 
            parc_space=space,
            downsample_vol=2,
            centroids=False,
            surf_euclidean=False,
            n_proc=-1,
            dtype=np.float32
        )
        if not isinstance(dist_mat, tuple):
            pd.DataFrame(dist_mat).to_csv(
                nispace_source_data_path / "parcellation" / parc / space / f"parc-{parc}_space-{space}.dist.csv.gz", 
                header=None, index=None
            )
        else:
            for mat, hemi in zip(dist_mat, ["L", "R"]):
                pd.DataFrame(mat).to_csv(
                    nispace_source_data_path / "parcellation" / parc / space / f"parc-{parc}_space-{space}_hemi-{hemi}.dist.csv.gz", 
                    header=None, index=None
                )
# %%
