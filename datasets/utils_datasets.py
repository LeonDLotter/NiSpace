import sys
import pathlib
import pickle
import gzip
import requests
import numpy as np
import pandas as pd
import tempfile

from nispace.modules.constants import _PARCS_NICE
from nispace.io import parcellate_data
from nispace.nulls import generate_null_maps
from nispace.utils import _rm_ext
from nispace.datasets import fetch_parcellation


def download(url, path=None):
    r = requests.get(url)
    r.raise_for_status()
    if path is None:
        path = pathlib.Path(tempfile.gettempdir()) / pathlib.Path(url).name
    with open(str(path), "wb") as f:
        f.write(r.content)
    return path


def parcellate_reference_dataset(reference_name, reference_files, nispace_data_path, parcs=_PARCS_NICE, nulls=False):
    nispace_data_path = pathlib.Path(nispace_data_path)
    reference_data_path = nispace_data_path / "reference" / reference_name

    for parc in parcs:
        print(parc)
        
        # get parcellation
        parc_loaded, parc_labels, parc_space, parc_distmat = \
            fetch_parcellation(parc, return_dist_mat=True, return_space=True, return_loaded=True)

        # parcellate  
        tab = parcellate_data(
            parcellation=parc_loaded,
            parc_labels=parc_labels,
            parc_space=parc_space,
            parc_hemi=["L", "R"],
            resampling_target="data" if parc_space == "MNI152" else "parcellation",
            data=reference_files,
            data_labels=[_rm_ext(f.name) for f in reference_files],
            data_space="MNI152",
            drop_background_parcels=True,
            min_num_valid_datapoints=5,
            min_fraction_valid_datapoints=0.1,
            n_proc=-1,
            dtype=np.float32,
        )
        tab.index.name = "map"
        tab.to_csv(reference_data_path / "tab" / f"{reference_name}_{parc}.csv.gz")
        
        if nulls:
            # null maps
            null_maps, _ = generate_null_maps(
                "moran",
                tab, 
                parcellation=parc_loaded,
                dist_mat=parc_distmat,
                parc_space=parc_space,
                n_nulls=10000,
                dtype=np.float16,
                n_proc=-1,
                seed=42
            )
            with gzip.open(reference_data_path / "null" / f"{reference_name}_{parc}.pkl.gz", "wb") as f:
                pickle.dump(null_maps, f, pickle.HIGHEST_PROTOCOL)
        
        
        