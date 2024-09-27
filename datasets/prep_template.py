# %% Init

import sys
import pathlib
import numpy as np
import pandas as pd
from templateflow import api as tf
from nilearn.plotting import plot_anat, plot_surf
from nilearn import image
from nilearn import datasets 
import matplotlib.pyplot as plt

from nispace.io import parcellate_data
from nispace.modules.constants import _PARCS_NICE
from nispace.datasets import fetch_template
#from nispace.utils.utils_datasets import parcellate_reference_dataset

# nispace data path in home dir
nispace_data_path = pathlib.Path.cwd() / "nispace-data"


# %% MNI152 - We use: MNI152NLin2009cAsym as does fMRIprep by default!

# MNI 152 templates in 1 and 2 mm resolution are fetched directly from templateflow.
# We will only generate 3 and 4 mm resolution templates from the 1mm templateflow version

tpl_1mm_T1 = fetch_template("MNI152", res="1mm", desc="T1")
tpl_1mm_gmprob = fetch_template("MNI152", res="1mm", desc="gmprob")
tpl_1mm_mask = fetch_template("MNI152", res="1mm", desc="mask")

for voxsize in [3, 4]:
    for tpl_file, interp in [(tpl_1mm_T1, "linear"), (tpl_1mm_gmprob, "linear"), (tpl_1mm_mask, "nearest")]:
        # resample
        tpl_resampled = image.resample_img(
            image.load_img(tpl_file),
            target_affine=np.diag([voxsize, voxsize, voxsize]), 
            interpolation=interp
        )
        # save
        path = nispace_data_path / "template" / "mni152" / "map" / tpl_file.name.replace("1mm", f"{voxsize}mm")
        tpl_resampled.to_filename(path)



# %% GM parcellated tissue probability data

# parcellate_reference_dataset(
#     reference_name="gmprob",
#     reference_files=[nispace_data_path / "template" / "mni152" / "map" / f"MNI152NLin2009cAsym_desc-gmprob_res-1mm.nii.gz"],
#     reference_data_path=nispace_data_path / "template" / "mni152",
#     nispace_data_path=nispace_data_path,
#     data_labels=["gmprob"],
#     parcs=_PARCS_NICE,
# )

# %% FSAVERAGE - We use: fsaverage5 as does nilearn by default!


# %% parcellated thickness data

# parcellate_reference_dataset(
#     reference_name="thick",
#     reference_files=[
#         (nispace_data_path / "template" / "fsaverage" / "map" / f"fsaverage_desc-thick_hemi-L_res-10k.gii.gz",
#          nispace_data_path / "template" / "fsaverage" / "map" / f"fsaverage_desc-thick_hemi-R_res-10k.gii.gz")
#     ],
#     reference_data_path=nispace_data_path / "template" / "fsaverage",
#     nispace_data_path=nispace_data_path,
#     data_labels=["thick"],
#     parcs=["Destrieux", "DesikanKilliany"],
# )

# %%
