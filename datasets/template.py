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

from utils_datasets import parcellate_reference_dataset

# nispace data path in home dir
nispace_data_path = pathlib.Path.cwd() / "nispace-data"


# %% MNI152 - We use: MNI152NLin2009cAsym as does fMRIprep by default!

# fetch template flow data in 1mm space and save to directory
mni152 = {
    "t1": tf.get("MNI152NLin2009cAsym", suffix="T1w", desc="brain", resolution=1, extension=".nii.gz"),
    "gmprob": tf.get("MNI152NLin2009cAsym", suffix="probseg", label="GM", resolution=1, extension=".nii.gz"),
    "mask": tf.get("MNI152NLin2009cAsym", suffix="mask", desc="brain", resolution=1, extension=".nii.gz"),
}

# plot
for f in mni152:
    plot_anat(mni152[f], title=f)
    plt.show()
    
# save
for f in mni152:
    
    # load image
    img = image.load_img(mni152[f])
    
    # file type to save space
    if f=="t1":
        img = image.new_img_like(img, img.get_fdata().astype(np.int32), copy_header=True)
    elif f=="gmprob":
        img = image.new_img_like(img, img.get_fdata().astype(np.float16), copy_header=True)
    
    # save
    img.to_filename(nispace_data_path / "template" / "mni152" / "map" / f"MNI152NLin2009cAsym_desc-{f}_res-1mm.nii.gz")


# %% GM parcellated tissue probability data

parcellate_reference_dataset(
    reference_name="gmprob",
    reference_files=[nispace_data_path / "template" / "mni152" / "map" / f"MNI152NLin2009cAsym_desc-gmprob_res-1mm.nii.gz"],
    reference_data_path=nispace_data_path / "template" / "mni152",
    nispace_data_path=nispace_data_path,
    data_labels=["gmprob"],
    parcs=_PARCS_NICE,
)

# %% FSAVERAGE - We use: fsaverage5 as does nilearn by default!

# fetch template flow data in 10k resolution, except for midthickness in 164k
fsa_nilearn = datasets.fetch_surf_fsaverage("fsaverage5")
fsa = {
    "pial": ( pathlib.Path(fsa_nilearn["pial_left"]), pathlib.Path(fsa_nilearn["pial_right"]) ),
    "infl": ( pathlib.Path(fsa_nilearn["infl_left"]), pathlib.Path(fsa_nilearn["infl_right"]) ),
    "thick": ( pathlib.Path(fsa_nilearn["thick_left"]), pathlib.Path(fsa_nilearn["thick_right"]) )
}

# plot
for f in fsa:
    if f != "thick":
        plot_surf(fsa[f][0], title=f, hemi="left")
    else:
        plot_surf(fsa["infl"][0], fsa["thick"][0], title="thickness on infl", hemi="left")
    plt.show()

# save
for f in fsa:
    for hemi, f_hemi in zip(["L", "R"], fsa[f]):
        dest = nispace_data_path / "template" / "fsaverage" / "map" / f"fsaverage_desc-{f}_hemi-{hemi}_res-10k.gii.gz"
        dest.write_bytes(f_hemi.read_bytes())


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
