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

# add nispace to path
sys.path.append(str(pathlib.Path.cwd().parent))
from nispace.io import parcellate_data

# nispace data path in home dir
nispace_data_path = pathlib.Path.home() / "nispace-data"


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

for parc in (nispace_data_path / "parcellation").glob("parc*[!txt]"): 
    print(parc)
    
    if not parc.is_dir():
        gmprob_tab = parcellate_data(
            data=nispace_data_path / "template" / "mni152" / "map" / f"MNI152NLin2009cAsym_desc-gmprob_res-1mm.nii.gz",
            data_labels=["GM_prob"],
            data_space="MNI152",
            parcellation=parc,
            parc_space="MNI152",
            parc_labels=np.loadtxt(str(parc).replace("nii.gz", "txt"), str).tolist(), 
            resampling_target="data",
            n_proc=-1,
            dtype=np.float32
        )
    else:
        gmprob_tab = parcellate_data(
            data=nispace_data_path / "template" / "mni152" / "map" / f"MNI152NLin2009cAsym_desc-gmprob_res-1mm.nii.gz",
            data_labels=["GM_prob"],
            data_space="MNI152",
            parcellation=(
                parc / (parc.name + "_hemi-L.gii.gz"),
                parc / (parc.name + "_hemi-R.gii.gz")
            ),
            parc_space="fsaverage",
            parc_hemi=["L", "R"],
            parc_labels= \
                np.loadtxt(str(parc / (parc.name + "_hemi-L.txt")), str).tolist() + \
                np.loadtxt(str(parc / (parc.name + "_hemi-R.txt")), str).tolist(),
            resampling_target="parcellation",
            n_proc=-1,
            dtype=np.float32
        )
    
    gmprob_tab.to_csv(nispace_data_path / "template" / "mni152" / "tab" / f"gmprob_{parc.name.split('_')[0].split('-')[1]}.csv")


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

for parc in (nispace_data_path / "parcellation").glob("parc*[!txt]"): 
    print(parc)
    
    if parc.is_dir():
        thick_tab = parcellate_data(
            data=(
                nispace_data_path / "template" / "fsaverage" / "map" / f"fsaverage_desc-thick_hemi-L_res-10k.gii.gz",
                nispace_data_path / "template" / "fsaverage" / "map" / f"fsaverage_desc-thick_hemi-R_res-10k.gii.gz"
            ),
            data_labels=["thick"],
            data_space="fsaverage",
            parcellation=(
                parc / (parc.name + "_hemi-L.gii.gz"),
                parc / (parc.name + "_hemi-R.gii.gz")
            ),
            parc_space="fsaverage",
            parc_hemi=["L", "R"],
            parc_labels= \
                np.loadtxt(str(parc / (parc.name + "_hemi-L.txt")), str).tolist() + \
                np.loadtxt(str(parc / (parc.name + "_hemi-R.txt")), str).tolist(),
            resampling_target="parcellation",
            n_proc=-1,
            dtype=np.float32
        )
    else:
        print("skip")
        continue
    
    thick_tab.to_csv(nispace_data_path / "template" / "fsaverage" / "tab" / f"thick_{parc.name.split('_')[0].split('-')[1]}.csv")


# %%
