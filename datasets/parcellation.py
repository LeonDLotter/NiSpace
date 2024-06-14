# %% Init

import sys
import os
import shutil
import pathlib
import numpy as np
import pandas as pd
from nilearn import datasets
from neuromaps import images
import abagen

from utils_datasets import download

# import NiSpace functions
from nispace.nulls import get_distance_matrix

# nispace data path in home dir
nispace_data_path = pathlib.Path.cwd() / "nispace-data"


# %% Get parcellations

# parcellation info
parc_info = pd.DataFrame(columns=["parcellation", "n_parcels", "space", "resolution", "publication"])

# ==================================================================================================
# Schaefer + Melbourne
for schaefer, tian in [(100, "S1"), (200, "S2"), (300, "S3")]:
    print(f"Schaefer {schaefer} + Melbourne {tian}")
    parc_path = download(
        f"https://github.com/yetianmed/subcortex/raw/master/Group-Parcellation/3T/Cortex-Subcortex/"
        f"MNIvolumetric/Schaefer2018_{schaefer}Parcels_7Networks_order_Tian_Subcortex_{tian}_3T_"
        "MNI152NLin2009cAsym_1mm.nii.gz"
    )
    labs_tian = pd.read_csv(
        f"https://github.com/yetianmed/subcortex/raw/master/Group-Parcellation/3T/Subcortex-Only/"
        f"Tian_Subcortex_{tian}_3T_label.txt", 
        header=None
    )[0].to_list()
    labs_schaefer = pd.read_csv(
        f"https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/brain_parcellation/"
        f"Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/"
        f"Schaefer2018_{schaefer}Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv"
    )["ROI Name"].to_list()
    labs = []
    for l in labs_tian:
        l = l.split("-")
        labs.append(l[-1].upper() + "_SC_" + "-".join(l[:-1]))
    for l in labs_schaefer:
        l = l.split("_")
        labs.append(l[1] + "_CX_" + "_".join(l[2:]))
    labs = [f"{i}_{l}" for i, l in enumerate(labs, start=1)]
    parc_name = f"parc-Schaefer{schaefer}Melbourne{tian}_n-{len(labs)}_space-mni152_res-1mm.nii.gz"
    info = {
        "parcellation": parc_name, 
        "n_parcels": len(labs), 
        "space": "MNI152", # "FSL" (schaefer), "MNI152NLin2009cAsym" (Melbourne)
        "resolution": "1mm", 
        "publication": "10.1093/cercor/bhx179; 0.1038/s41593-020-00711-6",
        "license": "MIT"
    }
    # Move the downloaded parcellation file to the desired directory
    destination_path = nispace_data_path / "parcellation" / parc_name
    shutil.move(parc_path, destination_path)
    with open(nispace_data_path / "parcellation" / (info["parcellation"]).replace("nii.gz", "txt"), "w") as f:
        f.write("\n".join(labs))
    parc_info.loc[len(parc_info)] = info
# ==================================================================================================

# ==================================================================================================
# HCPex
print("HCPex")
info = {
    "parcellation": "parc-HCPex_n-426_space-mni152_res-1mm.nii.gz", 
    "n_parcels": 426, 
    "space": "MNI152", # "MNI152NLin2009cAsym"
    "resolution": "1mm", 
    "publication": "10.1038/nature18933; 10.1007/s00429-021-02421-6",
    "license": "GPL-3.0"
}
parc_path = download(
    "https://github.com/wayalan/HCPex/raw/main/HCPex_v1.1/HCPex.nii.gz",
    nispace_data_path / "parcellation" / info["parcellation"]
)
lab_path = download(
    "https://github.com/wayalan/HCPex/raw/main/HCPex_v1.1/HCPex.nii.txt",
    str(nispace_data_path / "parcellation" / info["parcellation"]).replace(".nii.gz", ".txt")
)
labs = np.loadtxt(lab_path, str).tolist()
labs = [f"{l[0]}_{l[1]}H_{'CX' if int(l[0]) < 361 else 'SC'}_{l[2]}" for l in labs]
with open(lab_path, "w") as f:
    f.write("\n".join(labs))
parc_info.loc[len(parc_info)] = info
# ==================================================================================================

# ==================================================================================================
# DesikanKilliany
print("DesikanKilliany")
info = {
    "parcellation": "parc-DesikanKilliany_n-68_space-fsaverage_res-10k", 
    "n_parcels": 68, 
    "space": "fsaverage",
    "resolution": "10k", 
    "publication": "10.1016/j.neuroimage.2006.01.021",
    "license": "free"
}
parc_path = nispace_data_path / "parcellation" / info["parcellation"]
if not parc_path.exists():
    parc_path.mkdir()
desikan = abagen.fetch_desikan_killiany(surface=True)
lab_dict = images.load_gifti(desikan["image"][0]).labeltable.get_labels_as_dict()
labs = [l for _, l in lab_dict.items()]
parc = images.relabel_gifti(
    (images.construct_shape_gii(images.load_data(desikan["image"][0]), labels=labs, intent='NIFTI_INTENT_LABEL'), 
     images.construct_shape_gii(images.load_data(desikan["image"][1]), labels=labs, intent='NIFTI_INTENT_LABEL')), 
     background=["Unknown"]
)
parc[0].to_filename(str(parc_path / (info["parcellation"] + "_hemi-L.gii.gz")))
parc[1].to_filename(str(parc_path / (info["parcellation"] + "_hemi-R.gii.gz")))
labs = [l for l in labs if l.lower() not in ["unknown", "medial_wall"]]
labs_left = [f"{i}_LH_CX_{l}" for i,l in enumerate(labs, start=1)]
labs_right = [f"{i}_RH_CX_{l}" for i,l in enumerate(labs, start=35)]
with open(parc_path / (info["parcellation"] + "_hemi-L.txt"), "w") as f:
    f.write("\n".join(labs_left))
with open(parc_path / (info["parcellation"] + "_hemi-R.txt"), "w") as f:
    f.write("\n".join(labs_right))
parc_info.loc[len(parc_info)] = info
# ==================================================================================================

# ==================================================================================================
# Destrieux
print("Destrieux")
info = {
    "parcellation": "parc-Destrieux_n-148_space-fsaverage_res-10k", 
    "n_parcels": 148, 
    "space": "fsaverage",
    "resolution": "10k", 
    "publication": "10.1016/j.neuroimage.2010.06.010",
    "license": "free"
}
destrieux = datasets.fetch_atlas_surf_destrieux()
labs = [l.decode() for l in destrieux['labels']]
parc_left = images.construct_shape_gii(destrieux['map_left'], labels=labs, intent='NIFTI_INTENT_LABEL')
parc_right = images.construct_shape_gii(destrieux['map_right'], labels=labs, intent='NIFTI_INTENT_LABEL')
parc = images.relabel_gifti((parc_left, parc_right), background=['Medial_wall', "Unknown"])
parc[0].to_filename(nispace_data_path / "parcellation" / info["parcellation"] / (info["parcellation"] + "_hemi-L.gii.gz"))
parc[1].to_filename(nispace_data_path / "parcellation" / info["parcellation"] / (info["parcellation"] + "_hemi-R.gii.gz"))
labs.remove("Unknown") 
labs.remove("Medial_wall")
labs_left = [f"{i}_LH_CX_{l}" for i,l in enumerate(labs, start=1)]
labs_right = [f"{i}_RH_CX_{l}" for i,l in enumerate(labs, start=75)]
with open(nispace_data_path / "parcellation" / info["parcellation"] / (info["parcellation"] + "_hemi-L.txt"), "w") as f:
    f.write("\n".join(labs_left))
with open(nispace_data_path / "parcellation" / info["parcellation"] / (info["parcellation"] + "_hemi-R.txt"), "w") as f:
    f.write("\n".join(labs_right))
parc_info.loc[len(parc_info)] = info
# ==================================================================================================

# Save info

parc_info = parc_info.sort_values("parcellation").reset_index(drop=True)
parc_info.to_csv(nispace_data_path / "parcellation" / "metadata.csv", index=None)


# %% Generate distance matrices

parc_info = pd.read_csv(nispace_data_path / "parcellation" / "metadata.csv")

for parc in parc_info.parcellation:
    print(parc)
    parc_path = nispace_data_path / "parcellation" / parc
    
    # Volume
    if not parc_path.is_dir():
        parc_file = images.load_nifti(parc_path)
    # Surface
    else:
        parc_file = (images.load_gifti(parc_path / f"{parc}_hemi-L.gii.gz"),
                     images.load_gifti(parc_path / f"{parc}_hemi-R.gii.gz"))
        
    # distance matrix 
    dist_mat = get_distance_matrix(
        parc_file, 
        parc_space=parc.split("space-")[1].split("_")[0],
        downsample_vol=False,
        centroids=False,
        surf_euclidean=False,
        n_proc=-1,
        dtype=np.float32
    )
    if not isinstance(dist_mat, tuple):
        pd.DataFrame(dist_mat).to_csv(str(parc_path).replace(".nii.gz", ".csv.gz"), header=None, index=None)
    else:
        for mat, hemi in zip(dist_mat, ["L", "R"]):
            pd.DataFrame(mat).to_csv(parc_path / f"{parc}_hemi-{hemi}.csv.gz", header=None, index=None)


# %%
