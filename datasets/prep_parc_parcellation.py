# %% Init

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from neuromaps import images

wd = Path.cwd().parent
print(f"Working dir: {wd}")
sys.path.append(wd.as_posix())

# import NiSpace functions
from nispace.utils.utils_datasets import download

# nispace data path 
nispace_source_data_path = wd / "datasets" / "nispace-data_source"


# %% Get parcellations

# parcellation info
parc_info = {}
#pd.DataFrame(columns=["parcellation", "n_parcels", "space", "resolution", "publication"])


# ==================================================================================================
# Schaefer + Melbourne

for schaefer, tian in [(100, "S1"), (200, "S2"), (400, "S3")]:
    print(f"Schaefer {schaefer} + Melbourne {tian}")
    
    # name
    name = f"Schaefer{schaefer}Melbourne{tian}"
    
    # labels Melbourne
    labs_tian = pd.read_csv(
        f"https://github.com/yetianmed/subcortex/raw/master/Group-Parcellation/3T/Subcortex-Only/"
        f"Tian_Subcortex_{tian}_3T_label.txt", 
        header=None
    )[0].to_list()
    
    # labels Schaefer
    labs_schaefer = pd.read_csv(
        f"https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/brain_parcellation/"
        f"Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/"
        f"Schaefer2018_{schaefer}Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv"
    )["ROI Name"].to_list()
    
    # combine and rename
    labs = []
    for l in labs_tian:
        l = l.split("-")
        labs.append(l[-1].upper() + "_SC_" + "-".join(l[:-1]))
    for l in labs_schaefer:
        l = l.split("_")
        labs.append(l[1] + "_CX_" + "_".join(l[2:]))
    labs = [f"{i}_{l}" for i, l in enumerate(labs, start=1)]
    
    # write info and labels, we have the same labels (obiously) for both resolutions
    for space in ["MNI152NLin2009cAsym", "MNI152NLin6Asym"]:
        parc_info[name, space] = {
            "n_parcels": len(labs), 
            "resolution": "1mm", 
            "publication": "10.1093/cercor/bhx179; 0.1038/s41593-020-00711-6",
            "license": "MIT"
        }
        # save labels
        path = nispace_source_data_path / "parcellation" / name / space / f"parc-{name}_space-{space}.label.txt"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(labs))
        
# ==================================================================================================


# ==================================================================================================
# HCP

print("HCPex")

# name
name = "HCPex"
space = "MNI152NLin2009cAsym"

# labels
labs = np.loadtxt(
    download("https://github.com/wayalan/HCPex/raw/main/HCPex_v1.1/HCPex.nii.txt"), 
    str
).tolist()
labs = [f"{l[0]}_{l[1]}H_{'CX' if int(l[0]) < 361 else 'SC'}_{l[2]}" for l in labs]

# info
# space: "MNI152NLin2009cAsym"
parc_info[name, space] = {
    "n_parcels": len(labs), 
    "resolution": "1mm", 
    "publication": "10.1038/nature18933; 10.1007/s00429-021-02421-6",
    "license": "GPL-3.0"
}
# save labels
path = nispace_source_data_path / "parcellation" / name / space / f"parc-{name}_space-{space}.label.txt"
if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
with open(path, "w") as f:
    f.write("\n".join(labs))

# ==================================================================================================


# ==================================================================================================
# DesikanKilliany

print("DesikanKilliany")

# name
name = "DesikanKilliany"
space = "fsaverage"

# load parcellation from templateflow
parc = (
    images.load_gifti(download("https://templateflow.s3.amazonaws.com/tpl-fsaverage/tpl-fsaverage_hemi-L_den-10k_atlas-Desikan2006_seg-aparc_dseg.label.gii")),
    images.load_gifti(download("https://templateflow.s3.amazonaws.com/tpl-fsaverage/tpl-fsaverage_hemi-R_den-10k_atlas-Desikan2006_seg-aparc_dseg.label.gii"))
)

# get labels
labs = [l for _, l in parc[0].labeltable.get_labels_as_dict().items()]
labs_bg = ["unknown", "corpuscallosum"]

# relabel giftis
parc = images.relabel_gifti(
    (images.construct_shape_gii(images.load_data(parc[0]), labels=labs, intent='NIFTI_INTENT_LABEL'), 
     images.construct_shape_gii(images.load_data(parc[1]), labels=labs, intent='NIFTI_INTENT_LABEL')), 
     background=labs_bg
)

# info
parc_info[name, space] = {
    "n_parcels": len([l for l in labs if l != "unknown"]) * 2, 
    "resolution": "41k", 
    "publication": "https://doi.org/10.1016/j.neuroimage.2006.01.021",
    "license": "free"
}

# save maps and labels
save_dir = nispace_source_data_path / "parcellation" / name / space
if not save_dir.exists():
    save_dir.mkdir(parents=True, exist_ok=True)
# left
parc[0].to_filename(save_dir / f"parc-{name}_space-{space}_hemi-L.label.gii.gz")
with open(save_dir / f"parc-{name}_space-{space}_hemi-L.label.txt", "w") as f:
    f.write("\n".join([f"{i}_LH_CX_{l}" for i, l in enumerate([l for l in labs if l not in labs_bg], start=1)]))
# right
parc[1].to_filename(save_dir / f"parc-{name}_space-{space}_hemi-R.label.gii.gz")
with open(save_dir / f"parc-{name}_space-{space}_hemi-R.label.txt", "w") as f:
    f.write("\n".join([f"{i}_RH_CX_{l}" for i, l in enumerate([l for l in labs if l not in labs_bg], start=35)]))

# ==================================================================================================


# ==================================================================================================

# Destrieux
print("Destrieux")
# name
name = "Destrieux"
space = "fsaverage"

# load parcellation from templateflow
parc = (
    images.load_gifti(download("https://templateflow.s3.amazonaws.com/tpl-fsaverage/tpl-fsaverage_hemi-L_den-10k_atlas-Destrieux2009_dseg.label.gii")),
    images.load_gifti(download("https://templateflow.s3.amazonaws.com/tpl-fsaverage/tpl-fsaverage_hemi-R_den-10k_atlas-Destrieux2009_dseg.label.gii"))
)

# get labels
labs = [l for l in parc[0].labeltable.get_labels_as_dict().values()]
labs_bg = ["Unknown", "Medial_wall"]

# relabel giftis
parc = images.relabel_gifti(
    (images.construct_shape_gii(images.load_data(parc[0]), labels=labs, intent='NIFTI_INTENT_LABEL'), 
     images.construct_shape_gii(images.load_data(parc[1]), labels=labs, intent='NIFTI_INTENT_LABEL')), 
     background=labs_bg
)

# info
parc_info[name, space] = {
    "n_parcels": len([l for l in labs if l != "unknown"]) * 2, 
    "resolution": "41k", 
    "publication": "10.1016/j.neuroimage.2010.06.010",
    "license": "free"
}

# save maps and labels
save_dir = nispace_source_data_path / "parcellation" / name / space
if not save_dir.exists():
    save_dir.mkdir(parents=True, exist_ok=True)
# left
parc[0].to_filename(save_dir / f"parc-{name}_space-{space}_hemi-L.label.gii.gz")
with open(save_dir / f"parc-{name}_space-{space}_hemi-L.label.txt", "w") as f:
    f.write("\n".join([f"{i}_LH_CX_{l}" for i, l in enumerate([l for l in labs if l not in labs_bg], start=1)]))
# right
parc[1].to_filename(save_dir / f"parc-{name}_space-{space}_hemi-R.label.gii.gz")
with open(save_dir / f"parc-{name}_space-{space}_hemi-R.label.txt", "w") as f:
    f.write("\n".join([f"{i}_RH_CX_{l}" for i, l in enumerate([l for l in labs if l not in labs_bg], start=75)]))

# ==================================================================================================

# Save info

parc_info = pd.DataFrame.from_dict(parc_info).T
parc_info.index.names = ["parcellation", "space"]
parc_info.to_csv(nispace_source_data_path / "parcellation" / "metadata.csv")

# ==================================================================================================


# %%