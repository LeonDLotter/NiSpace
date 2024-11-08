# %% Init

import sys
from pathlib import Path
import numpy as np
from nilearn import image

wd = Path.cwd().parent
print(f"Working dir: {wd}")
sys.path.append(wd.as_posix())

from nispace.io import parcellate_data
from nispace.modules.constants import _PARCS_NICE
from nispace.datasets import fetch_template
#from nispace.utils.utils_datasets import parcellate_reference_dataset

# nispace data path 
nispace_source_data_path = wd / "datasets" / "nispace-data_source"


# %% MNI152NLin2009cAsym and MNI152NLin6Asym

# MNI152 templates in 1 and 2 mm resolution are fetched directly from templateflow.
# For MNI152NLin2009cAsym, we have T1w, brain, mask, and gmprob.
# For MNI152NLin6Asym, we have T1w, brain, mask.
# We will only generate 3mm templates from the 1mm templateflow version

# MNI152NLin2009cAsym - 3mm versions
for desc in ["T1w", "brain", "mask", "gmprob"]:
    tpl = fetch_template("MNI152NLin2009cAsym", res="1mm", desc=desc)
    tpl_resampled = image.resample_img(
        image.load_img(tpl),
        target_affine=np.diag([3, 3, 3]), 
        interpolation="nearest" if desc == "mask" else "linear"
    )
    path = nispace_source_data_path / "template" / "MNI152NLin2009cAsym" / "map" / desc / tpl.name.replace("1mm", "3mm")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {path}")
    tpl_resampled.to_filename(path)

# MNI152NLin6Asym - 3mm versions
for desc in ["T1w", "brain", "mask"]:
    tpl = fetch_template("MNI152NLin6Asym", res="1mm", desc=desc)
    tpl_resampled = image.resample_img(
        image.load_img(tpl),
        target_affine=np.diag([3, 3, 3]), 
        interpolation="nearest" if desc == "mask" else "linear"
    )
    path = nispace_source_data_path / "template" / "MNI152NLin6Asym" / "map" / desc / tpl.name.replace("1mm", "3mm")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {path}")
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
