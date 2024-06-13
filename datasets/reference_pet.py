# %% Init

import sys
import pathlib
import numpy as np
import pandas as pd
from nilearn import image
from nilearn import plotting
from neuromaps.resampling import resample_images
from nilearn.masking import compute_background_mask
from sklearn.preprocessing import minmax_scale

from utils_datasets import parcellate_reference_dataset

# add nispace to path
from nispace.datasets import fetch_template
from nispace.utils import _rm_ext
from nispace.io import write_json
from nispace.modules.constants import _PARCS_NICE

# nispace data path in home dir
nispace_data_path = pathlib.Path.cwd() / "nispace-data"


# %% PET image data --------------------------------------------------------------------------------
# TODO: TAKE LOOK INTO DIFFERENT MNI SPACE VERSIONS AND TRANSFORMATIONS FOR EXACT TRANSFORMATION

# paths
pet_local_info_path = pathlib.Path.home() / "projects" / "atlases" / "atlases.xlsx"
pet_local_path = pathlib.Path.home() / "projects" / "atlases" / "original"
nispace_pet_map_path = nispace_data_path / "reference" / "pet" / "map"

# PET info table
pet_local_info = pd.read_excel(pet_local_info_path)

# only files that can be shared and are obtained from HC
pet_local_info = pet_local_info.query("sharing_okay=='yes' and disorder=='hc'")
print(f"{len(pet_local_info)} PET files for {len(pet_local_info.target.unique())} targets.")

# template for resampling: MNI152 
mni152_mask = image.load_img(fetch_template("mni152", "mask"))
mni152_mask = {
    1: mni152_mask,
    2: image.resample_img(mni152_mask, target_affine=np.diag([2,2,2]), interpolation="nearest"),
    3: image.resample_img(mni152_mask, target_affine=np.diag([3,3,3]), interpolation="nearest")
}

# process files and save with new names
nispace_pet_map_path
pet_files = []
for f in pet_local_info.atlas_file[:]:
    # source file
    f_src = pet_local_path / f
    # dest file
    f_dest = nispace_pet_map_path / \
        "_".join(f"{desc}-{val}" 
                 for desc, val in zip(["target", "tracer", "n", "dx", "pub"], f.split("-")))
    pet_files.append(f_dest)
    
    # load image
    pet_img = image.load_img(f_src)
    
    # get rid of 4th dimension if present
    if len(pet_img.shape)==4:
        pet_img = image.index_img(pet_img, 0)
    #plotting.plot_img(pet_img, title=f"on load {pet_img.shape}", colorbar=True)
    
    # resample to mni152
    voxelsize_pet = pet_img.affine[0,0]
    voxelsize_mni152 = int(np.abs(np.round(voxelsize_pet)))
    print(f"Resampling {f_src.name} with voxelsize {voxelsize_pet} to "
          f"mni152 with voxelsize {voxelsize_mni152}.")
    pet_img, _ = resample_images(
        src=pet_img,
        src_space="mni152",
        trg=mni152_mask[voxelsize_mni152],
        trg_space="mni152",
        method="linear",
        resampling="transform_to_trg"
    )
    #plotting.plot_img(pet_img, title=f"after resampling {pet_img.shape}", colorbar=True)
    
    # get background mask
    mask = compute_background_mask(pet_img)   
    
    # multiply with bain mask
    mask = image.math_img("mask * mni_mask", mask=mask, mni_mask=mni152_mask[voxelsize_mni152])
    
    # rescale
    pet_data = pet_img.get_fdata()
    pet_data[mask.get_fdata()==0] = np.nan
    pet_data = minmax_scale(pet_data.flatten(), (1, 100)).reshape(pet_data.shape)
    pet_data = np.nan_to_num(pet_data)
    pet_img = image.new_img_like(pet_img, pet_data, copy_header=True)
    #plotting.plot_img(pet_img, title=f"after rescaling and masking {pet_img.shape}", colorbar=True)
    
    # change data type
    pet_img = image.new_img_like(pet_img, pet_img.get_fdata().astype(np.float32), copy_header=True)
    
    # save
    print(f"Saving to {f_dest}.")
    pet_img.to_filename(f_dest)

# file info to be stored with nispace data
pet_files_info = pet_local_info[["atlas_file", "target", "tracer", "disorder", "author",
                                 "age", "age_sd", "age_min", "age_max", "doi", "license"]].copy()
pet_files_info = pet_files_info.rename(columns={
    "atlas_file": "atlas", 
    "age": "age_mean", 
    "disorder":"diagnosis"
})
pet_files_info["atlas"] = [f.name for f in pet_files]
pet_files_info = pet_files_info.sort_values("atlas").reset_index(drop=True)
f_dest = nispace_data_path / "reference" / "pet" / "metadata.csv"
pet_files_info.to_csv(f_dest, index=None)
print(f"Saved metadata to {f_dest}.")


# %% PET tabulated data ----------------------------------------------------------------------------

files = list((nispace_data_path / "reference" / "pet" / "map").glob("target-*"))
files.sort()

parcellate_reference_dataset(
    reference_name="pet",
    reference_files=files,
    parcs=_PARCS_NICE,
)


# %% PET collections -------------------------------------------------------------------------------

pet_files = list((nispace_data_path / "reference" / "pet" / "map").glob("target-*"))
pet_files = sorted([_rm_ext(p.name) for p in pet_files])

# All
pd.Series(pet_files, name="map") \
    .to_csv(nispace_data_path / "reference" / "pet" / "collection-All.txt", index=None)
    
# AllTargetSets
pd.DataFrame({
    "set": [f.split("_")[0].split("-")[1] for f in pet_files],
    "x": pet_files,
    "weight": [f.split("_")[2].split("-")[1] for f in pet_files],
}) \
    .to_csv(nispace_data_path / "reference" / "pet" / "collection-AllTargetSets.csv", index=None)
    
# UniqueTracers
collection = {
    "General": [
        'target-CMRglu_tracer-fdg_n-20_dx-hc_pub-castrillon2023',
        'target-CBF_tracer-asl_n-31_dx-hc_pub-holiga2018', 
        'target-SV2A_tracer-ucbj_n-10_dx-hc_pub-finnema2016', 
        'target-HDAC_tracer-martinostat_n-8_dx-hc_pub-wey2016'],
    "Immunity": [
        'target-TSPO_tracer-pbr28_n-6_dx-hc_pub-lois2018', 
        'target-COX1_tracer-ps13_n-11_dx-hc_pub-kim2020'
    ],
    "Glutamate": [
        'target-mGluR5_tracer-abp688_n-73_dx-hc_pub-smart2019', 
        'target-NMDA_tracer-ge179_n-29_dx-hc_pub-galovic2021'
    ],
    "GABA": [
        'target-GABAa5_tracer-ro154513_n-10_dx-hc_pub-lukow2022', 
        'target-GABAa_tracer-flumazenil_n-16_dx-hc_pub-norgaard2020'
    ],
    "Dopamine": [
        'target-FDOPA_tracer-fluorodopa_n-12_dx-hc_pub-garciagomez2018', 
        'target-D1_tracer-sch23390_n-13_dx-hc_pub-kaller2017', 
        'target-D2_tracer-flb457_n-55_dx-hc_pub-sandiego2015', 
        'target-DAT_tracer-fpcit_n-174_dx-hc_pub-dukart2018'
    ],
    "Serotonin": [
        'target-5HT1a_tracer-cumi101_n-8_dx-hc_pub-beliveau2017', 
        'target-5HT1b_tracer-p943_n-65_dx-hc_pub-gallezot2010',
        'target-5HT2a_tracer-cimbi36_n-29_dx-hc_pub-beliveau2017',
        'target-5HT4_tracer-sb207145_n-59_dx-hc_pub-beliveau2017',
        'target-5HT6_tracer-gsk215083_n-30_dx-hc_pub-radhakrishnan2018',
        'target-5HTT_tracer-dasb_n-100_dx-hc_pub-beliveau2017'
    ],
    "Noradrenaline": [
        'target-NET_tracer-mrb_n-77_dx-hc_pub-ding2010',
    ],
    "Actylcholine": [
        'target-A4B2_tracer-flubatine_n-30_dx-hc_pub-hillmer2016',
        'target-M1_tracer-lsn3172176_n-24_dx-hc_pub-naganawa2021',
        'target-VAChT_tracer-feobv_n-18_dx-hc_pub-aghourian2017',
    ],
    "Opiods/Endocannabinoids": [
        'target-MOR_tracer-carfentanil_n-204_dx-hc_pub-kantonen2020',
        'target-KOR_tracer-ly2795050_n-28_dx-hc_pub-vijay2018',
        'target-CB1_tracer-omar_n-77_dx-hc_pub-normandin2015',
    ],
    "Histamine": [
        'target-H3_tracer-gsk189254_n-8_dx-hc_pub-gallezot2017',
    ]
}
write_json(
    collection,
    nispace_data_path / "reference" / "pet" / "collection-UniqueTracers.json"
)

# UniqueTracerSets
collection = [
    'target-5HT1a_tracer-cumi101_n-8_dx-hc_pub-beliveau2017',
    'target-5HT1b_tracer-p943_n-23_dx-hc_pub-savli2012',
    'target-5HT1b_tracer-p943_n-65_dx-hc_pub-gallezot2010',
    'target-5HT2a_tracer-cimbi36_n-29_dx-hc_pub-beliveau2017',
    'target-5HT4_tracer-sb207145_n-59_dx-hc_pub-beliveau2017',
    'target-5HT6_tracer-gsk215083_n-30_dx-hc_pub-radhakrishnan2018',
    'target-5HTT_tracer-dasb_n-100_dx-hc_pub-beliveau2017',
    'target-5HTT_tracer-dasb_n-18_dx-hc_pub-savli2012',
    'target-A4B2_tracer-flubatine_n-30_dx-hc_pub-hillmer2016',
    'target-CB1_tracer-omar_n-77_dx-hc_pub-normandin2015',
    'target-CBF_tracer-asl_n-31_dx-hc_pub-holiga2018',
    'target-CMRglu_tracer-fdg_n-20_dx-hc_pub-castrillon2023',
    'target-COX1_tracer-ps13_n-11_dx-hc_pub-kim2020',
    'target-D1_tracer-sch23390_n-13_dx-hc_pub-kaller2017',
    'target-D2_tracer-flb457_n-37_dx-hc_pub-smith2019',
    'target-D2_tracer-flb457_n-55_dx-hc_pub-sandiego2015',
    'target-DAT_tracer-fpcit_n-174_dx-hc_pub-dukart2018',
    'target-DAT_tracer-fpcit_n-30_dx-hc_pub-garciagomez2013',
    'target-FDOPA_tracer-fluorodopa_n-12_dx-hc_pub-garciagomez2018',
    'target-GABAa_tracer-flumazenil_n-16_dx-hc_pub-norgaard2020',
    'target-GABAa_tracer-flumazenil_n-6_dx-hc_pub-dukart2018',
    'target-GABAa5_tracer-ro154513_n-10_dx-hc_pub-lukow2022',
    'target-H3_tracer-gsk189254_n-8_dx-hc_pub-gallezot2017',
    'target-HDAC_tracer-martinostat_n-8_dx-hc_pub-wey2016',
    'target-KOR_tracer-ly2795050_n-28_dx-hc_pub-vijay2018',
    'target-M1_tracer-lsn3172176_n-24_dx-hc_pub-naganawa2021',
    'target-mGluR5_tracer-abp688_n-22_dx-hc_pub-rosaneto',
    'target-mGluR5_tracer-abp688_n-28_dx-hc_pub-dubois2015',
    'target-mGluR5_tracer-abp688_n-73_dx-hc_pub-smart2019'
    'target-MOR_tracer-carfentanil_n-204_dx-hc_pub-kantonen2020',
    'target-MOR_tracer-carfentanil_n-39_dx-hc_pub-turtonen2021',
    'target-NET_tracer-mrb_n-10_dx-hc_pub-hesse2017',
    'target-NET_tracer-mrb_n-77_dx-hc_pub-ding2010',
    'target-NMDA_tracer-ge179_n-29_dx-hc_pub-galovic2021',
    'target-SV2A_tracer-ucbj_n-10_dx-hc_pub-finnema2016',
    'target-TSPO_tracer-pbr28_n-6_dx-hc_pub-lois2018',
    'target-VAChT_tracer-feobv_n-18_dx-hc_pub-aghourian2017',
    'target-VAChT_tracer-feobv_n-4_dx-hc_pub-tuominen',
    'target-VAChT_tracer-feobv_n-5_dx-hc_pub-bedard2019',
 ]
pd.DataFrame({
    "set": [f.split("_")[0].split("-")[1] for f in collection],
    "x": collection,
    "weight": [f.split("_")[2].split("-")[1] for f in collection],
}) \
    .to_csv(nispace_data_path / "reference" / "pet" / "collection-UniqueTracerSets.csv", index=None)

# %%
