# %% Init

import sys
import pathlib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from nilearn.datasets import fetch_abide_pcp

# add nispace to path
sys.path.append(str(pathlib.Path.cwd().parent))
from nispace.datasets import fetch_parcellation
from nispace.io import parcellate_data
from nispace import NiSpace

# nispace data path in home dir
nispace_data_path = pathlib.Path.home() / "nispace-data"


# %% Download ABIDE data from Nilearn
deriv = "falff"

data = fetch_abide_pcp(
    data_dir=None, 
    n_subjects=None, 
    pipeline="ccs", 
    band_pass_filtering=True, 
    global_signal_regression=False, 
    derivatives=[deriv], 
    quality_checked=True, 
    verbose=1, 
    legacy_format=False, 
)

data_description = data["description"]
print(data_description)

data_rsfmri = data[deriv]
print(data_rsfmri[:3])

data_pheno = data["phenotypic"] \
    .loc[:, ["SUB_ID", "SITE_ID", "DX_GROUP", "DSM_IV_TR", 
             "qc_rater_1", "qc_func_rater_2", "qc_func_rater_3", 
             "AGE_AT_SCAN", "SEX", 
             "ADI_R_SOCIAL_TOTAL_A", "ADI_R_VERBAL_TOTAL_BV", "ADI_RRB_TOTAL_C", 
             "ADOS_TOTAL", "SRS_RAW_TOTAL", "SCQ_TOTAL", "AQ_TOTAL"]] \
    .rename(columns={
        "SUB_ID": "subject", 
        "SITE_ID": "site", 
        "DX_GROUP": "dx_num", 
        "DSM_IV_TR": "dsm_iv_tr", 
        "AGE_AT_SCAN": "age", 
        "SEX": "sex_num", 
        "ADI_R_SOCIAL_TOTAL_A": "adi_r_social_total_a", 
        "ADI_R_VERBAL_TOTAL_BV": "adi_r_verbal_total_bv", 
        "ADI_RRB_TOTAL_C": "adi_rrb_total_c", 
        "ADOS_TOTAL": "ados_total", 
        "SRS_RAW_TOTAL": "srs_raw_total", 
        "SCQ_TOTAL": "scq_total", 
        "AQ_TOTAL": "aq_total"
    })
data_pheno["dx"] = data_pheno["dx_num"].map({1: "ASD", 2: "CTRL"})
data_pheno["sex"] = data_pheno["sex_num"].map({1: "M", 2: "F"})
data_pheno["site_num"] = pd.Categorical(data_pheno["site"]).codes
data_pheno = data_pheno[["subject", "site", "site_num", "dx", "dx_num", "dsm_iv_tr", "age", "sex", "sex_num",
                         "qc_rater_1", "qc_func_rater_2", "qc_func_rater_3", 
                         "adi_r_social_total_a", "adi_r_verbal_total_bv", "adi_rrb_total_c", 
                         "ados_total", "srs_raw_total", "scq_total", "aq_total"]]
data_pheno = data_pheno.replace(-9999, np.nan)
print(data_pheno.head(5))

# %% Parcellate

# parcellation
parc, parc_labels = fetch_parcellation("Schaefer200MelbourneS1")

 # parcellate  
abide_tab = parcellate_data(
    data=data_rsfmri,
    data_labels=data_pheno["subject"],
    data_space="MNI152",
    parcellation=parc,
    parc_labels=parc_labels,
    parc_space="MNI152",
    resampling_target="data",
    drop_background_parcels=True,
    min_num_valid_datapoints=10,
    min_fraction_valid_datapoints=0.1,
    n_proc=-1,
    dtype=np.float32,
)

# %% Save

# save parcellated data
abide_tab.to_csv(nispace_data_path / "example" / "example-abide_parc-Schaefer200MelbourneS1.csv.gz")
# save phenotypic data
data_pheno.to_csv(nispace_data_path / "example" / "example-abide_info.csv", index=False)
    
 

# %%
