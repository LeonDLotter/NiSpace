# %% Init

import sys
import pathlib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from nilearn.datasets import fetch_abide_pcp

# add nispace to path
from nispace.datasets import fetch_parcellation, fetch_reference
from nispace.io import parcellate_data

# nispace data path in home dir
nispace_data_path = pathlib.Path.cwd() / "nispace-data"


# %% EXAMPLE: ABIDE --------------------------------------------------------------------------------

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

# parcellation to use
parc_name = "Schaefer200"

# get parcellation
parc, parc_labels = fetch_parcellation(parc_name, return_loaded=True)

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
    min_num_valid_datapoints=5,
    min_fraction_valid_datapoints=0.1,
    n_proc=-1,
    dtype=np.float32,
)

# save parcellated data
abide_tab.to_csv(nispace_data_path / "example" / f"example-abide_parc-{parc_name}.csv.gz")
# save phenotypic data
data_pheno.to_csv(nispace_data_path / "example" / "example-abide_info.csv", index=False)
    
 
# %% EXAMPLE: ENIGMA -------------------------------------------------------------------------------

# Most from ENIGMA TOOLBOX on GitHub
disorders = {
    "MDD": "mddadult_case-controls_CortThick.csv",
    "PTSD": None,
    "AN": None,
    "ADHD": "adhdallages_case-controls_CortThick.csv",
    "ASD": "asd_mega-analysis_case-controls_CortThick.csv",
    "OCD": "ocdadults_case-controls_CortThick.csv",
    "BD": "bd_case-controls_CortThick_adult.csv",
    "SCZ": "scz_case-controls_CortThick.csv",
    "22q11.2": "22q_case-controls_CortThick.csv",
    "Epilepsy - all": "allepi_case-controls_CortThick.csv",
    "Epilepsy - temporal": "gge_case-controls_CortThick.csv",
    "Epilepsy - generalized": "tlemtsl_case-controls_CortThick.csv",
    "PD": None
}
enigma_tab = pd.DataFrame(columns=disorders.keys())
for disorder, file in disorders.items():
    if file is not None:
        print(disorder)
        tab = pd.read_csv("https://github.com/MICA-MNI/ENIGMA/raw/master/enigmatoolbox/"
                        f"datasets/summary_statistics/{file}")
        enigma_tab[disorder] = tab.set_index("Structure")["d_icv"]
    
# Add PTSD data from https://doi.org/10.21203/rs.3.rs-2085479/v1
print("PTSD")
enigma_tab["PTSD"] = pd.read_csv("_archive/enigma_ptsd_case-controls_CortThick.csv")\
    .set_index("Structure")["d"]

# Add Parkinson data from https://doi.org/10.1002/mds.28706
print("PD")
tab = pd.read_csv("_archive/enigma_pd_case-controls_CortThick.csv")
tab.Structure = [r.split(" ")[0] + "_" + "".join(r.split(" ")[1:]).lower()
                 for r in tab.Structure]
enigma_tab["PD"] = tab.set_index("Structure")["d"]

# Add Anorexia data from 
print("AN")
tab = pd.read_csv("_archive/enigma_an_case-controls_CortThick.csv")
tab.Structure = [r.replace("_thickavg","") for r in tab.Structure]
enigma_tab["AN"] = tab.set_index("Structure")["d"]

# Check
if enigma_tab.isna().sum().sum() > 0:
    print("NaN in table!")
    print(enigma_tab.isna().sum())

# Dtype
enigma_tab = enigma_tab.round(3).astype(np.float32)

# save parcellated data
enigma_tab.T.to_csv(nispace_data_path / "example" / "example-enigma_parc-DesikanKilliany.csv.gz")    


# %% EXAMPLE: HAPPY --------------------------------------------------------------------------------

# parcellation we will use
parc = "Schaefer200"

# number of subjects for each group (happy vs normal)
n_subs = 50

# get happy source data
tab_happy = fetch_reference(
    "pet",
    maps=["MOR", "KOR", "CB1"],
    parcellation=parc,
    nispace_data_dir=nispace_data_path
)

# get all data
tab_all = fetch_reference(
    "pet",
    parcellation=parc,
    nispace_data_dir=nispace_data_path
)

# generate our 100 subjects
# dataframe: first 50 are happy, second 50 are random
data_happy = pd.DataFrame(
    columns=tab_happy.columns, 
    index=[f"sub-{i:03d}H" for i in range(1, n_subs+1)] + [f"sub-{i:03d}C" for i in range(n_subs+1, n_subs*2+1)]
)

rng = np.random.default_rng(42)  
for i in range(n_subs):  
    
    # get a randomly weighted combination of our happy maps
    weights = rng.random(tab_happy.shape[0]) + 0.5
    data = ( tab_happy.T * weights ).mean(axis=1)
    data_happy.iloc[i, :] = data

    # get the mean of random subsets of the whole dataset -> "realistic noise"
    
    subset = rng.choice(tab_all.index, 30)
    data = ( tab_all.loc[subset, :].T ).mean(axis=1) + rng.random(tab_all.shape[1])
    data_happy.iloc[i + n_subs, :] = data

# save
data_happy.to_csv(nispace_data_path / "example" / f"example-happy_parc-{parc}.csv.gz")


# Test Happy Data

# from nispace import NiSpace

# coloc = "spearman"
# stat = "rho"
# group_comparison = "zscore(a,b)" 

# nsp = NiSpace(
#     x=fetch_reference("pet", collection="UniqueTracer", parcellation=parc, nispace_data_dir=nispace_data_path),
#     y=data_happy,
#     parcellation=parc,
#     n_proc=8,
# ).fit()

# nsp.transform_y(group_comparison, groups=[0]*n_subs + [1]*n_subs)
# nsp.colocalize(coloc, Y_transform=group_comparison)
# nsp.permute("groups", coloc, Y_transform=group_comparison, n_perm=1000)
# nsp.plot(
#     method=coloc, 
#     Y_transform=group_comparison,
#     plot_kwargs={"sort_categories": False},
#     permute_what="groups"
# )


# %%
