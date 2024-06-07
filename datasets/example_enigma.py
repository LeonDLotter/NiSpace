# %% Init

import sys
import pathlib
import numpy as np
import pandas as pd


# nispace data path in home dir
nispace_data_path = pathlib.Path.home() / "nispace-data"


# %% Get ENIGMA summary stats

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

# %% Save

# save parcellated data
enigma_tab.T.to_csv(nispace_data_path / "example" / "example-enigma_parc-DesikanKilliany.csv.gz")    
 

# %%
