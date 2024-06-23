# %% Init

import sys
import pathlib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# add nispace to path
sys.path.append(str(pathlib.Path.cwd().parent))
from nispace.datasets import fetch_reference
from nispace.stats.misc import zscore_df
from nispace import NiSpace

# nispace data path in home dir
nispace_data_path = pathlib.Path.cwd() / "nispace-data"


# %% Create fake tabulated dataset for testing: The "Happy Data"

# parcellation we will use
parc = "Schaefer200MelbourneS2"

# number of subjects for each group (happy vs normal)
n_subs = 50

# get happy source data
tab_happy = fetch_reference(
    "pet",
    maps=["MOR", "KOR", "CB1"],
    parcellation=parc
)

# get all data
tab_all = fetch_reference(
    "pet",
    parcellation=parc
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


# %% Test Happy Data
coloc = "spearman"
stat = "rho"
group_comparison = "zscore(a,b)" 

nsp = NiSpace(
    x=fetch_reference("pet", collection="UniqueTracer", parcellation=parc),
    y=data_happy,
    parcellation=parc,
    n_proc=8
).fit()
nsp.transform_y(group_comparison, groups=[0]*n_subs + [1]*n_subs)
nsp.colocalize(coloc, Y_transform=group_comparison)
nsp.permute("groups", coloc, Y_transform=group_comparison, n_perm=1000)
nsp.plot(
    method=coloc, 
    Y_transform=group_comparison,
    plot_kwargs={"sort_categories": False},
    permute_what="groups"
)


# %%
