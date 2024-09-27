# %% Init

import sys
import pathlib
import numpy as np
import pandas as pd
import zipfile
from abagen import get_expression_data
from tqdm.auto import tqdm
from itertools import combinations
from joblib import Parallel, delayed

from nispace.utils.utils_datasets import download

from nispace.modules.constants import _PARCS_NICE
from nispace.datasets import fetch_parcellation
from nispace.stats.coloc import corr
from nispace.io import load_labels

# nispace data path
nispace_source_data_path = pathlib.Path.cwd() / "nispace-data"


# %% mRNA tabulated data ---------------------------------------------------------------------------
corr_threshold = 0.2

def par_fun(parc):
        
    # space
    if parc in ["Destrieux", "DesikanKilliany"]:
        space = "fsaverage"
    else:
        space = "mni152"

    # load parc
    parc_path, labels_path = fetch_parcellation(parc, space=space, return_loaded=False)
    if isinstance(parc_path, tuple):
        parc_path = (str(parc_path[0]), str(parc_path[1]))
    else:
        parc_path = str(parc_path)
    parc_labels = load_labels(labels_path)

    # parc info
    parc_info = pd.DataFrame({
        "id": [int(l.split("_")[0]) for l in parc_labels],
        "label": parc_labels,
        "hemisphere": [l.split("_")[1][0] for l in parc_labels],
        "structure": ["cortex" if "_CX_" in l else "subcortex/brainstem" for l in parc_labels]
    })
        
    # get data for each donor
    mRNA_dict = get_expression_data(
        atlas=parc_path,
        atlas_info=parc_info,
        lr_mirror="bidirectional",
        n_proc=1,
        verbose=False, # 1
        return_donors=True 
    )

    # get donor combinations
    donor_combinations = list(combinations(mRNA_dict.keys(), 2))

    # get crosscorr matrix
    gene_crosscorr = pd.DataFrame(
        columns=pd.MultiIndex.from_tuples(donor_combinations, names=["donor1", "donor2"]), 
        index=pd.Index(mRNA_dict["9861"].columns, name="map"),
        dtype=np.float16
    )
    for gene in gene_crosscorr.index:
        for comb in donor_combinations:
            df = pd.concat([mRNA_dict[comb[0]][gene], mRNA_dict[comb[1]][gene]], axis=1).dropna()
            if len(df) > 1:
                gene_crosscorr.loc[gene, comb] = corr(df.values[:,0], df.values[:,1], rank=True) 
            else:
                gene_crosscorr.loc[gene, comb] = np.nan

    # get combined data for all donors        
    mRNA_tab = get_expression_data(
        atlas=parc_path,
        atlas_info=parc_info,
        lr_mirror="bidirectional",
        n_proc=1,
        verbose=False, # 1
    )

    # process dataset        
    mRNA_tab.index = parc_info.label
    mRNA_tab = mRNA_tab.T
    mRNA_tab.index.name = "map"
    mRNA_tab = mRNA_tab.astype(np.float32)

    # subset dataset
    n_genes_prior = mRNA_tab.shape[0]
    genes_overthresh = gene_crosscorr.loc[gene_crosscorr.mean(axis=1) > corr_threshold].index
    mRNA_tab = mRNA_tab.loc[genes_overthresh]
    print(f"Parcellation: {parc}. Originally {n_genes_prior} genes.\n"
            f"After correlation threshold of >= {corr_threshold}, {mRNA_tab.shape[0]} genes remain.")

    # save
    mRNA_tab.to_csv(nispace_source_data_path / "reference" / "mrna" / "tab" / 
                    f"mrna_{parc}.csv.gz")
    gene_crosscorr.to_csv(nispace_source_data_path / "reference" / "mrna" / "tab" / 
                            f"mrna_crosscorr_{parc}.csv.gz")

# Run in parallel
Parallel(n_jobs=-1)(
    delayed(par_fun)(parc) 
    for parc in tqdm(_PARCS_NICE)
)

    

# %%
