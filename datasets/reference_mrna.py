# %% Init

import sys
import pathlib
import numpy as np
import pandas as pd
import zipfile
from abagen import get_expression_data

from utils_datasets import download

from nispace.modules.constants import _PARCS_NICE
from nispace.datasets import fetch_parcellation
from nispace.io import write_json

# nispace data path
nispace_data_path = pathlib.Path.cwd() / "nispace-data"


# %% mRNA tabulated data ---------------------------------------------------------------------------

for parc in _PARCS_NICE: 
    print(parc)
    parc_loaded, parc_labels = fetch_parcellation(parc, return_loaded=True)
    
    # parc info
    parc_info = pd.DataFrame({
        "id": [int(l.split("_")[0]) for l in parc_labels],
        "label": parc_labels,
        "hemisphere": [l.split("_")[1][0] for l in parc_labels],
        "structure": ["cortex" if "_CX_" in l else "subcortex/brainstem" for l in parc_labels]
    })
        
    # get data
    mRNA_tab = get_expression_data(
        atlas=parc_loaded,
        atlas_info=parc_info,
        lr_mirror="bidirectional",
        n_proc=1,
        verbose=1      
    )
    mRNA_tab.index = parc_info.label
    mRNA_tab = mRNA_tab.T
    mRNA_tab.index.name = "map"
    mRNA_tab = mRNA_tab.astype(np.float32)
    
    # save
    mRNA_tab.to_csv(nispace_data_path / "reference" / "mrna" / "tab" / 
                    f"mrna_{parc}.csv.gz")
    
# %% Collections (= gene sets) ---------------------------------------------------------------------
# TODO: new ABA cell types, brainspan, add weighted cell types
# TODO: add GWAS from PGC after mapping to genes ("35 kb upstream and 10 kb downstream")

# All genes
mRNA_tab_files = (nispace_data_path / "reference" / "mrna" / "tab").glob("*.csv.gz")
all_genes = set()
for f in mRNA_tab_files:
    all_genes.update(pd.read_csv(f, index_col=0).index.unique())
all_genes = pd.Series(sorted(list(all_genes)), name="map")
all_genes.to_csv(nispace_data_path / "reference" / "mrna" / "collection-All.txt", index=False)
 
# PsychEncode cell types: Darmanis 2015 / Lake 2016 vs.  Lake 2018
for collection, save_name in zip(
    [pd.read_excel("http://resource.psychencode.org/Datasets/Derived/SC_Decomp/DER-19_Single_cell_markergenes_TPM.xlsx") \
        .rename(columns=dict(GeneName="gene", CellType="set")),
     pd.read_excel("http://resource.psychencode.org/Datasets/Derived/SC_Decomp/DER-21_Single_cell_markergenes_UMI.xlsx", header=1) \
        .rename(columns=dict(Gene="gene", Cluster="set"))],
    ["CellTypesPsychEncodeTPM", 
     "CellTypesPsychEncodeUMI"]
):
    collection = collection.astype(str)
    collection = {k: sorted(collection.query("set==@k").gene.unique()) for k in collection.set.unique()}
    all_genes = sum([collection[k] for k in collection], [])
    print(len(collection), "sets,", len(all_genes), "genes,", len(set(all_genes)), "unique.")
    write_json(collection, nispace_data_path / "reference" / "mrna" / f"collection-{save_name}.json")

# SynGO
url = "https://syngoportal.org/data/SynGO_bulk_download_release_20231201.zip"
path = download(url)
zip_file = zipfile.ZipFile(path)
with zip_file.open("syngo_ontologies.xlsx") as file:
    df = pd.read_excel(file)
collection = {name: genes.split(", ") 
              for id, name, genes in zip(df["id"], df["name"], df["hgnc_symbol"])}
all_genes = sum([collection[k] for k in collection], [])
print(len(collection), "sets,", len(all_genes), "genes,", len(set(all_genes)), "unique.")
write_json(collection, nispace_data_path / "reference" / "mrna" / f"collection-SynGO.json")

# Chromosome location
# for now, get from ABAnnotate (original source: DAVID)
df = pd.read_csv("/Users/llotter/projects/ABAnnotate/raw_datasets/DAVID/OFFICIAL_GENE_SYMBOL2CHROMOSOME.txt", sep="\t", header=None)
df.columns = ["gene", "chrom"]
sets = [str(i) for i in np.arange(1,23,1)] + ["X","Y"]
collection = {k: sorted(df.query("chrom==@k").gene.unique()) for k in sets}
all_genes = sum([collection[k] for k in collection], [])
print(len(collection), "sets,", len(all_genes), "genes,", len(set(all_genes)), "unique.")
write_json(collection, nispace_data_path / "reference" / "mrna" / f"collection-Chromosome.json")

# %%
