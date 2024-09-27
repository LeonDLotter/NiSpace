# %% imports ---------------------------------------------------------------------------------------

import pathlib
import zipfile
import numpy as np
import pandas as pd
import requests
import io
from nispace.datasets import fetch_reference
from nispace.io import write_json
from nispace.utils.utils import _rm_ext
from nispace.utils.utils_datasets import download

# nispace data path
nispace_source_data_path = pathlib.Path.cwd() / "nispace-data"


# %% PET collections -------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

pet_files = fetch_reference("pet")
pet_files = [_rm_ext(f.name) for f in pet_files]

# All
pd.Series(pet_files, name="map") \
    .to_csv(nispace_source_data_path / "reference" / "pet" / "collection-All.collect", index=None)
    
# AllTargetSets
pd.DataFrame({
    "set": [f.split("_")[0].split("-")[1] for f in pet_files],
    "x": pet_files,
    "weight": [f.split("_")[2].split("-")[1] for f in pet_files],
}) \
    .to_csv(nispace_source_data_path / "reference" / "pet" / "collection-AllTargetSets.collect", index=None)
    
# UniqueTracers
collection = {
    "General": [
        "target-CMRglu_tracer-fdg_n-20_dx-hc_pub-castrillon2023_space-mni152",
        # "CBF"
        "target-SV2A_tracer-ucbj_n-10_dx-hc_pub-finnema2016_space-mni152",
        "target-HDAC_tracer-martinostat_n-8_dx-hc_pub-wey2016_space-mni152"
    ],
    "Immunity": [
        'target-TSPO_tracer-pbr28_n-6_dx-hc_pub-lois2018_space-mni152', 
        'target-COX1_tracer-ps13_n-11_dx-hc_pub-kim2020_space-mni152'
    ],
    "Glutamate": [
        "target-mGluR5_tracer-abp688_n-73_dx-hc_pub-smart2019_space-mni152",
        "target-NMDA_tracer-ge179_n-29_dx-hc_pub-galovic2021_space-mni152"
    ],
    "GABA": [
        "target-GABAa5_tracer-ro154513_n-10_dx-hc_pub-lukow2022_space-mni152",
        "target-GABAa_tracer-flumazenil_n-6_dx-hc_pub-dukart2018_space-mni152",
    ],
    "Dopamine": [
        "target-FDOPA_tracer-fluorodopa_n-12_dx-hc_pub-garciagomez2018_space-mni152",
        "target-D1_tracer-sch23390_n-13_dx-hc_pub-kaller2017_space-mni152",
        "target-D2_tracer-flb457_n-55_dx-hc_pub-sandiego2015_space-mni152",
        "target-DAT_tracer-fpcit_n-174_dx-hc_pub-dukart2018_space-mni152",
    ],
    "Serotonin": [
        "target-5HT1a_tracer-way100635_n-35_dx-hc_pub-savli2012_space-mni152",
        "target-5HT1b_tracer-p943_n-23_dx-hc_pub-savli2012_space-mni152",
        "target-5HT2a_tracer-altanserin_n-19_dx-hc_pub-savli2012_space-mni152",
        "target-5HT4_tracer-sb207145_n-59_dx-hc_pub-beliveau2017_space-mni152",
        "target-5HT6_tracer-gsk215083_n-30_dx-hc_pub-radhakrishnan2018_space-mni152",
        "target-5HTT_tracer-dasb_n-18_dx-hc_pub-savli2012_space-mni152",
    ],
    "Noradrenaline/Acetylcholine": [
        "target-NET_tracer-mrb_n-77_dx-hc_pub-ding2010_space-mni152",
        "target-A4B2_tracer-flubatine_n-30_dx-hc_pub-hillmer2016_space-mni152",
        "target-M1_tracer-lsn3172176_n-24_dx-hc_pub-naganawa2020_space-mni152",
        "target-VAChT_tracer-feobv_n-18_dx-hc_pub-aghourian2017_space-mni152",
    ],
    "Opiods/Endocannabinoids": [
        "target-MOR_tracer-carfentanil_n-204_dx-hc_pub-kantonen2020_space-mni152",
        "target-KOR_tracer-ly2795050_n-28_dx-hc_pub-vijay2018_space-mni152",
        "target-CB1_tracer-omar_n-77_dx-hc_pub-normandin2015_space-mni152",
    ],
    "Histamine": [
        "target-H3_tracer-gsk189254_n-8_dx-hc_pub-gallezot2017_space-mni152",
    ]
}
write_json(
    collection,
    nispace_source_data_path / "reference" / "pet" / "collection-UniqueTracers.collect"
)

# UniqueTracerSets
collection = [
    'target-5HT1a_tracer-way100635_n-35_dx-hc_pub-savli2012_space-mni152',
    'target-5HT1b_tracer-p943_n-23_dx-hc_pub-savli2012_space-mni152',
    'target-5HT1b_tracer-p943_n-65_dx-hc_pub-gallezot2010_space-mni152',
    'target-5HT2a_tracer-altanserin_n-19_dx-hc_pub-savli2012_space-mni152',
    'target-5HT4_tracer-sb207145_n-59_dx-hc_pub-beliveau2017_space-mni152',
    'target-5HT6_tracer-gsk215083_n-30_dx-hc_pub-radhakrishnan2018_space-mni152',
    'target-5HTT_tracer-dasb_n-100_dx-hc_pub-beliveau2017_space-mni152',
    'target-5HTT_tracer-dasb_n-18_dx-hc_pub-savli2012_space-mni152',
    'target-A4B2_tracer-flubatine_n-30_dx-hc_pub-hillmer2016_space-mni152',
    'target-CB1_tracer-omar_n-77_dx-hc_pub-normandin2015_space-mni152',
    'target-CMRglu_tracer-fdg_n-20_dx-hc_pub-castrillon2023_space-mni152',
    'target-COX1_tracer-ps13_n-11_dx-hc_pub-kim2020_space-mni152',
    'target-D1_tracer-sch23390_n-13_dx-hc_pub-kaller2017_space-mni152',
    'target-D2_tracer-flb457_n-37_dx-hc_pub-smith2017_space-mni152',
    'target-D2_tracer-flb457_n-55_dx-hc_pub-sandiego2015_space-mni152',
    'target-DAT_tracer-fpcit_n-174_dx-hc_pub-dukart2018_space-mni152',
    'target-DAT_tracer-fpcit_n-30_dx-hc_pub-garciagomez2013_space-mni152',
    'target-FDOPA_tracer-fluorodopa_n-12_dx-hc_pub-garciagomez2018_space-mni152',
    'target-GABAa_tracer-flumazenil_n-16_dx-hc_pub-norgaard2021_space-mni152',
    'target-GABAa_tracer-flumazenil_n-6_dx-hc_pub-dukart2018_space-mni152',
    'target-GABAa5_tracer-ro154513_n-10_dx-hc_pub-lukow2022_space-mni152',
    'target-H3_tracer-gsk189254_n-8_dx-hc_pub-gallezot2017_space-mni152',
    'target-HDAC_tracer-martinostat_n-8_dx-hc_pub-wey2016_space-mni152',
    'target-KOR_tracer-ly2795050_n-28_dx-hc_pub-vijay2018_space-mni152',
    'target-M1_tracer-lsn3172176_n-24_dx-hc_pub-naganawa2020_space-mni152',
    'target-mGluR5_tracer-abp688_n-22_dx-hc_pub-rosaneto_space-mni152',
    'target-mGluR5_tracer-abp688_n-28_dx-hc_pub-dubois2015_space-mni152',
    'target-mGluR5_tracer-abp688_n-73_dx-hc_pub-smart2019_space-mni152',
    'target-MOR_tracer-carfentanil_n-204_dx-hc_pub-kantonen2020_space-mni152',
    'target-MOR_tracer-carfentanil_n-39_dx-hc_pub-turtonen2021_space-mni152',
    'target-NET_tracer-mrb_n-10_dx-hc_pub-hesse2017_space-mni152',
    'target-NET_tracer-mrb_n-77_dx-hc_pub-ding2010_space-mni152',
    'target-SV2A_tracer-ucbj_n-10_dx-hc_pub-finnema2016_space-mni152',
    'target-TSPO_tracer-pbr28_n-6_dx-hc_pub-lois2018_space-mni152',
    'target-VAChT_tracer-feobv_n-18_dx-hc_pub-aghourian2017_space-mni152',
    'target-VAChT_tracer-feobv_n-4_dx-hc_pub-tuominen_space-mni152',
    'target-VAChT_tracer-feobv_n-5_dx-hc_pub-bedard2019_space-mni152',
]
pd.DataFrame({
    "set": [f.split("_")[0].split("-")[1] for f in collection],
    "x": collection,
    "weight": [f.split("_")[2].split("-")[1] for f in collection],
}) \
    .to_csv(nispace_source_data_path / "reference" / "pet" / "collection-UniqueTracerSets.collect", index=None)


# %% mRNA collections -------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# TODO: brainspan, add weighted cell types
# TODO: add GWAS from PGC after mapping to genes ("35 kb upstream and 10 kb downstream")

# All genes
mRNA_tab_files = (nispace_source_data_path / "reference" / "mrna" / "tab").glob("*.csv.gz")
all_genes = set()
for f in mRNA_tab_files:
    all_genes.update(pd.read_csv(f, index_col=0).index.unique())
all_genes = pd.Series(sorted(list(all_genes)), name="map")
all_genes.to_csv(nispace_source_data_path / "reference" / "mrna" / "collection-All.collect", index=False)

# Siletti et al cell types
df = (
    pd.read_excel("https://raw.githubusercontent.com/linnarsson-lab/adult-human-brain/refs/heads/main/tables/cluster_annotation.xlsx")
    .dropna(how="all")
    .loc[:, ["Supercluster", "Cluster name", "Top Enriched Genes"]]
)
sets = {
    'Upper-layer intratelencephalic': 'Upper-layer IT',
    'Deep-layer intratelencephalic': 'Deep-layer IT',
    'Deep-layer near-projecting': 'Deep-layer NP', 
    'Deep-layer corticothalamic and 6b': 'Deep-layer CT/6b',
    'MGE interneuron': 'MGE interneuron', 
    'CGE interneuron': 'CGE interneuron',
    'LAMP5-LHX6 and Chandelier': 'LAMP5-LHX6/Chandelier', 
    'Hippocampal CA1-3': "Hippocampus CA1-3",
    'Hippocampal CA4': "Hippocampus CA4", 
    'Hippocampal dentate gyrus': "Hippocampus DG",
    'Amygdala excitatory': "Amygdala excitatory",
    'Medium spiny neuron': "MSN", 
    'Eccentric medium spiny neuron': "Eccentric MSN", 
    'Splatter': "Splatter",
    'Mammillary body': "Mammillary body", 
    'Thalamic excitatory': "Thalamic excitatory",
    'Midbrain-derived inhibitory': "Midbrain-derived inhibitory",
    'Upper rhombic lip': "Upper rhombic lip", 
    'Cerebellar inhibitory': "Cerebellar inhibitory", 
    'Lower rhombic lip': "Lower rhombic lip",
    'Astrocyte': "Astrocyte", 
    'Oligodendrocyte': "Oligodendrocyte",
    'Oligodendrocyte precursor': "OPC",
    'Committed oligodendrocyte precursor': "Committed OPC", 
    'Microglia': "Microglia", 
    'Bergmann glia': "Bergmann glia", 
    'Vascular': "Vascular", 
    'Choroid plexus': "Choroid plexus",
    'Fibroblast': "Fibroblast", 
    'Ependymal': "Ependymal",  
    'Miscellaneous': "Miscellaneous",
}
# Clusters
collection = {}
for s in sets.keys():
    for x in df.query("Supercluster==@s")["Cluster name"].unique().tolist():
        collection[f"{sets[s]} - {x}"] = df.query("`Cluster name`==@x")["Top Enriched Genes"].str.split(", ").explode().unique().tolist()
write_json(
    collection,
    nispace_source_data_path / "reference" / "mrna" / "collection-CellTypesSilettiClusters.collect"
)
# Superclusters
collection = {}
for s in sets.keys():
    if s != "Miscellaneous":
        collection[sets[s]] = (
            df
            .query("Supercluster==@s")["Top Enriched Genes"]
            .str
            .split(", ").explode().unique().tolist()
        )
write_json(
    collection,
    nispace_source_data_path / "reference" / "mrna" / "collection-CellTypesSilettiSuperclusters.collect"
)

# PsychEncode cell types: Darmanis 2015 / Lake 2016 vs.  Lake 2018
for collection, save_name in zip(
    [pd.read_excel("http://resource.psychencode.org/Datasets/Derived/SC_Decomp/"
                   "DER-19_Single_cell_markergenes_TPM.xlsx") \
        .rename(columns=dict(GeneName="gene", CellType="set")),
     pd.read_excel("http://resource.psychencode.org/Datasets/Derived/SC_Decomp/"
                   "DER-21_Single_cell_markergenes_UMI.xlsx", header=1) \
        .rename(columns=dict(Gene="gene", Cluster="set"))],
    ["CellTypesPsychEncodeTPM", 
     "CellTypesPsychEncodeUMI"]
):
    collection = collection.astype(str)
    collection = {k: sorted(collection.query("set==@k").gene.unique()) for k in collection.set.unique()}
    all_genes = sum([collection[k] for k in collection], [])
    print(len(collection), "sets,", len(all_genes), "genes,", len(set(all_genes)), "unique.")
    write_json(collection, nispace_source_data_path / "reference" / "mrna" / f"collection-{save_name}.collect")

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
write_json(collection, nispace_source_data_path / "reference" / "mrna" / f"collection-SynGO.collect")

# Chromosome location
# for now, get from ABAnnotate (original source: DAVID)
df = pd.read_csv("/Users/llotter/projects/ABAnnotate/raw_datasets/DAVID/OFFICIAL_GENE_SYMBOL2CHROMOSOME.txt", sep="\t", header=None)
df.columns = ["gene", "chrom"]
sets = [str(i) for i in np.arange(1,23,1)] + ["X","Y"]
collection = {k: sorted(df.query("chrom==@k").gene.unique()) for k in sets}
all_genes = sum([collection[k] for k in collection], [])
print(len(collection), "sets,", len(all_genes), "genes,", len(set(all_genes)), "unique.")
write_json(collection, nispace_source_data_path / "reference" / "mrna" / f"collection-Chromosome.collect")


# %% RSN collections -------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

rsn_files = fetch_reference("rsn")
rsn_files = [_rm_ext(f.name) for f in rsn_files]

# All
pd.Series(rsn_files, name="map") \
    .to_csv(nispace_source_data_path / "reference" / "rsn" / "collection-All.collect", index=None)
    

# %%
