# %% Init

import sys
import pathlib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from nimare.dataset import Dataset

from utils_datasets import parcellate_reference_dataset

# add nispace to path
sys.path.append(str(pathlib.Path.cwd().parent))
from nispace.utils import _rm_ext
from nispace.io import write_json
from nispace.modules.constants import _PARCS_NICE

# nispace data path in home dir
nispace_data_path = pathlib.Path.home() / "nispace-data"
nispace_brainmap_map_path = nispace_data_path / "reference" / "brainmap" / "map"
nispace_brainmap_tab_path = nispace_data_path / "reference" / "brainmap" / "tab"

# ALE or SCALE?
meta_method = "mkdaChi2"


# %% Load BrainMap databank

# load
brainmap_df = pd.read_csv(pathlib.Path().cwd() / "_archive" / "brainmap_func_01feb2024.txt",
                          sep="\t", header=0)
# "Activations" only
brainmap_df = brainmap_df.query("ACTIVATIONS == 'Y'")
# Talairach and MNI only
brainmap_df["BRAIN_TEMPLATE"] = brainmap_df["BRAIN_TEMPLATE"].replace(
    {
        'Talairach 1988': 'TAL',
        'Unknown': 'MNI',
        'ICBM152': 'MNI',
        'ICBM452': 'MNI',
        'Talairach 1967:HD6': 'TAL',
        'Talairach 1967:vf25': 'TAL',
        'Talairach 1993': 'TAL',
        'MNI305': 'MNI',
        'SPM EPI': 'MNI',
        'MNI152': 'MNI',
        'East Asian': 'DROP',
        'MNIN27': 'MNI',
        'MNI T1': 'DROP',
        'EPI': 'DROP'
    }
)
brainmap_df = brainmap_df.query("BRAIN_TEMPLATE != 'DROP'")
# Unique papers
unique_papers = brainmap_df["PAPER_ID"].unique()

# Create dataset
dset_dict = {}
for paper in tqdm(unique_papers):
    brainmap_df_paper = brainmap_df.query("PAPER_ID == @paper")
    dset_dict[paper] = {}
    for EXPERIMENT_ID in brainmap_df_paper["EXPERIMENT_ID"].unique():
        brainmap_df_paper_exp = brainmap_df_paper.query("EXPERIMENT_ID == @EXPERIMENT_ID")
        dset_dict[paper]["contrasts"] = {EXPERIMENT_ID: {
            "coords": {
                "space": brainmap_df_paper_exp["BRAIN_TEMPLATE"].values[0],
                "x": brainmap_df_paper_exp["X"].to_list(),
                "y": brainmap_df_paper_exp["Y"].to_list(),
                "z": brainmap_df_paper_exp["Z"].to_list(),
            },
            "metadata": {
                "sample_sizes": brainmap_df_paper_exp["MIN_SUBJ_TOT"].values[0],
            } | {
                desc.lower(): brainmap_df_paper_exp[desc].values[0] 
                for desc in ['FIRST_AUTHOR', 'YEAR', 'DIAGNOSES', 'STIMULUS_MODALITY', 
                             'STIMULUS_TYPE', 'RESPONSE_MODALITY', 'RESPONSE_TYPE', 'INSTRUCTION', 
                             'EXTERNAL_ASSESSMENT', 'CONTEXT', 'IMAGING_MODALITY', 
                             'CONTRAST', 'PARADIGM_CLASS', 'BEHAVIORAL_DOMAIN',
                             'MIN_AGES', 'MAX_AGES', 'MEAN_AGES', 'AGE_CLASSIFICATIONS']
            }
        }}
        
# Convert dataset
dset = Dataset(dset_dict, target="mni152_2mm")


# %% Estimate meta-analyses

# Get a unique set of "Normal mapping" & MRI/nuclear imaging experiment ids
ids_normal = dset.metadata \
    .query("context == 'Normal Mapping' and imaging_modality not in ['EEG', 'MEG']") \
    ["id"].to_list()
ids_normal = set(ids_normal)
print(f"{len(ids_normal)} normal mapping MRI/nuclear imaging experiments")

# All domains
brainmap_domains = set()
for domain in brainmap_df["BEHAVIORAL_DOMAIN"].unique():
    if isinstance(domain, str):
        brainmap_domains.update(domain.split(", "))
brainmap_domains = sorted(brainmap_domains)
print(brainmap_domains)

# Meta-analysis per domain
from nimare.meta.cbma.ale import ALE, SCALE
from nimare.meta.cbma.mkda import MKDAChi2

for domain in brainmap_domains: 
    ids_domain = [id 
                  for id in ids_normal 
                  if domain in dset.metadata.query("id == @id")["behavioral_domain"].values[0]]
    print(f"{domain}: {len(ids_domain)} experiments")
    if len(ids_domain) < 15:
        print("Too few experiments; skipping.")
        continue
    
    # Domain dataset
    dset_domain = dset.slice(list(ids_domain))
    
    # Run ALE on domain coordinates
    if meta_method == "ale":
        meta = ALE(n_cores=8)
        results = meta.fit(dset_domain)
        results_map = results.get_map("z")
        
    # Run Specific ALE (SCALE) or MKDAChi2 on domain coordinates, using remaining coordinates as background
    # we actually dont use all coordinates as background but only those that do not contain the 
    # domain to any extent, which means that for, e.g., "Action.Execution" "Action.Execution.Speech"
    # will also be excluded from the background
    elif meta_method in ["scale", "mkdaChi2"]:
        ids_background = [id
                          for id in ids_normal
                          if domain not in dset.metadata.query("id == @id")["behavioral_domain"].values[0]]
        dset_background = dset.slice(list(ids_background))
        print(f"{domain} background: {len(ids_background)} experiments with "
              f"{len(dset_background.coordinates)} coordinates")
        
        if meta_method == "scale":
            meta = SCALE(xyz=dset.coordinates[["x", "y", "z"]].values, n_cores=8)
            results = meta.fit(dset_domain)
            results_map = results.get_map("z")
        
        elif meta_method == "mkdaChi2":
            meta = MKDAChi2()
            results = meta.fit(dset_domain, dset_background)   
            results_map = results.get_map("z_desc-association")
            
    # get results
    results_map.set_data_dtype(np.float32)
    # save
    domain_save = domain.replace('(GI/GU)','').replace(' ', '').replace('/','')
    results_map.to_filename(nispace_brainmap_map_path / f"domain-{domain_save}_n-{len(ids_domain)}.nii.gz")


# %% Tabulated data ----------------------------------------------------------------------------

files = list((nispace_data_path / "reference" / "brainmap" / "map").glob("domain-*"))
files.sort()

parcellate_reference_dataset(
    reference_name="brainmap",
    reference_files=files,
    parcs=_PARCS_NICE,
)

# %% Collections -------------------------------------------------------------------------------

files = list((nispace_data_path / "reference" / "brainmap" / "map").glob("domain-*"))
files = sorted([_rm_ext(f.name) for f in files])

# Define the order of categories
categories_order = [
    "Action.Observation",
    "Action.Imagination",
    "Action.Execution_",
    "Action.Execution.",
    "Action.Inhibition",
    "Perception.Vision_",
    "Perception.Vision.",
    "Perception.Audition",
    "Perception.Olfaction",
    "Perception.Gustation",
    "Perception.Somesthesis_",
    "Perception.Somesthesis.",
    "Interoception", 
    "Emotion.Positive_",
    "Emotion.Positive.",
    "Emotion.Negative_",
    "Emotion.Negative.",
    "Emotion.Valence",
    "Cognition.Attention",
    "Cognition.Spatial",
    "Cognition.Temporal",
    "Cognition.Reasoning",
    "Cognition.SocialCognition",
    "Cognition.Memory_",
    "Cognition.Memory.",
    "Cognition.Music",
    "Cognition.Language_",
    "Cognition.Language.",
]

# Sort files based on the defined order
files.sort(key=lambda f: next((i for i, category in enumerate(categories_order) if category in f), len(categories_order)))

# All
pd.Series(files, name="map") \
    .to_csv(nispace_data_path / "reference" / "brainmap" / "collection-All.txt", index=None)
    
# AllDomainSets
collection = {}
for category in [f.split("_")[0].split("-")[1].split(".")[0] for f in files]:
    collection[category] = [f for f in files if f"domain-{category}" in f]
write_json(collection, nispace_data_path / "reference" / "brainmap" / "collection-AllDomainSets.json")
    
# %%
