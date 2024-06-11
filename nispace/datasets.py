import pathlib
from typing import Union, List, Dict, Tuple
import requests
import pandas as pd
import numpy as np
from nilearn import image
from tqdm.auto import tqdm
import gzip
import pickle
import json

from . import lgr
from .modules.constants import _DSETS, _DSETS_NICE, _DSETS_MAP, _DSETS_TAB, _PARCS_DEFAULT
from nispace.stats.misc import zscore_df
from .utils import _rm_ext, set_log, load_img, load_distmat, load_labels

# OSF HANDLING =====================================================================================

def _download_file(url, path):
    """Download a file from a URL to a given path."""
    response = requests.get(url)
    response.raise_for_status()
    with open(path, "wb") as f:
        f.write(response.content)


def _ensure_data_available(local_path, remote_url):
    """Ensure that data is available, downloading it if necessary."""
    if not local_path.exists():
        lgr.info(f"Downloading data to {local_path}")
        _download_file(remote_url, local_path)
    else:
        lgr.info(f"Data already present at {local_path}")
        
# FILE HANDLING ====================================================================================

def _file_desc(fname, feature_position):
    if isinstance(fname, pathlib.Path):
        fname = fname.name
    fname = fname.split(".")[0]
    if isinstance(feature_position, int):
        return fname.split("_")[feature_position].split("-")[1]
    elif isinstance(feature_position, str):
        return fname.split(f"{feature_position}-")[1].split("_")[0]
    
# BRAIN TEMPLATES ==================================================================================

def fetch_template(template: str = "mni152", 
                   desc: str = None,
                   parcellation: str = None,
                   hemi: Union[List[str], str] = ["L", "R"],
                   nispace_data_dir: Union[str, pathlib.Path] = None):
    """
    Fetch a brain template.
    
    Parameters
    ----------
    template : str, optional
        The template to fetch. Default is "mni152".
        
    desc : str, optional
        The description of the template to fetch. Default is None.
        
    parcellation : str, optional
        The parcellation to fetch. Default is None.
        
    hemi : list of str, optional
        The hemispheres to fetch. Default is ["L", "R"].
        
    nispace_data_dir : str or pathlib.Path, optional
        The directory containing the NiSpace data. Default is None.
        
    Returns
    -------
    The template.
    """
    
    # hard-coded template type    
    if "mni" in template.lower():
        template = "mni152"
    elif "fsa" in template.lower():
        template = "fsaverage"
    else:
        raise ValueError("'template' must be 'MNI152' or 'fsaverage'!")
    
    # paths        
    if not nispace_data_dir:
        base_dir = pathlib.Path.home() / "nispace-data" / "template" / template
    else:
        base_dir = pathlib.Path(nispace_data_dir) / "template" / template
    map_dir = base_dir / "map"
    tab_dir = base_dir / "tab"
    
    # # OSF CODE
    # remote_base_url = "https://osf.io/path_to_your_osf_data/"  # Example OSF base URL

    # # Ensure base directory exists
    # base_dir.mkdir(parents=True, exist_ok=True)

    # # Check and download parcellation data if necessary
    # _ensure_data_available(base_dir / "your_parcellation_data_file", remote_base_url + "your_parcellation_data_file")
    
    # get files
    # mni
    if template == "mni152":
        # desc
        if (not desc) and (not parcellation):
            desc = "T1"
        elif (not desc) and parcellation:
            desc = "gmprob"
        lgr.info(f"Loading MNI152NLin2009cAsym {desc} template in 1mm resolution.")
        # get files
        tpl_files = list(map_dir.glob(f"MNI152NLin2009cAsym_desc-{desc}_res-1mm.nii.gz"))
        # check
        if len(tpl_files) != 1:
            raise ValueError(f"'desc' = {desc} not defined. Choose one of 'T1', 'gmprob', or 'mask'!")
        # fin
        tpl_file = tpl_files[0]

    # fsa
    elif template == "fsaverage":
         # desc
        if (not desc) and (not parcellation):
            desc = "pial"
        elif (not desc) and parcellation:
            desc = "thick"
        lgr.info(f"Loading FSaverage {desc} template in 10k resolution.")
        # get files
        tpl_files = list(map_dir.glob(f"fsaverage_desc-{desc}_hemi-*_res-10k.gii.gz"))
        # check
        if len(tpl_files) != 2:
            raise ValueError(f"'desc' = {desc} not defined. Choose one of 'pial', 'infl', or 'thick'!")
        # fin
        tpl_file = ()
        if "L" in hemi:
            tpl_file += [f for f in tpl_files if "hemi-L" in f.name][0],
        if "R" in hemi:
            tpl_file += [f for f in tpl_files if "hemi-R" in f.name][0],
            
    
    # Load tabulated data if 'parcellation' is specified
    if (parcellation is not None) & (desc in ["gmprob", "thick"]):
        lgr.info(f"Loading parcellated data: {parcellation}")
        parcellation_file = tab_dir / f"{desc}_{parcellation}.csv"
        if parcellation_file.exists():
            tab_data = pd.read_csv(parcellation_file, index_col=0)

            if (template == "fsaverage") & (hemi in ["L", "R"]):
                if hemi in "L":
                    tab_data = tab_data.iloc[:, :int(tab_data.shape[1] / 2)]
                else:
                    tab_data = tab_data.iloc[:, int(tab_data.shape[1] / 2):]
                    
            # return
            return tab_data
        
        else:
            lgr.warning(f"Parcellated {desc} data for {parcellation} not found!")
    
    # feedback
    elif (parcellation is not None) & (desc not in ["gmprob", "thick"]):
        lgr.warning("Parcellated data only available for desc = 'gmprob' (MNI152) or 'thick' (fsaverage)! "
                    "Will return path to requested file instead.")
        
    # return
    return tpl_file

# PARCELLATIONS ====================================================================================

def fetch_parcellation(parcellation: str = None, 
                       space: str = None, 
                       n_parcels: Union[int, str] = None,
                       resolution: str = None,
                       hemi: Union[List[str], str] = ["L", "R"],
                       cortex_only: bool = False,
                       return_space: bool = False,
                       return_resolution: bool = False,
                       return_dist_mat: bool = False,
                       return_loaded: bool = False,
                       nispace_data_dir: Union[str, pathlib.Path] = None):
    
    if not nispace_data_dir:
        base_dir = pathlib.Path.home() / "nispace-data" / "parcellation"
    else:
        base_dir = pathlib.Path(nispace_data_dir) / "parcellation"
    
    # # OSF CODE
    # remote_base_url = "https://osf.io/path_to_your_osf_data/"  # Example OSF base URL

    # # Ensure base directory exists
    # base_dir.mkdir(parents=True, exist_ok=True)

    # # Check and download parcellation data if necessary
    # _ensure_data_available(base_dir / "your_parcellation_data_file", remote_base_url + "your_parcellation_data_file")
 
    # descriptors
    if parcellation not in ["", None, False, True, "*", "**", "***"]:
        parcellation = f"*{parcellation}*"
    else:
        parcellation = "*"
        
    if not n_parcels:
        n_parcels = "*"
        
    if not space:
        space = "*"
    elif "mni" in space.lower():
        space = "mni152"
 
    if not resolution:
        resolution = "*"
    
    # Locate parcellation file
    parcellation_files = list(base_dir.glob(f"parc-{parcellation}_n-{n_parcels}_space-{space}_res-{resolution}"))
    parcellation_files.sort()
    parcellation_files = [f for f in parcellation_files if f.name.endswith(".nii.gz") | f.is_dir()]
    
    # print parcellation (not found) info and errors
    def parc_info_str(parc_file, newline=False):
        string = \
            f"{_file_desc(parc_file, 0)} " \
            f"({', '.join([desc + ': ' + _file_desc(parc_file, desc) for desc in ['n', 'space', 'res']])})"
        if newline:
            string = "\n" + string
        return string
    
    if not parcellation_files:
        raise FileNotFoundError("No matching parcellation found. Available: "
                                f"{''.join([parc_info_str(f, True) for f in base_dir.glob('parc-*[!.txt]')])}")
        
    elif len(parcellation_files) > 1:
        raise FileNotFoundError("Multiple files match your criteria: "
                                f"{''.join([parc_info_str(f, True) for f in parcellation_files])}")

    # Select the first matching file (assuming unique naming)
    parcellation_file = parcellation_files[0]
    lgr.info(f"Loading parcellation: {parc_info_str(parcellation_file)}")
    
    # surface
    if parcellation_file.is_dir():
        space = "fsaverage"
        
        # get hemispheres
        parcellation_file_surf = ()
        dist_mat = ()
        if "L" in hemi:
            parcellation_file_surf += list(parcellation_file.glob("*hemi-L.gii.gz"))[0],
            dist_mat += list(parcellation_file.glob("*hemi-L.csv.gz"))[0],
        if "R" in hemi:
            parcellation_file_surf += list(parcellation_file.glob("*hemi-R.gii.gz"))[0],
            dist_mat += list(parcellation_file.glob("*hemi-R.csv.gz"))[0],

        # get labels
        labels = ()
        for parc_file in parcellation_file_surf:
            label_file = base_dir / parcellation_file / parc_file.name.replace(".gii.gz", ".txt")
            if label_file.exists():
                labels += label_file,
        labels = None if labels==() else labels
        
        # density
        density = parcellation_file_surf[0].name.split(".")[0].split("_")[3].split("-")[1]
         
        # return
        parcellation_file = parcellation_file_surf
    
    # volume
    else:
        space = "MNI152"
        
        # dist mat
        dist_mat = base_dir / parcellation_file.name.replace(".nii.gz", ".csv.gz")

        # get labels
        label_file = base_dir / parcellation_file.name.replace(".nii.gz", ".txt")
        if label_file.exists():
            labels = label_file
        else:
            labels = None
        
        # density
        density = parcellation_file.name.split(".")[0].split("_")[3].split("-")[1]
        
        # cortex only:
        if cortex_only:
            # subcortical indices
            labels = load_labels(labels)
            idc_sc = [int(l.split("_")[0]) for l in labels if "_SC_" in l]
            lgr.info(f"Removing {len(idc_sc)} subcortical parcels. Will return Nifti1 object instead of path!")
            # drop from parcellation
            parc = load_img(parcellation_file)
            parc_array = parc.get_fdata()
            for idx in idc_sc:
                parc_array[parc_array==idx] = 0
            parc = image.new_img_like(parc, parc_array, copy_header=True)
            parcellation_file = parc
            # drop from labels
            labels = [l for l in labels if "_CX_" in l]
            # drop from dist mat
            bool_sc = np.array([True if "_SC_" in l else False for l in labels])
            dist_mat = load_distmat(dist_mat)
            dist_mat = dist_mat[np.ix_(~bool_sc, ~bool_sc)]
            
    # return
   
    if return_loaded:
        parcellation_file = load_img(parcellation_file)
        if labels is not None:
            labels = load_labels(labels)
    out = parcellation_file, labels 
    if return_space:
        out += space,
    if return_resolution:
        out += density,
    if return_dist_mat:
        if return_loaded:
            dist_mat = load_distmat(dist_mat)
        out += dist_mat,
    return out

# REFERENCE DATA - PRIVATE =========================================================================

def _filter_maps(map_files: List[pathlib.Path], 
                 maps: Union[str, List[str], Dict[str, Union[str, list]]]) -> List[pathlib.Path]:
    
    def matches_filters(file_path: pathlib.Path, filters: Dict[str, Union[str, List[str]]]) -> bool:
        for filter_name, filter_content in filters.items():
            if filter_content not in [None, False, "", []]:
                if isinstance(filter_content, (str, int)):
                    filter_content = [filter_content]
                filter_content = list(map(str, filter_content))
                if filter_name == "n" and filter_content[0].startswith(">"):
                    try:
                        filter_n = int(filter_content[0].replace(">", ""))
                        n_value = int(_file_desc(file_path, 2))
                        if n_value <= filter_n:
                            return False
                    except (ValueError, IndexError):
                        continue  # Skip this filter if parsing fails
                else:
                    if not any(f"{filter_name}-{content}".lower() in file_path.name.lower() for content in filter_content):
                        return False
        return True

    if isinstance(maps, str):
        maps = [maps]
    if isinstance(maps, list):
        maps = list(set(maps))
        filtered_maps = [f for f in map_files if any(map_str in f.name for map_str in maps)]
    elif isinstance(maps, dict):
        filtered_maps = [f for f in map_files if matches_filters(f, maps)]
    else:
        filtered_maps = map_files
        
    return filtered_maps


def _fetch_collection(collection_path):
    
    # if path, read file
    if isinstance(collection_path, (str, pathlib.Path)):
        collection_path = pathlib.Path(collection_path)
        ext = collection_path.suffix
            
        # if json, load into dict
        if ext == ".json":
            with open(collection_path) as f:
                collection = json.load(f)
                
        # else, try to directly load as table file
        else:
            with open(collection_path) as f:
                header = f.readline().strip("\n")
                if any([h in header for h in ["set", "map", "weight"]]):
                    header = 0
                else: 
                    header = None
            collection = pd.read_csv(collection_path, header=header)
    else:
        collection = collection_path
        
    # if array, convert all do df
    if isinstance(collection, (np.ndarray, pd.DataFrame, pd.Series, list)):   
        collection = pd.DataFrame(collection)
        
    # if dict, convert to df as well
    elif isinstance(collection, dict):   
        collection = pd.concat([pd.DataFrame({0:k, 1:v}) for k, v in collection.items()])
        
    # else
    else:
        raise TypeError(f"Datatype {type(collection_path)} not accepted for argument 'collection'.")
        
    # process depending on number of columns
    n_cols = collection.shape[1]
    if n_cols == 0:
        raise ValueError("No columns detected in collection file?!")
    elif n_cols == 1:
        collection.columns = ["map"]
    elif n_cols == 2:
        collection.columns = ["set", "map"]
    elif n_cols == 3:
        collection.columns = ["set", "map", "weight"]
    else:
        raise ValueError(f"Collection file with > 3 columns not supported ({n_cols} columns)!")
    
    # return
    return collection.reset_index(drop=True)


def _apply_collection_filter(map_files: List[Union[str, pathlib.Path]], 
                             collection: str,
                             base_dir: pathlib.Path,
                             set_size_range: Union[None, Tuple[int, int]] = None) -> List[pathlib.Path]:
    
    # Check if path to custom file
    collection_file = pathlib.Path(collection)
    if not collection_file.exists():
        # If not exists, search integrated collections
        collection_file = list(base_dir.glob(f"collection-{collection.replace('collection-', '')}.*"))
        if len(collection_file) == 0:
            lgr.warning(f"Collection '{collection}' not found! Available: "
                        f"{[f.name.replace('collection-', '').replace(f.suffix, '') for f in base_dir.glob('collection-*.*')]}")
            return map_files, None
        elif len(collection_file) > 1:
            lgr.warning("Found more than one collection file matching your search; using first:", 
                        collection_file)
        collection_file = collection_file[0]

    # Load collection file; 1-column df (= maps) or 2-column df (= set and maps)
    collection_df = _fetch_collection(collection_file)
    lgr.debug(f"Collection df shape: {collection_df.shape}; "
              f"index names: {collection_df.index.names}; "
              f"column names: {collection_df.columns.names}")

    # Apply collection filter
    lgr.info(f"Applying collection filter from: {collection_file}.")
    if isinstance(map_files[0], pathlib.Path):
        map_names = [_rm_ext(f.name) for f in map_files]
        filtered_map_files = [f for f, f_name in zip(map_files, map_names) 
                              if f_name in collection_df["map"].unique()]
        collection_df = collection_df[collection_df["map"].isin(map_names)]
    else:
        filtered_map_files = [f for f in map_files if f in collection_df["map"].unique()]
        collection_df = collection_df[collection_df["map"].isin(filtered_map_files)]
        
    # Apply size filter
    if set_size_range is not None:
        if "set" in collection_df.columns and isinstance(set_size_range, (tuple, list)):
            set_size_range = [x if x is not None else x_ 
                              for x, x_ in zip(set_size_range, (1, np.inf))]
            lgr.info(f"Filtering to collection sets with between {set_size_range[0]} and "
                     f"{set_size_range[1]} maps.")
            collection_df = collection_df.groupby("set") \
                .filter(lambda x: set_size_range[0] <= x.shape[0] <= set_size_range[1])            
            filtered_map_files = [f for f in map_files if f in collection_df["map"].unique()]

    return filtered_map_files, collection_df


def _load_parcellated_data(dataset: str, 
                           tab_dir: pathlib.Path, 
                           parcellation: str, 
                           map_files: List[str],
                           collection_df: pd.DataFrame,
                           return_nulls: bool, 
                           nulls_dir: pathlib.Path, 
                           cortex_only: bool,
                           standardize: bool) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    lgr.info(f"Loading parcellated data: {parcellation}")
    parcellation_file = tab_dir / f"{dataset}_{parcellation}.csv.gz"
    lgr.debug(f"Checking for {parcellation_file}")
    if not parcellation_file.exists():
        lgr.critical_raise(f"Parcellated data file for '{parcellation}' not found!",
                           FileNotFoundError)

    # Load parcellated data
    data = pd.read_csv(parcellation_file, index_col=0)
    lgr.debug(f"Loaded parcellated data of shape {data.shape}")
    lgr.debug(f"First 5 map names: {data.index.to_list()[:5]}")

    # Apply filter to the dataframe index
    lgr.debug(f"Applying filtering based on maps, first 5: {map_files[:5]}")
    if isinstance(map_files[0], pathlib.Path):
        map_files = [_rm_ext(f.name) for f in map_files]
    data = data.loc[data.index.intersection(map_files)]
    lgr.debug(f"Shape after filtering based on map_names: {data.shape}")
    
    # Apply collection index (-> handles maps that are present multiple times in different sets)
    if collection_df is not None:
        maps_intersection = data.index.intersection(collection_df["map"].unique())
        collection_df_intersection = collection_df.query("map in @maps_intersection")
        data = data.loc[collection_df_intersection["map"]]     
        data.index = pd.MultiIndex.from_frame(collection_df_intersection)
        
    # Load null maps if requested
    if return_nulls:
        lgr.info("Loading precomputed null maps.")
        try:
            # load
            with gzip.open(nulls_dir / f"{dataset}_{parcellation}.pkl.gz", "rb") as f:
                null_maps = pickle.load(f)
            # filter to selected maps
            if collection_df is None:
                null_maps = {name: null_maps[name] for name in data.index}
            else:
                null_maps = {idx: null_maps[name] 
                             for idx, name in zip(data.index, data.index.get_level_values("map"))}
        except FileNotFoundError:
            lgr.warning("No precomputed null map data found. Did you download it?")
            return_nulls = False

    # Filter to keep only cortical parcels if requested
    if cortex_only:
        lgr.info("Keeping only cortical parcels.")
        bool_cx = [True if "_CX_" in c else False for c in data.columns]
        data = data.loc[:, bool_cx]
        if return_nulls:
            null_maps = {name: null_maps[:, bool_cx] 
                         for name, null_maps in null_maps.items()}
            
    # Standardize
    if standardize:
        lgr.info("Standardizing parcellated data.")
        data = zscore_df(data, along="rows")
        if return_nulls:
            null_maps = {name: zscore_df(null_maps, along="rows", force_df=False) 
                         for name, null_maps in null_maps.items()}

    if return_nulls:
        return data, null_maps
    else:
        return data
    

def _print_references(dataset: str, meta: pd.DataFrame = None):
    
    # PET
    if dataset.lower() == "pet":
        msg = """
The NiSpace "PET" dataset is based on openly available nuclear imaging maps from various independent sources.
The accompanying metadata table contains detailed information about tracers, source samples, original publications and data 
sources, as well as the publication licenses. The licenses determine in which contexts the maps can be used. Every map, 
except for those with a "free" or "CC0 1.0" license, must be cited when used. The responsibility for this lies with the user!
We additionally recommend to cite Dukart et al., 2021 (https://doi.org/10.1002/hbm.25244) and Hansen et al., 2022 
(https://doi.org/10.1038/s41593-022-01186-3), as they provided the majority of these data.
"""
        if meta is not None:
            atlas_maxlen = max([len(x) for x in meta["atlas"]])
            author_maxlen = max([len(x) for x in meta["author"]])
            lic_maxlen = max([len(x) for x in meta["license"]])
            for atlas, pub, doi, lic in zip(meta["atlas"], meta["author"], meta["doi"], meta["license"]):
                atlas = atlas.ljust(atlas_maxlen)
                author = pub.capitalize().ljust(author_maxlen)
                lic = lic.ljust(lic_maxlen)
                msg += f"- {atlas}  Source: {author}  {lic}  https://doi.org/{doi}\n"
    
    # mRNA
    elif dataset.lower() == "mrna":
        msg = """
The NiSpace "mRNA" dataset is based on Allen Human Brain Atlas (AHBA) gene expression data published in Hawrylycz et al., 2015
(https://doi.org/10.1038/nn.4171). The AHBA dataset consists of mRNA expression data from postmortem brain tissue of 6 donors, 
mapped into imaging space using the abagen toolbox (Markello et al., 2021, https://doi.org/10.7554/eLife.72129).
In addition to those two publications, please cite publications associated with gene set collections as appropriate.
"""
        if meta is not None:
            if len(meta) > 0:
                collection_maxlen = max([len(x) for x in meta["collection"]])
                author_maxlen = max([len(x) for x in meta["author"]])
                for collection, pub, doi in zip(meta["collection"], meta["author"], meta["doi"]):
                    collection = collection.ljust(collection_maxlen)
                    author = pub.capitalize().ljust(author_maxlen)
                    msg += f"- {collection}  Source: {author}  https://doi.org/{doi}\n"
    
    # BrainMap
    elif dataset.lower() == "brainmap":
        msg = """
The NiSpace "BrainMap" dataset is consists of meta-analytic maps generated from coordinates from the BrainMap database
(http://brainmap.org). Please cite the following references when using these maps.
"""
        if meta is not None:
            if len(meta) > 0:
                author_maxlen = max([len(x) for x in meta["author"]])
                for pub, doi in zip(meta["author"], meta["doi"]):
                    author = pub.capitalize().ljust(author_maxlen)
                    msg += f"- {author}  https://doi.org/{doi}\n"

    # print
    if msg[-2:] != "\n":
        msg += "\n"
    print(msg)
    
    
# REFERENCE DATA - PUBLIC ==========================================================================

def fetch_reference(dataset: str,
                    maps: Union[None, str, List[str], Dict[str, Union[str, list]]] = None,
                    collection: str = None,
                    set_size_range: Union[None, Tuple[int, int]] = None,
                    parcellation: str = None,
                    standardize_parcellated: bool = True,
                    cortex_only: bool = False,
                    return_nulls: bool = False,
                    return_metadata: bool = False,
                    print_references: bool = True,
                    verbose: bool = True,
                    nispace_data_dir: Union[str, pathlib.Path] = None):
    verbose = set_log(lgr, verbose)

    if isinstance(dataset, str):
        dataset = dataset.lower()
        if dataset not in _DSETS:
            lgr.critical_raise(f"Dataset '{dataset}' not found! Available datasets: {_DSETS_NICE}",
                               ValueError)
    else:
        lgr.critical_raise(f"Invalid dataset type; expecting string. Available datasets: {_DSETS_NICE}",
                           TypeError)
    lgr.info(f"Loading {dataset} maps.")
    
    # Define the base directories
    if not nispace_data_dir:
        base_dir = pathlib.Path.home() / f"nispace-data" / "reference" / dataset
    else:
        base_dir = pathlib.Path(nispace_data_dir) / f"reference" / dataset
    map_dir = base_dir / "map"
    tab_dir = base_dir / "tab"
    nulls_dir = base_dir / "null"
    
    # TODO: HANDLE DOWNLOAD
    
    # Get list of map files
    if dataset in _DSETS_MAP:
        map_files = [f for f in map_dir.glob("*.nii.gz")]
        map_files.sort()
    elif dataset in _DSETS_TAB:
        if parcellation is None:
            lgr.warning("mRNA data requires a parcellation. Defaulting to: 'Schaefer200MelbourneS1'.")
            parcellation = _PARCS_DEFAULT
        if return_nulls:
            lgr.warning("Precomputed null maps are not available for mRNA data. Will not return any.")
            return_nulls = False
        map_files = pd.read_csv(base_dir / "collection-All.txt", header=0)["map"].to_list()
    lgr.debug(f"Loaded {len(map_files)} unfiltered map{' files' if dataset in _DSETS_MAP else 's'}. "
              f"First 5: {map_files[:5] if len(map_files) >= 5 else map_files[:len(map_files)]}")

    # Filter by 'maps'
    if maps:
        n_tmp = len(map_files)
        lgr.info(f"Applying filter: {maps}")
        if dataset in _DSETS_MAP:
            map_files = _filter_maps(map_files, maps)
        elif dataset in _DSETS_TAB:
            if isinstance(maps, str):
                maps = [maps]
            elif not isinstance(maps, (list, tuple, set, pd.Series)):
                lgr.warning(f"For dataset '{dataset}', 'maps' must be list-like. Skipping filter.")
                maps = map_files
            map_files = list(set(map_files).intersection(maps))
        lgr.info(f"Filtered from {n_tmp} to {len(map_files)} maps.")
            
    # Filter by 'collection'
    if collection == "All":
        collection = None
    if collection:
        map_files, collection_df = _apply_collection_filter(map_files, collection, base_dir, 
                                                            set_size_range)
    else:
        collection_df = None

    # Load tabulated data if 'parcellation' is specified
    if parcellation:
        # for now, no null data included. TODO: re-evaluate cost/benefits
        if return_nulls:
            lgr.warning("Pre-calculated null maps are currently not available. Will not return any.")
            return_nulls = False
        data = _load_parcellated_data(
            dataset=dataset, 
            tab_dir=tab_dir, 
            parcellation=parcellation, 
            map_files=map_files, 
            collection_df=collection_df,
            return_nulls=return_nulls, 
            nulls_dir=nulls_dir, 
            cortex_only=cortex_only,
            standardize=standardize_parcellated
        )
    else:
        data = map_files
        
    # Print references
    # for maps if "pet", or for sets if "mrna"
    if return_metadata or print_references:
        if dataset == "pet":
            meta = fetch_metadata(dataset, map_files)
        elif dataset == "mrna" and collection_df is not None:
            meta = fetch_metadata(dataset, collection=collection)
        elif dataset == "brainmap":
            meta = fetch_metadata(dataset)
        else: 
            meta = None
 
        if return_metadata:
            data = (data + (meta,)) if isinstance(data, tuple) else (data, meta)
        if print_references & verbose:
            _print_references(dataset, meta)

    return data


def fetch_metadata(dataset: str, maps: Union[str, list] = None, collection: str = None):
    if isinstance(dataset, str):
        dataset = dataset.lower()
        if dataset not in _DSETS:
            return None
    else:
        return None
    
    base_dir = pathlib.Path.home() / "nispace-data" / "reference" / dataset
    meta = pd.read_csv(base_dir / "metadata.csv", header=0)
    
    if dataset == "pet" and maps is not None:
        if isinstance(maps, str):
            maps = [maps]
        if isinstance(maps[0], pathlib.Path):
            maps = [_rm_ext(f.name) for f in maps]
        meta = meta[meta.atlas.str.contains("|".join(maps), na=False)]
    elif dataset == "mrna" and collection is not None:
        meta = meta.query("collection == @collection")
    elif dataset == "brainmap":
        pass
            
    return meta


# EXAMPLE DATA =====================================================================================
 
def fetch_example(example: str,
                  return_associated_data: bool = True,
                  nispace_data_dir: Union[str, pathlib.Path] = None):
    
    # Define the base directories
    if not nispace_data_dir:
        base_dir = pathlib.Path.home() / "nispace-data" / "example"
    else:
        base_dir = pathlib.Path(nispace_data_dir) / "example"

    example = example.lower()
    lgr.info(f"Loading example dataset: {example}")
    
    # # OSF CODE
    # remote_base_url = "https://osf.io/path_to_your_osf_data/"  # Example OSF base URL
    
    # # Ensure base directory exists
    # base_dir.mkdir(parents=True, exist_ok=True)

    # # Check and download data if necessary
    # _ensure_data_available(base_dir / "your_pet_data_file", remote_base_url + "your_pet_data_file")

    # Get parcellated data
    example_file = list(base_dir.glob(f"example-{example}*.csv.gz"))
    if len(example_file) == 0:
        lgr.warning(f"Example dataset '{example}' not found! Available: "
                    f"{[f.name.split('_')[0].replace('example-','') for f in base_dir.glob('example-*.csv.gz')]}")
    example_file = example_file[0]
    
    # load
    if example_file.exists():
        example_data = pd.read_csv(example_file, index_col=0)
        
        # parcellation
        lgr.info(f"The {example} dataset was parcellated using the "
                 f"{_rm_ext(example_file.name.split('_')[-1].split('-')[1])} parcellation.")
        
        # Check for info data 
        if return_associated_data:
            example_info_file = example_file.with_name(example_file.name.split("_")[0] + "_info.csv")
            if example_info_file.exists():
                example_info = pd.read_csv(example_info_file, header=0)
                lgr.info("Returning parcellated and associated subject data.")
                
                return example_data, example_info
        
        return example_data


# ==================================================================================================
# BACKUP OLD FUNCTIONS =============================================================================
# ==================================================================================================

# def fetch_pet(maps: Union[None, str, List[str], Dict[str, Union[str, list]]] = None,
#               collection: str = None,
#               parcellation: str = None,
#               cortex_only: bool = False,
#               return_nulls: bool = False,
#               nispace_data_dir: Union[str, pathlib.Path] = None):
#     # TODO: modularize collection filter
    
#     # Define the base directories
#     if not nispace_data_dir:
#         base_dir = pathlib.Path.home() / "nispace-data" / "reference" / "pet"
#     else:
#         base_dir = pathlib.Path(nispace_data_dir) / "reference" / "pet"
#     map_dir = base_dir / "map"
#     tab_dir = base_dir / "tab"
#     nulls_dir = base_dir / "null"
    
#     lgr.info("Loading PET maps.")
    
#     # # OSF CODE
#     # remote_base_url = "https://osf.io/path_to_your_osf_data/"  # Example OSF base URL
    
#     # # Ensure base directory exists
#     # base_dir.mkdir(parents=True, exist_ok=True)

#     # # Check and download data if necessary
#     # _ensure_data_available(base_dir / "your_pet_data_file", remote_base_url + "your_pet_data_file")

#     # Get list of PET map files
#     pet_maps = [f for f in map_dir.glob("*.nii.gz")]
#     pet_maps.sort()

#     # Filter by 'maps'    
#     if maps:
#         def _filter_pet_maps(pet_maps: List[pathlib.Path],
#                             filters: Dict[str, Union[str, List[str]]]) -> List[pathlib.Path]:
            
#             def matches_filters(file_path: pathlib.Path) -> bool:
#                 for filter_name, filter_content in filters.items():
#                     if filter_content not in [None, False, "", []]:
#                         if isinstance(filter_content, (str, int)):
#                             filter_content = [filter_content]
#                         filter_content = list(map(str, filter_content))
#                         if filter_name == "n" and filter_content[0].startswith(">"):
#                             try:
#                                 filter_n = int(filter_content[0].replace(">", ""))
#                                 n_value = int(_file_desc(file_path, 2))
#                                 if n_value <= filter_n:
#                                     return False
#                             except (ValueError, IndexError):
#                                 continue  # Skip this filter if parsing fails
#                         else:
#                             if not any(f"{filter_name}-{content}".lower() in file_path.name.lower() for content in filter_content):
#                                 return False
#                 return True

#             # Apply filters
#             filtered_pet_maps = [f for f in pet_maps if matches_filters(f)]
#             return filtered_pet_maps
        
#         lgr.info(f"Applying filter: {maps}")
#         if isinstance(maps, str):
#             maps = [maps]
#         if isinstance(maps, list):
#             maps = set(maps)
#             pet_maps = [f for f in pet_maps if any(map_str in f.name for map_str in maps)]
#         elif isinstance(maps, dict):
#             pet_maps = _filter_pet_maps(pet_maps, maps)

#     # Filter by 'collection'
#     if collection:
                
#         # check if path to custom file
#         collection_file = pathlib.Path(collection)
#         if collection_file.exists():
#             pass
        
#         # if not exists, search integrated collections
#         else:
#             collection_file = list(base_dir.glob(f"collection-{collection.replace('collection-','')}.*"))
#             if len(collection_file) == 0:
#                 lgr.warning(f"Collection '{collection}' not found! Available: "
#                             f"{[f.name.replace('collection-','').replace(f.suffix,'') for f in base_dir.glob('collection-*.*')]}")
#             elif len(collection_file) > 1:
#                 lgr.warning("Found more than one collection file matching your search; using first:", collection_file)
#             collection_file = collection_file[0]
        
#         # load; 1-column df (= maps) or 2-column df (= set and maps)
#         collection_df = _fetch_collection_old(collection_file)

#         # apply
#         lgr.info(f"Applying collection filter from: {collection_file}.")
#         pet_maps = [f for f in pet_maps if _rm_ext(f.name) in collection_df["map"].unique()]
            
#     # Load tabulated data if 'parcellation' is specified
#     if parcellation:
#         lgr.info(f"Loading parcellated data: {parcellation}")
#         parcellation_file = tab_dir / f"pet_{parcellation}.csv"
#         if parcellation_file.exists():
#             pet_data = pd.read_csv(parcellation_file, index_col=0)

#             # Apply filter to the dataframe index
#             pet_data = pet_data.loc[
#                 pet_data.index.intersection([_rm_ext(f.name) for f in pet_maps])
#             ]     
            
#             # Apply collection index (-> handles maps that are present multiple times in different sets)
#             if collection:
#                 pet_data = pet_data.loc[collection_df["map"], :]
#                 pet_data_index_orig = pet_data.index.to_list()
#                 pet_data.index = pd.MultiIndex.from_frame(collection_df)
#                 pet_data_index_new = pet_data.index.to_list()
                
#              # Load null maps
#             if return_nulls:
#                 lgr.info("Loading precomputed null maps.")
#                 try:
#                     with gzip.open(nulls_dir / f"pet_{parcellation}.pkl.gz", "rb") as f:
#                         tmp = pickle.load(f)
#                     if collection:
#                         null_maps = {}
#                         for name, nulls in tmp.items():
#                             if name in pet_data_index_orig:
#                                 null_maps[pet_data_index_new[pet_data_index_orig.index(name)]] = nulls
#                     else:
#                         null_maps = tmp
#                 except FileNotFoundError:
#                     lgr.warning("No precomputed null map data found. Did you download it?")
#                     return_nulls = False
                    
#             # cortex only
#             if cortex_only:
#                 lgr.info("Keeping only cortical parcels.")
#                 bool_cx = [True if "_CX_" in c else False for c in pet_data.columns]
#                 pet_data = pet_data.loc[:, bool_cx]
#                 if return_nulls:
#                     null_maps = {name: null_maps[:, bool_cx] for name, null_maps in null_maps.items()}
            
#         else:
#             lgr.warning(f"Parcellated PET data for {parcellation} not found!")
            
#     # Return the results
#     out = pet_maps if not parcellation else pet_data
#     if return_nulls:
#         out = (out, null_maps)
#     return out
    

# def fetch_mrna(genes: Union[str, List[str]] = None,
#                collection: str = None,
#                parcellation: str = "HCPex",
#                cortex_only: bool = False,
#                nispace_data_dir: Union[str, pathlib.Path] = None):
#     # TODO: add filtering information
#     # TODO: modularize collection filter
    
#     # Define the base directories
#     if not nispace_data_dir:
#         base_dir = pathlib.Path.home() / "nispace-data" / "reference" / "mrna"
#     else:
#         base_dir = pathlib.Path(nispace_data_dir) / "reference" / "mrna"
#     tab_dir = base_dir / "tab"
    
#     lgr.info(f"Loading mRNA maps for parcellation {parcellation}.")
    
#     # # OSF CODE
#     # remote_base_url = "https://osf.io/path_to_your_osf_data/"  # Example OSF base URL
    
#     # # Ensure base directory exists
#     # base_dir.mkdir(parents=True, exist_ok=True)

#     # # Check and download data if necessary
#     # _ensure_data_available(base_dir / "your_pet_data_file", remote_base_url + "your_pet_data_file")

#     # Genes
#     if isinstance(genes, str):
#         genes = [genes]
    
#     # Load tabulated data
#     gene_file = tab_dir / f"mrna_{parcellation}.csv.gz"
    
#     if gene_file.exists():
#         mrna_data = pd.read_csv(gene_file, index_col=0)

#         # Apply 'genes' filter to the dataframe index
#         if genes:
#             lgr.info(f"Filtering to {len(genes)} gene(s).")
#             mrna_data = mrna_data.loc[mrna_data.index.intersection(genes)]
#     else:
#         raise FileNotFoundError(f"Parcellated mRNA data for {parcellation} not found!")

#     # Filter by 'collection'
#     if collection:
#         if collection.lower() != "all":
                
#             # check if path to custom file
#             collection_file = pathlib.Path(collection)
#             if collection_file.exists():
#                 pass
            
#             # if not exists, search integrated collections
#             else:
#                 collection_file = list(base_dir.glob(f"collection-{collection.replace('collection-','')}.*"))
#                 if len(collection_file) == 0:
#                     lgr.warning(f"Collection '{collection}' not found! Available: "
#                                 f"{[f.name.replace('collection-','').replace(f.suffix,'') for f in base_dir.glob('collection-*.*')]}")
#                 elif len(collection_file) > 1:
#                     lgr.warning("Found more than one collection file matching your search; using first:", collection_file)
#                 collection_file = collection_file[0]
            
#             # load; 1-column df (= maps) or 2-column df (= set and maps) or 3-columns df (= set, maps, and weights)
#             collection_df = _fetch_collection_old(collection_file)

#             # apply
#             lgr.info(f"Applying collection filter from: {collection_file}.")
#             genes_intersection = mrna_data.index.intersection(collection_df["map"].unique())
#             collection_df_intersection = collection_df.query("map in @genes_intersection")
#             mrna_data = mrna_data.loc[collection_df_intersection["map"]]     
#             mrna_data.index = pd.MultiIndex.from_frame(collection_df_intersection)
            
#     # cortex only
#     if cortex_only:
#         lgr.info("Keeping only cortical parcels.")
#         mrna_data = mrna_data[[c for c in mrna_data.columns if "_CX_" in c]]
    
#     # return
#     return mrna_data


# def fetch_neuroquery(queries: Union[str, List[str]] = None,
#                      collection: str = None,
#                      parcellation: str = "HCPex",
#                      cortex_only: bool = False,
#                      return_nulls: bool = False,
#                      n_proc: int = 1,
#                      generated_maps_dir: Union[str, pathlib.Path] = None,
#                      nispace_data_dir: Union[str, pathlib.Path] = None):
    
#     # Define the base directories
#     if not nispace_data_dir:
#         base_dir = pathlib.Path.home() / "nispace-data" / "reference" / "neuroquery"
#     else:
#         base_dir = pathlib.Path(nispace_data_dir) / "reference" / "neuroquery"
#     model_dir = base_dir / "model"
#     tab_dir = base_dir / "tab"
#     nulls_dir = base_dir / "null"
    
#     lgr.info(f"Loading/Generating Neuroquery maps.")
    
#     # # OSF CODE
#     # remote_base_url = "https://osf.io/path_to_your_osf_data/"  # Example OSF base URL
    
#     # # Ensure base directory exists
#     # base_dir.mkdir(parents=True, exist_ok=True)

#     # # Check and download data if necessary
#     # _ensure_data_available(base_dir / "your_pet_data_file", remote_base_url + "your_pet_data_file")

#     # Queries
#     if not queries:
#         with open(base_dir / f"collection-All.txt", "r") as file:
#             queries = set(file.read().splitlines())
#     if isinstance(queries, str):
#         queries = [queries]
#     nq_data = None
    
#     # Filter by 'collection'
#     if collection:
#         if collection.lower() != "all":
                
#             # check if path to custom file
#             collection_file = pathlib.Path(collection)
#             if collection_file.exists():
#                 pass
            
#             # if not exists, search integrated collections
#             else:
#                 collection_file = list(base_dir.glob(f"collection-{collection.replace('collection-','')}.*"))
#                 if len(collection_file) == 0:
#                     lgr.warning(f"Collection '{collection}' not found! Available: "
#                         f"{[f.name.replace('collection-','').replace(f.suffix,'') for f in base_dir.glob('collection-*.*')]}")
#                 elif len(collection_file) > 1:
#                     lgr.warning("Found more than one collection file matching your search; using first:", collection_file)
#                 collection_file = collection_file[0]
            
#             # load; 1-column df (= maps) or 2-column df (= set and maps) or 3-columns df (= set, maps, and weights)
#             collection_df = _fetch_collection(collection_file)

#             # apply
#             lgr.info(f"Applying collection filter from: {collection_file}.")
#             queries = [q for q in queries if q in collection_df["map"].unique()]
#             collection_df_intersection = collection_df.query("map in @queries")
#             #mrna_data = mrna_data.loc[collection_df_intersection["map"]]     
#             #mrna_data.index = pd.MultiIndex.from_frame(collection_df_intersection)
        
            
#     # Load tabulated data if 'parcellation' is specified
#     if parcellation:
#         lgr.info(f"Loading parcellated data: {parcellation}")
#         parcellation_file = tab_dir / f"neuroquery_{parcellation}.csv.gz"
#         if parcellation_file.exists():
#             nq_data = pd.read_csv(parcellation_file, index_col=0)
                
#             # Apply 'queries' and/or 'collection' filter to the dataframe index
#             if queries:
#                 queries_intersection = nq_data.index.intersection(queries)
#                 if len(queries_intersection) < len(queries):
#                     lgr.warning(f"Only {len(queries_intersection)} of {len(queries)} queries are present in the "
#                                 "parcellated data. These will be kept. Run with 'parcellation=None' to generate "
#                                 "new maps, but be sure to check online if Neuroquery produces meaningful output!")
#                 nq_data = nq_data.loc[queries_intersection]
#                 nq_data_index_orig = nq_data.index.to_list()
#                 if "collection_df_intersection" in locals():
#                     nq_data = nq_data.loc[collection_df_intersection["map"]]
#                     nq_data.index = pd.MultiIndex.from_frame(collection_df_intersection)
#                     nq_data_index_new = nq_data.index.to_list()

#                 # Load null maps
#                 if return_nulls:
#                     lgr.info("Loading precomputed null maps.")
#                     try:
#                         with gzip.open(nulls_dir / f"neuroquery_{parcellation}.pkl.gz", "rb") as f:
#                             tmp = pickle.load(f)                        
#                         if not all([query in tmp.keys() for query in queries]):
#                             lgr.warning("Null maps not available for all selected queries. Returning available.")
#                         if "nq_data_index_new" in locals():
#                             null_maps = {}
#                             for name, nulls in tmp.items():
#                                 if name in nq_data_index_orig:
#                                     null_maps[nq_data_index_new[nq_data_index_orig.index(name)]] = nulls
#                         else:
#                             null_maps = tmp
#                     except FileNotFoundError:
#                         lgr.warning("No precomputed null map data found. Did you download it?")
#                         return_nulls = False
                          
#             # cortex only
#             if cortex_only:
#                 lgr.info("Keeping only cortical parcels.")
#                 bool_cx = [True if "_CX_" in c else False for c in nq_data.columns]
#                 nq_data = nq_data.loc[:, bool_cx]
#                 if return_nulls:
#                     null_maps = {name: null_maps[:, bool_cx] for name, null_maps in null_maps.items()}
            
#         else:
#             lgr.warning(f"Parcellated PET data for {parcellation} not found!")
            
#     # Regenerate if data not available
#     if nq_data is None:
#         lgr.info(f"Using the Neuroquery decoder to generate {len(queries)} brain maps.")
        
#         # save dir
#         if not generated_maps_dir:
#             # create temp dir if not provided
#             generated_maps_dir = (pathlib.Path(tempfile.mkdtemp()) / "neuroquery_maps")
#             generated_maps_dir.mkdir()
        
#         # load model
#         nq_encoder = NeuroQueryModel.from_data_dir(model_dir)

#         # parallelization function
#         def par_fun(query):
#             img = nq_encoder(query)["brain_map"]
#             if not np.allclose(img.get_fdata(), 0):
#                 img_path = generated_maps_dir / (re.sub("[!@#$%^&*()[]{};:,./<>?\|`~-=+ ]", "_", query) + ".nii.gz")
#                 img.to_filename(img_path)
#                 return img_path
#             else:
#                 pass
        
#         # run
#         nq_files = Parallel(n_proc)(delayed(par_fun)(query) for query in tqdm(queries, desc=f"Generating ({n_proc}) proc"))
#         nq_files = [f for f in nq_files if f]
#         lgr.info(f"Generated {len(nq_files)} valid Neuroquery maps.")
        
#         # return
#         return nq_files
    
#     # return
#     else:
#         out = nq_data
#         if "null_maps" in locals():
#             out = (out, null_maps)
#         return out
    
    

