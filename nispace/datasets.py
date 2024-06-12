from typing import Union, List, Dict, Tuple
import requests
import pathlib
import os
import io
import zipfile
import pandas as pd
import numpy as np
from nilearn import image
import gzip
import pickle

from . import lgr
from .modules.constants import (_DSETS_VERSION, _DSETS, _DSETS_NICE, _DSETS_MAP, _DSETS_TAB,
                                _PARCS_DEFAULT)
from .stats.misc import zscore_df
from .utils import _rm_ext, set_log
from .io import read_json, load_img, load_distmat, load_labels


# OSF HANDLING =====================================================================================

def download_datasets(datasets="all", 
                      nispace_data_dir: Union[str, pathlib.Path] = None, 
                      osf_id: str = "derpj"):
    """
    Download all data associated with NiSpace.

    Parameters
    ----------
    datasets : str or list of str, optional
        The datasets to download. Options are "template", "parcellation", "reference", "example", or "all".
        Default is "all".
    nispace_data_dir : str or pathlib.Path, optional
        The directory where the data will be downloaded. Default is `~HOME/nispace-data`.
    osf_id : str, optional
        The OSF project ID from which to download the data. Default is "derpj".

    Raises
    ------
    ValueError
        If the specified datasets are not in the list of available datasets.
    FileNotFoundError
        If the specified directory does not exist and cannot be created.
    requests.exceptions.RequestException
        If there is an issue with the HTTP request to the OSF API.

    Notes
    -----
    This function downloads approximately 250 MB of data. This process will be updated in the future
    by integrating downloads into the "fetch_...()" functions and only downloading necessary data.
    """
    
    lgr.info("Downloading all data associated with NiSpace (~250 MB). This will take some time.\n"
             "By default, data will be downloaded to ~HOME/nispace-data. "
             "You can also download the data manually from https://osf.io/derpj\n"
             "This strategy will change in the future to a more reliable, version-tracked one, "
             "integrated into the fetch_...() functions.")
    
    if nispace_data_dir is None:
        nispace_data_dir = pathlib.Path.home() / "nispace-data"
        
    dsets_all = ["template", "parcellation", "reference", "example"]
    if isinstance(datasets, str):
        datasets = [datasets]
    if datasets in [["all"], [None]]:
        datasets = dsets_all
    else:
        if not all(dset in dsets_all for dset in datasets):
            raise ValueError(f"Datasets must be 'all' or one or more of {dsets_all}")
    
    if not os.path.exists(nispace_data_dir):
        os.makedirs(nispace_data_dir)
    
    # Get a list of files in the project
    response = requests.get(f'https://api.osf.io/v2/nodes/{osf_id}/files/osfstorage/')
    response.raise_for_status()
    files = response.json()['data']
    
    for file in files:
        if file["attributes"]["kind"] == "folder":
            folder_name = file["attributes"]["name"]
            if folder_name in datasets:
                folder_id = file["attributes"]["path"]
                local_path = pathlib.Path(nispace_data_dir, folder_name)
                local_path.mkdir(parents=True, exist_ok=True)
                lgr.info(f"Downloading '{folder_name}' data to '{local_path}'")
                response = requests.get(f'https://files.osf.io/v1/resources/{osf_id}/providers/osfstorage/{folder_id}/?zip=')
                response.raise_for_status()
                # Open the ZIP file
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    # Extract all files to the specified directory
                    z.extractall(local_path)

def _check_base_dir(base_dir, data_type):
    base_dir = pathlib.Path(base_dir)
    if not base_dir.exists():
        lgr.critical_raise(f"{data_type} data not found. Run 'nispace.datasets.download_datasets()' "
                           "to download it or adjust 'nispace_data_path'!",
                           FileNotFoundError)
    elif len(os.listdir(base_dir)) < 1:
        lgr.critical_raise(f"{data_type} directory exists but data not found! Maybe something "
                           "went wrong with the download?!",
                           FileNotFoundError)
        
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
                   nispace_data_dir: Union[str, pathlib.Path] = None,
                   dset_version: str = None):
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
        raise ValueError("'template' should be 'MNI152' or 'fsaverage'!")
    
    # paths        
    if nispace_data_dir is None:
        nispace_data_dir = pathlib.Path.home() / "nispace-data"
    base_dir = pathlib.Path(nispace_data_dir) / "template" / template
    map_dir = base_dir / "map"
    tab_dir = base_dir / "tab"
    
    # OSF
    _check_base_dir(base_dir, "Template")
    
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
    
    # OSF
    _check_base_dir(base_dir, "Parcellation")
    
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
            collection = read_json(collection_path)
                
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

    # OSF
    _check_base_dir(base_dir, "Reference")
    
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
    
    # OSF
    _check_base_dir(base_dir, "Reference")
    
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
    
    # OSF
    _check_base_dir(base_dir, "Example")

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
