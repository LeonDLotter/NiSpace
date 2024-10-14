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
from typing import Literal

from sklearn.decomposition import non_negative_factorization

from . import lgr
from .modules.constants import (_DSETS, _DSETS_NICE, _DSETS_TAB_ONLY,
                                _DSETS_CX_ONLY, _DSETS_SC_ONLY,
                                _PARCS_DEFAULT)
from .stats.misc import zscore_df
from .utils.utils import _rm_ext, set_log
from .utils.utils_datasets import get_file
from .io import read_json, load_img, load_distmat, load_labels
from .nulls import _img_density_for_neuromaps

datalib_dir = pathlib.Path(__file__).parent / "datalib"
reference_lib = read_json(datalib_dir / "reference.json")
template_lib = read_json(datalib_dir / "template.json")
parcellation_lib = read_json(datalib_dir / "parcellation.json")
example_lib = read_json(datalib_dir / "example.json")

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
                   res: str = None,
                   desc: str = None,
                   #parcellation: str = None,
                   hemi: Union[List[str], str] = ["L", "R"],
                   nispace_data_dir: Union[str, pathlib.Path] = None,
                   verbose: bool = True):
    """
    Fetch a brain template.
    
    Parameters
    ----------
    template : str, optional
        The template to fetch. Default is "mni152".
        
    res : str, optional
        The resolution of the template to fetch. If None, will default to "1mm" for MNI152 and 
        "10k" for fsaverage.
        
    desc : str, optional
        The description of the template to fetch. If None, will default to "T1" for MNI152 and 
        "pial" for fsaverage.
        
    hemi : list of str, optional
        The hemispheres to fetch. Default is ["L", "R"].
        
    nispace_data_dir : str or pathlib.Path, optional
        The directory containing the NiSpace data. Default is None.
        
    Returns
    -------
    The template.
    """
    verbose = set_log(lgr, verbose)
    
    # hard-coded template type    
    if "mni" in template.lower():
        template = "mni152"
    elif "fsa" in template.lower():
        template = "fsaverage"
    else:
        raise ValueError("template should be 'MNI152' or 'fsaverage'!")
    
    # paths        
    if nispace_data_dir is None:
        nispace_data_dir = pathlib.Path.home() / "nispace-data"
    base_dir = pathlib.Path(nispace_data_dir) / "template" / template
    map_dir = base_dir / "map"
    
    # get files
    # mni
    if template == "mni152":
        # default resolution
        if res is None:
            res = "1mm"
        if res not in template_lib["mni152"]:
            raise ValueError(f"res = '{res}' not defined. Choose one of '1mm', '2mm', or '3mm'!")
        # default desc
        if desc is None:
            desc = "T1"
        if desc not in template_lib["mni152"][res]:
            raise ValueError(f"desc = '{desc}' not defined. Choose one of {list(template_lib['mni152'][res].keys())}!")
        # get file
        lgr.info(f"Loading MNI152NLin2009cAsym '{desc}' template in '{res}' resolution.")
        tpl_path = map_dir / f"MNI152NLin20009cAsym_desc-{desc}_res-{res}.nii.gz"
        tpl_file = get_file(tpl_path, **template_lib["mni152"][res][desc])

    # fsa
    elif template == "fsaverage":
        # default resolution
        if res is None:
            res = "10k"
        if res not in template_lib["fsaverage"]:
            raise ValueError(f"res = '{res}' not defined. Choose one of {list(template_lib['fsaverage'].keys())}!")
        # default desc
        if desc is None:
            desc = "pial"
        if desc not in template_lib["fsaverage"][res]:
            raise ValueError(f"desc = '{desc}' not defined. Choose one of {list(template_lib['fsaverage'][res].keys())}!")
        # hemi
        if isinstance(hemi, str):
            hemi = [hemi]
        if hemi not in [["L"], ["R"], ["L", "R"]]:
            raise ValueError(f"hemi = '{hemi}' not defined. Choose one of 'L', 'R', or ['L', 'R']!")
        # get file(s)
        lgr.info(f"Loading fsaverage '{desc}' template in '{res}' resolution.")
        tpl_file = ()
        for h in hemi:
            tpl_path = map_dir / f"fsaverage_desc-{desc}_res-{res}_hemi-{h}.surf.gii"
            tpl_file += get_file(tpl_path, **template_lib["fsaverage"][res][desc][h]),
        if len(tpl_file) == 1:
            tpl_file = tpl_file[0]
            
    # return
    return tpl_file

# PARCELLATIONS ====================================================================================

def fetch_parcellation(parcellation: str = _PARCS_DEFAULT, 
                       space: Literal["mni152", "fsaverage"] = None,
                       #n_parcels: Union[int, str] = None,
                       #resolution: str = None,
                       hemi: Union[List[str], str] = ["L", "R"],
                       cortex_only: bool = False,
                       subcortex_only: bool = False,
                       return_labels: bool = True,
                       return_space: bool = False,
                       return_resolution: bool = False,
                       return_dist_mat: bool = False,
                       return_loaded: bool = False,
                       nispace_data_dir: Union[str, pathlib.Path] = None):
    
    # Check available
    if parcellation not in parcellation_lib:
        lgr.critical_raise(f"Parcellation '{parcellation}' not found. Available: {list(parcellation_lib.keys())}",
                           ValueError)
    # Check space
    if space is None:
        # get default space -> first space listed in parcellation_lib
        space = list(parcellation_lib[parcellation].keys())[0]
    else:
        if space.lower() not in parcellation_lib[parcellation]:
            lgr.critical_raise(f"Space '{space}' not found for parcellation '{parcellation}'. "
                               f"Available: {list(parcellation_lib[parcellation].keys())}",
                               ValueError)
        
    # base dir
    if not nispace_data_dir:
        base_dir = pathlib.Path.home() / "nispace-data" / "parcellation" / parcellation / space
    else:
        base_dir = pathlib.Path(nispace_data_dir) / "parcellation" / parcellation / space
    
    # LOAD
    lgr.info(f"Loading parcellation '{parcellation}' in '{space}' space.")
    
    # volume
    if space.lower() == "mni152":
        
        # get files
        parcellation_file = get_file(
            base_dir / f"{parcellation}_space-{space}.label.nii.gz", 
            **parcellation_lib[parcellation][space]["map"]
        )
        if return_labels or cortex_only or subcortex_only:
            label_file = get_file(
                base_dir / f"{parcellation}_space-{space}.label.txt",
                **parcellation_lib[parcellation][space]["label"]
            )
        if return_dist_mat:
            distmat_file = get_file(
                base_dir / f"{parcellation}_space-{space}.dist.csv.gz",
            **parcellation_lib[parcellation][space]["distmat"]
        )
    
        # cortex only:
        if cortex_only and subcortex_only:
            lgr.error("Cannot set both 'cortex_only' and 'subcortex_only' to True. Returning all!")
            cortex_only = False
            subcortex_only = False
            
        if cortex_only or subcortex_only:
            # specify the indices * we want to remove *
            labels = load_labels(label_file)
            str_rm = "_SC_" if cortex_only else "_CX_"
            idc_rm = [int(l.split("_")[0]) for l in labels if str_rm in l]
            lgr.info(f"Removing {len(idc_rm)} {'subcortical' if str_rm=='_SC_' else 'cortical'} "
                     "parcels. Will return Nifti1 object instead of path!")
            # drop from parcellation
            parc = load_img(parcellation_file)
            parc_array = parc.get_fdata()
            for idx in idc_rm:
                parc_array[parc_array==idx] = 0
            parc = image.new_img_like(parc, parc_array, copy_header=True)
            # drop from labels
            labels = [l for l in labels if str_rm not in l]
            # replace vars
            parcellation_file, label_file = parc, labels
            # drop from dist mat
            if return_dist_mat:
                bool_keep = np.array([False if str_rm in l else True for l in labels])
                distmat = load_distmat(distmat_file)
                distmat = distmat[np.ix_(bool_keep, bool_keep)]
                distmat_file = distmat
            
            
    # surface
    elif space.lower() in ["fsaverage", "fslr"]:
        
        # check hemis
        if isinstance(hemi, str):
            hemi = [hemi]
        if hemi not in [["L"], ["R"], ["L", "R"]]:
            raise ValueError(f"hemi = '{hemi}' not defined. Choose one of 'L', 'R', or ['L', 'R']!")

        # get files
        parcellation_file, label_file, distmat_file = (), (), ()
        for h in hemi:
            parcellation_file += get_file(
                base_dir / f"{parcellation}_space-{space}_hemi-{h}.label.gii.gz", 
                **parcellation_lib[parcellation][space]["map"][h]
            ),
            if return_labels or cortex_only or subcortex_only:
                label_file += get_file(
                    base_dir / f"{parcellation}_space-{space}_hemi-{h}.label.txt",
                    **parcellation_lib[parcellation][space]["label"][h]
                ),
            if return_dist_mat:
                distmat_file += get_file(
                    base_dir / f"{parcellation}_space-{space}_hemi-{h}.dist.csv.gz",
                    **parcellation_lib[parcellation][space]["distmat"][h]
                ),
        if len(parcellation_file) == 1:
            parcellation_file, label_file, distmat_file = parcellation_file[0], label_file[0], distmat_file[0]
                   
                     
    # return
    # parc
    out = (load_img(parcellation_file) if return_loaded else parcellation_file),
    # label
    if return_labels:
        out += (load_labels(label_file) if return_loaded else label_file),
    # space
    if return_space:
        out += space,
    # res
    if return_resolution:
        out += _img_density_for_neuromaps(load_img(parcellation_file)),
    # distmat
    if return_dist_mat:
        out += (load_distmat(distmat_file) if return_loaded else distmat_file),
    
    return out

# REFERENCE DATA - PRIVATE =========================================================================

def _filter_maps(maps_avail: List[str], 
                 maps: Union[str, List[str], Dict[str, Union[str, list]]]) -> List[pathlib.Path]:
    
    def matches_filters(map_name: str, filters: Dict[str, Union[str, List[str]]]) -> bool:
        for filter_name, filter_content in filters.items():
            if filter_content not in [None, False, "", []]:
                if isinstance(filter_content, (str, int)):
                    filter_content = [filter_content]
                filter_content = list(map(str, filter_content))
                if filter_name == "n" and filter_content[0].startswith(">"):
                    try:
                        filter_n = int(filter_content[0].replace(">", ""))
                        n_value = int(_file_desc(map_name, 2))
                        if n_value <= filter_n:
                            return False
                    except (ValueError, IndexError):
                        continue  # Skip this filter if parsing fails
                else:
                    if not any(f"{filter_name}-{content}".lower() in map_name.lower() 
                               for content in filter_content):
                        return False
        return True

    if isinstance(maps, str):
        maps = [maps]
    if isinstance(maps, list):
        maps = list(set(maps))
        filtered_maps = [f for f in maps_avail if any(map_str in f for map_str in maps)]
    elif isinstance(maps, dict):
        filtered_maps = [f for f in maps_avail if matches_filters(f, maps)]
    else:
        filtered_maps = maps_avail
        
    return filtered_maps


def _fetch_collection(collection_path):
    
    # if path, read file
    if isinstance(collection_path, (str, pathlib.Path)):
        collection_path = pathlib.Path(collection_path)
        ext = collection_path.suffix
        
        # if "collect" file, detect if dict or table
        if ext == ".collect":
            with open(collection_path) as f:
                header = f.readline()
                if header.startswith("{"):
                    ext = ".json"
                else:
                    ext = ".csv"
        
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
            collection = pd.read_csv(collection_path, header=header, sep=",")
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


def _apply_collection_filter(dataset: str,
                             map_files: List[Union[str, pathlib.Path]], 
                             collection: str,
                             base_dir: pathlib.Path,
                             set_size_range: Union[None, Tuple[int, int]] = None) -> List[pathlib.Path]:
    
    # Check if path to custom file
    collection_path = pathlib.Path(collection)
    if not collection_path.exists():
        # If not exists, search integrated collections
        if collection in reference_lib[dataset]["collection"]:
            collection_path = base_dir / f"collection-{collection}.collect"
            collection_file = get_file(collection_path, **reference_lib[dataset]["collection"][collection])
        else:
            lgr.warning(f"Collection '{collection}' not found! Available: "
                        f"{list(reference_lib[dataset]['collection'].keys())}")
            return map_files, None

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
            set_size_range = [
                x if x is not None else x_ 
                for x, x_ 
                in zip(set_size_range, (1, np.inf))
            ]
            lgr.info(f"Filtering to collection sets with between {set_size_range[0]} and "
                     f"{set_size_range[1]} maps.")
            collection_df = (
                collection_df
                .groupby("set")
                .filter(lambda x: set_size_range[0] <= x.shape[0] <= set_size_range[1])   
            )         
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
                           subcortex_only: bool,
                           standardize: bool) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    lgr.info(f"Loading parcellated data: {parcellation}")
    parcellation_file = tab_dir / f"{dataset}_{parcellation}.csv.gz"
    lgr.debug(f"Loading {parcellation_file}")
    
    # Load parcellated data
    data = pd.read_csv(
        get_file(parcellation_file, **reference_lib[dataset]["tab"][parcellation]), 
        index_col=0
    )
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
    if cortex_only and subcortex_only:
        lgr.error("Cannot set both 'cortex_only' and 'subcortex_only' to True. Returning all!")
        cortex_only = False
        subcortex_only = False
    
    if cortex_only or subcortex_only:
        str_rm = "_SC_" if cortex_only else "_CX_"
        lgr.info(f"Removing {'subcortical' if str_rm=='_SC_' else 'cortical'} parcels.")
        bool_keep = np.array([False if str_rm in c else True for c in data.columns])
        data = data.loc[:, bool_keep]
        if return_nulls:
            null_maps = {name: null_maps[:, bool_keep] 
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
    
    # info file
    def get_ref_info(dataset):
        get_line = False
        msg = ""
        with open(datalib_dir / "reference.txt", "r") as f:
            for line in f:
                if line.lower().startswith(f"# {dataset.lower()}"):
                    get_line = True
                    continue                
                if get_line and line == "\n":
                    break
                if get_line:
                    msg += line
        msg += "\n"
        return msg
                    
    # PET
    if dataset.lower() == "pet":
        msg = get_ref_info(dataset)
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
        msg = get_ref_info(dataset)
        if meta is not None:
            if len(meta) > 0:
                collection_maxlen = max([len(x) for x in meta["collection"]])
                author_maxlen = max([len(x) for x in meta["author"]])
                for collection, pub, doi in zip(meta["collection"], meta["author"], meta["doi"]):
                    collection = collection.ljust(collection_maxlen)
                    author = pub.capitalize().ljust(author_maxlen)
                    msg += f"- {collection}  Source: {author}  https://doi.org/{doi}\n"
    
    # RSN
    elif dataset.lower() == "rsn":
        msg = get_ref_info(dataset)
        if meta is not None:
            if len(meta) > 0:
                author_maxlen = max([len(x) for x in meta["author"]])
                for pub, doi in zip(meta["author"], meta["doi"]):
                    author = pub.capitalize().ljust(author_maxlen)
                    msg += f"- {author}  https://doi.org/{doi}\n"

    # print
    # if msg[-2:] != "\n":
    #     msg += "\n"
    print(msg)
    
    
# REFERENCE DATA - PUBLIC ==========================================================================

def fetch_reference(dataset: str,
                    maps: Union[None, str, List[str], Dict[str, Union[str, list]]] = None,
                    collection: str = None,
                    set_size_range: Union[None, Tuple[int, int]] = None,
                    parcellation: str = None,
                    standardize_parcellated: bool = True,
                    cortex_only: bool = False,
                    subcortex_only: bool = False,
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

    # Get list of maps
    if dataset not in _DSETS_TAB_ONLY:
        maps_avail = list(reference_lib[dataset]["map"].keys())
    else:
        if parcellation is None:
            lgr.warning(f"mRNA data requires a parcellation. Defaulting to: '{_PARCS_DEFAULT}'.")
            parcellation = _PARCS_DEFAULT
        if return_nulls:
            lgr.warning("Precomputed null maps are not available for mRNA data. Will not return any.")
            return_nulls = False
        maps_avail = pd.read_csv(
            get_file(tab_dir / f"{dataset}_{parcellation}.csv.gz", **reference_lib[dataset]["tab"][parcellation]), 
            index_col=0
        ).index.to_list()
        
    lgr.debug(f"Loaded {len(maps_avail)} unfiltered map(s). "
              f"First 5: {maps_avail[:5] if len(maps_avail) >= 5 else maps_avail[:len(maps_avail)]}")

    # Filter by 'maps'
    if maps:
        n_tmp = len(maps_avail)
        lgr.info(f"Applying filter: {maps}")
        if dataset not in _DSETS_TAB_ONLY:
            maps_avail = _filter_maps(maps_avail, maps)
        else:
            if isinstance(maps, str):
                maps = [maps]
            elif not isinstance(maps, (list, tuple, set, pd.Series)):
                lgr.warning(f"For dataset '{dataset}', 'maps' must be list-like. Skipping filter.")
                maps = maps_avail
            maps_avail = list(set(maps_avail).intersection(maps))
        lgr.info(f"Filtered from {n_tmp} to {len(maps_avail)} maps.")
            
    # Filter by 'collection'
    if collection == "All":
        collection = None
    if collection:
        maps_avail, collection_df = _apply_collection_filter(dataset, maps_avail, collection, 
                                                             base_dir, set_size_range)
    else:
        collection_df = None

    # Load tabulated data if 'parcellation' is specified
    if parcellation:
        # for now, no null data included. TODO: re-evaluate cost/benefits
        if return_nulls:
            lgr.warning("Pre-calculated null maps are currently not available. Will not return any.")
            return_nulls = False
        # cortex only for specific datasets
        if dataset in _DSETS_CX_ONLY:
            lgr.warning(f"Dataset '{dataset}' is cortex-only. Will not return subcortical parcels.")
            cortex_only = True
            subcortex_only = False
        # subcortex only for specific datasets
        if dataset in _DSETS_SC_ONLY:
            lgr.warning(f"Dataset '{dataset}' is subcortex-only. Will not return cortical parcels.")
            cortex_only = False
            subcortex_only = True
        # get data
        data = _load_parcellated_data(
            dataset=dataset, 
            tab_dir=tab_dir, 
            parcellation=parcellation, 
            map_files=maps_avail, 
            collection_df=collection_df,
            return_nulls=return_nulls, 
            nulls_dir=nulls_dir, 
            cortex_only=cortex_only,
            subcortex_only=subcortex_only,
            standardize=standardize_parcellated
        )
        
    # Fetch paths to maps if no 'parcellation' is specified
    else:
        data = [
            get_file(
                local_path=map_dir / f"{m}.nii.gz", 
                **reference_lib[dataset]["map"][m], 
                process_img=True,
                override_file_format="nifti"
            ) 
            for m in maps_avail
        ]
        
    # Print references
    # for maps if "pet", or for sets if "mrna"
    if return_metadata or print_references:
        if dataset == "pet":
            meta = fetch_metadata(dataset, maps_avail)
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
    meta = pd.read_csv(get_file(base_dir / "metadata.csv", **reference_lib[dataset]["metadata"]))
    
    if dataset == "pet" and maps is not None:
        if isinstance(maps, str):
            maps = [maps]
        meta = meta[meta.atlas.str.contains("|".join(maps), na=False)]
    elif dataset == "mrna" and collection is not None:
        meta = meta.query("collection == @collection")
    elif dataset == "rsn":
        meta = None
            
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

    # check available
    example = example.lower()
    if example not in example_lib:
        lgr.critical_raise(f"Example '{example}' not found. Available: {list(example_lib.keys())}",
                           ValueError)
    parc_name = example_lib[example]["parc"]
    
    # load
    lgr.info(f"Loading example dataset: '{example}'. The data was parcellated with: '{parc_name}'.")
    example_data = pd.read_csv(
        get_file(base_dir / f"example-{example}_parc-{parc_name}.csv.gz", **example_lib[example]["tab"]), 
        index_col=0
    )

    # Check for info data 
    if return_associated_data and "info" in example_lib[example]:
        lgr.info("Returning parcellated and associated subject data.")
        example_info = pd.read_csv(
            get_file(base_dir / f"example-{example}_info.csv", **example_lib[example]["info"]), 
            index_col=0
        )
        return example_data, example_info
    else:
        return example_data
