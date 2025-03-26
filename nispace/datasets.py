from typing import Union, List, Dict, Tuple
import pathlib
import pandas as pd
import numpy as np
from nilearn import image
import shutil
from typing import Literal

from . import lgr
from .modules.constants import _PARC_DEFAULT, _SPACE_DEFAULT
from .stats.misc import zscore_df
from .utils.utils import _rm_ext, set_log
from .utils.utils_datasets import get_file
from .io import read_json, load_img, load_distmat, load_labels, load_l2rmap
from .nulls import _img_density_for_neuromaps

datalib_dir = pathlib.Path(__file__).parent / "datalib"
reference_lib = read_json(datalib_dir / "reference.json")
template_lib = read_json(datalib_dir / "template.json")
parcellation_lib = read_json(datalib_dir / "parcellation.json")
example_lib = read_json(datalib_dir / "example.json")

def keys2list(dct):
    return list(dct.keys())

def keys2str(dct, sep=", "):
    return sep.join(list(dct.keys()))


# EMPTY NISPACE DATA DIR ===========================================================================

_EMPTY_DATA_CONFIRMED = False
def empty_nispace_data_dir(nispace_data_dir: Union[str, pathlib.Path] = None):
    global _EMPTY_DATA_CONFIRMED
    if nispace_data_dir is None:
        nispace_data_dir = pathlib.Path.home() / "nispace-data"
    if not _EMPTY_DATA_CONFIRMED:
        lgr.warning("If you call this function again, it will remove all contents of your NiSpace "
                    f"data directory at {nispace_data_dir}.")
        lgr.warning("Call it again to proceed.")
        _EMPTY_DATA_CONFIRMED = True
    else:
        lgr.warning(f"Emptying nispace data dir at {nispace_data_dir}.")
        shutil.rmtree(nispace_data_dir)
        nispace_data_dir.mkdir(parents=True, exist_ok=True)


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

def fetch_template(template: str = _SPACE_DEFAULT, 
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
        The template to fetch. Default is "MNI152NLin2009cAsym".
        
    res : str, optional
        The resolution of the template to fetch. If None, will default to "1mm" for MNI152 and 
        "10k" for fsaverage.
        
    desc : str, optional
        The description of the template to fetch. If None, will default to "T1w" for MNI152 and 
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
    
    # check if template exists
    if template not in template_lib:
        raise ValueError(f"Template '{template}' not found. Available: {keys2str(template_lib)}")
    
    # paths        
    if nispace_data_dir is None:
        nispace_data_dir = pathlib.Path.home() / "nispace-data"
    base_dir = pathlib.Path(nispace_data_dir) / "template" / template
    map_dir = base_dir / "map"
    
    # set defaults:
    if "mni" in template.lower():
        res = "1mm" if res is None else res
        desc = "T1w" if desc is None else desc
        hemi = None
    elif "fsa" in template.lower():
        res = "41k" if res is None else res
        desc = "pial" if desc is None else desc
        if hemi is None:
            hemi = ["L", "R"]
    
    # check settings
    if res not in template_lib[template]:
        raise ValueError(f"res = '{res}' not defined. Choose one of {keys2str(template_lib[template])}!")
    if desc not in template_lib[template][res]:
        raise ValueError(f"desc = '{desc}' not defined. Choose one of {keys2str(template_lib[template][res])}!")
    if hemi is not None:
        if isinstance(hemi, str):
            hemi = [hemi]
        if hemi not in [["L"], ["R"], ["L", "R"]]:
            raise ValueError(f"hemi = '{hemi}' not defined. Choose one of 'L', 'R', or ['L', 'R']!")
    
    # get file
    lgr.info(f"Loading {template} '{desc}' template in '{res}' resolution.")
    if "mni" in template.lower():
        tpl_path = map_dir / desc / f"tpl-{template}_desc-{desc}_res-{res}.nii.gz"
        tpl_file = get_file(tpl_path, **template_lib[template][res][desc])
    else:
        tpl_file = ()
        for h in hemi:
            tpl_path = map_dir / desc / f"tpl-{template}_desc-{desc}_res-{res}_hemi-{h}.surf.gii"
            tpl_file += get_file(tpl_path, **template_lib[template][res][desc][h]),
        if len(tpl_file) == 1: 
            tpl_file = tpl_file[0]
    
    # return
    return tpl_file

# PARCELLATIONS ===================================================================================

def _parc_alias(parcellation: str):
    if "alias" in parcellation_lib[parcellation]:
        parc = parcellation_lib[parcellation]["alias"]
        cortex = parcellation_lib[parcellation]["cortex"]
        subcortex = parcellation_lib[parcellation]["subcortex"]
    else:
        parc = parcellation
        cortex, subcortex = True, True
    return parc, cortex, subcortex

def _parc_symmetric(parc_labels):
    labels_lh = [l.split("_LH_")[1] for l in parc_labels if "_LH_" in l]
    labels_rh = [l.split("_RH_")[1] for l in parc_labels if "_RH_" in l]
    if not labels_lh or not labels_rh:
        return False
    if labels_lh == labels_rh:
        return True
    return False

def fetch_parcellation(parcellation: str = _PARC_DEFAULT, 
                       space: str = None,
                       hemi: Union[List[str], str] = ["L", "R"],
                       return_labels: bool = True,
                       return_space: bool = False,
                       return_resolution: bool = False,
                       return_symmetric: bool = False,
                       return_l2rmap: bool = False,
                       return_dist_mat: bool = False,
                       return_loaded: bool = False,
                       nispace_data_dir: Union[str, pathlib.Path] = None):
    
    # Check if in main parcellation list
    if parcellation not in parcellation_lib:
        lgr.critical_raise(f"Parcellation '{parcellation}' not found. Available: {keys2str(parcellation_lib)}",
                           ValueError)
        
    # Check if alias and set data to retrieve
    # variable "parcellation" is now what the user sees, "parc" is what we go with internally
    parc, cortex, subcortex = _parc_alias(parcellation)
        
    # Check space
    if space is None:
        # get default space -> first space listed in parcellation_lib
        space = list(parcellation_lib[parc].keys())[0]
    else:
        if space not in parcellation_lib[parc]:
            lgr.critical_raise(f"Space '{space}' not found for parcellation '{parcellation}'. "
                               f"Available: {keys2str(parcellation_lib[parc])}",
                               ValueError)
    
    # Symmetry
    if "l2rmap" in parcellation_lib[parc][space]:
        symmetric = False
    else:
        symmetric = True
    
    # base dir
    if not nispace_data_dir:
        base_dir = pathlib.Path.home() / "nispace-data" / "parcellation" / parc / space
    else:
        base_dir = pathlib.Path(nispace_data_dir) / "parcellation" / parc / space
    
    # LOAD
    lgr.info(f"Loading parcellation '{parcellation}' in '{space}' space.")
    
    # volume
    if "mni" in space.lower():
        
        # get files
        parcellation_file = get_file(
            base_dir / f"parc-{parc}_space-{space}.label.nii.gz", 
            **parcellation_lib[parc][space]["map"]
        )
        if return_labels or not cortex or not subcortex:
            label_file = get_file(
                base_dir / f"parc-{parc}_space-{space}.label.txt",
                **parcellation_lib[parc][space]["label"]
            )
        if return_l2rmap and not symmetric:
            l2rmap_file = get_file(
                base_dir / f"parc-{parc}_space-{space}.l2rmap.csv.gz",
                **parcellation_lib[parc][space]["l2rmap"]
            )
        elif return_l2rmap and symmetric:
            l2rmap_file = None
        if return_dist_mat:
            distmat_file = get_file(
                base_dir / f"parc-{parc}_space-{space}.dist.csv.gz",
                **parcellation_lib[parc][space]["distmat"]
            )
    
        # cortex only:
        if not cortex and not subcortex:
            lgr.error("Cannot set both 'cortex' and 'subcortex' to False. Returning all!")
            cortex, subcortex = True, True
        if not cortex or not subcortex:
            lgr.info(f"{parcellation} is a {['cortex', 'subcortex'][not cortex]} version of the "
                     f"whole-brain parcellation {parc}.")
            # get the labels we want to keep
            labels_all = load_labels(label_file)
            str_to_keep = "_CX_" if cortex else "_SC_"
            labels_to_keep = [l for l in labels_all if str_to_keep in l]
            # get the indices we want to remove
            idc_rm = [int(l.split("_")[0]) for l in labels_all if l not in labels_to_keep]
            lgr.info(f"Removing {len(idc_rm)} {['cortical', 'subcortical'][not cortex]} parcels and "
                     "returning Nifti1 object instead of path!")
            # drop indices from parcellation
            parc = load_img(parcellation_file)
            parc_array = parc.get_fdata()
            for idx in idc_rm:
                parc_array[parc_array==idx] = 0
            parc = image.new_img_like(parc, parc_array, copy_header=True)
            # replace vars
            parcellation_file, label_file = parc, labels_to_keep
            # drop from left-to-right mapping
            if return_l2rmap and not symmetric:
                l2rmap = load_l2rmap(l2rmap_file)
                l2rmap = l2rmap.loc[l2rmap.index.intersection(labels_to_keep), 
                                    l2rmap.columns.intersection(labels_to_keep)]
                l2rmap_file = l2rmap
            # drop from dist mat
            if return_dist_mat:
                bool_keep = np.array([True if l in labels_to_keep else False for l in labels_all])
                distmat = load_distmat(distmat_file)
                distmat = distmat[np.ix_(bool_keep, bool_keep)]
                distmat_file = distmat
            
    # surface
    else:
        
        # check hemis
        if isinstance(hemi, str):
            hemi = [hemi]
        if hemi not in [["L"], ["R"], ["L", "R"]]:
            raise ValueError(f"hemi = '{hemi}' not defined. Choose one of 'L', 'R', or ['L', 'R']!")

        # get files
        parcellation_file, label_file, distmat_file = (), (), ()
        for h in hemi:
            parcellation_file += get_file(
                base_dir / f"parc-{parc}_space-{space}_hemi-{h}.label.gii.gz", 
                **parcellation_lib[parc][space]["map"][h]
            ),
            if return_labels:
                label_file += get_file(
                    base_dir / f"parc-{parc}_space-{space}_hemi-{h}.label.txt",
                    **parcellation_lib[parc][space]["label"][h]
                ),
            if return_dist_mat:
                if "fslr" in space.lower():
                    lgr.warning("Distance matrices for fslr spaces are currently not available. Returning None.")
                    distmat_file += None,
                else:
                    distmat_file += get_file(
                        base_dir / f"parc-{parc}_space-{space}_hemi-{h}.dist.csv.gz",
                        **parcellation_lib[parc][space]["distmat"][h]
                    ),
        if return_l2rmap and not symmetric:
            l2rmap_file = get_file(
                base_dir / f"parc-{parc}_space-{space}.l2rmap.csv.gz",
                **parcellation_lib[parc][space]["l2rmap"]
            )
        elif return_l2rmap and symmetric:
            l2rmap_file = None
        if len(parcellation_file) == 1:
            parcellation_file, label_file, distmat_file, l2rmap_file = parcellation_file[0], label_file[0], distmat_file[0], None
        
    
    # return      
    
    # build output
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
    # symmetric
    if return_symmetric:
        out += symmetric,
    # l2rmap
    if return_l2rmap:
        out += (load_l2rmap(l2rmap_file) if return_loaded else l2rmap_file),
    # distmat
    if return_dist_mat:
        out += (load_distmat(distmat_file) if return_loaded else distmat_file),
    # index into tuple if length is 1
    if len(out) == 1:
        out = out[0]
    
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
                        f"{keys2str(reference_lib[dataset]['collection'])}")
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
                           parc: str, 
                           map_files: List[str],
                           collection_df: pd.DataFrame,
                           cortex: bool,
                           subcortex: bool,
                           standardize: bool) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    lgr.info(f"Loading parcellated data: {parc}")
    parcellation_file = tab_dir / f"dset-{dataset}_parc-{parc}.csv.gz"
    lgr.debug(f"Loading {parcellation_file}")
    
    # Load parcellated data
    data = pd.read_csv(
        get_file(parcellation_file, **reference_lib[dataset]["tab"][parc]), 
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
        
    # Filter to keep only cortical parcels if requested
    if not cortex and not subcortex:
        lgr.error("Cannot set both 'cortex' and 'subcortex' to False. Returning all!")
        cortex, subcortex = True, True
    if not cortex or not subcortex:
        str_to_keep = "_CX_" if cortex else "_SC_"
        bool_keep = np.array([True if str_to_keep in c else False for c in data.columns])
        lgr.info(f"Removing {bool_keep.sum()} {['cortical', 'subcortical'][not cortex]} parcels.")
        data = data.loc[:, bool_keep]
        
    # Standardize
    if standardize:
        lgr.info("Standardizing parcellated data.")
        data = zscore_df(data, along="rows")

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
            author_maxlen = max([len(x) for x in meta["publication"]])
            license_maxlen = max([len(x) for x in meta["license"]])
            for atlas, pub, doi, license, note in zip(
                meta["atlas"], meta["publication"], meta["doi"], meta["license"], meta["note"]
                ):
                
                doi_list = [f"https://doi.org/{doi}" for doi in doi.replace(" ", "").split(";")]
                if "" in doi_list: doi_list.remove("")
                doi_str = ", ".join(doi_list)
                atlas = atlas.ljust(atlas_maxlen)
                author = pub.capitalize().ljust(author_maxlen)
                license = license.ljust(license_maxlen)
                msg += f"- {atlas}  Source: {author}  {license}  {doi_str}\n"
                
                if not pd.isna(note):
                    msg += f"    CAVE: {note}\n"
    
    # mRNA
    elif dataset.lower() in ["mrna", "magicc"]:
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
                    space: str = _SPACE_DEFAULT,
                    collection: str = None,
                    set_size_range: Union[None, Tuple[int, int]] = None,
                    parcellation: str = None,
                    standardize_parcellated: bool = True,
                    return_metadata: bool = False,
                    print_references: bool = True,
                    verbose: bool = True,
                    nispace_data_dir: Union[str, pathlib.Path] = None,
                    osf_config_file: str = None,
                    github_config_file: str = None):
    verbose = set_log(lgr, verbose)

    # Check dataset availability
    if isinstance(dataset, str):
        dataset = dataset.lower()
        if dataset not in reference_lib:
            lgr.critical_raise(f"Dataset '{dataset}' not found! Available datasets: {keys2str(reference_lib)}",
                               ValueError)
        elif parcellation is None and "map" not in reference_lib[dataset]:
            lgr.critical_raise(f"Dataset '{dataset}' is only available as parcellated data, choose a parcellation!",
                               ValueError)
    else:
        lgr.critical_raise(f"Invalid dataset type; expecting string.",
                           TypeError)
    lgr.info(f"Loading {dataset} maps.")
    
    # Define the base directories
    if not nispace_data_dir:
        base_dir = pathlib.Path.home() / f"nispace-data" / "reference" / dataset
    else:
        base_dir = pathlib.Path(nispace_data_dir) / f"reference" / dataset
    map_dir = base_dir / "map"
    tab_dir = base_dir / "tab"
    
    # Check if parcellation is defined correctly and load map lists
    if parcellation is not None:
        
        # check if parcellation is defined correctly and set alias settings
        if parcellation not in parcellation_lib:
            lgr.critical_raise(f"Parcellation '{parcellation}' not found. Available: {keys2str(parcellation_lib)}",
                               ValueError)
        # check parcellation aliases
        parc, cortex, subcortex = _parc_alias(parcellation)
        
        # load maps from tabulated data (index col)
        maps_avail = pd.read_csv(
            get_file(tab_dir / f"dset-{dataset}_parc-{parc}.csv.gz", **reference_lib[dataset]["tab"][parc]), 
            index_col=0
        ).index.to_list()
    
    # Check space availability and load map lists   
    else:
            
        # get list of map image files
        maps_avail = [m for m, v in reference_lib[dataset]["map"].items() if space in v]
        if len(maps_avail) == 0:
            lgr.critical_raise(f"Found no maps for space '{space}' in dataset '{dataset}'.",
                               ValueError)
        
    lgr.debug(f"Loaded {len(maps_avail)} unfiltered map(s). "
              f"First 5: {maps_avail[:5] if len(maps_avail) >= 5 else maps_avail[:len(maps_avail)]}")

    # Remove private maps
    if "map" in reference_lib[dataset]:
        if not osf_config_file and not github_config_file:
            if "mni152" in space.lower():
                maps_avail = [
                    m for m in maps_avail 
                    if reference_lib[dataset]["map"][m][space]["host"] not in ["osfprivate", "github-nispace-private"]
                ]
            else:
                maps_avail = [
                    m for m in maps_avail 
                    if reference_lib[dataset]["map"][m][space]["L"]["host"] not in ["osfprivate", "github-nispace-private"]
                ]
    
    # Filter by 'maps'
    if maps:
        n_tmp = len(maps_avail)
        lgr.info(f"Applying filter: {maps}")
        maps_avail = _filter_maps(maps_avail, maps)
        # if "map" not in reference_lib[dataset]:
        #     maps_avail = _filter_maps(maps_avail, maps)
        # else:
        #     if isinstance(maps, str):
        #         maps = [maps]
        #     elif not isinstance(maps, (list, tuple, set, pd.Series)):
        #         lgr.warning(f"For dataset '{dataset}', 'maps' must be list-like. Skipping filter.")
        #         maps = maps_avail
        #     maps_avail = list(set(maps_avail).intersection(maps))
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
        data = _load_parcellated_data(
            dataset=dataset, 
            tab_dir=tab_dir, 
            parc=parc, 
            map_files=maps_avail, 
            collection_df=collection_df,
            cortex=cortex,
            subcortex=subcortex,
            standardize=standardize_parcellated
        )
        
    # Fetch paths to maps if no 'parcellation' is specified
    else:
        # MNI: one file per map
        if "mni152" in space.lower():
            data = [
                get_file(
                    local_path=map_dir / m / f"{m}_space-{space}.nii.gz", 
                    **reference_lib[dataset]["map"][m][space], 
                    compress_nifti=True,
                    osf_config_file=osf_config_file,
                    github_config_file=github_config_file
                ) 
                for m in maps_avail
            ]
        # surface: two files per map
        else:
            data = [
                (get_file(
                     local_path=map_dir / m / f"{m}_space-{space}_hemi-L.surf.gii", 
                     **reference_lib[dataset]["map"][m][space]["L"], 
                     osf_config_file=osf_config_file,
                     github_config_file=github_config_file
                 ),
                 get_file(
                     local_path=map_dir / m / f"{m}_space-{space}_hemi-R.surf.gii", 
                     **reference_lib[dataset]["map"][m][space]["R"], 
                     osf_config_file=osf_config_file,
                     github_config_file=github_config_file
                 ))
                for m in maps_avail
            ]
        
    # Print references
    # for maps if "pet", or for sets if "mrna"
    if return_metadata or print_references:
        if dataset == "pet":
            meta = fetch_metadata(dataset, maps_avail)
        elif dataset in ["mrna", "magicc"] and collection_df is not None:
            meta = fetch_metadata(dataset, collection=collection)
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
        if dataset not in reference_lib:
            return None
    else:
        return None
    
    base_dir = pathlib.Path.home() / "nispace-data" / "reference" / dataset
    meta = pd.read_csv(get_file(base_dir / "metadata.csv", **reference_lib[dataset]["metadata"]))
    
    if dataset == "pet" and maps is not None:
        if isinstance(maps, str):
            maps = [maps]
        meta = meta[meta.atlas.str.contains("|".join(maps), na=False)]
    elif dataset in ["mrna", "magicc"] and collection is not None:
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
