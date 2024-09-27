from pathlib import Path
import shutil
import pickle
import gzip
import requests
import numpy as np
import pandas as pd
import tempfile
from typing import Literal, Union
from nilearn import image
from neuromaps.datasets import fetch_annotation
from neuromaps.resampling import resample_images
from nilearn.masking import compute_background_mask
from sklearn.preprocessing import minmax_scale
from nibabel import Nifti1Image

import nispace.datasets as datasets
import nispace.io as io
    
    
def download(url, path=None):
    r = requests.get(url)
    r.raise_for_status()
    if path is None:
        path = Path(tempfile.gettempdir()) / Path(url).name
    with open(str(path), "wb") as f:
        f.write(r.content)
    return path


def download_file(host: Literal["url", "github", "osf", "neuromaps"] = "url", 
                  remote: Union[str, Path, tuple[str, str], tuple[str, str, str]] = None, 
                  save_path: Union[str, Path] = None):
    
    # errors
    if host not in ["url", "github", "osf", "neuromaps"]:
        raise ValueError(f"'host' must be one of 'url', 'github', 'osf', or 'neuromaps; not '{host}'.")
    if remote is not None:
        if isinstance(remote, (str, Path)):
            if str(remote).lower() in ["", "none"]:
                raise ValueError(f"'remote' must be provided.")
        elif isinstance(remote, (tuple, list)):
            if any(v is None for v in remote) or any(v in ["", "none"] for v in remote):
                raise ValueError(f"'remote' must not contain None or empty strings.")
        else:
            raise ValueError(f"'remote' must be str, Path, or tuple of str/Paths; not '{remote}'.")
    else:
        raise ValueError("'remote' must be provided.")
    if host == "url":
        if not isinstance(remote, (str, Path)):
            raise ValueError("'remote' must be a string or pathlib.Path for url")
    elif host == "github":
        if not isinstance(remote, (tuple, list)):
            raise ValueError("'remote' must be a tuple of (repo, branch, path) for github")
        else:
            repo, branch, path = remote
            remote = Path(path)
    elif host == "osf":
        if not isinstance(remote, (tuple, list)):
            raise ValueError("'remote' must be a tuple of (osf_repo, osf_id) for osf")
        else:
            osf_repo, osf_id = remote
            remote = Path(osf_id)
    elif host == "neuromaps":
        if not isinstance(remote, (tuple, list)):
            raise ValueError("'remote' must be a tuple of (source, target, space) for neuromaps")
        else:
            source, tracer, space = remote
    
    if host != "neuromaps":
        
        # save path
        if isinstance(save_path, (str, Path)):
            save_path = Path(save_path)
            if save_path.is_dir():
                save_path = save_path / Path(remote).name
        elif save_path == "cwd":
            save_path = Path.cwd() / Path(remote).name
        elif save_path is None:
            save_path = Path(tempfile.gettempdir()) / Path(remote).name
        else:
            raise ValueError("'save_path' must be a string, pathlib.Path, or 'cwd'")
        
        # get url
        if host == "url":
            url = str(remote)        
        elif host == "github":
            url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
        elif host == "osf":
            url = f"https://files.osf.io/v1/resources/{osf_repo}/providers/osfstorage/{osf_id}"
        
        # download
        return download(url, save_path)
    
    else:
        path = fetch_annotation(source=source, desc=tracer, space=space)
        if isinstance(path, str):
            return path
        else: 
            raise ValueError(f"Unexpected neuromaps output for "
                             f"source={source}, desc={tracer}, space={space}: {path}")

def process_ref_img(image_path, save_path=None, override_file_format=False):
    if not isinstance(image_path, (str, Path)):
        raise ValueError(f"'image_path' must be a string or pathlib.Path; not '{image_path}'.")
    image_path = Path(image_path)
    
    # load image
    img = io.load_img(image_path, override_file_format)
    
    # volumetric processing
    if isinstance(img, Nifti1Image):
        
        # image voxelsize
        voxsize = int(np.abs(np.round(img.affine[0,0])))
        
        # get rid of 4th dimension if present
        if img.ndim==4:
            img = image.index_img(img, 0)
        
        # load mask and resample to voxsize
        mask = io.load_img(datasets.fetch_template("mni152", res=f"{voxsize}mm", desc="mask", verbose=False))
        
        # resample image to mask space
        img, _ = resample_images(
            src=img,
            src_space="mni152",
            trg=mask,
            trg_space="mni152",
            method="linear",
            resampling="transform_to_trg"
        )
        
        # get background mask
        bg_mask = compute_background_mask(img)   
        bg_mask = image.math_img("bg_mask * mni_mask", bg_mask=bg_mask, mni_mask=mask)
        
        # rescale and adjust data type
        img_data = img.get_fdata()
        img_data[bg_mask.get_fdata() == 0] = np.nan
        img_data = minmax_scale(img_data.flatten(), (1, 100)).reshape(img_data.shape)
        img_data = np.nan_to_num(img_data)
        img = image.new_img_like(img, img_data.astype(np.float32), copy_header=True)
        
        if save_path is None:
            return img
        else:
            img.to_filename(save_path)
            return save_path
            
    # surface processing
    else:
        raise NotImplementedError("Surface processing not implemented.")


def get_file(local_path, host, remote, process_img=False, override_file_format=False):
    
    local_path = Path(local_path)
    if local_path.is_dir():
        raise ValueError(f"'local_path' must be a file path, not a directory path; not '{local_path}'.")
    
    if not local_path.exists():
        
        print(f"Downloading '{local_path}'.")
        if not local_path.parent.exists():
            local_path.parent.mkdir(parents=True)
        tmp_path = download_file(host, remote)
        
        if process_img:
            print("Processing image.")
            local_path = process_ref_img(tmp_path, local_path, override_file_format)
        else:
            shutil.copy(tmp_path, local_path)
            
    return local_path