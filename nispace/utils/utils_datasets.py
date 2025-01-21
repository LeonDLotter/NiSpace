import os
from pathlib import Path
import shutil
import pickle
import gzip
import requests
import numpy as np
import pandas as pd
import tempfile
import re
import configparser
import hashlib
import subprocess

from typing import Literal, Union
from nilearn import image
from neuromaps.datasets import fetch_annotation
from neuromaps.resampling import resample_images
from nilearn.masking import compute_background_mask
from sklearn.preprocessing import minmax_scale
from nibabel import Nifti1Image

import nispace.datasets as datasets
import nispace.io as io
    
try:
    import osfclient
    _OSF_AVAIL = True
except:
    _OSF_AVAIL = False
    
from nispace.config import DATA_REPO, DATA_REPO_COMMIT, DATA_REPO_PRIVATE, DATA_REPO_PRIVATE_COMMIT
    
    
def download(url, path=None, headers=None):
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    if path is None:
        path = Path(tempfile.gettempdir()) / Path(url).name
    with open(str(path), "wb") as f:
        f.write(r.content)
    return path


def download_via_osfclient(osf_repo, osf_file_id, save_path,
                           osf_username=None, osf_password=None, osf_token=None):
    osf = osfclient.OSF(username=osf_username, password=osf_password, token=osf_token)
    project = osf.project(osf_repo)
    storage = project.storage()
    remote_files = {remote_file.id: remote_file for remote_file in storage.files}
    remote_file = remote_files[osf_file_id]
    with open(save_path, "wb") as local_file:
        remote_file.write_to(local_file)
    return save_path


def download_file(host: Literal["url", "github", "github-nispace", "github-nispace-private", 
                                "osf", "osfprivate", "neuromaps"] = "url", 
                  remote: Union[str, Path, tuple[str, str], tuple[str, str, str]] = None, 
                  save_path: Union[str, Path] = None,
                  osf_config_file: str = None,
                  github_config_file: str = None):
    
    # errors
    hosts_avail = ["url", "github", "github-nispace", "github-nispace-private", 
                   "osf", "osfprivate", "neuromaps"]
    if host not in hosts_avail:
        raise ValueError(f"'host' must be one of {hosts_avail}; not '{host}'.")
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
    elif host in ["github-nispace", "github-nispace-private"]:
        if not isinstance(remote, str):
            raise ValueError("'remote' must be a string for github-nispace (path in NiSpace-data repo)")
        else:
            remote = Path(remote)
        if host == "github-nispace-private":
            if not Path(github_config_file).exists():
                raise ValueError(f"Config file '{github_config_file}' does not exist.")
            else:
                config = configparser.ConfigParser()
                config.read(github_config_file)
                github_username = config["github"]["username"] if "username" in config["github"] else None
                github_token = config["github"]["token"] if "token" in config["github"] else None
    elif host == "osf":
        if not isinstance(remote, (tuple, list)):
            raise ValueError("'remote' must be a tuple of (osf_repo, osf_id) for osf")
        else:
            osf_repo, osf_id = remote
            remote = Path(osf_id)
    elif host == "osfprivate":
        if not isinstance(remote, (tuple, list)):
            raise ValueError("'remote' must be a tuple of (osf_repo, osf_id) for osfprivate")
        elif not Path(osf_config_file).exists():
            raise ValueError(f"Config file '{osf_config_file}' does not exist.")
        else:
            config = configparser.ConfigParser()
            config.read(osf_config_file)
            osf_username = config["osf"]["username"] if "username" in config["osf"] else None
            osf_token = config["osf"]["token"] if "token" in config["osf"] else None
            osf_password = config["osf"]["password"] if "password" in config["osf"] else None
            osf_repo, osf_id = remote
            remote = Path(osf_id)
    elif host == "neuromaps":
        if not isinstance(remote, (tuple, list)):
            raise ValueError("'remote' must be a tuple of (source, tracer, space, "
                             "{hemi: 'L' or 'R' if surface space}) for neuromaps")
        else:
            if len(remote) == 3:
                remote = (remote[0], remote[1], remote[2], ["L", "R"])
            source, tracer, space, hemi = remote
           
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
        
        # download if not osfprivate
        if host not in ["osfprivate", "github-nispace-private"]:
            
            # get url
            if host == "url":
                url = str(remote)        
            elif host == "github":
                url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
            elif host == "github-nispace":
                url = f"https://raw.githubusercontent.com/{DATA_REPO}/{DATA_REPO_COMMIT}/{remote}"                
            elif host == "osf":
                url = f"https://files.osf.io/v1/resources/{osf_repo}/providers/osfstorage/{osf_id}"
        
            # download
            return download(url, save_path)
        
        # github-nispace-private
        elif host == "github-nispace-private":
            print(f"Downloading private GitHub file.")
            url = f"https://raw.githubusercontent.com/{DATA_REPO_PRIVATE}/{DATA_REPO_PRIVATE_COMMIT}/{remote}"
            headers = {"Authorization": f"token {github_token}"}
            if not Path(github_config_file).exists():
                raise ValueError(f"Config file '{github_config_file}' does not exist.")
            else:
                return download(url, save_path, headers=headers)
                
        # download if osfprivate via osfclient
        elif host == "osfprivate":
            print(f"Downloading private OSF file via osfclient (this will be slow).")
            if not _OSF_AVAIL:
                raise ImportError("'osfclient' is not installed. Install it with, e.g., 'pip install osfclient'.")
            return download_via_osfclient(
                osf_repo=osf_repo, 
                osf_file_id=osf_id, 
                save_path=save_path, 
                osf_username=osf_username, osf_password=osf_password, osf_token=osf_token
            )
            
    else:
        path = fetch_annotation(source=source, desc=tracer, space=space, hemi=hemi)
        # should be a string or pathlib.Path
        if isinstance(path, (str, Path)):
            return path
        else: 
            raise ValueError(f"Unexpected neuromaps output for "
                             f"source={source}, desc={tracer}, space={space}: {path}")


def _compress_nifti(file_path, save_path, dtype=np.float32):
    # try to load
    try:
        img = io.load_img(file_path, override_file_format=".nii.gz")
    except:
        try:
            img = io.load_img(file_path, override_file_format=".nii")
        except Exception as e:
            raise ValueError(f"Could not load file '{file_path}': {e}")
    # change dtype
    img_dat = img.get_fdata().astype(dtype)
    img = image.new_img_like(img, img_dat, copy_header=True)
    # save
    img.to_filename(save_path)


def get_file(local_path, host, remote, 
             compress_nifti=False,
             osf_config_file=None,
             github_config_file=None):
    
    local_path = Path(local_path)
    if local_path.is_dir():
        raise ValueError(f"'local_path' must be a file path, not a directory path; not '{local_path}'.")
    
    if not local_path.exists():
        
        print(f"Downloading {local_path.resolve()}.")
        if not local_path.parent.exists():
            local_path.parent.mkdir(parents=True)
        tmp_path = download_file(
            host, remote, 
            osf_config_file=osf_config_file,
            github_config_file=github_config_file
        )
        
        if compress_nifti:
            _compress_nifti(tmp_path, local_path)
        else:
            shutil.copy(tmp_path, local_path)
            
    return local_path


def calculate_file_hash(file_path):
    """Calculate the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def sync_osf(local_path, osf_id, username=None, password=None, token=None,
             dry_run=False, exclude=["^\."], config_file=None, 
             skip_new_file_url_error=False, skip_file_exists_error=False,
             use_R=False):
    
    # check if osfclient is installed
    if not _OSF_AVAIL:
        raise ImportError("'osfclient' is not installed. Install it with, e.g., 'pip install osfclient'.")
    
    # Initialize OSF client
    print(f"Syncing local::{local_path} to osf::{osf_id}")
    if config_file is not None:
        if Path(config_file).exists():
            print(f"Attempting to read config file {config_file}")
            config = configparser.ConfigParser()
            config.read(config_file)
            username = config["osf"]["username"] if "username" in config["osf"] else None
            password = config["osf"]["password"] if "password" in config["osf"] else None
            token = config["osf"]["token"] if "token" in config["osf"] else None
        else:
            print(f"Config {config_file} does not exist. Trying to proceed without.")
    osf = osfclient.OSF(username, password, token)
    project = osf.project(osf_id)
    storage = project.storage()

    # Helper function to get remote files and folders
    def get_remote_files_and_folders(storage):
        remote_files = {}
        for file in storage.files:
            remote_files[file.path] = file
        return remote_files

    # Get remote files and folders
    print(f"Loading osf::{osf_id} remote files and folders (this will take a while...)")
    if not use_R:
        remote_files = get_remote_files_and_folders(storage)
    else:
        print("Using R via command line to get osf ids")
        if not Path("get_osf_ids.R").exists():
            raise FileNotFoundError("'get_osf_ids.R' not found in the current working directory.")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "osf_ids.json"
            subprocess.run(["Rscript", "get_osf_ids.R", str(temp_file)])
            remote_files = io.read_json(temp_file)
    print(remote_files)

    # Prepare local path
    local_path = Path(local_path)
    if not local_path.exists() or not local_path.is_dir():
        raise FileNotFoundError(f"Local path::{local_path} does not exist")
    if isinstance(exclude, str):
        exclude = [exclude]
    ids = {}
        
    # Traverse local path
    print(f"Traversing local::{local_path}")
    new_file_url_error_dict = {}
    for root, dirs, files in os.walk(local_path):
        for name in files:
            if any(re.match(pattern, name) for pattern in exclude):
                continue
            local_file_path = Path(root) / name
            remote_file_path = f"/{local_file_path.relative_to(local_path)}"

            # check if file exists on remote
            if remote_file_path in remote_files:
                remote_file = remote_files[remote_file_path]
                
                # get hash 
                local_file_hash = calculate_file_hash(local_file_path)
                if not use_R:
                    remote_file_hash = remote_file.hashes.get('md5')
                else:
                    remote_file_hash = remote_files[remote_file_path]["md5"]
                
                # Check if the local file is different from the remote file and update if so
                if local_file_hash != remote_file_hash:
                    print(f"local::{local_file_path.relative_to(local_path)}: Updating remote")
                    if not dry_run:
                        with open(local_file_path, 'rb') as local_file:
                            remote_file.update(local_file)
                else:
                    print(f"local::{local_file_path.relative_to(local_path)}: Remote is up to date")
                    
            # file does not exist on remote
            else:
                # Upload new file
                print(f"local::{local_file_path.relative_to(local_path)}: Uploading to remote")
                # print("this remote file:", remote_file_path)
                # print("all remote files:")
                # print(remote_files)
                if not dry_run:
                    
                    # try upload as is
                    if not skip_new_file_url_error and not skip_file_exists_error:
                        with open(local_file_path, 'rb') as local_file:
                            storage.create_file(remote_file_path, local_file)
                            
                    # skip new file url error
                    else:
                        
                        # try upload as is
                        try:
                            with open(local_file_path, 'rb') as local_file:
                                storage.create_file(remote_file_path, local_file)
                                
                        except AttributeError as e:
                            if skip_new_file_url_error:
                                print(f"AttributeError: {e}")
                                new_file_url_error_dict[local_file_path] = remote_file_path
                            else:
                                raise e
                        
                        except FileExistsError as e:
                            if skip_file_exists_error:
                                print(f"FileExistsError: {e}")
                            else:
                                raise e
                            
    # Remove remote files not present locally
    for remote_file_path in remote_files.keys():
        if remote_file_path.startswith("/"):
            remote_file_path = remote_file_path[1:]
        if not (local_path / remote_file_path).exists():
            print(f"remote::{remote_file_path}: Deleting remote")
            if not dry_run:
                remote_files["/" + remote_file_path].remove()
                
    # print new file url errors
    if len(new_file_url_error_dict) > 0:
        print(f"There were {len(new_file_url_error_dict)} new_file_url errors:")
        for local_file_path, remote_file_path in new_file_url_error_dict.items():
            print(f"local::{local_file_path}: remote::{remote_file_path}")
                
    # get file ids
    print(f"Loading updated osf::{osf_id} remote files and folders")
    if not use_R:
        remote_files = get_remote_files_and_folders(storage)
        for remote_file_path in sorted(remote_files.keys()):
            ids[remote_file_path] = {
                "id": remote_files[remote_file_path].id,
                "md5": remote_files[remote_file_path].hashes.get('md5')
            }
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "osf_ids.json"
            subprocess.run(["Rscript", "get_osf_ids.R", str(temp_file)])
            ids = io.read_json(temp_file)
        
    return ids
                    