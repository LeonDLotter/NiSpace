import os
import logging
from pathlib import Path
import shutil
from threading import local
import requests
import numpy as np
import tempfile
import re
import configparser
import hashlib
import subprocess

from typing import Literal, Union

lgr = logging.getLogger(__name__)
from nilearn import image
from neuromaps.datasets import fetch_annotation

from nispace.io import read_json, load_img

datalib_dir = Path(__file__).parent.parent / "datalib"
hash_lib = read_json(datalib_dir / "file_hashes.json")

try:
    import osfclient
    _OSF_AVAIL = True
except:
    _OSF_AVAIL = False
    
from nispace.config import DATA_REPO, DATA_REPO_COMMIT, DATA_REPO_PRIVATE, DATA_REPO_PRIVATE_COMMIT
    

def _check_hash(local: Union[str, Path], remote: Union[str, Path] = None) -> bool:
    
    # hash of local file
    hash_local = calculate_sha256_hash(local)
    
    # hash of remote file
    if remote is None:
        remote = local
    remote = str(remote)
    if remote not in hash_lib:
        raise ValueError(f"Hash not found for {remote}. Problem with nispace updating?")
    hash_remote = hash_lib[remote]
    
    # compare hashes
    if hash_local == hash_remote:
        return True
    else:
        lgr.debug(f"Hash mismatch: {local} -> {hash_local} != {remote} -> {hash_remote}")
        return False
     
    
def download(url, path=None, headers=None, suffix=""):
    """
    Stream-download a file from a URL.

    Parameters
    ----------
    url : str
        URL to download from.
    path : str or os.PathLike, optional
        Destination file path. If not given, saved under a temp directory using
        the URL's basename (plus `suffix`).
    headers : dict, optional
        HTTP headers to send with the request (e.g. an ``Authorization`` token).
    suffix : str, default ""
        Appended to the auto-derived filename when `path` is not given.

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    from urllib.parse import urlparse
    r = requests.get(url, headers=headers, stream=True)
    r.raise_for_status()
    if path is None:
        name = Path(urlparse(url).path).name
        path = Path(tempfile.gettempdir()) / (name + suffix)
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return path


def download_via_osfclient(osf_repo, osf_file_id, save_path,
                           osf_username=None, osf_password=None, osf_token=None):
    """
    Download a single file from a (typically private) OSF project via ``osfclient``.

    Slower than the direct ``files.osf.io`` URL used by public downloads (see
    :func:`download_file`'s ``"osf"`` host), but supports authentication.

    Parameters
    ----------
    osf_repo : str
        OSF project (repo) ID.
    osf_file_id : str
        ID of the remote file within the project's storage.
    save_path : str or os.PathLike
        Local destination file path.
    osf_username : str, optional
        OSF account username, for password authentication.
    osf_password : str, optional
        OSF account password, for password authentication.
    osf_token : str, optional
        OSF personal access token, for token authentication.

    Returns
    -------
    save_path
        The same `save_path` passed in.
    """
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
    """
    Download a single file from one of several supported hosts.

    Dispatches to the right download mechanism based on `host`; used internally
    by :func:`get_file` as the actual download backend for every dataset host
    NiSpace supports.

    Parameters
    ----------
    host : {"url", "github", "github-nispace", "github-nispace-private", "osf", "osfprivate", "neuromaps"}, default "url"
        Which download mechanism/source to use.
    remote : str, os.PathLike, or tuple, required
        The remote file identifier; shape depends on `host`:

        - ``"url"``: a full URL string/path.
        - ``"github"``: ``(repo, branch, path)`` — any public GitHub repo.
        - ``"github-nispace"``: a path string within the NiSpace-data repo
          (repo/commit taken from :mod:`nispace.config`).
        - ``"github-nispace-private"``: same as ``"github-nispace"`` but against
          the private NiSpace-data repo; requires `github_config_file`.
        - ``"osf"``: ``(osf_repo, osf_id)`` for a public OSF file, downloaded
          directly via the ``files.osf.io`` URL.
        - ``"osfprivate"``: ``(osf_repo, osf_id)`` for a private OSF file,
          downloaded via ``osfclient``; requires `osf_config_file`.
        - ``"neuromaps"``: ``(source, tracer, space)`` or ``(source, tracer, space, hemi)``,
          forwarded to ``neuromaps.datasets.fetch_annotation``.
    save_path : str, os.PathLike, or "cwd", optional
        Local destination. A directory path is combined with the remote file's
        basename. ``"cwd"`` uses the current working directory. If not given,
        defaults to a temp directory (ignored for `host="neuromaps"`, which
        requires an explicit `save_path`).
    osf_config_file : str, optional
        Path to an INI file with an ``[osf]`` section (``username``/``password``/
        ``token``); required when `host="osfprivate"`.
    github_config_file : str, optional
        Path to an INI file with a ``[github]`` section (``username``/``token``);
        required when `host="github-nispace-private"`.

    Returns
    -------
    Path
        Path to the downloaded (or copied, for `host="neuromaps"`) file.

    Raises
    ------
    ValueError
        If `host` is not one of the supported values, `remote` is missing/malformed
        for the given `host`, or a required config file doesn't exist.
    ImportError
        If `host="osfprivate"` and the optional ``osfclient`` dependency is not installed.
    """
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
        else:
            remote = Path(remote)
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
                remote = (remote[0], remote[1], remote[2], None)
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
                url = remote.as_posix().replace("https:/", "https://").replace("http:/", "http://")  
            elif host == "github":
                url = f"https://raw.githubusercontent.com/{repo}/{branch}/{remote.as_posix()}"
            elif host == "github-nispace":
                url = f"https://raw.githubusercontent.com/{DATA_REPO}/{DATA_REPO_COMMIT}/{remote.as_posix()}"                
            elif host == "osf":
                url = f"https://files.osf.io/v1/resources/{osf_repo}/providers/osfstorage/{osf_id}"
        
            # download
            return download(url, save_path)
        
        # github-nispace-private
        elif host == "github-nispace-private":
            print(f"Downloading private GitHub file.")
            url = f"https://raw.githubusercontent.com/{DATA_REPO_PRIVATE}/{DATA_REPO_PRIVATE_COMMIT}/{remote.as_posix()}"
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
            shutil.copy(path, save_path)
            return Path(save_path)
        else: 
            raise ValueError(f"Unexpected neuromaps output for "
                             f"source={source}, desc={tracer}, space={space}, hemi={hemi}: {path}")


def _compress_nifti(file_path, save_path, dtype=np.float32):
    # try to load
    try:
        img = load_img(file_path, override_file_format=".nii.gz")
    except:
        try:
            img = load_img(file_path, override_file_format=".nii")
        except Exception as e:
            raise ValueError(f"Could not load file '{file_path}': {e}")
    # change dtype
    img_dat = img.get_fdata().astype(dtype)
    img = image.new_img_like(img, img_dat, copy_header=True)
    # save
    img.to_filename(save_path)
    

def _compress_gifti(file_path, save_path):
    if isinstance(file_path, (str, Path)):
        file_path = file_path, 
        save_path = save_path, 
    for fp, sp in zip(file_path, save_path):
        # try to load
        try:
            img = load_img(fp, override_file_format=".gii.gz")
        except:
            try:
                img = load_img(fp, override_file_format=".gii")
            except Exception as e:
                raise ValueError(f"Could not load file '{fp}': {e}")
        # save
        img.to_filename(sp)


def _get_file_ext(remote):
    remote = str(remote)
    gz = ".gz" if remote.endswith(".gz") else ""
    gii_extra = ""
    for s in [".func.", ".shape.", ".label.", ".surf."]:
        if s in remote:
            gii_extra = s[1:]
            break
    ext_nogz = remote.replace(gii_extra, "").replace(gz, "").split(".")[-1]
    return f"{gii_extra}{ext_nogz}{gz}"


def get_file(local_path, host, remote,
             ext=None,
             osf_config_file=None,
             github_config_file=None,
             hash_check=True,
             overwrite=False,
             **_ignored):
    """
    Return a local, up-to-date copy of a remote file, downloading only if needed.

    This is the caching entry point used throughout :mod:`nispace.datasets` for
    every remote file lookup — it downloads via :func:`download_file` only when
    the local file is missing, stale (per `hash_check`), or `overwrite=True`.

    Parameters
    ----------
    local_path : str or os.PathLike
        Local destination path. If its filename contains the literal placeholder
        ``"%s"``, it is filled in with the extension inferred from `remote`
        (via an internal helper) when `ext` is not given.
    host : str
        Download host; forwarded to :func:`download_file` (see its docstring for
        the supported values and corresponding `remote` shapes).
    remote : str, os.PathLike, or tuple
        Remote file identifier; forwarded to :func:`download_file`.
    ext : str, optional
        File extension to use for `local_path` instead of the inferred one.
    osf_config_file : str, optional
        Forwarded to :func:`download_file` (used for `host="osfprivate"`).
    github_config_file : str, optional
        Forwarded to :func:`download_file` (used for `host="github-nispace-private"`).
    hash_check : bool, default True
        If True and `host="github-nispace"`, re-download when the local file's
        SHA-256 hash no longer matches the hash recorded in NiSpace's data-library
        hash file (i.e. the remote file was updated upstream). Ignored for other
        hosts, for which no reference hash is available.
    overwrite : bool, default False
        If True, always re-download regardless of whether the local file exists
        or matches its hash.
    **_ignored
        Extra keyword arguments are accepted and silently ignored, so callers can
        pass a shared kwargs dict across different host types.

    Returns
    -------
    Path
        Path to the (now-local) file.

    Raises
    ------
    ValueError
        If `local_path` points to an existing directory.
    """
    # local path
    local_path = Path(local_path)
    # infer file extension if necessary
    if local_path.name.endswith(".%s") and not ext:
        ext = _get_file_ext(remote)
        local_path = local_path.parent / (local_path.name % ext)
    elif ext:
        local_path = local_path.parent / (local_path.stem + f".{ext.lstrip('.')}")
    # check if directory
    if local_path.is_dir():
        raise ValueError(f"'local_path' must be a file path, not a directory path: '{local_path}'.")
    
    redownload = False
    msg = "Downloading"
    # download always if overwrite is enabled
    if overwrite:
        redownload = True
    else:
        # download if not exists
        if not local_path.exists():
            redownload = True
        # download if hash check is enabled and hash is different
        else:
            # check hash, this is only possible with github-nispace
            if hash_check and host == "github-nispace":
                if not _check_hash(local_path, remote):
                    redownload = True
                    msg = "Updating"
    
    # download if not exists or overwriting
    if redownload:
        
        print(f"{msg} {local_path.resolve()}")
        if not local_path.parent.exists():
            local_path.parent.mkdir(parents=True)
        tmp_path = download_file(
            host, remote, 
            save_path=local_path,
            osf_config_file=osf_config_file,
            github_config_file=github_config_file
        )

        # save
        #shutil.copy(tmp_path, local_path)
        #tmp_path.unlink()
            
    return local_path


def calculate_md5_hash(file_path):
    """Calculate the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def calculate_sha256_hash(file_path):
    """Calculate the SHA-256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def sync_osf(local_path, osf_id, username=None, password=None, token=None,
             dry_run=False, exclude=[r"^\."], config_file=None,
             skip_new_file_url_error=False, skip_file_exists_error=False,
             use_R=False):
    """
    Two-way sync a local directory with an OSF project's storage (dev tool).

    For every local file under `local_path`: uploads it if missing remotely,
    updates it if its MD5 hash differs from the remote copy, and leaves it
    otherwise. Remote files with no corresponding local file are deleted.
    Used to publish/update NiSpace's reference data on OSF, not part of the
    end-user data-fetching path.

    Parameters
    ----------
    local_path : str or os.PathLike
        Local directory to sync from; must already exist.
    osf_id : str
        OSF project (repo) ID to sync to.
    username : str, optional
        OSF account username, for password authentication.
    password : str, optional
        OSF account password, for password authentication.
    token : str, optional
        OSF personal access token, for token authentication.
    dry_run : bool, default False
        If True, only print what would be uploaded/updated/deleted without
        making any remote changes.
    exclude : list of str, default [r"^\\."]
        Regex patterns; local filenames matching any pattern are skipped
        (default excludes dotfiles).
    config_file : str, optional
        Path to an INI file with an ``[osf]`` section (``username``/``password``/
        ``token``), used instead of the individual credential arguments if given.
    skip_new_file_url_error : bool, default False
        If True, catch and record (rather than raise) an ``AttributeError``
        sometimes raised by ``osfclient`` when uploading brand-new files.
    skip_file_exists_error : bool, default False
        If True, catch (rather than raise) a ``FileExistsError`` during upload.
    use_R : bool, default False
        If True, shell out to a local ``get_osf_ids.R`` script (via ``Rscript``)
        to enumerate remote file IDs instead of using ``osfclient`` directly.

    Returns
    -------
    dict
        Mapping of remote file path to ``{"id": ..., "md5": ...}`` for every file
        present on OSF after the sync.

    Raises
    ------
    ImportError
        If the optional ``osfclient`` dependency is not installed.
    FileNotFoundError
        If `local_path` doesn't exist / isn't a directory, or `use_R=True` and
        ``get_osf_ids.R`` isn't found in the current working directory.
    """
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
            remote_files = read_json(temp_file)
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
                local_file_hash = calculate_md5_hash(local_file_path)
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
            ids = read_json(temp_file)
        
    return ids
                    