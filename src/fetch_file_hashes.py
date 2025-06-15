import os
import json
import urllib.request
import urllib.error
import re

# where to save the file hashes.json
FILE_HASHES_PATH = "nispace/datalib/file_hashes.json"

def get_config_values():
    # get the parent directory of the current file
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(parent_dir, "nispace", "config.py")
    
    # read config.py as text
    with open(config_path, "r") as f:
        content = f.read()
    
    # extract values using regex
    data_repo = re.search(r'DATA_REPO\s*=\s*"([^"]+)"', content).group(1)
    data_repo_commit = re.search(r'DATA_REPO_COMMIT\s*=\s*"([^"]+)"', content).group(1)
    
    return data_repo, data_repo_commit

def fetch_file_hashes():
    # get config values
    DATA_REPO, DATA_REPO_COMMIT = get_config_values()
    
    # construct github url
    print(DATA_REPO, DATA_REPO_COMMIT)
    github_url = f"https://raw.githubusercontent.com/{DATA_REPO}/{DATA_REPO_COMMIT}/file_hashes.json"
    print(github_url)
    
    try:
        # get the file hashes from the data repository using urllib
        with urllib.request.urlopen(github_url) as response:
            file_hashes = json.loads(response.read().decode('utf-8'))
        
        # save the file hashes to the file_hashes.json file
        with open(FILE_HASHES_PATH, "w") as f:
            json.dump(file_hashes, f, indent=4)
    except urllib.error.URLError as e:
        raise Exception(f"Failed to fetch file hashes: {e}")


if __name__ == "__main__":
    fetch_file_hashes()