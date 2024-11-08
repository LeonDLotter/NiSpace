# %%
import sys
from pathlib import Path

wd = Path.cwd().parent
print(f"Working dir: {wd}")

sys.path.append(wd.as_posix())
from nispace.utils.utils_datasets import sync_osf
from nispace.io import write_json


# %%
ids = sync_osf(
    local_path=wd / "datasets" / "nispace-data_source", 
    osf_id="derpj", 
    config_file=wd / "datasets" / ".osfcli.config",
)

write_json(ids, wd / "datasets" / "osf_ids.json")

# %%
