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
    local_path=wd / "datasets" / "nispace-private_source", 
    osf_id="m2ctx", 
    config_file=wd / "datasets" / ".osfcli.config",
)

write_json(ids, wd / "datasets" / "osfprivate_ids.json")

# %%
