"""Pinned remote inputs for the logo, downloaded once into a cache folder and checked by MD5.
Usage as a script: remote.py CACHE_DIR NAME  -> prints the local path of NAME."""
import hashlib
import sys
import urllib.request
from pathlib import Path

GIN = "https://gin.g-node.org/llotter/mni_freesurfer/raw/4e7d399d6f10f28b5310189798e40b85d23ac438/freesurfer/MNI152NLin6Asym"
JBM = "https://raw.githubusercontent.com/JetBrains/JetBrainsMono/02bb50b082dad9ef8a0f33ac393839202b760223/fonts/ttf"

SOURCES = {
    # FreeSurfer recon of the MNI152NLin6Asym template (GIN: llotter/mni_freesurfer)
    "rh.pial": (f"{GIN}/surf/rh.pial", "7a255d18bd2a4bf0be12838e4d700cde"),
    "rh.thickness": (f"{GIN}/surf/rh.thickness", "bf847cabd14c0f98cb759c9bb8ac0a52"),
    "rh.aparc.annot": (f"{GIN}/label/rh.aparc.annot", "9cda966fc97d9348f777ea4c7e8c8ead"),
    # JetBrains Mono Bold (wordmark) and ExtraBold (full name) (SIL Open Font License 1.1)
    "JetBrainsMono-Bold.ttf": (f"{JBM}/JetBrainsMono-Bold.ttf", "79e9a0365a86aeb48c8d51212b215c9b"),
    "JetBrainsMono-ExtraBold.ttf": (f"{JBM}/JetBrainsMono-ExtraBold.ttf", "8d83c4826bd219efd19fe257fe0165ec"),
}


def fetch(name, cache):
    url, md5 = SOURCES[name]
    path = Path(cache) / name
    if not path.exists() or hashlib.md5(path.read_bytes()).hexdigest() != md5:
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
        got = hashlib.md5(path.read_bytes()).hexdigest()
        if got != md5:
            raise RuntimeError(f"checksum mismatch for {name}: {got} != {md5} ({url})")
    return str(path)


if __name__ == "__main__":
    print(fetch(sys.argv[2], sys.argv[1]))
