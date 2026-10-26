"""Generate `wb_resample_fixture.npz`: Connectome Workbench resampling outputs on small synthetic spheres.

Lets CI (no Workbench installed) check that nispace's pure-Python ADAP_BARY_AREA re-implementation matches
``wb_command -metric-resample`` / ``-label-resample`` (with ``-area-metrics`` and ``-current-roi``).
Regenerate locally with Workbench installed:

    python tests/data/make_wb_resample_fixture.py
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.spatial import ConvexHull


def sphere_mesh(n, jitter, seed):
    i = np.arange(n) + 0.5
    phi, theta = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    xyz = np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    xyz += np.random.default_rng(seed).normal(scale=jitter, size=xyz.shape)
    xyz = 100 * xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    return xyz.astype(np.float32), ConvexHull(xyz).simplices.astype(np.int32)


def save(path, *arrays, intents, labeltable=None):
    darrays = [nib.gifti.GiftiDataArray(a, intent=it) for a, it in zip(arrays, intents)]
    nib.save(nib.GiftiImage(darrays=darrays, labeltable=labeltable), path)


def main():
    rng = np.random.default_rng(42)
    meshes = {"a": sphere_mesh(1200, 0.01, 1), "b": sphere_mesh(500, 0.02, 2)}
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, (xyz, faces) in meshes.items():
            n = len(xyz)
            area = rng.uniform(0.5, 1.5, n).astype(np.float32)
            roi = (xyz[:, 2] < 70).astype(np.float32)                      # "medial wall" cap excluded
            data = (np.sin(3 * xyz[:, 0] / 100) + xyz[:, 1] / 100 + 0.1 * rng.standard_normal(n)).astype(np.float32)
            labels = (1 + (xyz[:, 0] > 0) + 2 * (xyz[:, 1] > 30) + 4 * (np.sin(5 * xyz[:, 2] / 100) > 0)).astype(np.int32)
            save(tmp / f"{name}.surf.gii", xyz, faces, intents=["NIFTI_INTENT_POINTSET", "NIFTI_INTENT_TRIANGLE"])
            save(tmp / f"{name}_area.func.gii", area, intents=["NIFTI_INTENT_NONE"])
            save(tmp / f"{name}_roi.func.gii", roi, intents=["NIFTI_INTENT_NONE"])
            save(tmp / f"{name}_data.func.gii", data, intents=["NIFTI_INTENT_NONE"])
            lt = nib.gifti.GiftiLabelTable()
            for k in range(9):
                lab = nib.gifti.GiftiLabel(key=k, red=0.5, green=0.5, blue=0.5, alpha=1)
                lab.label = f"p{k}"
                lt.labels.append(lab)
            save(tmp / f"{name}_labels.label.gii", labels, intents=["NIFTI_INTENT_LABEL"], labeltable=lt)
            out.update({f"{name}_xyz": xyz, f"{name}_faces": faces, f"{name}_area": area, f"{name}_roi": roi,
                        f"{name}_data": data, f"{name}_labels": labels})
        for src, trg in [("a", "b"), ("b", "a")]:
            common = [tmp / f"{src}.surf.gii", tmp / f"{trg}.surf.gii", "ADAP_BARY_AREA"]
            opts = ["-area-metrics", tmp / f"{src}_area.func.gii", tmp / f"{trg}_area.func.gii",
                    "-current-roi", tmp / f"{src}_roi.func.gii"]
            mo, lo = tmp / "m.func.gii", tmp / "l.label.gii"
            subprocess.run(["wb_command", "-metric-resample", tmp / f"{src}_data.func.gii", *common, mo, *opts],
                           check=True, capture_output=True)
            subprocess.run(["wb_command", "-label-resample", tmp / f"{src}_labels.label.gii", *common, lo, *opts],
                           check=True, capture_output=True)
            out[f"wb_{src}{trg}_data"] = nib.load(mo).darrays[0].data
            out[f"wb_{src}{trg}_labels"] = nib.load(lo).darrays[0].data
    version = subprocess.run(["wb_command", "-version"], capture_output=True, text=True).stdout
    out["wb_version"] = np.array([line for line in version.splitlines() if "Version" in line][0])
    np.savez_compressed(Path(__file__).parent / "wb_resample_fixture.npz", **out)


if __name__ == "__main__":
    main()
