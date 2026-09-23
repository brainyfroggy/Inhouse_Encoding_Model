from __future__ import annotations

import argparse
import gzip
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = Path(
    r"N:\Experimental_Data\yujunchen\projects\Inhouse_encoding_model"
    r"\TRIBEv2_encoding_regression\outputs\ckvideo_surface_reg_20260709_local"
)

PREDICTORS = ("valence", "arousal")
N_LEFT_VERTICES = 10242
GRID_X = 102
GRID_Y = 101
GRID_CAPACITY_PER_HEMI = GRID_X * GRID_Y
PSEUDOVOL_SHAPE = (GRID_X, GRID_Y, 2)


def make_header(shape: tuple[int, int, int], dtype: np.dtype) -> bytes:
    dtype = np.dtype(dtype)
    if dtype == np.dtype("float32"):
        datatype = 16
        bitpix = 32
    elif dtype == np.dtype("uint8"):
        datatype = 2
        bitpix = 8
    else:
        raise ValueError(f"Unsupported NIfTI dtype: {dtype}")

    hdr = bytearray(348)
    struct.pack_into("<i", hdr, 0, 348)
    struct.pack_into("<8h", hdr, 40, 3, shape[0], shape[1], shape[2], 1, 1, 1, 1)
    struct.pack_into("<h", hdr, 70, datatype)
    struct.pack_into("<h", hdr, 72, bitpix)
    struct.pack_into("<8f", hdr, 76, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    struct.pack_into("<f", hdr, 108, 352.0)
    struct.pack_into("<f", hdr, 112, 1.0)
    struct.pack_into("<f", hdr, 116, 0.0)
    struct.pack_into("<h", hdr, 252, 1)
    struct.pack_into("<h", hdr, 254, 1)
    struct.pack_into("<4f", hdr, 280, 1.0, 0.0, 0.0, 0.0)
    struct.pack_into("<4f", hdr, 296, 0.0, 1.0, 0.0, 0.0)
    struct.pack_into("<4f", hdr, 312, 0.0, 0.0, 1.0, 0.0)
    hdr[344:348] = b"n+1\0"
    return bytes(hdr) + b"\0\0\0\0"


def save_nifti_gz(volume: np.ndarray, path: Path, dtype: np.dtype) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(volume, dtype=dtype)
    header = make_header(tuple(int(v) for v in data.shape), data.dtype)
    endian_dtype = np.dtype(data.dtype).newbyteorder("<")
    with gzip.open(path, "wb") as f:
        f.write(header)
        f.write(np.asarray(data, dtype=endian_dtype).tobytes(order="F"))


def surface_to_pseudovolume(values: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[0] != N_LEFT_VERTICES * 2:
        raise ValueError(f"Expected {N_LEFT_VERTICES * 2} vertices, found {values.shape[0]}")
    volume = np.full(PSEUDOVOL_SHAPE, fill_value, dtype=values.dtype)
    for hemi_idx, start in enumerate((0, N_LEFT_VERTICES)):
        hemi_values = values[start : start + N_LEFT_VERTICES]
        padded = np.full(GRID_CAPACITY_PER_HEMI, fill_value, dtype=values.dtype)
        padded[:N_LEFT_VERTICES] = hemi_values
        volume[:, :, hemi_idx] = padded.reshape((GRID_X, GRID_Y), order="F")
    return volume


def write_vertex_map(out_dir: Path) -> None:
    rows = []
    for global_idx in range(N_LEFT_VERTICES * 2):
        hemi = "lh" if global_idx < N_LEFT_VERTICES else "rh"
        hemi_vertex = global_idx if hemi == "lh" else global_idx - N_LEFT_VERTICES
        rows.append(
            {
                "vertex_global": global_idx,
                "hemisphere": hemi,
                "vertex_hemi": hemi_vertex,
                "x": int(hemi_vertex % GRID_X),
                "y": int(hemi_vertex // GRID_X),
                "z": 0 if hemi == "lh" else 1,
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "pseudovolume_vertex_index_map.csv", index=False)


def export(run_dir: Path) -> Path:
    out_dir = run_dir / "nifti_pseudovolume"
    out_dir.mkdir(parents=True, exist_ok=True)

    beta = np.load(run_dir / "beta_intercept_valence_arousal_vertices.npy")
    t_values = np.load(run_dir / "t_cluster_intercept_valence_arousal_vertices.npy")
    p_values = np.load(run_dir / "p_cluster_valence_arousal_vertices.npy")
    q_values = np.load(run_dir / "q_fdr_valence_arousal_vertices.npy")
    fdr_masks = np.load(run_dir / "fdr05_mask_valence_arousal_vertices.npy")

    write_vertex_map(out_dir)

    manifest: dict[str, object] = {
        "source_run_dir": str(run_dir),
        "warning": (
            "These are fsaverage5 surface vertices laid out as a two-slice pseudo-volume "
            "for MRIcroGL inspection. They are not anatomical MNI152 voxel maps."
        ),
        "shape_xyz": list(PSEUDOVOL_SHAPE),
        "slice_z_0": "left hemisphere fsaverage5 vertices",
        "slice_z_1": "right hemisphere fsaverage5 vertices",
        "grid_layout": "x = vertex_hemi % 102, y = vertex_hemi // 102",
        "files": [],
    }

    for idx, predictor in enumerate(PREDICTORS):
        beta_vec = beta[idx + 1]
        t_vec = t_values[idx + 1]
        p_vec = p_values[idx]
        q_vec = q_values[idx]
        fdr_vec = fdr_masks[idx]
        pos_vec = fdr_vec & (beta_vec > 0)
        neg_vec = fdr_vec & (beta_vec < 0)

        maps = {
            f"{predictor}_beta_pseudovolume.nii.gz": (beta_vec, np.float32),
            f"{predictor}_t_cluster_pseudovolume.nii.gz": (t_vec, np.float32),
            f"{predictor}_p_cluster_pseudovolume.nii.gz": (p_vec, np.float32),
            f"{predictor}_q_fdr_pseudovolume.nii.gz": (q_vec, np.float32),
            f"{predictor}_fdr05_mask_pseudovolume.nii.gz": (fdr_vec.astype(np.uint8), np.uint8),
            f"{predictor}_fdr05_positive_mask_pseudovolume.nii.gz": (pos_vec.astype(np.uint8), np.uint8),
            f"{predictor}_fdr05_negative_mask_pseudovolume.nii.gz": (neg_vec.astype(np.uint8), np.uint8),
            f"{predictor}_t_fdr05_positive_pseudovolume.nii.gz": (np.where(pos_vec, t_vec, 0.0), np.float32),
            f"{predictor}_t_fdr05_negative_pseudovolume.nii.gz": (np.where(neg_vec, t_vec, 0.0), np.float32),
            f"{predictor}_beta_fdr05_positive_pseudovolume.nii.gz": (np.where(pos_vec, beta_vec, 0.0), np.float32),
            f"{predictor}_beta_fdr05_negative_pseudovolume.nii.gz": (np.where(neg_vec, beta_vec, 0.0), np.float32),
        }
        for filename, (vec, dtype) in maps.items():
            path = out_dir / filename
            save_nifti_gz(surface_to_pseudovolume(vec, fill_value=0), path, np.dtype(dtype))
            manifest["files"].append(filename)

    readme = """# MRIcroGL Pseudo-Volume Exports

These `.nii.gz` files are MRIcroGL-readable pseudo-volumes created from TRIBEv2
fsaverage5 surface vertices.

They are useful for quick inspection of coefficient/significance patterns, but
they are **not anatomical MNI152 volumes** and should not be overlaid on an MNI
template as if each value were a real voxel location.

Layout:

- `z = 0`: left hemisphere fsaverage5 vertices
- `z = 1`: right hemisphere fsaverage5 vertices
- `x = vertex_hemi % 102`
- `y = vertex_hemi // 102`

Use `pseudovolume_vertex_index_map.csv` to map any pseudo-volume coordinate back
to the original fsaverage5 vertex index.

For significance viewing, start with:

- `valence_t_fdr05_positive_pseudovolume.nii.gz`
- `valence_t_fdr05_negative_pseudovolume.nii.gz`
- `arousal_t_fdr05_positive_pseudovolume.nii.gz`
- `arousal_t_fdr05_negative_pseudovolume.nii.gz`
"""
    (out_dir / "README_MRICROGL_PSEUDOVOLUME.md").write_text(readme, encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    return parser.parse_args()


def main() -> None:
    out_dir = export(parse_args().run_dir)
    print(f"Wrote MRIcroGL pseudo-volume NIfTIs to: {out_dir}")


if __name__ == "__main__":
    main()
