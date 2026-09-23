from __future__ import annotations

import argparse
import gzip
import json
import struct
from pathlib import Path

import numpy as np


DEFAULT_RUN_DIR = Path(
    r"N:\Experimental_Data\yujunchen\projects\Inhouse_encoding_model"
    r"\TRIBEv2_encoding_regression\outputs\ckvideo_surface_reg_20260709_local"
)
DEFAULT_FSAVERAGE5_DIR = Path(
    r"N:\Experimental_Data\yujunchen\projects\data\NSD_atlas\nsddata\freesurfer\fsaverage5"
)
DEFAULT_TEMPLATE = Path(r"C:\MRIcroGL\Resources\standard\mni152.nii.gz")

PREDICTORS = ("valence", "arousal")
N_LEFT_VERTICES = 10242
SURFACE_MAGIC_TRIANGLE = 16777214


def read3(f) -> int:
    return int.from_bytes(f.read(3), byteorder="big")


def read_freesurfer_surface(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic = read3(f)
        if magic != SURFACE_MAGIC_TRIANGLE:
            raise ValueError(f"{path} is not a FreeSurfer triangle surface; magic={magic}")
        f.readline()
        f.readline()
        n_vertices = struct.unpack(">i", f.read(4))[0]
        n_faces = struct.unpack(">i", f.read(4))[0]
        coords = np.frombuffer(f.read(n_vertices * 3 * 4), dtype=">f4").reshape(n_vertices, 3)
        # Consume faces so malformed/truncated files fail loudly.
        faces = np.frombuffer(f.read(n_faces * 3 * 4), dtype=">i4")
        if faces.size != n_faces * 3:
            raise ValueError(f"{path} ended before all faces were read.")
    return coords.astype(np.float64)


def load_surface_pairs(fsaverage5_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    white_coords = []
    pial_coords = []
    for hemi in ("lh", "rh"):
        white = read_freesurfer_surface(fsaverage5_dir / "surf" / f"{hemi}.white")
        pial = read_freesurfer_surface(fsaverage5_dir / "surf" / f"{hemi}.pial")
        if white.shape != pial.shape:
            raise ValueError(f"{hemi} white and pial surfaces have different shapes")
        white_coords.append(white)
        pial_coords.append(pial)
    all_white = np.vstack(white_coords)
    all_pial = np.vstack(pial_coords)
    if all_white.shape != (N_LEFT_VERTICES * 2, 3):
        raise ValueError(f"Expected 20484 fsaverage5 vertices, found {all_white.shape}")
    return all_white, all_pial


def make_projection_coords(
    fsaverage5_dir: Path,
    mode: str,
    ribbon_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    white, pial = load_surface_pairs(fsaverage5_dir)
    n_vertices = white.shape[0]
    vertex_indices = np.arange(n_vertices, dtype=np.int64)

    if mode == "midgray":
        return (white + pial) / 2.0, vertex_indices
    if mode != "ribbon":
        raise ValueError(f"Unknown projection mode: {mode}")
    if ribbon_steps < 2:
        raise ValueError("--ribbon-steps must be at least 2 for ribbon mode")

    coords = []
    indices = []
    for depth in np.linspace(0.0, 1.0, ribbon_steps):
        coords.append((1.0 - depth) * white + depth * pial)
        indices.append(vertex_indices)
    return np.vstack(coords), np.concatenate(indices)


def read_nifti_header(path: Path) -> tuple[bytes, tuple[int, int, int], np.ndarray, float]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        header = bytearray(f.read(348))
        extension = f.read(4)
    sizeof_hdr = struct.unpack("<i", header[:4])[0]
    if sizeof_hdr != 348:
        raise ValueError(f"{path} is not a little-endian NIfTI-1 image.")
    dim = struct.unpack("<8h", header[40:56])
    shape = (int(dim[1]), int(dim[2]), int(dim[3]))
    sform_code = struct.unpack("<h", header[254:256])[0]
    if sform_code > 0:
        affine = np.array(
            [
                struct.unpack("<4f", header[280:296]),
                struct.unpack("<4f", header[296:312]),
                struct.unpack("<4f", header[312:328]),
                (0.0, 0.0, 0.0, 1.0),
            ],
            dtype=np.float64,
        )
    else:
        pixdim = struct.unpack("<8f", header[76:108])
        affine = np.diag([pixdim[1], pixdim[2], pixdim[3], 1.0]).astype(np.float64)
    vox_offset = float(struct.unpack("<f", header[108:112])[0])
    if int(round(vox_offset)) < 352:
        vox_offset = 352.0
    return bytes(header + extension), shape, affine, vox_offset


def make_float32_header(template_header: bytes, description: str) -> bytes:
    header = bytearray(template_header[:352])
    struct.pack_into("<h", header, 70, 16)
    struct.pack_into("<h", header, 72, 32)
    struct.pack_into("<f", header, 108, 352.0)
    struct.pack_into("<f", header, 112, 1.0)
    struct.pack_into("<f", header, 116, 0.0)
    desc = description.encode("ascii", errors="ignore")[:79]
    header[148:228] = b"\0" * 80
    header[148 : 148 + len(desc)] = desc
    header[344:348] = b"n+1\0"
    if len(header) < 352:
        header.extend(b"\0" * (352 - len(header)))
    return bytes(header[:352])


def save_like_template(volume: np.ndarray, path: Path, template_header: bytes, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = make_float32_header(template_header, description)
    data = np.asarray(volume, dtype=np.dtype("<f4"))
    with gzip.open(path, "wb") as f:
        f.write(header)
        f.write(data.tobytes(order="F"))


def vertex_voxels(coords_mm: np.ndarray, affine: np.ndarray, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    hom = np.column_stack([coords_mm, np.ones(coords_mm.shape[0])])
    vox = hom @ np.linalg.inv(affine).T
    ijk = np.rint(vox[:, :3]).astype(np.int64)
    in_bounds = np.all((ijk >= 0) & (ijk < np.array(shape)[None, :]), axis=1)
    return ijk, in_bounds


def make_neighbor_offsets(radius_vox: int) -> np.ndarray:
    offsets = []
    for dx in range(-radius_vox, radius_vox + 1):
        for dy in range(-radius_vox, radius_vox + 1):
            for dz in range(-radius_vox, radius_vox + 1):
                if dx * dx + dy * dy + dz * dz <= radius_vox * radius_vox:
                    offsets.append((dx, dy, dz))
    return np.asarray(offsets, dtype=np.int64)


def rasterize_vertices(
    values: np.ndarray,
    ijk: np.ndarray,
    in_bounds: np.ndarray,
    shape: tuple[int, int, int],
    radius_vox: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    volume = np.zeros(shape, dtype=np.float32)
    offsets = make_neighbor_offsets(radius_vox)

    for vertex_idx in np.flatnonzero(in_bounds):
        value = float(values[vertex_idx])
        if value == 0.0 or not np.isfinite(value):
            continue
        center = ijk[vertex_idx]
        for offset in offsets:
            pos = center + offset
            if np.any(pos < 0) or np.any(pos >= np.array(shape)):
                continue
            x, y, z = (int(v) for v in pos)
            old = volume[x, y, z]
            if abs(value) > abs(float(old)):
                volume[x, y, z] = value
    return volume


def export(
    run_dir: Path,
    fsaverage5_dir: Path,
    template_path: Path,
    radius_vox: int,
    mode: str,
    ribbon_steps: int,
    out_name: str | None,
) -> Path:
    if out_name is None:
        out_name = "nifti_mni152_surface_projected" if mode == "midgray" else f"nifti_mni152_surface_ribbon_r{radius_vox}"
    out_dir = run_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    template_header, template_shape, affine, _ = read_nifti_header(template_path)
    coords, value_vertex_indices = make_projection_coords(fsaverage5_dir, mode=mode, ribbon_steps=ribbon_steps)
    ijk, in_bounds = vertex_voxels(coords, affine, template_shape)

    beta = np.load(run_dir / "beta_intercept_valence_arousal_vertices.npy")
    t_values = np.load(run_dir / "t_cluster_intercept_valence_arousal_vertices.npy")
    q_values = np.load(run_dir / "q_fdr_valence_arousal_vertices.npy")
    fdr_masks = np.load(run_dir / "fdr05_mask_valence_arousal_vertices.npy")

    manifest: dict[str, object] = {
        "export_type": "fsaverage5 surface values rasterized into MRIcroGL MNI152 template grid",
        "template_path": str(template_path),
        "template_shape_xyz": list(template_shape),
        "template_affine": affine.tolist(),
        "surface_space": "fsaverage5 coordinates from local FreeSurfer fsaverage5 white/pial surfaces",
        "projection_mode": mode,
        "ribbon_steps": int(ribbon_steps) if mode == "ribbon" else 1,
        "surface_to_volume_note": (
            "This is a cortical surface rasterization for MRIcroGL viewing, not a true volumetric "
            "statistical analysis in MNI voxel space."
        ),
        "radius_vox": int(radius_vox),
        "n_surface_vertices": int(N_LEFT_VERTICES * 2),
        "n_projection_points": int(coords.shape[0]),
        "n_projection_points_in_template_bounds": int(in_bounds.sum()),
        "files": [],
    }

    np.save(out_dir / "fsaverage5_projection_coords_mm.npy", coords.astype(np.float32))
    np.save(out_dir / "fsaverage5_projection_voxels_mni152.npy", ijk.astype(np.int16))
    np.save(out_dir / "fsaverage5_projection_vertex_indices.npy", value_vertex_indices.astype(np.int32))
    np.save(out_dir / "fsaverage5_projection_points_in_mni152_bounds.npy", in_bounds)

    for idx, predictor in enumerate(PREDICTORS):
        beta_vec = beta[idx + 1]
        t_vec = t_values[idx + 1]
        q_vec = q_values[idx]
        fdr_vec = fdr_masks[idx]
        pos_vec = fdr_vec & (beta_vec > 0)
        neg_vec = fdr_vec & (beta_vec < 0)
        maps = {
            f"{predictor}_beta_mni152_surface.nii.gz": beta_vec,
            f"{predictor}_t_cluster_mni152_surface.nii.gz": t_vec,
            f"{predictor}_q_fdr_mni152_surface.nii.gz": q_vec,
            f"{predictor}_t_fdr05_positive_mni152_surface.nii.gz": np.where(pos_vec, t_vec, 0.0),
            f"{predictor}_t_fdr05_negative_mni152_surface.nii.gz": np.where(neg_vec, t_vec, 0.0),
            f"{predictor}_beta_fdr05_positive_mni152_surface.nii.gz": np.where(pos_vec, beta_vec, 0.0),
            f"{predictor}_beta_fdr05_negative_mni152_surface.nii.gz": np.where(neg_vec, beta_vec, 0.0),
            f"{predictor}_fdr05_positive_mask_mni152_surface.nii.gz": pos_vec.astype(np.float32),
            f"{predictor}_fdr05_negative_mask_mni152_surface.nii.gz": neg_vec.astype(np.float32),
        }
        for filename, vec in maps.items():
            projection_values = np.asarray(vec)[value_vertex_indices]
            volume = rasterize_vertices(projection_values, ijk, in_bounds, template_shape, radius_vox=radius_vox)
            save_like_template(volume, out_dir / filename, template_header, filename.replace(".nii.gz", ""))
            manifest["files"].append(filename)

    readme = f"""# MNI152 Surface-Projected NIfTI Exports

These files use the same NIfTI grid/header as:

`{template_path}`

They should overlay on MRIcroGL's `mni152` background without the square-grid
artifact from the earlier pseudo-volume export.

Important: TRIBEv2 predictions are fsaverage5 cortical surface vertices. These
NIfTIs are surface values rasterized into the MNI152 display grid using the local
fsaverage5 white/pial geometry. They are for visualization in MRIcroGL, not a
true voxelwise MNI-space statistical model.

Start with:

- `valence_t_fdr05_positive_mni152_surface.nii.gz`
- `valence_t_fdr05_negative_mni152_surface.nii.gz`
- `arousal_t_fdr05_positive_mni152_surface.nii.gz`
- `arousal_t_fdr05_negative_mni152_surface.nii.gz`

Projection mode: `{mode}`.
Ribbon steps: `{ribbon_steps if mode == "ribbon" else 1}`.
Rasterization radius: `{radius_vox}` voxel(s).
Projection points in template bounds: `{int(in_bounds.sum())}` / `{coords.shape[0]}`.
"""
    (out_dir / "README_MRICROGL_MNI152_SURFACE.md").write_text(readme, encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--fsaverage5-dir", type=Path, default=DEFAULT_FSAVERAGE5_DIR)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--radius-vox", type=int, default=1)
    parser.add_argument("--mode", choices=["midgray", "ribbon"], default="midgray")
    parser.add_argument("--ribbon-steps", type=int, default=7)
    parser.add_argument("--out-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = export(
        args.run_dir,
        args.fsaverage5_dir,
        args.template,
        args.radius_vox,
        args.mode,
        args.ribbon_steps,
        args.out_name,
    )
    print(f"Wrote MNI152 surface-projected NIfTIs to: {out_dir}")


if __name__ == "__main__":
    main()
