from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


DEFAULT_RUN_DIR = Path(
    r"N:\Experimental_Data\yujunchen\projects\Inhouse_encoding_model"
    r"\TRIBEv2_encoding_regression\outputs\ckvideo_surface_reg_20260709_local"
)
DEFAULT_FSAVERAGE5_DIR = Path(
    r"N:\Experimental_Data\yujunchen\projects\data\NSD_atlas\nsddata\freesurfer\fsaverage5"
)
DEFAULT_JULIAN_DIR = Path(
    r"N:\Experimental_Data\yujunchen\projects\data\masks\Julian2012\face_parcels\face_parcels"
)

SURFACE_MAGIC_TRIANGLE = 16777214
N_LEFT_VERTICES = 10242


def read3(f) -> int:
    return int.from_bytes(f.read(3), byteorder="big")


def read_freesurfer_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        magic = read3(f)
        if magic != SURFACE_MAGIC_TRIANGLE:
            raise ValueError(f"{path} is not a FreeSurfer triangle surface; magic={magic}")
        f.readline()
        f.readline()
        n_vertices = struct.unpack(">i", f.read(4))[0]
        n_faces = struct.unpack(">i", f.read(4))[0]
        coords = np.frombuffer(f.read(n_vertices * 3 * 4), dtype=">f4").reshape(n_vertices, 3)
        faces = np.frombuffer(f.read(n_faces * 3 * 4), dtype=">i4").reshape(n_faces, 3)
    return coords.astype(np.float64), faces.astype(np.int64)


def load_surfaces(fsaverage5_dir: Path, surface: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        hemi: read_freesurfer_surface(fsaverage5_dir / "surf" / f"{hemi}.{surface}")
        for hemi in ("lh", "rh")
    }


def load_surface_pairs(fsaverage5_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        hemi: (
            read_freesurfer_surface(fsaverage5_dir / "surf" / f"{hemi}.white")[0],
            read_freesurfer_surface(fsaverage5_dir / "surf" / f"{hemi}.pial")[0],
        )
        for hemi in ("lh", "rh")
    }


def load_freesurfer_label(path: Path, n_vertices: int) -> np.ndarray:
    vertices = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 1:
                try:
                    vertices.append(int(parts[0]))
                except ValueError:
                    continue
    mask = np.zeros(n_vertices, dtype=bool)
    valid = [v for v in vertices if 0 <= v < n_vertices]
    mask[valid] = True
    return mask


def sample_volume_mask_to_surface(
    volume_path: Path,
    white: np.ndarray,
    pial: np.ndarray,
    ribbon_steps: int,
    radius_vox: int,
) -> np.ndarray:
    img = nib.load(str(volume_path))
    data = np.asarray(img.get_fdata()) != 0
    inv_affine = np.linalg.inv(img.affine)
    mask = np.zeros(white.shape[0], dtype=bool)
    offsets = []
    for dx in range(-radius_vox, radius_vox + 1):
        for dy in range(-radius_vox, radius_vox + 1):
            for dz in range(-radius_vox, radius_vox + 1):
                if dx * dx + dy * dy + dz * dz <= radius_vox * radius_vox:
                    offsets.append((dx, dy, dz))
    offsets = np.asarray(offsets, dtype=np.int64)

    shape = np.asarray(data.shape, dtype=np.int64)
    for depth in np.linspace(0.0, 1.0, ribbon_steps):
        coords = (1.0 - depth) * white + depth * pial
        hom = np.column_stack([coords, np.ones(coords.shape[0])])
        vox = np.rint((hom @ inv_affine.T)[:, :3]).astype(np.int64)
        for offset in offsets:
            pos = vox + offset
            in_bounds = np.all((pos >= 0) & (pos < shape[None, :]), axis=1)
            idx = pos[in_bounds]
            hit = np.zeros(pos.shape[0], dtype=bool)
            hit[in_bounds] = data[idx[:, 0], idx[:, 1], idx[:, 2]]
            mask |= hit
    return mask


def face_colors_from_values(values: np.ndarray, faces: np.ndarray, norm: colors.Normalize, cmap_name: str) -> np.ndarray:
    face_values = np.nanmean(values[faces], axis=1)
    cmap = plt.get_cmap(cmap_name)
    rgba = np.tile(np.array([0.74, 0.74, 0.74, 1.0]), (faces.shape[0], 1))
    active = np.isfinite(face_values) & (face_values != 0)
    rgba[active] = cmap(norm(face_values[active]))
    return rgba


def boundary_segments(mask: np.ndarray, coords: np.ndarray, faces: np.ndarray) -> np.ndarray:
    segments = []
    seen: set[tuple[int, int]] = set()
    for tri in faces:
        tri_mask = mask[tri]
        if tri_mask.all() or not tri_mask.any():
            continue
        for a, b in ((int(tri[0]), int(tri[1])), (int(tri[1]), int(tri[2])), (int(tri[2]), int(tri[0]))):
            if mask[a] != mask[b]:
                key = (min(a, b), max(a, b))
                if key not in seen:
                    seen.add(key)
                    segments.append([coords[a], coords[b]])
    if not segments:
        return np.zeros((0, 2, 3), dtype=np.float64)
    return np.asarray(segments, dtype=np.float64)


def lifted_coords(coords: np.ndarray, lift: float = 1.018) -> np.ndarray:
    center = coords.mean(axis=0, keepdims=True)
    return center + (coords - center) * lift


def setup_view(ax, coords: np.ndarray, hemi: str, view: str, title: str) -> None:
    xyz_min = coords.min(axis=0)
    xyz_max = coords.max(axis=0)
    center = (xyz_min + xyz_max) / 2.0
    extent = float(np.max(xyz_max - xyz_min)) / 2.0
    ax.set_xlim(center[0] - extent, center[0] + extent)
    ax.set_ylim(center[1] - extent, center[1] + extent)
    ax.set_zlim(center[2] - extent, center[2] + extent)
    if hemi == "lh" and view == "lateral":
        ax.view_init(elev=0, azim=180)
    elif hemi == "lh" and view == "medial":
        ax.view_init(elev=0, azim=0)
    elif hemi == "rh" and view == "lateral":
        ax.view_init(elev=0, azim=0)
    elif hemi == "rh" and view == "medial":
        ax.view_init(elev=0, azim=180)
    ax.set_title(title, fontsize=12)
    ax.set_axis_off()
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.65)
    except TypeError:
        ax.set_box_aspect((1, 1, 1))


def plot_tmap_with_borders(
    out_path: Path,
    surfaces: dict[str, tuple[np.ndarray, np.ndarray]],
    t_values: np.ndarray,
    sts_mask: np.ndarray,
) -> None:
    active = t_values[t_values != 0]
    vmax = float(np.percentile(active, 99)) if active.size else 1.0
    vmax = max(vmax, 1.0)
    norm = colors.Normalize(vmin=0.0, vmax=vmax)
    views = [("lh", "lateral"), ("lh", "medial"), ("rh", "medial"), ("rh", "lateral")]

    fig = plt.figure(figsize=(21, 10), constrained_layout=False)
    for col, (hemi, view) in enumerate(views):
        ax = fig.add_subplot(1, 4, col + 1, projection="3d")
        coords, faces = surfaces[hemi]
        sl = slice(0, N_LEFT_VERTICES) if hemi == "lh" else slice(N_LEFT_VERTICES, None)
        hemi_t = t_values[sl]
        hemi_sts = sts_mask[sl]

        surface = Poly3DCollection(
            coords[faces],
            facecolors=face_colors_from_values(hemi_t, faces, norm, "coolwarm"),
            edgecolors="none",
            linewidths=0,
            antialiased=False,
        )
        ax.add_collection3d(surface)

        for mask, color, width in ((hemi_sts, "#00D7FF", 2.6),):
            segs = boundary_segments(mask, lifted_coords(coords), faces)
            if segs.size:
                ax.add_collection3d(Line3DCollection(segs, colors="black", linewidths=width + 2.2, alpha=1.0))
                ax.add_collection3d(Line3DCollection(segs, colors=color, linewidths=width, alpha=1.0))
        setup_view(ax, coords, hemi, view, f"{hemi.upper()} {view}")

    fig.suptitle("Conjunction t-stat map with Julian2012 STS border on fsaverage5", fontsize=16)
    mappable = cm.ScalarMappable(norm=norm, cmap="coolwarm")
    cbar_ax = fig.add_axes([0.31, 0.095, 0.38, 0.025])
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("min(|t_valence|, |t_arousal|), both FDR-significant")
    legend = [
        Line2D([0], [0], color="#00D7FF", linewidth=3, label="Julian2012 STS border"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=1, frameon=False, fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.16, wspace=0.00)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--fsaverage5-dir", type=Path, default=DEFAULT_FSAVERAGE5_DIR)
    parser.add_argument("--julian-dir", type=Path, default=DEFAULT_JULIAN_DIR)
    parser.add_argument("--surface", choices=["inflated", "pial", "white"], default="inflated")
    parser.add_argument("--ribbon-steps", type=int, default=7)
    parser.add_argument("--radius-vox", type=int, default=1)
    args = parser.parse_args()

    out_dir = args.run_dir / "figures_fsaverage5_roi_borders"
    data_dir = args.run_dir / "roi_borders_fsaverage5"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    surfaces = load_surfaces(args.fsaverage5_dir, args.surface)
    surface_pairs = load_surface_pairs(args.fsaverage5_dir)
    sts_l = sample_volume_mask_to_surface(
        args.julian_dir / "lSTS.img",
        surface_pairs["lh"][0],
        surface_pairs["lh"][1],
        ribbon_steps=args.ribbon_steps,
        radius_vox=args.radius_vox,
    )
    sts_r = sample_volume_mask_to_surface(
        args.julian_dir / "rSTS.img",
        surface_pairs["rh"][0],
        surface_pairs["rh"][1],
        ribbon_steps=args.ribbon_steps,
        radius_vox=args.radius_vox,
    )
    sts_mask = np.concatenate([sts_l, sts_r])

    t_map = np.load(args.run_dir / "conjunction_valence_arousal" / "valence_arousal_both_fdr05_min_abs_t_vertices.npy")

    np.save(data_dir / "julian2012_sts_projected_fsaverage5_mask.npy", sts_mask)
    summary = {
        "space": "fsaverage5",
        "stat_map": "valence_arousal_both_fdr05_min_abs_t_vertices.npy",
        "julian_lsts_source": str(args.julian_dir / "lSTS.img"),
        "julian_rsts_source": str(args.julian_dir / "rSTS.img"),
        "julian_projection": f"sampled through white-pial ribbon, {args.ribbon_steps} depths, radius {args.radius_vox} voxel",
        "sts_vertices": int(sts_mask.sum()),
    }
    (data_dir / "sts_mt_border_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    out_path = out_dir / "fsaverage5_min_t_with_julian_sts_border.png"
    plot_tmap_with_borders(out_path, surfaces, t_map, sts_mask)
    (out_dir / "README_STS_MT_BORDERS.md").write_text(
        "# fsaverage5 t-map with STS border\n\n"
        "Background: conjunction min-t map, `min(abs(t_valence), abs(t_arousal))`, shown only where both predictors are FDR-significant.\n\n"
        "Border:\n"
        "- Cyan: Julian2012 lSTS/rSTS projected from Analyze volume masks to fsaverage5 surface.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
