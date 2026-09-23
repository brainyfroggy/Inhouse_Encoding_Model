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
from matplotlib import colors
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree


DEFAULT_RUN_DIR = Path(
    r"N:\Experimental_Data\yujunchen\projects\Inhouse_encoding_model"
    r"\TRIBEv2_encoding_regression\outputs\ckvideo_surface_reg_20260709_local"
)
DEFAULT_NSD_FS_DIR = Path(r"N:\Experimental_Data\yujunchen\projects\data\NSD_atlas\nsddata\freesurfer")

SURFACE_MAGIC_TRIANGLE = 16777214
N_LEFT_VERTICES_FSAVERAGE5 = 10242

KASTNER_LABELS = {
    1: "V1v",
    2: "V1d",
    3: "V2v",
    4: "V2d",
    5: "V3v",
    6: "V3d",
    7: "hV4",
    8: "VO1",
    9: "VO2",
    10: "PHC1",
    11: "PHC2",
    12: "TO2",
    13: "TO1",
    14: "LO2",
    15: "LO1",
    16: "V3B",
    17: "V3A",
    18: "IPS0",
    19: "IPS1",
    20: "IPS2",
    21: "IPS3",
    22: "IPS4",
    23: "IPS5",
    24: "SPL1",
    25: "FEF",
}

# Category-selective probabilistic ROIs in the NSD fsaverage bundle that match
# the Julian/fLoc-style category atlas used in the earlier local notes.
JULIAN_ROIS = [
    "EBA",
    "FBA-2",
    "OFA",
    "FFA-1",
    "FFA-2",
    "OPA",
    "PPA",
    "RSC",
    "OWFA",
    "VWFA-1",
    "VWFA-2",
]


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


def load_surfaces(fs_dir: Path, subject: str, surface: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        hemi: read_freesurfer_surface(fs_dir / subject / "surf" / f"{hemi}.{surface}")
        for hemi in ("lh", "rh")
    }


def fsaverage_to_fsaverage5_indices(fs_dir: Path) -> dict[str, np.ndarray]:
    mapping = {}
    for hemi in ("lh", "rh"):
        src_coords, _ = read_freesurfer_surface(fs_dir / "fsaverage" / "surf" / f"{hemi}.sphere.reg")
        dst_coords, _ = read_freesurfer_surface(fs_dir / "fsaverage5" / "surf" / f"{hemi}.sphere.reg")
        tree = cKDTree(src_coords)
        _, idx = tree.query(dst_coords, k=1)
        mapping[hemi] = idx.astype(np.int64)
    return mapping


def load_mgz_vector(path: Path) -> np.ndarray:
    data = np.asarray(nib.load(str(path)).get_fdata()).squeeze()
    if data.ndim != 1:
        raise ValueError(f"Expected a 1D surface vector in {path}, found shape {data.shape}")
    return data


def resample_kastner(fs_dir: Path, mapping: dict[str, np.ndarray]) -> np.ndarray:
    labels = []
    for hemi in ("lh", "rh"):
        src = load_mgz_vector(fs_dir / "fsaverage" / "label" / f"{hemi}.Kastner2015.mgz")
        labels.append(np.rint(src[mapping[hemi]]).astype(np.int16))
    return np.concatenate(labels)


def resample_julian(fs_dir: Path, mapping: dict[str, np.ndarray], threshold: float) -> tuple[np.ndarray, np.ndarray]:
    all_labels = []
    all_max_prob = []
    for hemi in ("lh", "rh"):
        probs = []
        for roi in JULIAN_ROIS:
            src = load_mgz_vector(fs_dir / "fsaverage" / "label" / f"{hemi}.probmap_{roi}.mgz")
            probs.append(src[mapping[hemi]])
        prob_arr = np.vstack(probs)
        winner = np.argmax(prob_arr, axis=0) + 1
        max_prob = np.max(prob_arr, axis=0)
        winner[max_prob < threshold] = 0
        all_labels.append(winner.astype(np.int16))
        all_max_prob.append(max_prob.astype(np.float32))
    return np.concatenate(all_labels), np.concatenate(all_max_prob)


def discrete_palette(n: int) -> list[tuple[float, float, float, float]]:
    palette = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(cmap_name)
        palette.extend([cmap(i) for i in range(cmap.N)])
    return palette[:n]


def label_face_colors(labels: np.ndarray, faces: np.ndarray, color_lookup: dict[int, tuple[float, float, float, float]]) -> np.ndarray:
    rgba = np.tile(np.array([0.72, 0.72, 0.72, 1.0]), (faces.shape[0], 1))
    face_labels = labels[faces]
    for label, color in color_lookup.items():
        active = np.any(face_labels == label, axis=1)
        rgba[active] = color
    return rgba


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
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.65)
    except TypeError:
        ax.set_box_aspect((1, 1, 1))


def plot_atlas(
    out_path: Path,
    surfaces: dict[str, tuple[np.ndarray, np.ndarray]],
    labels: np.ndarray,
    label_names: dict[int, str],
    title: str,
    max_legend_items: int = 30,
) -> None:
    active_labels = [int(v) for v in sorted(np.unique(labels)) if int(v) != 0]
    palette = discrete_palette(len(active_labels))
    color_lookup = {label: palette[idx] for idx, label in enumerate(active_labels)}

    fig = plt.figure(figsize=(21, 10), constrained_layout=False)
    views = [("lh", "lateral"), ("lh", "medial"), ("rh", "medial"), ("rh", "lateral")]
    for col, (hemi, view) in enumerate(views):
        ax = fig.add_subplot(1, 4, col + 1, projection="3d")
        coords, faces = surfaces[hemi]
        hemi_labels = labels[:N_LEFT_VERTICES_FSAVERAGE5] if hemi == "lh" else labels[N_LEFT_VERTICES_FSAVERAGE5:]
        collection = Poly3DCollection(
            coords[faces],
            facecolors=label_face_colors(hemi_labels, faces, color_lookup),
            edgecolors="none",
            linewidths=0,
            antialiased=False,
        )
        ax.add_collection3d(collection)
        setup_view(ax, coords, hemi, view, f"{hemi.upper()} {view}")

    handles = [
        Patch(facecolor=color_lookup[label], edgecolor="black", label=f"{label}: {label_names.get(label, str(label))}")
        for label in active_labels[:max_legend_items]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.18, wspace=0.00)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260, facecolor="white")
    plt.close(fig)


def write_label_csv(path: Path, labels: np.ndarray, label_names: dict[int, str], max_prob: np.ndarray | None = None) -> None:
    rows = ["vertex_global,hemisphere,vertex_hemi,label,label_name,max_probability"]
    for idx, label in enumerate(labels.astype(int)):
        hemi = "lh" if idx < N_LEFT_VERTICES_FSAVERAGE5 else "rh"
        hemi_vertex = idx if hemi == "lh" else idx - N_LEFT_VERTICES_FSAVERAGE5
        prob = "" if max_prob is None else f"{float(max_prob[idx]):.6g}"
        rows.append(f"{idx},{hemi},{hemi_vertex},{label},{label_names.get(label, 'Unknown')},{prob}")
    path.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--fs-dir", type=Path, default=DEFAULT_NSD_FS_DIR)
    parser.add_argument("--surface", choices=["inflated", "pial", "white"], default="inflated")
    parser.add_argument("--julian-threshold", type=float, default=0.10)
    args = parser.parse_args()

    out_dir = args.run_dir / "figures_fsaverage5_atlases"
    data_dir = args.run_dir / "atlases_fsaverage5"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    surfaces = load_surfaces(args.fs_dir, "fsaverage5", args.surface)
    mapping = fsaverage_to_fsaverage5_indices(args.fs_dir)
    np.save(data_dir / "lh_fsaverage5_to_fsaverage_nearest_sphere_reg.npy", mapping["lh"])
    np.save(data_dir / "rh_fsaverage5_to_fsaverage_nearest_sphere_reg.npy", mapping["rh"])

    kastner_labels = resample_kastner(args.fs_dir, mapping)
    np.save(data_dir / "kastner2015_labels_fsaverage5.npy", kastner_labels)
    write_label_csv(data_dir / "kastner2015_labels_fsaverage5.csv", kastner_labels, KASTNER_LABELS)
    plot_atlas(
        out_dir / "fsaverage5_kastner2015_atlas.png",
        surfaces,
        kastner_labels,
        KASTNER_LABELS,
        "Kastner2015 visual-field atlas resampled to fsaverage5",
    )

    julian_names = {idx + 1: roi for idx, roi in enumerate(JULIAN_ROIS)}
    julian_labels, julian_prob = resample_julian(args.fs_dir, mapping, threshold=args.julian_threshold)
    np.save(data_dir / "julian2012_category_labels_fsaverage5.npy", julian_labels)
    np.save(data_dir / "julian2012_category_max_probability_fsaverage5.npy", julian_prob)
    write_label_csv(data_dir / "julian2012_category_labels_fsaverage5.csv", julian_labels, julian_names, julian_prob)
    plot_atlas(
        out_dir / "fsaverage5_julian2012_category_atlas.png",
        surfaces,
        julian_labels,
        julian_names,
        f"Julian2012 category-selective atlas resampled to fsaverage5 (prob >= {args.julian_threshold:g})",
    )

    summary = {
        "space": "fsaverage5",
        "surface": args.surface,
        "resampling": "nearest neighbor on FreeSurfer sphere.reg from fsaverage to fsaverage5",
        "kastner_source": str(args.fs_dir / "fsaverage" / "label" / "[lh/rh].Kastner2015.mgz"),
        "julian_source": str(args.fs_dir / "fsaverage" / "label" / "[lh/rh].probmap_<ROI>.mgz"),
        "julian_probability_threshold": args.julian_threshold,
        "kastner_nonzero_vertices": int(np.count_nonzero(kastner_labels)),
        "julian_nonzero_vertices": int(np.count_nonzero(julian_labels)),
        "julian_rois": JULIAN_ROIS,
    }
    (out_dir / "README_FSAVERAGE5_ATLASES.md").write_text(
        "# fsaverage5 Atlas Figures\n\n"
        "These figures show atlas labels in the same fsaverage5 surface space as the TRIBEv2 outputs.\n\n"
        "- `fsaverage5_kastner2015_atlas.png`\n"
        "- `fsaverage5_julian2012_category_atlas.png`\n\n"
        "Kastner labels are discrete atlas labels. Julian2012 labels are winner-take-all category ROI labels from probabilistic maps after thresholding.\n",
        encoding="utf-8",
    )
    (data_dir / "atlas_resampling_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote atlas figures to: {out_dir}")


if __name__ == "__main__":
    main()
