from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


DEFAULT_RUN_DIR = Path(
    r"N:\Experimental_Data\yujunchen\projects\Inhouse_encoding_model"
    r"\TRIBEv2_encoding_regression\outputs\ckvideo_surface_reg_20260709_local"
)
DEFAULT_FSAVERAGE5_DIR = Path(
    r"N:\Experimental_Data\yujunchen\projects\data\NSD_atlas\nsddata\freesurfer\fsaverage5"
)

PREDICTORS = ("valence", "arousal")
N_LEFT_VERTICES = 10242
SURFACE_MAGIC_TRIANGLE = 16777214


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


def face_values(vertex_values: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return np.nanmean(vertex_values[faces], axis=1)


def face_colors(
    vertex_values: np.ndarray,
    faces: np.ndarray,
    norm: colors.Normalize,
    cmap_name: str = "coolwarm",
) -> np.ndarray:
    values = face_values(vertex_values, faces)
    cmap = plt.get_cmap(cmap_name)
    rgba = np.tile(np.array([0.78, 0.78, 0.78, 1.0]), (faces.shape[0], 1))
    active = np.isfinite(values) & (values != 0)
    rgba[active] = cmap(norm(values[active]))
    return rgba


def plot_hemi_view(
    ax,
    coords: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    norm: colors.Normalize,
    hemi: str,
    view: str,
    title: str,
    cmap_name: str = "coolwarm",
) -> None:
    collection = Poly3DCollection(
        coords[faces],
        facecolors=face_colors(values, faces, norm, cmap_name=cmap_name),
        edgecolors="none",
        linewidths=0.0,
        antialiased=False,
    )
    ax.add_collection3d(collection)

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
    elif view == "dorsal":
        ax.view_init(elev=90, azim=90)
    elif view == "ventral":
        ax.view_init(elev=-90, azim=90)
    else:
        raise ValueError(f"Unknown hemi/view combination: {hemi} {view}")

    ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.65)
    except TypeError:
        ax.set_box_aspect((1, 1, 1))


def plot_signed_fdr_panel(
    out_path: Path,
    surfaces: dict[str, tuple[np.ndarray, np.ndarray]],
    t_values: np.ndarray,
    fdr_masks: np.ndarray,
    vmax: float,
) -> None:
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig = plt.figure(figsize=(15, 7.5), constrained_layout=False)
    views = [("lh", "lateral"), ("lh", "medial"), ("rh", "medial"), ("rh", "lateral")]

    for row, predictor in enumerate(PREDICTORS):
        signed = np.where(fdr_masks[row], t_values[row + 1], 0.0)
        for col, (hemi, view) in enumerate(views):
            ax = fig.add_subplot(2, 4, row * 4 + col + 1, projection="3d")
            coords, faces = surfaces[hemi]
            hemi_values = signed[:N_LEFT_VERTICES] if hemi == "lh" else signed[N_LEFT_VERTICES:]
            plot_hemi_view(
                ax,
                coords,
                faces,
                hemi_values,
                norm,
                hemi=hemi,
                view=view,
                title=f"{predictor.capitalize()} {hemi.upper()} {view}",
            )

    fig.suptitle("TRIBEv2 CK-video encoding regression: FDR q < 0.05 signed t maps on fsaverage5", fontsize=15)
    mappable = cm.ScalarMappable(norm=norm, cmap="coolwarm")
    cbar_ax = fig.add_axes([0.28, 0.045, 0.44, 0.025])
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Cluster-robust t value, FDR q < 0.05")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.10, wspace=0.00, hspace=0.08)
    fig.savefig(out_path, dpi=240, facecolor="white")
    plt.close(fig)


def plot_predictor_detail(
    out_path: Path,
    surfaces: dict[str, tuple[np.ndarray, np.ndarray]],
    signed: np.ndarray,
    predictor: str,
    vmax: float,
) -> None:
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig = plt.figure(figsize=(15, 8.5), constrained_layout=False)
    views = [("lh", "lateral"), ("lh", "medial"), ("rh", "medial"), ("rh", "lateral")]
    for col, (hemi, view) in enumerate(views):
        ax = fig.add_subplot(1, 4, col + 1, projection="3d")
        coords, faces = surfaces[hemi]
        hemi_values = signed[:N_LEFT_VERTICES] if hemi == "lh" else signed[N_LEFT_VERTICES:]
        plot_hemi_view(ax, coords, faces, hemi_values, norm, hemi, view, f"{hemi.upper()} {view}", cmap_name="magma")
    fig.suptitle(f"{predictor.capitalize()} FDR q < 0.05 signed t map on fsaverage5 inflated surface", fontsize=15)
    mappable = cm.ScalarMappable(norm=norm, cmap="coolwarm")
    cbar_ax = fig.add_axes([0.28, 0.08, 0.44, 0.025])
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Cluster-robust t value")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.13, wspace=0.00)
    fig.savefig(out_path, dpi=260, facecolor="white")
    plt.close(fig)


def plot_sign_count_bar(out_path: Path, summary: dict[str, object]) -> None:
    counts = summary["significant_counts"]
    labels = []
    values = []
    bar_colors = []
    for predictor in PREDICTORS:
        labels.extend([f"{predictor}\npositive", f"{predictor}\nnegative"])
        values.extend([counts[predictor]["positive"], counts[predictor]["negative"]])
        bar_colors.extend(["#c43b3b", "#356aa0"])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(labels, values, color=bar_colors)
    ax.set_ylabel("Significant fsaverage5 vertices")
    ax.set_title("FDR q < 0.05 significant surface vertices")
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:,}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, facecolor="white")
    plt.close(fig)


def plot_conjunction_category_panel(
    out_path: Path,
    surfaces: dict[str, tuple[np.ndarray, np.ndarray]],
    categories: np.ndarray,
    counts: dict[str, int],
) -> None:
    category_colors = {
        1: "#2166ac",  # both negative
        2: "#67a9cf",  # valence negative, arousal positive
        3: "#ef8a62",  # valence positive, arousal negative
        4: "#b2182b",  # both positive
    }
    category_labels = {
        1: "Both negative",
        2: "Valence -, arousal +",
        3: "Valence +, arousal -",
        4: "Both positive",
    }

    def category_face_colors(vertex_categories: np.ndarray, faces: np.ndarray) -> np.ndarray:
        rgba = np.tile(np.array([0.78, 0.78, 0.78, 1.0]), (faces.shape[0], 1))
        vals = vertex_categories[faces]
        for category, color in category_colors.items():
            active = np.any(vals == category, axis=1)
            rgba[active] = colors.to_rgba(color)
        return rgba

    fig = plt.figure(figsize=(15, 7.5), constrained_layout=False)
    views = [("lh", "lateral"), ("lh", "medial"), ("rh", "medial"), ("rh", "lateral")]
    for col, (hemi, view) in enumerate(views):
        ax = fig.add_subplot(1, 4, col + 1, projection="3d")
        coords, faces = surfaces[hemi]
        hemi_categories = categories[:N_LEFT_VERTICES] if hemi == "lh" else categories[N_LEFT_VERTICES:]
        collection = Poly3DCollection(
            coords[faces],
            facecolors=category_face_colors(hemi_categories, faces),
            edgecolors="none",
            linewidths=0.0,
            antialiased=False,
        )
        ax.add_collection3d(collection)
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
        ax.set_title(f"{hemi.upper()} {view}", fontsize=11)
        ax.set_axis_off()
        try:
            ax.set_box_aspect((1, 1, 1), zoom=1.65)
        except TypeError:
            ax.set_box_aspect((1, 1, 1))

    handles = [
        Patch(facecolor=category_colors[key], edgecolor="black", label=f"{category_labels[key]} ({counts[str(key)]:,})")
        for key in (1, 2, 3, 4)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle("Vertices where valence and arousal coefficients are both FDR-significant", fontsize=15)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.16, wspace=0.00)
    fig.savefig(out_path, dpi=260, facecolor="white")
    plt.close(fig)


def plot_conjunction_min_panel(
    out_path: Path,
    surfaces: dict[str, tuple[np.ndarray, np.ndarray]],
    min_abs_t: np.ndarray,
    vmax: float,
) -> None:
    norm = colors.Normalize(vmin=0.0, vmax=vmax)
    fig = plt.figure(figsize=(15, 8.5), constrained_layout=False)
    views = [("lh", "lateral"), ("lh", "medial"), ("rh", "medial"), ("rh", "lateral")]
    for col, (hemi, view) in enumerate(views):
        ax = fig.add_subplot(1, 4, col + 1, projection="3d")
        coords, faces = surfaces[hemi]
        hemi_values = min_abs_t[:N_LEFT_VERTICES] if hemi == "lh" else min_abs_t[N_LEFT_VERTICES:]
        plot_hemi_view(ax, coords, faces, hemi_values, norm, hemi, view, f"{hemi.upper()} {view}")
    fig.suptitle("Conjunction t-stat map: min(|t_valence|, |t_arousal|) where both are FDR-significant", fontsize=15)
    mappable = cm.ScalarMappable(norm=norm, cmap="magma")
    cbar_ax = fig.add_axes([0.28, 0.08, 0.44, 0.025])
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Minimum absolute t statistic across valence and arousal")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.13, wspace=0.00)
    fig.savefig(out_path, dpi=260, facecolor="white")
    plt.close(fig)


def save_conjunction_outputs(
    run_dir: Path,
    out_dir: Path,
    surfaces: dict[str, tuple[np.ndarray, np.ndarray]],
    t_values: np.ndarray,
    beta: np.ndarray,
    fdr_masks: np.ndarray,
) -> None:
    val_sig = fdr_masks[0]
    aro_sig = fdr_masks[1]
    conjunction = val_sig & aro_sig
    val_t = t_values[1]
    aro_t = t_values[2]
    val_beta = beta[1]
    aro_beta = beta[2]

    categories = np.zeros(conjunction.shape, dtype=np.uint8)
    categories[conjunction & (val_beta < 0) & (aro_beta < 0)] = 1
    categories[conjunction & (val_beta < 0) & (aro_beta > 0)] = 2
    categories[conjunction & (val_beta > 0) & (aro_beta < 0)] = 3
    categories[conjunction & (val_beta > 0) & (aro_beta > 0)] = 4

    min_abs_t = np.minimum(np.abs(val_t), np.abs(aro_t))
    same_sign = np.sign(val_beta) == np.sign(aro_beta)
    min_abs_t_conjunction = np.where(conjunction, min_abs_t, 0.0).astype(np.float32)
    min_signed_t = np.zeros_like(val_t, dtype=np.float32)
    min_signed_t[conjunction & same_sign] = (
        min_abs_t[conjunction & same_sign] * np.sign(val_beta[conjunction & same_sign])
    ).astype(np.float32)
    # The primary min-t map is unsigned so opposite-sign conjunction vertices still
    # appear with their weaker t statistic. The signed version is saved for reuse.

    out_data = run_dir / "conjunction_valence_arousal"
    out_data.mkdir(exist_ok=True)
    np.save(out_data / "valence_arousal_both_fdr05_mask_vertices.npy", conjunction)
    np.save(out_data / "valence_arousal_both_fdr05_sign_categories_vertices.npy", categories)
    np.save(out_data / "valence_arousal_both_fdr05_min_abs_t_vertices.npy", min_abs_t_conjunction)
    np.save(out_data / "valence_arousal_both_fdr05_min_signed_t_same_sign_vertices.npy", min_signed_t)

    counts = {
        "total": int(conjunction.sum()),
        "1": int(np.sum(categories == 1)),
        "2": int(np.sum(categories == 2)),
        "3": int(np.sum(categories == 3)),
        "4": int(np.sum(categories == 4)),
        "both_negative": int(np.sum(categories == 1)),
        "valence_negative_arousal_positive": int(np.sum(categories == 2)),
        "valence_positive_arousal_negative": int(np.sum(categories == 3)),
        "both_positive": int(np.sum(categories == 4)),
    }
    (out_data / "conjunction_summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")

    active = min_abs_t_conjunction[min_abs_t_conjunction != 0]
    vmax = float(np.percentile(active, 99)) if active.size else 1.0
    vmax = max(vmax, 1.0)
    plot_conjunction_category_panel(
        out_dir / "fsaverage5_valence_arousal_both_fdr05_sign_categories.png",
        surfaces,
        categories,
        counts,
    )
    plot_conjunction_min_panel(
        out_dir / "fsaverage5_valence_arousal_both_fdr05_min_abs_t.png",
        surfaces,
        min_abs_t_conjunction,
        vmax,
    )


def run(run_dir: Path, fsaverage5_dir: Path, surface: str) -> Path:
    out_dir = run_dir / "figures_fsaverage5"
    out_dir.mkdir(parents=True, exist_ok=True)

    surfaces = load_surfaces(fsaverage5_dir, surface)
    beta = np.load(run_dir / "beta_intercept_valence_arousal_vertices.npy")
    t_values = np.load(run_dir / "t_cluster_intercept_valence_arousal_vertices.npy")
    fdr_masks = np.load(run_dir / "fdr05_mask_valence_arousal_vertices.npy")
    with open(run_dir / "regression_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    signed_all = np.vstack([np.where(fdr_masks[idx], t_values[idx + 1], 0.0) for idx in range(2)])
    finite_active = np.abs(signed_all[np.isfinite(signed_all) & (signed_all != 0)])
    vmax = float(np.percentile(finite_active, 99)) if finite_active.size else 1.0
    vmax = max(vmax, 1.0)

    plot_signed_fdr_panel(out_dir / "fsaverage5_signed_t_fdr05_valence_arousal.png", surfaces, t_values, fdr_masks, vmax)
    for idx, predictor in enumerate(PREDICTORS):
        signed = np.where(fdr_masks[idx], t_values[idx + 1], 0.0)
        plot_predictor_detail(out_dir / f"fsaverage5_{predictor}_signed_t_fdr05.png", surfaces, signed, predictor, vmax)
    plot_sign_count_bar(out_dir / "fsaverage5_significant_vertex_counts.png", summary)
    save_conjunction_outputs(run_dir, out_dir, surfaces, t_values, beta, fdr_masks)

    notes = f"""# fsaverage5 Surface Figures

These figures show the intrinsic TRIBEv2 regression result on the fsaverage5
surface, not an MNI volume.

Surface used: `{surface}`.

Values shown on the cortical surface are cluster-robust t values after FDR
q < 0.05 across fsaverage5 vertices, separately for valence and arousal.

Positive coefficients are red; negative coefficients are blue; non-significant
vertices are gray.

Generated files:

- `fsaverage5_signed_t_fdr05_valence_arousal.png`
- `fsaverage5_valence_signed_t_fdr05.png`
- `fsaverage5_arousal_signed_t_fdr05.png`
- `fsaverage5_significant_vertex_counts.png`
- `fsaverage5_valence_arousal_both_fdr05_sign_categories.png`
- `fsaverage5_valence_arousal_both_fdr05_min_abs_t.png`
"""
    (out_dir / "README_FSAVERAGE5_FIGURES.md").write_text(notes, encoding="utf-8")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--fsaverage5-dir", type=Path, default=DEFAULT_FSAVERAGE5_DIR)
    parser.add_argument("--surface", choices=["inflated", "pial", "white"], default="inflated")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = run(args.run_dir, args.fsaverage5_dir, args.surface)
    print(f"Wrote fsaverage5 surface figures to: {out_dir}")


if __name__ == "__main__":
    main()
