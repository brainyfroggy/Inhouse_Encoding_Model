from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from nibabel.freesurfer import read_geometry, read_morph_data, write_morph_data


N_HEMI_VERTICES = 10242


def surface_face_colors(
    faces: np.ndarray,
    sulc: np.ndarray,
    values: np.ndarray,
    significant: np.ndarray,
    norm: colors.Normalize,
) -> np.ndarray:
    face_sulc = sulc[faces].mean(axis=1)
    background = np.empty((faces.shape[0], 4), dtype=np.float64)
    background[face_sulc <= 0] = (0.78, 0.78, 0.78, 1.0)
    background[face_sulc > 0] = (0.56, 0.56, 0.56, 1.0)

    vertex_values = np.where(significant, values, np.nan)
    valid_count = np.sum(np.isfinite(vertex_values[faces]), axis=1)
    face_value = np.divide(
        np.nansum(vertex_values[faces], axis=1),
        valid_count,
        out=np.full(faces.shape[0], np.nan),
        where=valid_count > 0,
    )
    active = valid_count > 0
    background[active] = plt.get_cmap("inferno")(norm(face_value[active]))
    return background


def plot_view(
    axis: plt.Axes,
    coords: np.ndarray,
    faces: np.ndarray,
    facecolors: np.ndarray,
    hemi: str,
    view: str,
) -> None:
    collection = Poly3DCollection(
        coords[faces],
        facecolors=facecolors,
        edgecolors="none",
        linewidths=0.0,
        antialiased=True,
    )
    axis.add_collection3d(collection)
    xyz_min = coords.min(axis=0)
    xyz_max = coords.max(axis=0)
    center = (xyz_min + xyz_max) / 2.0
    extent = float(np.max(xyz_max - xyz_min)) / 2.0
    axis.set_xlim(center[0] - extent, center[0] + extent)
    axis.set_ylim(center[1] - extent, center[1] + extent)
    axis.set_zlim(center[2] - extent, center[2] + extent)

    angles = {
        ("lh", "lateral"): (0, 180),
        ("lh", "medial"): (0, 0),
        ("rh", "medial"): (0, 180),
        ("rh", "lateral"): (0, 0),
    }
    elevation, azimuth = angles[(hemi, view)]
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_title(f"{hemi.upper()} {view}", fontsize=13, weight="bold", pad=0)
    axis.set_axis_off()
    try:
        axis.set_box_aspect((1, 1, 1), zoom=1.55)
    except TypeError:
        axis.set_box_aspect((1, 1, 1))


def run(
    stats_path: Path,
    fsaverage5_dir: Path,
    output_png: Path,
    output_svg: Path,
    overlay_dir: Path,
    p_threshold: float,
    title: str,
) -> None:
    saved = np.load(stats_path)
    fisher_z = np.asarray(saved["fisher_z"], dtype=np.float64)
    rho = np.asarray(saved["rho"], dtype=np.float64)
    p_fwer = np.asarray(saved["p_fwer_positive"], dtype=np.float64)
    significant = np.isfinite(p_fwer) & (p_fwer < p_threshold) & (rho > 0)
    if fisher_z.shape != (2 * N_HEMI_VERTICES,):
        raise ValueError(f"Expected 20484 surface values, found {fisher_z.shape}")
    if not significant.any():
        raise ValueError(f"No vertices survive positive FWER p < {p_threshold}")

    visible = fisher_z[significant]
    norm = colors.Normalize(vmin=float(visible.min()), vmax=float(visible.max()))
    overlays = np.where(significant, fisher_z, 0.0).astype(np.float32)
    threshold_tag = f"fwer{int(round(p_threshold * 100)):02d}"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    write_morph_data(overlay_dir / f"lh.rsa_fisher_z_{threshold_tag}.curv", overlays[:N_HEMI_VERTICES])
    write_morph_data(overlay_dir / f"rh.rsa_fisher_z_{threshold_tag}.curv", overlays[N_HEMI_VERTICES:])

    surfaces: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for hemi, offset in (("lh", 0), ("rh", N_HEMI_VERTICES)):
        coords, faces = read_geometry(fsaverage5_dir / "surf" / f"{hemi}.inflated")
        sulc = read_morph_data(fsaverage5_dir / "surf" / f"{hemi}.sulc")
        hemi_values = fisher_z[offset : offset + N_HEMI_VERTICES]
        hemi_significant = significant[offset : offset + N_HEMI_VERTICES]
        facecolors = surface_face_colors(faces, sulc, hemi_values, hemi_significant, norm)
        surfaces[hemi] = (coords, faces, sulc, facecolors)

    views = (("lh", "lateral"), ("lh", "medial"), ("rh", "medial"), ("rh", "lateral"))
    fig = plt.figure(figsize=(19, 7.4), facecolor="white")
    for column, (hemi, view) in enumerate(views):
        axis = fig.add_subplot(1, 4, column + 1, projection="3d")
        coords, faces, _, facecolors = surfaces[hemi]
        plot_view(axis, coords, faces, facecolors, hemi, view)

    mappable = cm.ScalarMappable(norm=norm, cmap="inferno")
    mappable.set_array([])
    colorbar_axis = fig.add_axes([0.26, 0.075, 0.48, 0.035])
    colorbar = fig.colorbar(mappable, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("RSA effect (Fisher z)", fontsize=13)
    colorbar.ax.tick_params(labelsize=11)
    fig.suptitle(
        f"{title} on FreeSurfer fsaverage5 — positive max-stat FWER p < {p_threshold:g}",
        fontsize=17,
        weight="bold",
        y=0.965,
    )
    fig.subplots_adjust(left=0.005, right=0.995, top=0.91, bottom=0.13, wspace=-0.08)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=260, facecolor="white", bbox_inches="tight")
    fig.savefig(output_svg, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    project = Path(r"N:\Experimental_Data\yujunchen\projects\Inhouse_Encoding_Model")
    output_dir = project / "outputs" / "iaps60_meta_static_3s" / "rsa_searchlight_va"
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=Path, default=output_dir / "rsa_surface_statistics.npz")
    parser.add_argument(
        "--fsaverage5-dir",
        type=Path,
        default=Path(r"N:\Experimental_Data\yujunchen\projects\data\NSD_atlas\nsddata\freesurfer\fsaverage5"),
    )
    parser.add_argument("--output-png", type=Path, default=output_dir / "rsa_fwer01_positive_fsaverage5_inflated.png")
    parser.add_argument("--output-svg", type=Path, default=output_dir / "rsa_fwer01_positive_fsaverage5_inflated.svg")
    parser.add_argument("--overlay-dir", type=Path, default=output_dir / "freesurfer_overlays")
    parser.add_argument("--p-threshold", type=float, default=0.01)
    parser.add_argument("--title", default="TRIBE v2 valence-arousal searchlight RSA")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.stats,
        args.fsaverage5_dir,
        args.output_png,
        args.output_svg,
        args.overlay_dir,
        args.p_threshold,
        args.title,
    )
