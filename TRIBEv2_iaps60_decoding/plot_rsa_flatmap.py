from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from nilearn import datasets, surface


N_HEMI_VERTICES = 10242


def render_hemisphere(
    axis: plt.Axes,
    flat_mesh_path: str,
    sulc_path: str,
    values: np.ndarray,
    significant: np.ndarray,
    cortex_vertices: np.ndarray,
    norm: Normalize,
    cmap: str,
    label: str,
) -> None:
    mesh = surface.load_surf_mesh(flat_mesh_path)
    coords = np.asarray(mesh.coordinates)
    faces = np.asarray(mesh.faces)
    sulc = np.asarray(surface.load_surf_data(sulc_path), dtype=np.float64)
    triangulation = mtri.Triangulation(coords[:, 0], coords[:, 1], faces)
    cortex_mask = np.zeros(coords.shape[0], dtype=bool)
    cortex_mask[cortex_vertices] = True
    triangulation.set_mask(~np.all(cortex_mask[faces], axis=1))

    # A quiet two-tone curvature background preserves sulcal context.
    sulc_background = (sulc > 0).astype(np.float64)
    axis.tripcolor(
        triangulation,
        sulc_background,
        shading="flat",
        cmap=ListedColormap(["#929292", "#D0D0D0"]),
        vmin=0.0,
        vmax=1.0,
        alpha=1.0,
        rasterized=True,
    )

    # Mask non-significant vertices, then interpolate colors linearly over each
    # retained surface triangle. This changes rendering only, not statistics.
    heat_triangulation = mtri.Triangulation(coords[:, 0], coords[:, 1], faces)
    heat_triangulation.set_mask(
        (~np.all(cortex_mask[faces], axis=1)) | (~np.any(significant[faces], axis=1))
    )
    display_values = np.where(significant, values, norm.vmin)
    heat_cmap = plt.get_cmap(cmap).copy()
    axis.tripcolor(
        heat_triangulation,
        display_values,
        shading="gouraud",
        cmap=heat_cmap,
        norm=norm,
        rasterized=True,
    )
    axis.set_aspect("equal")
    axis.set_xlim(coords[:, 0].min(), coords[:, 0].max())
    axis.set_ylim(coords[:, 1].min(), coords[:, 1].max())
    axis.set_axis_off()
    axis.text(
        0.02,
        0.98,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        weight="bold",
        color="#202020",
    )


def run(stats_path: Path, output_png: Path, output_svg: Path, p_threshold: float, title: str) -> None:
    saved = np.load(stats_path)
    fisher_z = np.asarray(saved["fisher_z"], dtype=np.float64)
    rho = np.asarray(saved["rho"], dtype=np.float64)
    p_fwer = np.asarray(saved["p_fwer_positive"], dtype=np.float64)
    if fisher_z.shape != (2 * N_HEMI_VERTICES,):
        raise ValueError(f"Expected 20484 surface values, found {fisher_z.shape}")
    significant = np.isfinite(p_fwer) & (p_fwer < p_threshold) & (rho > 0)
    if not significant.any():
        raise ValueError(f"No surface vertices survive positive FWER p < {p_threshold}")

    visible_values = fisher_z[significant]
    color_min = float(visible_values.min())
    color_max = float(visible_values.max())
    norm = Normalize(vmin=color_min, vmax=color_max)
    cmap = "inferno"
    fsaverage5 = datasets.fetch_surf_fsaverage("fsaverage5")
    fsaverage5_dir = Path(
        r"N:\Experimental_Data\yujunchen\projects\data\NSD_atlas\nsddata\freesurfer\fsaverage5"
    )
    cortex_left = np.loadtxt(
        fsaverage5_dir / "label" / "lh.cortex.label", skiprows=2, usecols=0, dtype=np.int64
    )
    cortex_right = np.loadtxt(
        fsaverage5_dir / "label" / "rh.cortex.label", skiprows=2, usecols=0, dtype=np.int64
    )

    fig, axes = plt.subplots(1, 2, figsize=(17, 8.7), facecolor="white")
    render_hemisphere(
        axes[0],
        fsaverage5.flat_left,
        fsaverage5.sulc_left,
        fisher_z[:N_HEMI_VERTICES],
        significant[:N_HEMI_VERTICES],
        cortex_left,
        norm,
        cmap,
        "LH",
    )
    render_hemisphere(
        axes[1],
        fsaverage5.flat_right,
        fsaverage5.sulc_right,
        fisher_z[N_HEMI_VERTICES:],
        significant[N_HEMI_VERTICES:],
        cortex_right,
        norm,
        cmap,
        "RH",
    )

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(
        scalar_mappable,
        ax=axes,
        orientation="horizontal",
        fraction=0.048,
        pad=0.035,
        aspect=45,
    )
    colorbar.set_label("RSA effect (Fisher z)", fontsize=13)
    colorbar.ax.tick_params(labelsize=11)
    fig.suptitle(
        f"{title} — positive max-stat FWER p < {p_threshold:g}",
        fontsize=18,
        weight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.015,
        f"{int(significant.sum()):,} significant fsaverage5 centers · smooth triangular color interpolation",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#444444",
    )
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.12, top=0.91, wspace=0.025)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=260, facecolor="white", bbox_inches="tight")
    fig.savefig(output_svg, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    output_dir = Path(
        r"N:\Experimental_Data\yujunchen\projects\Inhouse_Encoding_Model"
        r"\outputs\iaps60_meta_static_3s\rsa_searchlight_va"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=Path, default=output_dir / "rsa_surface_statistics.npz")
    parser.add_argument("--p-threshold", type=float, default=0.01)
    parser.add_argument("--output-png", type=Path, default=output_dir / "rsa_fwer01_positive_fsaverage5_flatmap.png")
    parser.add_argument("--output-svg", type=Path, default=output_dir / "rsa_fwer01_positive_fsaverage5_flatmap.svg")
    parser.add_argument("--title", default="TRIBE v2 valence-arousal searchlight RSA")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.stats, args.output_png, args.output_svg, args.p_threshold, args.title)
