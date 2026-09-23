from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

from plot_fsaverage5_surface_results import N_LEFT_VERTICES, load_surfaces, plot_hemi_view


DEFAULT_FSAVERAGE5_DIR = Path(
    r"N:\Experimental_Data\yujunchen\projects\data\NSD_atlas\nsddata\freesurfer\fsaverage5"
)


def load_regression_arrays(run_dir: Path) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    with open(run_dir / "regression_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    files = summary["output_files"]
    beta = np.load(run_dir / files["beta"])
    t_values = np.load(run_dir / files["t"])
    fdr_masks = np.load(run_dir / files["fdr_mask"])
    return summary, beta, t_values, fdr_masks


def active_vmax(t_values: np.ndarray, fdr_masks: np.ndarray) -> float:
    signed = np.vstack([np.where(fdr_masks[idx], t_values[idx + 1], 0.0) for idx in range(fdr_masks.shape[0])])
    finite_active = np.abs(signed[np.isfinite(signed) & (signed != 0)])
    vmax = float(np.percentile(finite_active, 99)) if finite_active.size else 1.0
    return max(vmax, 1.0)


def plot_all_predictors_panel(
    out_path: Path,
    surfaces: dict[str, tuple[np.ndarray, np.ndarray]],
    predictors: list[str],
    t_values: np.ndarray,
    fdr_masks: np.ndarray,
    vmax: float,
) -> None:
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig_height = max(4.8, 3.4 * len(predictors))
    fig = plt.figure(figsize=(15, fig_height), constrained_layout=False)
    views = [("lh", "lateral"), ("lh", "medial"), ("rh", "medial"), ("rh", "lateral")]

    for row, predictor in enumerate(predictors):
        signed = np.where(fdr_masks[row], t_values[row + 1], 0.0)
        for col, (hemi, view) in enumerate(views):
            ax = fig.add_subplot(len(predictors), 4, row * 4 + col + 1, projection="3d")
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
                title=f"{predictor} {hemi.upper()} {view}",
            )

    fig.suptitle("TRIBEv2 image encoding regression: FDR q < 0.05 signed t maps on fsaverage5", fontsize=15)
    mappable = cm.ScalarMappable(norm=norm, cmap="coolwarm")
    cbar_ax = fig.add_axes([0.28, 0.04, 0.44, 0.018])
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("OLS t value, FDR q < 0.05")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.08, wspace=0.00, hspace=0.08)
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
        plot_hemi_view(ax, coords, faces, hemi_values, norm, hemi, view, f"{hemi.upper()} {view}")
    fig.suptitle(f"{predictor} FDR q < 0.05 signed t map on fsaverage5 surface", fontsize=15)
    mappable = cm.ScalarMappable(norm=norm, cmap="coolwarm")
    cbar_ax = fig.add_axes([0.28, 0.08, 0.44, 0.025])
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("OLS t value")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.13, wspace=0.00)
    fig.savefig(out_path, dpi=260, facecolor="white")
    plt.close(fig)


def plot_sign_count_bar(out_path: Path, predictors: list[str], summary: dict[str, object]) -> None:
    counts = summary["significant_counts"]
    labels = []
    values = []
    bar_colors = []
    for predictor in predictors:
        labels.extend([f"{predictor}\npositive", f"{predictor}\nnegative"])
        values.extend([counts[predictor]["positive"], counts[predictor]["negative"]])
        bar_colors.extend(["#c43b3b", "#356aa0"])

    fig_width = max(7.5, 1.5 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    ax.bar(labels, values, color=bar_colors)
    ax.set_ylabel("Significant fsaverage5 vertices")
    ax.set_title("FDR q < 0.05 significant surface vertices")
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:,}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, facecolor="white")
    plt.close(fig)


def run(run_dir: Path, fsaverage5_dir: Path, surface: str) -> Path:
    out_dir = run_dir / "figures_fsaverage5"
    out_dir.mkdir(parents=True, exist_ok=True)

    surfaces = load_surfaces(fsaverage5_dir, surface)
    summary, _beta, t_values, fdr_masks = load_regression_arrays(run_dir)
    predictors = list(summary["predictors"])
    vmax = active_vmax(t_values, fdr_masks)
    slug = str(summary["predictor_slug"])

    plot_all_predictors_panel(
        out_dir / f"fsaverage5_signed_t_fdr05_{slug}.png",
        surfaces,
        predictors,
        t_values,
        fdr_masks,
        vmax,
    )
    for idx, predictor in enumerate(predictors):
        signed = np.where(fdr_masks[idx], t_values[idx + 1], 0.0)
        plot_predictor_detail(out_dir / f"fsaverage5_{predictor}_signed_t_fdr05.png", surfaces, signed, predictor, vmax)
    plot_sign_count_bar(out_dir / "fsaverage5_significant_vertex_counts.png", predictors, summary)

    generated = [f"`fsaverage5_signed_t_fdr05_{slug}.png`", "`fsaverage5_significant_vertex_counts.png`"]
    generated.extend(f"`fsaverage5_{predictor}_signed_t_fdr05.png`" for predictor in predictors)
    generated_lines = "\n".join(f"- {item}" for item in generated)
    notes = f"""# fsaverage5 Image Surface Figures

These figures show the intrinsic TRIBEv2 image regression result on the
fsaverage5 surface, not an MNI volume.

Surface used: `{surface}`.

Values shown on the cortical surface are OLS t values after FDR q < 0.05 across
fsaverage5 vertices, separately for each image-level predictor.

Positive coefficients are red; negative coefficients are blue; non-significant
vertices are gray.

Generated files:

{generated_lines}
"""
    (out_dir / "README_FSAVERAGE5_FIGURES.md").write_text(notes, encoding="utf-8")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot image-regression TRIBEv2 maps on fsaverage5 surfaces.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fsaverage5-dir", type=Path, default=DEFAULT_FSAVERAGE5_DIR)
    parser.add_argument("--surface", choices=["inflated", "pial", "white"], default="inflated")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = run(args.run_dir, args.fsaverage5_dir, args.surface)
    print(f"Wrote fsaverage5 surface figures to: {out_dir}")


if __name__ == "__main__":
    main()
