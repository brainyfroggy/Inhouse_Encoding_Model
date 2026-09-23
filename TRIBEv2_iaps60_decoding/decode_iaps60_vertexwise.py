from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from matplotlib.colors import ListedColormap
from nibabel.freesurfer.io import write_morph_data
from nilearn import datasets, surface


N_HEMI_VERTICES = 10242
N_VERTICES = 2 * N_HEMI_VERTICES
CONTRASTS = (
    ("pleasant_vs_neutral", "pleasant"),
    ("unpleasant_vs_neutral", "unpleasant"),
)
CLASS_CODES = {"neutral": 0, "pleasant": 1, "unpleasant": 2}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cortex_mask(fsaverage5_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    mask = np.zeros(N_VERTICES, dtype=bool)
    hemispheres: dict[str, np.ndarray] = {}
    for hemi, offset in (("lh", 0), ("rh", N_HEMI_VERTICES)):
        vertices = np.loadtxt(
            fsaverage5_dir / "label" / f"{hemi}.cortex.label",
            skiprows=2,
            usecols=0,
            dtype=np.int64,
        )
        hemispheres[hemi] = vertices
        mask[vertices + offset] = True
    return mask, hemispheres


def loo_nearest_centroid_accuracy(
    responses: np.ndarray,
    labels: np.ndarray,
    positive_code: int,
) -> np.ndarray:
    selected = (labels == 0) | (labels == positive_code)
    x = responses[selected]
    y = labels[selected]
    if x.shape[0] != 40 or np.sum(y == 0) != 20 or np.sum(y == positive_code) != 20:
        raise ValueError("Each contrast must contain 20 neutral and 20 emotional images.")
    sum_neutral = x[y == 0].sum(axis=0, dtype=np.float64)
    sum_positive = x[y == positive_code].sum(axis=0, dtype=np.float64)
    correct = np.zeros(x.shape[1], dtype=np.uint8)
    for sample, sample_label in zip(x, y, strict=True):
        if sample_label == 0:
            own_mean = (sum_neutral - sample) / 19.0
            other_mean = sum_positive / 20.0
        else:
            own_mean = (sum_positive - sample) / 19.0
            other_mean = sum_neutral / 20.0
        correct += np.abs(sample - own_mean) <= np.abs(sample - other_mean)
    return correct.astype(np.float64) / 40.0


def joint_max_permutation_null(
    responses: np.ndarray,
    labels: np.ndarray,
    n_permutations: int,
    seed: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint max-accuracy null across every vertex and both contrasts."""
    rng = np.random.default_rng(seed)
    n_images, n_vertices = responses.shape
    if n_images != 60:
        raise ValueError(f"Expected 60 images, found {n_images}")
    joint_max = np.empty(n_permutations, dtype=np.float32)
    contrast_max = np.empty((2, n_permutations), dtype=np.float32)
    output_index = 0
    while output_index < n_permutations:
        batch_n = min(batch_size, n_permutations - output_index)
        permuted = np.stack([rng.permutation(labels) for _ in range(batch_n)], axis=0)
        membership = [(permuted == code).astype(np.float32) for code in range(3)]
        class_sums = [member @ responses for member in membership]
        batch_contrast_max = np.empty((2, batch_n), dtype=np.float32)
        for contrast_index, positive_code in enumerate((1, 2)):
            correct = np.zeros((batch_n, n_vertices), dtype=np.uint8)
            for image_index in range(n_images):
                sample = responses[image_index][None, :]
                image_labels = permuted[:, image_index]
                neutral_rows = np.flatnonzero(image_labels == 0)
                if neutral_rows.size:
                    own = (class_sums[0][neutral_rows] - sample) / 19.0
                    other = class_sums[positive_code][neutral_rows] / 20.0
                    correct[neutral_rows] += np.abs(sample - own) <= np.abs(sample - other)
                positive_rows = np.flatnonzero(image_labels == positive_code)
                if positive_rows.size:
                    own = (class_sums[positive_code][positive_rows] - sample) / 19.0
                    other = class_sums[0][positive_rows] / 20.0
                    correct[positive_rows] += np.abs(sample - own) <= np.abs(sample - other)
            batch_contrast_max[contrast_index] = (correct.astype(np.float32) / 40.0).max(axis=1)
        sl = slice(output_index, output_index + batch_n)
        contrast_max[:, sl] = batch_contrast_max
        joint_max[sl] = batch_contrast_max.max(axis=0)
        output_index += batch_n
        if output_index % max(batch_size, 250) == 0 or output_index == n_permutations:
            print(f"Permutations: {output_index}/{n_permutations}", flush=True)
    return joint_max, contrast_max


def maxstat_p_values(observed: np.ndarray, joint_max_null: np.ndarray) -> np.ndarray:
    # Accuracy has exactly 40 possible held-out decisions. Compare integer
    # correct counts so float32 representations of values such as 33/40 cannot
    # turn an equality into a strict inequality in the permutation p-value.
    sorted_null = np.sort(np.rint(np.asarray(joint_max_null) * 40.0).astype(np.int16))
    observed_counts = np.rint(np.asarray(observed) * 40.0).astype(np.int16)
    first_ge = np.searchsorted(sorted_null, observed_counts, side="left")
    n_ge = sorted_null.size - first_ge
    return (n_ge + 1.0) / (sorted_null.size + 1.0)


def render_flat_hemi(
    axis,
    mesh_path: str,
    sulc_path: str,
    cortex_vertices: np.ndarray,
    values: np.ndarray,
    visible: np.ndarray,
    norm: colors.Normalize,
    cmap: str,
    label: str,
) -> None:
    mesh = surface.load_surf_mesh(mesh_path)
    coords, faces = np.asarray(mesh.coordinates), np.asarray(mesh.faces)
    sulc = np.asarray(surface.load_surf_data(sulc_path))
    cortex = np.zeros(N_HEMI_VERTICES, dtype=bool)
    cortex[cortex_vertices] = True
    base_tri = mtri.Triangulation(coords[:, 0], coords[:, 1], faces)
    base_tri.set_mask(~np.all(cortex[faces], axis=1))
    axis.tripcolor(
        base_tri,
        (sulc > 0).astype(float),
        shading="flat",
        cmap=ListedColormap(["#929292", "#D0D0D0"]),
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    heat_tri = mtri.Triangulation(coords[:, 0], coords[:, 1], faces)
    heat_tri.set_mask((~np.all(cortex[faces], axis=1)) | (~np.any(visible[faces], axis=1)))
    display_values = np.where(visible, values, norm.vmin)
    axis.tripcolor(
        heat_tri,
        display_values,
        shading="gouraud",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    axis.set_aspect("equal")
    axis.set_xlim(coords[:, 0].min(), coords[:, 0].max())
    axis.set_ylim(coords[:, 1].min(), coords[:, 1].max())
    axis.set_axis_off()
    axis.text(0.02, 0.98, label, transform=axis.transAxes, ha="left", va="top", fontsize=13, weight="bold")


def plot_flatmap(
    output_path: Path,
    accuracies: np.ndarray,
    visible: np.ndarray,
    cortex_by_hemi: dict[str, np.ndarray],
    norm: colors.Normalize,
    cmap: str,
    title: str,
    colorbar_label: str,
    footer: str,
) -> None:
    fsaverage5 = datasets.fetch_surf_fsaverage("fsaverage5")
    fig, axes = plt.subplots(2, 2, figsize=(17, 15), facecolor="white")
    titles = ("Pleasant vs Neutral", "Unpleasant vs Neutral")
    for row, contrast_title in enumerate(titles):
        render_flat_hemi(
            axes[row, 0], fsaverage5.flat_left, fsaverage5.sulc_left, cortex_by_hemi["lh"],
            accuracies[row, :N_HEMI_VERTICES], visible[row, :N_HEMI_VERTICES], norm, cmap,
            f"{contrast_title} · LH",
        )
        render_flat_hemi(
            axes[row, 1], fsaverage5.flat_right, fsaverage5.sulc_right, cortex_by_hemi["rh"],
            accuracies[row, N_HEMI_VERTICES:], visible[row, N_HEMI_VERTICES:], norm, cmap,
            f"{contrast_title} · RH",
        )
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    colorbar = fig.colorbar(mappable, ax=axes, orientation="horizontal", fraction=0.025, pad=0.025, aspect=55)
    colorbar.set_label(colorbar_label, fontsize=12)
    colorbar.ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.suptitle(title, fontsize=17, weight="bold", y=0.98)
    fig.text(0.5, 0.018, footer, ha="center", va="bottom", fontsize=11, color="#444444")
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.085, top=0.94, wspace=0.025, hspace=0.015)
    fig.savefig(output_path, dpi=250, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = args.responses.resolve()
    manifest_path = args.manifest.resolve()
    responses = np.load(responses_path).astype(np.float32)
    manifest = pd.read_csv(manifest_path)
    if responses.shape != (60, N_VERTICES):
        raise ValueError(f"Expected (60, 20484) responses, found {responses.shape}")
    if "emotion_label" not in manifest or len(manifest) != 60:
        raise ValueError("Manifest must contain 60 rows and an emotion_label column.")
    labels = manifest["emotion_label"].map(CLASS_CODES).to_numpy()
    if np.any(pd.isna(labels)) or np.bincount(labels.astype(int)).tolist() != [20, 20, 20]:
        raise ValueError("Expected 20 neutral, 20 pleasant, and 20 unpleasant images.")
    labels = labels.astype(np.int8)
    cortex_mask, cortex_by_hemi = load_cortex_mask(args.fsaverage5_dir.resolve())
    cortex_responses = np.ascontiguousarray(responses[:, cortex_mask], dtype=np.float32)

    accuracy_cortex = np.vstack(
        [loo_nearest_centroid_accuracy(cortex_responses, labels, positive_code) for positive_code in (1, 2)]
    )
    joint_null, contrast_null = joint_max_permutation_null(
        cortex_responses,
        labels,
        n_permutations=args.n_permutations,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    p_fwer_cortex = np.vstack([maxstat_p_values(values, joint_null) for values in accuracy_cortex])
    significant_cortex = (p_fwer_cortex <= args.alpha) & (accuracy_cortex > 0.5)

    accuracy = np.full((2, N_VERTICES), np.nan, dtype=np.float32)
    p_fwer = np.ones((2, N_VERTICES), dtype=np.float32)
    significant = np.zeros((2, N_VERTICES), dtype=bool)
    accuracy[:, cortex_mask] = accuracy_cortex.astype(np.float32)
    p_fwer[:, cortex_mask] = p_fwer_cortex.astype(np.float32)
    significant[:, cortex_mask] = significant_cortex
    np.save(output_dir / "vertexwise_loo_accuracy_vertices.npy", accuracy)
    np.save(output_dir / "vertexwise_joint_maxstat_p_fwer_vertices.npy", p_fwer)
    np.save(output_dir / "vertexwise_joint_maxstat_fwer05_mask_vertices.npy", significant)
    np.savez_compressed(
        output_dir / "vertexwise_permutation_max_null.npz",
        joint_max_accuracy=joint_null,
        pleasant_vs_neutral_max_accuracy=contrast_null[0],
        unpleasant_vs_neutral_max_accuracy=contrast_null[1],
    )

    overlay_dir = output_dir / "freesurfer_overlays"
    overlay_dir.mkdir(exist_ok=True)
    for contrast_index, (contrast, _) in enumerate(CONTRASTS):
        for hemi, offset in (("lh", 0), ("rh", N_HEMI_VERTICES)):
            full = np.nan_to_num(accuracy[contrast_index, offset : offset + N_HEMI_VERTICES], nan=0.0)
            thresholded = np.where(significant[contrast_index, offset : offset + N_HEMI_VERTICES], full, 0.0)
            write_morph_data(overlay_dir / f"{hemi}.{contrast}_accuracy.curv", full)
            write_morph_data(overlay_dir / f"{hemi}.{contrast}_accuracy_joint_fwer05.curv", thresholded)

    rows = []
    for contrast_index, (contrast, _) in enumerate(CONTRASTS):
        vertices = np.flatnonzero(significant[contrast_index])
        for vertex in vertices:
            rows.append(
                {
                    "contrast": contrast,
                    "surface_vertex": int(vertex),
                    "hemisphere": "lh" if vertex < N_HEMI_VERTICES else "rh",
                    "hemisphere_vertex": int(vertex % N_HEMI_VERTICES),
                    "loo_accuracy": float(accuracy[contrast_index, vertex]),
                    "joint_maxstat_p_fwer": float(p_fwer[contrast_index, vertex]),
                }
            )
    pd.DataFrame(
        rows,
        columns=[
            "contrast", "surface_vertex", "hemisphere", "hemisphere_vertex",
            "loo_accuracy", "joint_maxstat_p_fwer",
        ],
    ).to_csv(output_dir / "significant_vertexwise_decoding_centers_joint_fwer05.csv", index=False)

    finite_accuracy = accuracy[:, cortex_mask]
    delta = float(np.max(np.abs(finite_accuracy - 0.5)))
    unthresholded_norm = colors.TwoSlopeNorm(vmin=max(0.0, 0.5 - delta), vcenter=0.5, vmax=min(1.0, 0.5 + delta))
    plot_flatmap(
        output_dir / "iaps60_vertexwise_decoding_accuracy_unthresholded_fsaverage5_flatmap.png",
        accuracy,
        np.broadcast_to(cortex_mask, accuracy.shape),
        cortex_by_hemi,
        unthresholded_norm,
        "coolwarm",
        "IAPS-60 TRIBE v2 whole-cortex vertex-wise decoding · unthresholded",
        "Leave-one-out nearest-centroid accuracy",
        "Chance = 50% · one response value per vertex · smooth interpolation is display-only",
    )
    above_chance = np.broadcast_to(cortex_mask, accuracy.shape) & (accuracy >= 0.5)
    from50_norm = colors.Normalize(vmin=0.5, vmax=float(np.nanmax(accuracy)))
    plot_flatmap(
        output_dir / "iaps60_vertexwise_decoding_accuracy_from50_fsaverage5_flatmap.png",
        accuracy,
        above_chance,
        cortex_by_hemi,
        from50_norm,
        "inferno",
        "IAPS-60 TRIBE v2 whole-cortex vertex-wise decoding · descriptive accuracy from 50%",
        "Leave-one-out nearest-centroid accuracy",
        "No significance thresholding · vertices below 50% are left on the cortical background · smooth interpolation is display-only",
    )
    if significant.any():
        visible_values = accuracy[significant]
        thresholded_norm = colors.Normalize(vmin=float(visible_values.min()), vmax=float(visible_values.max()))
    else:
        thresholded_norm = colors.Normalize(vmin=0.5, vmax=1.0)
    plot_flatmap(
        output_dir / "iaps60_vertexwise_decoding_accuracy_joint_fwer05_fsaverage5_flatmap.png",
        accuracy,
        significant,
        cortex_by_hemi,
        thresholded_norm,
        "inferno",
        f"IAPS-60 TRIBE v2 whole-cortex vertex-wise decoding · joint max-stat FWER p ≤ {args.alpha:g}",
        "Leave-one-out nearest-centroid accuracy",
        f"{int(significant.sum()):,} significant vertices across both maps · {args.n_permutations:,} global three-class label permutations",
    )

    significant_counts = {
        CONTRASTS[index][0]: {
            "total": int(significant[index].sum()),
            "lh": int(significant[index, :N_HEMI_VERTICES].sum()),
            "rh": int(significant[index, N_HEMI_VERTICES:].sum()),
            "peak_accuracy": float(np.nanmax(accuracy[index])),
            "peak_vertex": int(np.nanargmax(accuracy[index])),
        }
        for index in range(2)
    }
    summary = {
        "status": "complete",
        "analysis": "whole-cortex univariate vertex-wise emotion decoding of deterministic TRIBE v2 responses",
        "surface_space": "fsaverage5; LH 10242 followed by RH 10242",
        "n_images": 60,
        "class_counts": {label: int(np.sum(labels == code)) for label, code in CLASS_CODES.items()},
        "contrasts": [name for name, _ in CONTRASTS],
        "classifier": "univariate nearest-centroid classifier independently at each vertex",
        "cross_validation": "leave-one-image-out; training class means recomputed for every held-out image",
        "accuracy_chance": 0.5,
        "inference": {
            "n_permutations": args.n_permutations,
            "seed": args.seed,
            "permutation_unit": "global permutation of the 60 three-class image labels",
            "tail": "positive decoding accuracy",
            "multiple_comparison_control": "maximum accuracy jointly across 18715 cortical vertices and both contrasts",
            "alpha_fwer": args.alpha,
            "joint_null_quantile_95": float(np.quantile(joint_null, 0.95, method="higher")),
            "joint_null_quantile_99": float(np.quantile(joint_null, 0.99, method="higher")),
        },
        "n_cortical_vertices": int(cortex_mask.sum()),
        "significant_counts": significant_counts,
        "response": str(responses_path),
        "response_sha256": sha256_file(responses_path),
        "manifest": str(manifest_path),
        "note": "This is vertex-wise cortical decoding, not volumetric whole-brain decoding; TRIBE v2 outputs fsaverage5 cortex only.",
    }
    (output_dir / "vertexwise_decoding_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme = f"""# IAPS-60 TRIBE v2 whole-cortex vertex-wise decoding

Each fsaverage5 cortical vertex was decoded independently with a univariate
nearest-centroid classifier. Pleasant-vs-neutral and unpleasant-vs-neutral each
use 40 unique images and leave-one-image-out cross-validation. Inference uses
{args.n_permutations:,} global three-class image-label permutations. The maximum
accuracy across all 18,715 cortical vertices and both contrast maps controls
joint family-wise error at p<={args.alpha:g}.

This is vertex-wise surface decoding, not volumetric whole-brain decoding,
because TRIBE v2 provides cortical fsaverage5 responses only. Surface color
interpolation affects display only.

`iaps60_vertexwise_decoding_accuracy_from50_fsaverage5_flatmap.png` is the
descriptive map requested for interpretation: it applies no significance test,
starts the color scale at 50% chance, and leaves below-chance vertices on the
cortical background.

Significant vertices:

- Pleasant vs neutral: {significant_counts['pleasant_vs_neutral']['total']}
- Unpleasant vs neutral: {significant_counts['unpleasant_vs_neutral']['total']}
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return output_dir


def parse_args() -> argparse.Namespace:
    project = Path(r"N:\Experimental_Data\yujunchen\projects\Inhouse_Encoding_Model")
    root = project / "outputs" / "iaps60_meta_static_3s"
    parser = argparse.ArgumentParser(description="Whole-cortex vertex-wise IAPS emotion decoding.")
    parser.add_argument("--responses", type=Path, default=root / "encoding" / "iaps60_tribev2_first_tr.npy")
    parser.add_argument("--manifest", type=Path, default=root / "iaps60_unique_manifest.csv")
    parser.add_argument(
        "--fsaverage5-dir",
        type=Path,
        default=Path(r"N:\Experimental_Data\yujunchen\projects\data\NSD_atlas\nsddata\freesurfer\fsaverage5"),
    )
    parser.add_argument("--output-dir", type=Path, default=root / "vertexwise_decoding")
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    if args.n_permutations < 99:
        parser.error("--n-permutations must be at least 99")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
