from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from iaps60_common import (
    DEFAULT_KEBO_PLEASANT,
    DEFAULT_KEBO_UNPLEASANT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SURFACE_ATLAS,
    DEFAULT_SURFACE_ATLAS_SUMMARY,
    EXPECTED_IMAGES,
    EXPECTED_VERTICES,
    ROI_SPECS,
    load_surface_rois,
    sha256_file,
    utc_now,
    write_json,
)


CONTRASTS: tuple[tuple[str, str], ...] = (
    ("pleasant_vs_neutral", "pleasant"),
    ("unpleasant_vs_neutral", "unpleasant"),
)
PAPER_THRESHOLD = 0.54
CHANCE = 0.50

# Figure 3 center lines are not tabulated. These values were visually
# digitized to the nearest percentage point and are always labeled approximate.
PUBLISHED_FIGURE3_APPROX = {
    "pleasant_vs_neutral": {
        "V1v": 0.60, "V1d": 0.64, "V2v": 0.59, "V2d": 0.62,
        "V3v": 0.60, "V3d": 0.62, "hV4": 0.63, "VO1": 0.62,
        "VO2": 0.61, "PHC1": 0.59, "PHC2": 0.56, "hMT": 0.66,
        "LO1": 0.66, "LO2": 0.66, "V3a": 0.64, "V3b": 0.62, "IPS": 0.61,
    },
    "unpleasant_vs_neutral": {
        "V1v": 0.63, "V1d": 0.63, "V2v": 0.60, "V2d": 0.62,
        "V3v": 0.61, "V3d": 0.62, "hV4": 0.63, "VO1": 0.60,
        "VO2": 0.61, "PHC1": 0.61, "PHC2": 0.58, "hMT": 0.63,
        "LO1": 0.65, "LO2": 0.64, "V3a": 0.65, "V3b": 0.58, "IPS": 0.61,
    },
}


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1 or not np.isfinite(p_values).all():
        raise ValueError("BH-FDR requires a finite one-dimensional p-value array.")
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def primary_estimator() -> SVC:
    return SVC(C=1.0, kernel="linear")


def local_reproduction_estimator(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("svm", LinearSVC(C=1.0, max_iter=50_000, dual="auto", random_state=seed)),
        ]
    )


def matlab_pattern_zscore(x: np.ndarray) -> np.ndarray:
    """Match MATLAB zscore(X, 0, 2): normalize each image across ROI features."""
    mean = x.mean(axis=1, keepdims=True)
    scale = x.std(axis=1, ddof=1, keepdims=True)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    return (x - mean) / scale


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join([header, divider, *rows])


def load_inputs(output_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    manifest_path = output_dir / "iaps60_unique_manifest.csv"
    responses_path = output_dir / "encoding" / "iaps60_tribev2_first_tr.npy"
    prediction_manifest_path = output_dir / "encoding" / "prediction_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run encoding first; missing {manifest_path}")
    if not responses_path.is_file():
        raise FileNotFoundError(f"Run encoding first; missing {responses_path}")
    manifest = pd.read_csv(manifest_path, dtype={"image_id": "string"})
    responses = np.load(responses_path)
    if len(manifest) != EXPECTED_IMAGES:
        raise ValueError(f"Decoding requires 60 unique images; manifest has {len(manifest)}")
    if responses.shape != (EXPECTED_IMAGES, EXPECTED_VERTICES):
        raise ValueError(f"Expected response shape (60, 20484), found {responses.shape}")
    if not np.isfinite(responses).all():
        raise ValueError("Response matrix contains NaN or infinite values.")
    if manifest["image_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate image IDs.")
    counts = manifest["emotion_label"].value_counts().to_dict()
    if counts != {"neutral": 20, "pleasant": 20, "unpleasant": 20}:
        raise ValueError(f"Unexpected class counts: {counts}")
    if prediction_manifest_path.is_file():
        predictions = pd.read_csv(prediction_manifest_path, dtype={"image_id": "string"})
        if predictions["image_id"].tolist() != manifest["image_id"].tolist():
            raise ValueError("Prediction and unique-image manifests are not in the same order.")
    return manifest, responses.astype(np.float64, copy=False)


def write_atlas_outputs(
    atlas_path: Path,
    atlas_summary_path: Path,
    decoding_dir: Path,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, Path]:
    _, rois, mapping = load_surface_rois(atlas_path)
    mapping_path = decoding_dir / "kastner17_fsaverage5_mapping.csv"
    mapping.to_csv(mapping_path, index=False)
    indices_path = decoding_dir / "kastner17_vertex_indices.npz"
    np.savez_compressed(indices_path, **rois)

    source_summary: dict[str, object] = {}
    if atlas_summary_path.is_file():
        source_summary = json.loads(atlas_summary_path.read_text(encoding="utf-8"))
    source_hashes: dict[str, str] = {}
    raw_source = source_summary.get("kastner_source")
    if isinstance(raw_source, str) and "[lh\\rh]" in raw_source:
        for hemisphere in ("lh", "rh"):
            source_path = Path(raw_source.replace("[lh\\rh]", hemisphere))
            if source_path.is_file():
                source_hashes[str(source_path.resolve())] = sha256_file(source_path)

    provenance = {
        "created_at": utc_now(),
        "atlas_kind": "Kastner2015 maximum-probability surface labels resampled to fsaverage5",
        "atlas_path": str(atlas_path.resolve()),
        "atlas_sha256": sha256_file(atlas_path.resolve()),
        "atlas_shape": [EXPECTED_VERTICES],
        "surface_vertex_order": "left hemisphere (10242), then right hemisphere (10242)",
        "resampling_summary_path": str(atlas_summary_path.resolve()) if atlas_summary_path.is_file() else None,
        "resampling_summary": source_summary,
        "source_atlas_hashes": source_hashes,
        "roi_rules": {
            name: list(label_ids) for name, label_ids in ROI_SPECS
        },
        "specific_rules": [
            "Both hemispheres are combined.",
            "IPS is the union of Kastner labels 18 through 23 (IPS0-IPS5).",
            "hMT is label 13; label 12 (MST) is excluded.",
            "SPL1 (24) and FEF (25) are excluded to match the 17 paper panels.",
            "The supplied volumetric kastner_dict.npy is not applied to surface predictions.",
        ],
        "mapping_csv": str(mapping_path.resolve()),
        "vertex_indices_npz": str(indices_path.resolve()),
    }
    provenance_path = decoding_dir / "atlas_provenance.json"
    write_json(provenance_path, provenance)
    return rois, mapping, provenance_path


def evaluate_roi(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_repeats: int,
    n_permutations: int,
    jobs: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    repeated_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=n_repeats, random_state=seed)
    fold_scores = cross_val_score(
        primary_estimator(),
        x,
        y,
        cv=repeated_cv,
        scoring="accuracy",
        n_jobs=jobs,
    )
    repeat_scores = fold_scores.reshape(n_repeats, 10).mean(axis=1)

    fixed_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    fixed_score, permutation_scores, p_value = permutation_test_score(
        primary_estimator(),
        x,
        y,
        cv=fixed_cv,
        n_permutations=n_permutations,
        n_jobs=jobs,
        random_state=seed,
        scoring="accuracy",
    )
    values = {
        "accuracy_mean_repeated_10fold": float(repeat_scores.mean()),
        "accuracy_sd_across_repeats": float(repeat_scores.std(ddof=1)),
        "accuracy_repeat_q025": float(np.quantile(repeat_scores, 0.025)),
        "accuracy_repeat_q975": float(np.quantile(repeat_scores, 0.975)),
        "accuracy_fixed_10fold_for_permutation": float(fixed_score),
        "permutation_p": float(p_value),
        "permutation_null_mean": float(permutation_scores.mean()),
        "permutation_null_q950": float(np.quantile(permutation_scores, 0.95)),
        "permutation_null_q990": float(np.quantile(permutation_scores, 0.99)),
    }
    return values, repeat_scores, permutation_scores


def evaluate_local_pipeline_sensitivity(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_repeats: int,
    jobs: int,
) -> tuple[dict[str, float], np.ndarray]:
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=n_repeats, random_state=seed)
    fold_scores = cross_val_score(
        local_reproduction_estimator(seed),
        x,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=jobs,
    )
    repeat_scores = fold_scores.reshape(n_repeats, 10).mean(axis=1)
    values = {
        "local_pipeline_sensitivity_accuracy": float(repeat_scores.mean()),
        "local_pipeline_sensitivity_sd_across_repeats": float(repeat_scores.std(ddof=1)),
    }
    return values, repeat_scores


def load_kebo_baseline(pleasant_path: Path, unpleasant_path: Path) -> pd.DataFrame:
    frames = []
    for path, expected in (
        (pleasant_path, "pleasant_vs_neutral"),
        (unpleasant_path, "unpleasant_vs_neutral"),
    ):
        frame = pd.read_csv(path)
        required = {"comparison", "roi", "mean_accuracy", "sem_accuracy", "n_subjects"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if set(frame["comparison"]) != {expected}:
            raise ValueError(f"Unexpected comparison label in {path}")
        frame = frame.copy()
        frame["kebo_source_file"] = str(path.resolve())
        frames.append(frame)
    baseline = pd.concat(frames, ignore_index=True)
    expected_rois = {name for name, _ in ROI_SPECS}
    for contrast, group in baseline.groupby("comparison"):
        if set(group["roi"]) != expected_rois:
            raise ValueError(f"Ke Bo baseline {contrast} has an unexpected ROI set.")
    return baseline


def comparison_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for contrast, group in results.groupby("contrast", sort=False):
        synth = group["accuracy_mean_repeated_10fold"].to_numpy()
        sensitivity = group["local_pipeline_sensitivity_accuracy"].to_numpy()
        measured = group["kebo_local_mean_accuracy"].to_numpy()
        pearson = pearsonr(synth, measured)
        spearman = spearmanr(synth, measured)
        rows.append(
            {
                "contrast": contrast,
                "synthetic_mean_across_rois": float(synth.mean()),
                "local_pipeline_sensitivity_mean_across_rois": float(sensitivity.mean()),
                "kebo_local_mean_across_rois": float(measured.mean()),
                "mean_delta_vs_local_percentage_points": float((synth - measured).mean() * 100),
                "mae_vs_local_percentage_points": float(np.mean(np.abs(synth - measured)) * 100),
                "rmse_vs_local_percentage_points": float(np.sqrt(np.mean((synth - measured) ** 2)) * 100),
                "pearson_r_vs_local_across_rois": float(pearson.statistic),
                "pearson_p_vs_local_across_rois": float(pearson.pvalue),
                "spearman_rho_vs_local_across_rois": float(spearman.statistic),
                "spearman_p_vs_local_across_rois": float(spearman.pvalue),
                "n_synthetic_above_50pct": int((synth > CHANCE).sum()),
                "n_synthetic_above_descriptive_54pct": int((synth > PAPER_THRESHOLD).sum()),
                "n_synthetic_fdr_q_lt_0_05": int((group["permutation_q_bh_34_tests"] < 0.05).sum()),
                "top_synthetic_roi": str(group.loc[group["accuracy_mean_repeated_10fold"].idxmax(), "roi"]),
            }
        )
    return pd.DataFrame(rows)


def plot_roi_comparison(results: pd.DataFrame, path: Path, n_repeats: int) -> None:
    order = [name for name, _ in ROI_SPECS]
    synthetic_sem = results["accuracy_sd_across_repeats"] / np.sqrt(n_repeats)
    y_min = max(0.0, min(0.45, float(results["bo_published_figure3_approx_accuracy"].min())) - 0.02)
    y_max = min(
        1.04,
        max(
            0.80,
            float((results["accuracy_mean_repeated_10fold"] + synthetic_sem).max()),
            float(results["bo_published_figure3_approx_accuracy"].max()),
        )
        + 0.04,
    )
    fig, axes = plt.subplots(1, 2, figsize=(19, 7.2), sharey=True)
    roi_colors = plt.get_cmap("tab20")(np.arange(len(order)))
    titles = {"pleasant_vs_neutral": "Pleasant vs Neutral", "unpleasant_vs_neutral": "Unpleasant vs Neutral"}
    for axis, (contrast, _) in zip(axes, CONTRASTS, strict=True):
        frame = results[results["contrast"] == contrast].set_index("roi").loc[order]
        x = np.arange(len(order))
        width = 0.36
        synthetic = frame["accuracy_mean_repeated_10fold"].to_numpy()
        sem = frame["accuracy_sd_across_repeats"].to_numpy() / np.sqrt(n_repeats)
        published = frame["bo_published_figure3_approx_accuracy"].to_numpy()
        axis.bar(
            x - width / 2,
            published - y_min,
            width,
            bottom=y_min,
            color=roi_colors,
            alpha=0.30,
            edgecolor=roi_colors,
            linewidth=1.0,
            hatch="//",
            zorder=2,
        )
        axis.bar(
            x + width / 2,
            synthetic,
            width,
            color=roi_colors,
            edgecolor="#333333",
            linewidth=0.45,
            yerr=sem,
            error_kw={"ecolor": "#222222", "elinewidth": 1.0, "capsize": 2, "capthick": 1.0},
            zorder=3,
        )
        significant = frame["permutation_q_bh_34_tests"].to_numpy() < 0.05
        for idx in np.flatnonzero(significant):
            axis.text(
                idx + width / 2,
                min(y_max - 0.012, synthetic[idx] + sem[idx] + 0.010),
                "*",
                ha="center",
                va="bottom",
                fontsize=11,
            )
        axis.axhline(CHANCE, color="#666666", linestyle=":", linewidth=1.2)
        axis.axhline(PAPER_THRESHOLD, color="#9E2A2B", linestyle="--", linewidth=1.1)
        axis.set_title(titles[contrast], fontsize=14, weight="bold")
        axis.set_xticks(x, order, rotation=55, ha="right")
        axis.set_ylim(y_min, y_max)
        axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        axis.grid(axis="y", alpha=0.2)
        axis.set_xlabel("Bilateral fsaverage5 Kastner/Wang ROI")
    axes[0].set_ylabel("Decoding accuracy")
    handles = [
        Patch(facecolor="#777777", edgecolor="#333333", label=f"TRIBE v2 synthetic (mean ± CV-repeat SEM; {n_repeats}×10-fold)"),
        Patch(facecolor="#AAAAAA", edgecolor="#777777", alpha=0.35, hatch="//", label="Published Figure 3 (approx. box center; SEM not reported)"),
        Line2D([0], [0], color="#666666", linestyle=":", linewidth=1.2, label="Chance (50%)"),
        Line2D([0], [0], color="#9E2A2B", linestyle="--", linewidth=1.1, label="Bo et al. group threshold (54%; descriptive here)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=2, frameon=False)
    fig.suptitle(
        "Unique-image emotion decoding from TRIBE v2 synthetic responses",
        fontsize=16,
        weight="bold",
        y=0.98,
    )
    fig.subplots_adjust(top=0.75, bottom=0.23, left=0.06, right=0.99, wspace=0.06)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_roi_scatter(results: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.2))
    titles = {"pleasant_vs_neutral": "Pleasant vs Neutral", "unpleasant_vs_neutral": "Unpleasant vs Neutral"}
    colors = {"pleasant_vs_neutral": "#E07A3F", "unpleasant_vs_neutral": "#3C78B4"}
    roi_order = [name for name, _ in ROI_SPECS]
    roi_number = {roi: index + 1 for index, roi in enumerate(roi_order)}
    for axis, (contrast, _) in zip(axes, CONTRASTS, strict=True):
        frame = results[results["contrast"] == contrast].reset_index(drop=True)
        x = frame["kebo_local_mean_accuracy"].to_numpy()
        y = frame["accuracy_mean_repeated_10fold"].to_numpy()
        axis.scatter(x, y, s=78, color=colors[contrast], edgecolor="white", linewidth=0.7, zorder=3)
        for row in frame.itertuples(index=False):
            axis.text(
                row.kebo_local_mean_accuracy,
                row.accuracy_mean_repeated_10fold,
                str(roi_number[row.roi]),
                ha="center",
                va="center",
                color="white",
                fontsize=6.5,
                weight="bold",
                zorder=4,
            )
        raw_low = min(0.45, float(x.min()), float(y.min()))
        raw_high = max(0.75, float(x.max()), float(y.max()))
        padding = max(0.015, (raw_high - raw_low) * 0.05)
        low = max(0.0, raw_low - padding)
        high = min(1.04, raw_high + padding)
        axis.plot([low, high], [low, high], color="#777777", linestyle="--", linewidth=1)
        pr = pearsonr(x, y)
        sr = spearmanr(x, y)
        axis.text(
            0.03,
            0.97,
            f"Pearson r={pr.statistic:.2f}\nSpearman ρ={sr.statistic:.2f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(titles[contrast], weight="bold")
        axis.set_xlabel("Local Ke Bo reproduction accuracy")
        axis.set_ylabel("TRIBE v2 synthetic accuracy")
        axis.grid(alpha=0.2)
    key_lines = []
    for start in (0, 6, 12):
        key_lines.append("   ".join(f"{index + 1} {roi_order[index]}" for index in range(start, min(start + 6, len(roi_order)))))
    fig.text(0.5, 0.08, "\n".join(key_lines), ha="center", va="center", fontsize=9)
    fig.suptitle("Across-ROI correspondence", fontsize=15, weight="bold", y=0.97)
    fig.subplots_adjust(top=0.86, bottom=0.22, left=0.08, right=0.98, wspace=0.24)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    mapping: pd.DataFrame,
    output_dir: Path,
    decoding_dir: Path,
    args: argparse.Namespace,
) -> Path:
    compact = results[
        [
            "contrast",
            "roi",
            "n_vertices",
            "accuracy_mean_repeated_10fold",
            "permutation_p",
            "permutation_q_bh_34_tests",
            "bo_published_figure3_approx_accuracy",
            "kebo_local_mean_accuracy",
            "local_pipeline_sensitivity_accuracy",
            "delta_vs_kebo_percentage_points",
        ]
    ].copy()
    for column in (
        "accuracy_mean_repeated_10fold",
        "permutation_p",
        "permutation_q_bh_34_tests",
        "bo_published_figure3_approx_accuracy",
        "kebo_local_mean_accuracy",
        "local_pipeline_sensitivity_accuracy",
        "delta_vs_kebo_percentage_points",
    ):
        compact[column] = compact[column].map(lambda value: f"{value:.4f}")

    overview = summary.copy()
    float_columns = overview.select_dtypes(include=["float"]).columns
    overview[float_columns] = overview[float_columns].map(lambda value: f"{value:.4f}")
    roi_counts = mapping[["roi", "kastner_label_ids", "n_vertices"]].copy()
    summary_by_contrast = summary.set_index("contrast")
    pleasant_summary = summary_by_contrast.loc["pleasant_vs_neutral"]
    unpleasant_summary = summary_by_contrast.loc["unpleasant_vs_neutral"]
    screening_verdict = (
        "**Screening verdict:** positive for decodable category information, but not yet positive for human "
        f"affect-neural fidelity. Pleasant-vs-neutral averaged {pleasant_summary.synthetic_mean_across_rois:.1%} "
        f"across ROIs ({int(pleasant_summary.n_synthetic_fdr_q_lt_0_05)}/17 permutation-FDR significant); "
        f"unpleasant-vs-neutral averaged {unpleasant_summary.synthetic_mean_across_rois:.1%} "
        f"({int(unpleasant_summary.n_synthetic_fdr_q_lt_0_05)}/17 significant). The much higher synthetic "
        "accuracies and weak across-ROI correspondence with measured results make image-content/V-JEPA "
        "confounding a leading explanation that requires a backbone control and real↔synthetic validation."
    )

    lines = [
        "# IAPS-60 TRIBE v2 static-image encoding and ROI decoding",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Outcome",
        "",
        "This report evaluates whether one deterministic TRIBE v2 population-average response per unique IAPS image contains linearly decodable pleasant-vs-neutral or unpleasant-vs-neutral information in 17 bilateral retinotopic ROIs. It is a synthetic screening analysis, not a subject-level replication of Bo et al.",
        "",
        screening_verdict,
        "",
        markdown_table(overview),
        "",
        "## Fixed encoding protocol",
        "",
        "- Sixty unique IAPS images (20 neutral, 20 pleasant, 20 unpleasant) were matched by `iaps_id` to filename.",
        "- Images were converted to grayscale and then replicated into RGB channels to match the stimuli shown to Bo et al.'s participants.",
        "- Each image was encoded as a 3-second identical-frame silent video, following the attached META/TRIBE paper.",
        "- TRIBE v2 produced a 1-Hz, fsaverage5, population-average response. The three-row series was resampled by origin-anchored linear interpolation to the Ke Bo acquisition TR (1.98 s), producing target offsets 0 and 1.98 s. The prespecified primary response is target row 0, exactly equal to native row 0 and beginning 5 seconds after stimulus onset.",
        "- No timepoint or temporal average was selected after inspecting decoding results. All three generated rows remain saved for later sensitivity analyses.",
        "",
        "## Decoding protocol",
        "",
        "- Separate pleasant-vs-neutral and unpleasant-vs-neutral linear SVMs (`C=1`) were fit in each ROI, using 40 unique images per contrast. Primary preprocessing matches the available MATLAB implementation: each image pattern is z-scored across ROI features (`zscore(X,0,2)`), which uses no information from other samples.",
        "- A prespecified sensitivity column repeats unique-image decoding with the newer local reproduction pipeline (training-fold `StandardScaler` plus `LinearSVC`). Permutation inference applies only to the primary MATLAB-style analysis.",
        f"- The descriptive accuracy is the mean of {args.n_repeats} repeated stratified 10-fold partitions. Figure error bars are SEM across the {args.n_repeats} repeat-level accuracies; they quantify cross-validation split variability, not uncertainty across subjects.",
        f"- Each permutation p-value uses one prespecified stratified 10-fold partition and {args.n_permutations} label permutations. BH-FDR is applied jointly to 34 ROI×contrast tests.",
        "- The five real repetitions per image were not duplicated synthetically: doing so would put identical deterministic maps in training and test folds.",
        "",
        "## ROI results",
        "",
        markdown_table(compact),
        "",
        "## Surface ROI mapping",
        "",
        "TRIBE predictions have 20,484 fsaverage5 vertices (10,242 LH followed by 10,242 RH). The volumetric `kastner_dict.npy` cannot index them. This analysis uses the existing Kastner2015 surface atlas resampled by nearest-neighbor sphere registration from fsaverage to fsaverage5, combines hemispheres, uses label 13 for hMT, and combines labels 18–23 for IPS.",
        "",
        markdown_table(roi_counts),
        "",
        "## How to read the comparison",
        "",
        "- Published Figure 3 values are approximate box-center digitizations (roughly ±0.5–1 percentage point; mean versus median is not guaranteed), because exact ROI means and SEM are not tabulated. The figure therefore shows no error bars for the published bars rather than inventing uncertainty.",
        "- ROI colors are held fixed across contrasts and shared by the synthetic and published bars. Solid bars are TRIBE v2; hatched bars are the published Figure 3 approximations.",
        "- The local Ke Bo reproduction remains in the result table as an exact saved numerical comparator, but is intentionally omitted from the figure.",
        "- The publication's 54% cutoff was a group-level permutation threshold over 20 people. Crossing 54% here is reported only descriptively and is not the same statistical test.",
        "- Synthetic accuracy can arise from image content or semantic regularities represented by V-JEPA2, even if TRIBE does not reproduce human affect-specific neural variation.",
        "- A stronger next validation would average the five measured responses per image, apply the same image-level folds, and test real↔synthetic generalization or representational correspondence.",
        "",
        "## Output files",
        "",
        f"- Full ROI results: `{(decoding_dir / 'roi_decoding_results.csv').resolve()}`",
        f"- Summary comparison: `{(decoding_dir / 'comparison_summary.csv').resolve()}`",
        f"- Main figure: `{(decoding_dir / 'roi_decoding_comparison.png').resolve()}`",
        f"- Atlas provenance: `{(decoding_dir / 'atlas_provenance.json').resolve()}`",
        f"- First-TR response matrix: `{(output_dir / 'encoding' / 'iaps60_tribev2_first_tr.npy').resolve()}`",
    ]
    report_path = output_dir / "RESULTS.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    decoding_dir = output_dir / "decoding"
    decoding_dir.mkdir(parents=True, exist_ok=True)
    manifest, responses = load_inputs(output_dir)
    rois, mapping, provenance_path = write_atlas_outputs(
        args.surface_atlas.resolve(), args.surface_atlas_summary.resolve(), decoding_dir
    )
    baseline = load_kebo_baseline(args.kebo_pleasant.resolve(), args.kebo_unpleasant.resolve())
    baseline.to_csv(decoding_dir / "kebo_local_baseline_copy.csv", index=False)
    digitized_rows = [
        {
            "contrast": contrast,
            "roi": roi,
            "figure3_center_approx_accuracy": accuracy,
            "visual_uncertainty_percentage_points": "approximately +/-0.5 to 1",
            "statistic_warning": "box center visually estimated; mean versus median is not guaranteed",
            "source": "Bo et al., Cerebral Cortex 2021, Figure 3",
        }
        for contrast, roi_values in PUBLISHED_FIGURE3_APPROX.items()
        for roi, accuracy in roi_values.items()
    ]
    pd.DataFrame(digitized_rows).to_csv(
        decoding_dir / "bo_published_figure3_approx_digitization.csv",
        index=False,
    )

    result_rows: list[dict[str, object]] = []
    repeat_outputs: dict[str, np.ndarray] = {}
    sensitivity_repeat_outputs: dict[str, np.ndarray] = {}
    null_outputs: dict[str, np.ndarray] = {}
    labels = manifest["emotion_label"].to_numpy()
    for contrast_index, (contrast, positive_label) in enumerate(CONTRASTS):
        selected = np.isin(labels, ["neutral", positive_label])
        y_text = labels[selected]
        y = (y_text == positive_label).astype(np.int8)
        if y.shape != (40,) or np.bincount(y).tolist() != [20, 20]:
            raise ValueError(f"Unexpected class balance for {contrast}: {np.bincount(y)}")
        for roi_index, (roi, _) in enumerate(ROI_SPECS):
            vertices = rois[roi]
            x_raw = responses[selected][:, vertices]
            x = matlab_pattern_zscore(x_raw)
            seed = args.seed + contrast_index * 10_000 + roi_index
            values, repeat_scores, permutation_scores = evaluate_roi(
                x,
                y,
                seed=seed,
                n_repeats=args.n_repeats,
                n_permutations=args.n_permutations,
                jobs=args.jobs,
            )
            sensitivity_values, sensitivity_repeat_scores = evaluate_local_pipeline_sensitivity(
                x_raw,
                y,
                seed=seed,
                n_repeats=args.n_repeats,
                jobs=args.jobs,
            )
            key = f"{contrast}__{roi}"
            repeat_outputs[key] = repeat_scores.astype(np.float32)
            sensitivity_repeat_outputs[key] = sensitivity_repeat_scores.astype(np.float32)
            null_outputs[key] = permutation_scores.astype(np.float32)
            result_rows.append(
                {
                    "contrast": contrast,
                    "positive_class": positive_label,
                    "reference_class": "neutral",
                    "roi": roi,
                    "n_vertices": int(vertices.size),
                    "n_unique_images": int(len(y)),
                    "n_images_per_class": int(np.bincount(y).min()),
                    "classifier": "MATLAB pattern zscore across ROI + linear SVC(C=1)",
                    "cv": f"RepeatedStratifiedKFold(10,{args.n_repeats})",
                    "seed": seed,
                    **values,
                    **sensitivity_values,
                    "above_chance_descriptive": bool(values["accuracy_mean_repeated_10fold"] > CHANCE),
                    "above_bo_54pct_descriptive": bool(values["accuracy_mean_repeated_10fold"] > PAPER_THRESHOLD),
                }
            )
            print(
                f"{contrast:24s} {roi:4s}: "
                f"accuracy={values['accuracy_mean_repeated_10fold']:.3f}, p_perm={values['permutation_p']:.4f}"
            )

    results = pd.DataFrame(result_rows)
    results["permutation_q_bh_34_tests"] = bh_fdr(results["permutation_p"].to_numpy())
    results["permutation_fdr_significant_0_05"] = results["permutation_q_bh_34_tests"] < 0.05
    baseline_subset = baseline[
        ["comparison", "roi", "mean_accuracy", "std_accuracy", "sem_accuracy", "n_subjects", "kebo_source_file"]
    ].rename(
        columns={
            "comparison": "contrast",
            "mean_accuracy": "kebo_local_mean_accuracy",
            "std_accuracy": "kebo_local_std_accuracy",
            "sem_accuracy": "kebo_local_sem_accuracy",
            "n_subjects": "kebo_local_n_subjects",
        }
    )
    results = results.merge(baseline_subset, on=["contrast", "roi"], how="left", validate="one_to_one")
    if results["kebo_local_mean_accuracy"].isna().any():
        raise ValueError("Could not match every ROI to the local Ke Bo baseline.")
    results["delta_vs_kebo_percentage_points"] = (
        results["accuracy_mean_repeated_10fold"] - results["kebo_local_mean_accuracy"]
    ) * 100
    results["bo_published_figure3_approx_accuracy"] = [
        PUBLISHED_FIGURE3_APPROX[contrast][roi]
        for contrast, roi in zip(results["contrast"], results["roi"], strict=True)
    ]
    results_path = decoding_dir / "roi_decoding_results.csv"
    results.to_csv(results_path, index=False)
    np.savez_compressed(decoding_dir / "cv_repeat_accuracy_distributions.npz", **repeat_outputs)
    np.savez_compressed(
        decoding_dir / "local_pipeline_sensitivity_cv_distributions.npz",
        **sensitivity_repeat_outputs,
    )
    np.savez_compressed(decoding_dir / "permutation_null_distributions.npz", **null_outputs)

    comparison = comparison_summary(results)
    comparison_path = decoding_dir / "comparison_summary.csv"
    comparison.to_csv(comparison_path, index=False)
    plot_roi_comparison(results, decoding_dir / "roi_decoding_comparison.png", args.n_repeats)
    report_path = write_report(results, comparison, mapping, output_dir, decoding_dir, args)

    run_summary = {
        "created_at": utc_now(),
        "status": "complete",
        "input_response": str((output_dir / "encoding" / "iaps60_tribev2_first_tr.npy").resolve()),
        "input_response_sha256": sha256_file(output_dir / "encoding" / "iaps60_tribev2_first_tr.npy"),
        "manifest": str((output_dir / "iaps60_unique_manifest.csv").resolve()),
        "surface_atlas": str(args.surface_atlas.resolve()),
        "atlas_provenance": str(provenance_path.resolve()),
        "n_repeats": args.n_repeats,
        "n_permutations": args.n_permutations,
        "seed": args.seed,
        "fdr_family": "34 tests (17 ROIs x 2 contrasts)",
        "results_csv": str(results_path.resolve()),
        "comparison_summary_csv": str(comparison_path.resolve()),
        "report_markdown": str(report_path.resolve()),
        "main_figure": str((decoding_dir / "roi_decoding_comparison.png").resolve()),
        "runtime_versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    write_json(decoding_dir / "decoding_summary.json", run_summary)
    print(f"Wrote report: {report_path}")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode IAPS emotion contrasts in fsaverage5 Kastner ROIs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--surface-atlas", type=Path, default=DEFAULT_SURFACE_ATLAS)
    parser.add_argument("--surface-atlas-summary", type=Path, default=DEFAULT_SURFACE_ATLAS_SUMMARY)
    parser.add_argument("--kebo-pleasant", type=Path, default=DEFAULT_KEBO_PLEASANT)
    parser.add_argument("--kebo-unpleasant", type=Path, default=DEFAULT_KEBO_UNPLEASANT)
    parser.add_argument("--n-repeats", type=int, default=100)
    parser.add_argument("--n-permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()
    if args.n_repeats < 2:
        parser.error("--n-repeats must be at least 2")
    if args.n_permutations < 99:
        parser.error("--n-permutations must be at least 99")
    return args


if __name__ == "__main__":
    run(parse_args())
