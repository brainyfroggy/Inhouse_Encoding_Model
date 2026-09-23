from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_ckvideo_surface_regression import fdr_bh, t_to_two_sided_p, zscore


PROJECT_ROOT = Path(r"N:\Experimental_Data\yujunchen\projects\Inhouse_encoding_model")
DEFAULT_RESPONSES_DIR = Path(r"N:\Experimental_Data\yujunchen\projects\data\TRIBEv2\images\outputs_image")
DEFAULT_OUT_ROOT = PROJECT_ROOT / "TRIBEv2_encoding_regression" / "outputs"
EXPECTED_VERTICES = 20484
N_LEFT_VERTICES = 10242
FDR_ALPHA = 0.05


def predictor_slug(predictors: list[str]) -> str:
    return "_".join(predictors)


def load_metadata(path: Path, image_id_column: str, predictors: list[str]) -> pd.DataFrame:
    meta = pd.read_csv(path)
    required = [image_id_column, *predictors]
    missing = [col for col in required if col not in meta.columns]
    if missing:
        raise ValueError(f"Missing metadata columns in {path}: {missing}")

    meta = meta[required].copy()
    meta[image_id_column] = meta[image_id_column].astype(str)
    for col in predictors:
        meta[col] = pd.to_numeric(meta[col], errors="coerce")
    meta = meta.dropna(subset=[image_id_column, *predictors]).reset_index(drop=True)
    meta = meta.rename(columns={image_id_column: "image_id"})
    if meta["image_id"].duplicated().any():
        dupes = sorted(meta.loc[meta["image_id"].duplicated(), "image_id"].unique())
        raise ValueError(f"Metadata image ids must be unique. Duplicates: {dupes[:20]}")
    return meta


def scan_response_files(responses_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    manifest_path = responses_dir / "image_prediction_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        if {"image_id", "predictions_path"}.issubset(manifest.columns):
            for row in manifest.itertuples(index=False):
                pred_path = Path(getattr(row, "predictions_path"))
                if pred_path.exists():
                    rows.append({"image_id": str(getattr(row, "image_id")), "predictions_path": str(pred_path)})

    if not rows:
        for folder in sorted(p for p in responses_dir.iterdir() if p.is_dir()):
            pred_path = folder / "predictions.npy"
            if pred_path.exists():
                rows.append({"image_id": folder.name, "predictions_path": str(pred_path)})

    if not rows:
        raise RuntimeError(f"No usable image TRIBEv2 predictions found in {responses_dir}")

    checked: list[dict[str, object]] = []
    for row in rows:
        pred_path = Path(str(row["predictions_path"]))
        arr = np.load(pred_path, mmap_mode="r")
        if arr.ndim != 2:
            print(f"Skipping {row['image_id']}: predictions are not 2D, shape={arr.shape}")
            continue
        checked.append(
            {
                "image_id": row["image_id"],
                "predictions_path": str(pred_path),
                "n_timepoints": int(arr.shape[0]),
                "n_vertices": int(arr.shape[1]),
            }
        )

    responses = pd.DataFrame(checked)
    if responses.empty:
        raise RuntimeError(f"No 2D prediction arrays found in {responses_dir}")
    if responses["n_vertices"].nunique() != 1:
        raise ValueError(
            "Prediction files do not all have the same number of vertices: "
            f"{sorted(responses['n_vertices'].unique())}"
        )
    n_vertices = int(responses["n_vertices"].iloc[0])
    if n_vertices != EXPECTED_VERTICES:
        raise ValueError(f"Expected {EXPECTED_VERTICES} fsaverage5 vertices, found {n_vertices}")
    return responses


def build_design_table(
    responses_dir: Path,
    metadata_csv: Path,
    image_id_column: str,
    predictors: list[str],
) -> pd.DataFrame:
    metadata = load_metadata(metadata_csv, image_id_column, predictors)
    responses = scan_response_files(responses_dir)
    table = responses.merge(metadata, on="image_id", how="inner")
    if table.empty:
        raise RuntimeError("No image ids overlap between metadata and TRIBEv2 prediction outputs.")

    table = table.sort_values("image_id").reset_index(drop=True)
    for predictor in predictors:
        table[f"{predictor}_z_image"] = zscore(table[predictor].to_numpy(dtype=np.float64))
    return table


def aggregate_prediction(path: Path, method: str) -> np.ndarray:
    arr = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64)
    if method == "mean":
        return arr.mean(axis=0)
    if method == "first":
        return arr[0]
    if method == "last":
        return arr[-1]
    raise ValueError(f"Unknown aggregation method: {method}")


def fit_vertexwise_ols(
    design_table: pd.DataFrame,
    predictors: list[str],
    aggregate: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    n_images = len(design_table)
    n_vertices = int(design_table["n_vertices"].iloc[0])
    coefficient_names = ["intercept", *predictors]
    n_coefficients = len(coefficient_names)
    if n_images <= n_coefficients:
        raise ValueError(
            f"Need more images than coefficients for OLS inference; got {n_images} images and "
            f"{n_coefficients} coefficients."
        )

    x = np.column_stack(
        [
            np.ones(n_images, dtype=np.float64),
            *[design_table[f"{predictor}_z_image"].to_numpy(dtype=np.float64) for predictor in predictors],
        ]
    )
    xtx = x.T @ x
    xtx_inv = np.linalg.inv(xtx)
    xty = np.zeros((n_coefficients, n_vertices), dtype=np.float64)

    for idx, row in enumerate(design_table.itertuples(index=False), start=1):
        y = aggregate_prediction(Path(row.predictions_path), aggregate)
        xty += x[idx - 1, :, None] * y[None, :]
        if idx % 250 == 0:
            print(f"First pass: {idx}/{n_images} images")

    beta = xtx_inv @ xty
    rss = np.zeros(n_vertices, dtype=np.float64)

    for idx, row in enumerate(design_table.itertuples(index=False), start=1):
        y = aggregate_prediction(Path(row.predictions_path), aggregate)
        fitted = x[idx - 1] @ beta
        resid = y - fitted
        rss += resid * resid
        if idx % 250 == 0:
            print(f"Second pass: {idx}/{n_images} images")

    df = n_images - n_coefficients
    sigma2 = rss / df
    se = np.sqrt(np.maximum(xtx_inv.diagonal()[:, None] * sigma2[None, :], 0.0))
    t_values = np.divide(beta, se, out=np.zeros_like(beta, dtype=np.float64), where=se > 0)
    return beta, se, t_values, sigma2, xtx, df


def vertex_table(
    mask: np.ndarray,
    beta: np.ndarray,
    t_values: np.ndarray,
    p_values: np.ndarray,
    q_values: np.ndarray,
) -> pd.DataFrame:
    vertices = np.flatnonzero(mask)
    hemi = np.where(vertices < N_LEFT_VERTICES, "lh", "rh")
    hemi_vertex = np.where(vertices < N_LEFT_VERTICES, vertices, vertices - N_LEFT_VERTICES)
    return pd.DataFrame(
        {
            "vertex_global": vertices.astype(int),
            "hemisphere": hemi,
            "vertex_hemi": hemi_vertex.astype(int),
            "beta": beta[vertices],
            "t_ols": t_values[vertices],
            "p_ols": p_values[vertices],
            "q_fdr": q_values[vertices],
        }
    ).sort_values("q_fdr")


def save_figures(
    out_dir: Path,
    predictors: list[str],
    beta: np.ndarray,
    t_values: np.ndarray,
    masks: dict[str, np.ndarray],
) -> None:
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(len(predictors), 2, figsize=(12, max(4.0, 3.5 * len(predictors))))
    axes = np.atleast_2d(axes)
    for idx, predictor in enumerate(predictors, start=1):
        row = idx - 1
        axes[row, 0].hist(beta[idx], bins=80, color="#356A9A", alpha=0.88)
        axes[row, 0].axvline(0, color="black", linewidth=1)
        axes[row, 0].set_title(f"{predictor}: beta distribution")
        axes[row, 0].set_xlabel("OLS beta")
        axes[row, 0].set_ylabel("vertices")

        axes[row, 1].hist(t_values[idx], bins=80, color="#8B5A2B", alpha=0.88)
        axes[row, 1].axvline(0, color="black", linewidth=1)
        axes[row, 1].set_title(f"{predictor}: t distribution")
        axes[row, 1].set_xlabel("OLS t")
        axes[row, 1].set_ylabel("vertices")

    fig.tight_layout()
    fig.savefig(figs_dir / "beta_t_histograms.png", dpi=180)
    plt.close(fig)

    labels = []
    counts = []
    bar_colors = []
    for predictor in predictors:
        for sign, color in [("positive", "#B33A3A"), ("negative", "#2F6F9F")]:
            labels.append(f"{predictor}\n{sign}")
            counts.append(int(masks[f"{predictor}_{sign}"].sum()))
            bar_colors.append(color)

    fig_width = max(7.5, 1.5 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    ax.bar(labels, counts, color=bar_colors)
    ax.set_ylabel("FDR q < 0.05 vertices")
    ax.set_title("Significant TRIBEv2 image surface vertices by predictor and sign")
    for i, count in enumerate(counts):
        ax.text(i, count, str(count), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(figs_dir / "significant_vertex_counts.png", dpi=180)
    plt.close(fig)


def write_readme(out_dir: Path, summary: dict[str, object]) -> None:
    predictors = list(summary["predictors"])
    sig = summary["significant_counts"]
    lines = [
        "# TRIBEv2 Image Surface Encoding Regression",
        "",
        f"Model: `{summary['model']}`",
        "",
        "This is an image-level encoding regression, not a decoding classifier.",
        "",
        "## Data",
        "",
        f"- Images: `{summary['n_images']}`",
        f"- Surface vertices: `{summary['n_vertices']}`",
        f"- Surface space: `{summary['surface_space']}`",
        f"- TRIBEv2 time aggregation per image: `{summary['prediction_time_aggregation']}`",
        "",
        "## Inference",
        "",
        "- Standard errors: ordinary least squares across images",
        f"- Residual df: `{summary['residual_df']}`",
        f"- FDR threshold: `q < {summary['fdr_alpha']}` across vertices, separately for each predictor",
        "- Positive and negative effects are saved separately after FDR.",
        "",
        "## FDR q < 0.05 Counts",
        "",
        "| Predictor | Positive vertices | Negative vertices | Total significant |",
        "|---|---:|---:|---:|",
    ]
    for predictor in predictors:
        pos = sig[predictor]["positive"]
        neg = sig[predictor]["negative"]
        lines.append(f"| {predictor} | {pos} | {neg} | {pos + neg} |")

    slug = str(summary["predictor_slug"])
    lines.extend(
        [
            "",
            "## Main Outputs",
            "",
            f"- `beta_intercept_{slug}_vertices.npy`: coefficient maps",
            f"- `t_ols_intercept_{slug}_vertices.npy`: OLS t maps",
            f"- `p_ols_{slug}_vertices.npy`: p maps for predictors",
            f"- `q_fdr_{slug}_vertices.npy`: FDR q maps for predictors",
            f"- `fdr05_mask_{slug}_vertices.npy`: FDR masks for predictors",
            "- `significant_vertices/*.csv`: positive/negative significant vertices with beta, t, p, q",
            "- `masks/*.npy`: boolean masks for each predictor/sign, including LH/RH split masks",
            "",
            "Caution: TRIBEv2 outputs are average-subject fsaverage5 cortical surface predictions,",
            "not MNI voxels and not human-subject group statistics.",
            "",
        ]
    )
    (out_dir / "README_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    predictors = args.predictors
    slug = predictor_slug(predictors)
    run_name = args.run_name or datetime.now().strftime(f"image_surface_reg_{slug}_%Y%m%d_%H%M%S")
    out_dir = args.out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "significant_vertices").mkdir()
    (out_dir / "masks").mkdir()

    design_table = build_design_table(args.responses_dir, args.metadata_csv, args.image_id_column, predictors)
    design_table.to_csv(out_dir / "design_image_table.csv", index=False)

    print(f"Images: {len(design_table)}")
    print(f"Vertices: {int(design_table['n_vertices'].iloc[0])}")
    print(f"Predictors: {predictors}")
    print(f"Outputs: {out_dir}")

    beta, se, t_values, sigma2, xtx, df = fit_vertexwise_ols(design_table, predictors, args.aggregate)
    xtx_inv = np.linalg.inv(xtx)

    predictor_t = t_values[1:]
    predictor_beta = beta[1:]
    p_values = t_to_two_sided_p(predictor_t, df=df)
    q_values = np.empty_like(p_values)
    fdr_masks = np.empty(p_values.shape, dtype=bool)
    for idx, predictor in enumerate(predictors):
        q_values[idx], fdr_masks[idx] = fdr_bh(p_values[idx], alpha=args.fdr_alpha)
        print(f"{predictor}: {int(fdr_masks[idx].sum())} vertices survive FDR q < {args.fdr_alpha}")

    np.save(out_dir / f"beta_intercept_{slug}_vertices.npy", beta.astype(np.float32))
    np.save(out_dir / f"se_ols_intercept_{slug}_vertices.npy", se.astype(np.float32))
    np.save(out_dir / f"t_ols_intercept_{slug}_vertices.npy", t_values.astype(np.float32))
    np.save(out_dir / f"p_ols_{slug}_vertices.npy", p_values.astype(np.float32))
    np.save(out_dir / f"q_fdr_{slug}_vertices.npy", q_values.astype(np.float32))
    np.save(out_dir / f"fdr05_mask_{slug}_vertices.npy", fdr_masks)
    np.save(out_dir / "residual_sigma2_vertices.npy", sigma2.astype(np.float32))
    np.save(out_dir / "xtx.npy", xtx.astype(np.float64))
    np.save(out_dir / "xtx_inv.npy", xtx_inv.astype(np.float64))

    masks: dict[str, np.ndarray] = {}
    significant_counts: dict[str, dict[str, int]] = {}
    for idx, predictor in enumerate(predictors):
        pos_mask = fdr_masks[idx] & (predictor_beta[idx] > 0)
        neg_mask = fdr_masks[idx] & (predictor_beta[idx] < 0)
        masks[f"{predictor}_positive"] = pos_mask
        masks[f"{predictor}_negative"] = neg_mask
        significant_counts[predictor] = {
            "positive": int(pos_mask.sum()),
            "negative": int(neg_mask.sum()),
        }

        for sign, mask in [("positive", pos_mask), ("negative", neg_mask)]:
            np.save(out_dir / "masks" / f"{predictor}_{sign}_mask.npy", mask)
            np.save(out_dir / "masks" / f"lh_{predictor}_{sign}_mask.npy", mask[:N_LEFT_VERTICES])
            np.save(out_dir / "masks" / f"rh_{predictor}_{sign}_mask.npy", mask[N_LEFT_VERTICES:])
            table = vertex_table(mask, predictor_beta[idx], predictor_t[idx], p_values[idx], q_values[idx])
            table.to_csv(out_dir / "significant_vertices" / f"{predictor}_{sign}_vertices.csv", index=False)

    predictor_correlation = None
    if len(predictors) > 1:
        z_cols = [f"{predictor}_z_image" for predictor in predictors]
        predictor_correlation = design_table[z_cols].corr().to_dict()

    coefficient_names = ["intercept", *predictors]
    summary = {
        "analysis": "TRIBEv2 image vertexwise encoding regression",
        "model": "surface_response_vertex ~ intercept + "
        + " + ".join(f"z_image({predictor})" for predictor in predictors),
        "dependent_variable": "TRIBEv2 predicted average-subject fsaverage5 cortical response, aggregated per image",
        "independent_variables": predictors,
        "predictors": predictors,
        "coefficient_names": coefficient_names,
        "predictor_slug": slug,
        "surface_space": "fsaverage5, 10242 LH + 10242 RH vertices",
        "n_images": int(len(design_table)),
        "n_vertices": int(design_table["n_vertices"].iloc[0]),
        "prediction_time_aggregation": args.aggregate,
        "standard_errors": "ordinary least squares across images",
        "residual_df": int(df),
        "fdr_alpha": float(args.fdr_alpha),
        "predictor_correlation": predictor_correlation,
        "significant_counts": significant_counts,
        "output_files": {
            "beta": f"beta_intercept_{slug}_vertices.npy",
            "se": f"se_ols_intercept_{slug}_vertices.npy",
            "t": f"t_ols_intercept_{slug}_vertices.npy",
            "p": f"p_ols_{slug}_vertices.npy",
            "q": f"q_fdr_{slug}_vertices.npy",
            "fdr_mask": f"fdr05_mask_{slug}_vertices.npy",
        },
        "beta_range": {
            predictor: [
                float(np.nanmin(predictor_beta[idx])),
                float(np.nanmax(predictor_beta[idx])),
            ]
            for idx, predictor in enumerate(predictors)
        },
        "t_range": {
            predictor: [
                float(np.nanmin(predictor_t[idx])),
                float(np.nanmax(predictor_t[idx])),
            ]
            for idx, predictor in enumerate(predictors)
        },
    }
    (out_dir / "regression_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(out_dir, summary)
    save_figures(out_dir, predictors, beta, t_values, masks)

    print("Done.")
    print(out_dir)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run image-level TRIBEv2 fsaverage5 surface encoding regression.")
    parser.add_argument("--responses-dir", type=Path, default=DEFAULT_RESPONSES_DIR)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--image-id-column", default="image_id")
    parser.add_argument("--predictors", nargs="+", default=["valence", "arousal"])
    parser.add_argument("--aggregate", choices=["mean", "first", "last"], default="mean")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--fdr-alpha", type=float, default=FDR_ALPHA)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
