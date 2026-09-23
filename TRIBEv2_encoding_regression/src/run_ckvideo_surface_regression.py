from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"N:\Experimental_Data\yujunchen\projects\Inhouse_encoding_model")
DEFAULT_RESPONSES_DIR = Path(
    r"N:\Experimental_Data\yujunchen\projects\data\TRIBEv2\ckvideos\outputs_ckvideo"
)
DEFAULT_METADATA_CSV = Path(
    r"N:\Experimental_Data\yujunchen\projects\data\original_ckvideo_data\CowenKeltnerEmotionalVideos.csv"
)
DEFAULT_OUT_ROOT = PROJECT_ROOT / "TRIBEv2_encoding_regression" / "outputs"

PREDICTORS = ("valence", "arousal")
FDR_ALPHA = 0.05
N_LEFT_VERTICES = 10242


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    sd = values.std(ddof=1)
    if sd == 0:
        raise ValueError("Cannot z-score a constant predictor.")
    return (values - values.mean()) / sd


def load_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path)
    required = ["Filename", "valence", "arousal"]
    missing = [col for col in required if col not in meta.columns]
    if missing:
        raise ValueError(f"Missing metadata columns in {path}: {missing}")

    meta = meta[required].copy()
    meta["video_id"] = meta["Filename"].str.replace(".mp4", "", regex=False)
    for col in PREDICTORS:
        meta[col] = pd.to_numeric(meta[col], errors="coerce")
    meta = meta.dropna(subset=["video_id", *PREDICTORS]).reset_index(drop=True)
    return meta


def scan_response_files(responses_dir: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    meta_ids = set(metadata["video_id"])
    rows: list[dict[str, object]] = []

    for folder in sorted(p for p in responses_dir.iterdir() if p.is_dir()):
        video_id = folder.name
        pred_path = folder / "predictions.npy"
        if video_id not in meta_ids or not pred_path.exists():
            continue

        arr = np.load(pred_path, mmap_mode="r")
        if arr.ndim != 2:
            print(f"Skipping {video_id}: predictions are not 2D, shape={arr.shape}")
            continue
        rows.append(
            {
                "video_id": video_id,
                "predictions_path": str(pred_path),
                "n_timepoints": int(arr.shape[0]),
                "n_vertices": int(arr.shape[1]),
            }
        )

    responses = pd.DataFrame(rows)
    if responses.empty:
        raise RuntimeError(f"No usable TRIBEv2 predictions found in {responses_dir}")

    table = responses.merge(metadata, on="video_id", how="inner")
    if table["n_vertices"].nunique() != 1:
        raise ValueError(
            "Prediction files do not all have the same number of vertices: "
            f"{sorted(table['n_vertices'].unique())}"
        )

    table = table.sort_values("video_id").reset_index(drop=True)
    for col in PREDICTORS:
        table[f"{col}_z_video"] = zscore(table[col].to_numpy(dtype=np.float64))
    return table


def build_xtx_xty(video_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int, int]:
    n_vertices = int(video_table["n_vertices"].iloc[0])
    n_rows = int(video_table["n_timepoints"].sum())
    n_predictors = 1 + len(PREDICTORS)
    xtx = np.zeros((n_predictors, n_predictors), dtype=np.float64)
    xty = np.zeros((n_predictors, n_vertices), dtype=np.float64)

    for idx, row in enumerate(video_table.itertuples(index=False), start=1):
        x = np.array([1.0, row.valence_z_video, row.arousal_z_video], dtype=np.float64)
        y = np.load(row.predictions_path, mmap_mode="r")
        y_sum = np.asarray(y, dtype=np.float64).sum(axis=0)
        xtx += int(row.n_timepoints) * np.outer(x, x)
        xty += x[:, None] * y_sum[None, :]
        if idx % 250 == 0:
            print(f"First pass: {idx}/{len(video_table)} videos")

    return xtx, xty, n_rows, n_vertices


def clustered_inference(
    video_table: pd.DataFrame,
    beta: np.ndarray,
    xtx_inv: np.ndarray,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cluster-robust SE and t maps for all coefficients.

    Clusters are videos. This keeps each time point in the regression while
    making the inferential unit the video rather than the individual time point.
    """
    n_coefficients, n_vertices = beta.shape
    meat = np.zeros((n_coefficients, n_coefficients, n_vertices), dtype=np.float64)
    rss = np.zeros(n_vertices, dtype=np.float64)

    for idx, row in enumerate(video_table.itertuples(index=False), start=1):
        x = np.array([1.0, row.valence_z_video, row.arousal_z_video], dtype=np.float64)
        y = np.asarray(np.load(row.predictions_path, mmap_mode="r"), dtype=np.float64)
        fitted = x @ beta
        resid = y - fitted[None, :]
        rss += np.sum(resid * resid, axis=0)

        score = x[:, None] * resid.sum(axis=0)[None, :]
        for a in range(n_coefficients):
            for b in range(n_coefficients):
                meat[a, b] += score[a] * score[b]

        if idx % 250 == 0:
            print(f"Second pass: {idx}/{len(video_table)} videos")

    n_clusters = len(video_table)
    correction = (n_clusters / (n_clusters - 1.0)) * ((n_rows - 1.0) / (n_rows - n_coefficients))
    se = np.zeros_like(beta, dtype=np.float64)
    for j in range(n_coefficients):
        cov_j = np.zeros(n_vertices, dtype=np.float64)
        for a in range(n_coefficients):
            for b in range(n_coefficients):
                cov_j += xtx_inv[j, a] * meat[a, b] * xtx_inv[b, j]
        se[j] = np.sqrt(np.maximum(correction * cov_j, 0.0))

    t_values = np.divide(beta, se, out=np.zeros_like(beta, dtype=np.float64), where=se > 0)
    sigma2 = rss / max(n_rows - n_coefficients, 1)
    return se, t_values, sigma2


def _betacf(a: float, b: float, x: float) -> float:
    max_iter = 200
    eps = 3.0e-12
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt) if log_bt > -745 else 0.0
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_to_two_sided_p(t_values: np.ndarray, df: int) -> np.ndarray:
    """Two-sided p-values.

    For the CK-video model the cluster df is large, so a normal approximation is
    stable and practically identical for thresholding. The beta-function branch
    is kept for small-df reuse.
    """
    t_flat = np.abs(np.asarray(t_values, dtype=np.float64)).ravel()
    p_flat = np.empty_like(t_flat)
    if df > 200:
        sqrt2 = math.sqrt(2.0)
        for idx, t_val in enumerate(t_flat):
            p_flat[idx] = math.erfc(float(t_val) / sqrt2) if np.isfinite(t_val) else np.nan
    else:
        a = df / 2.0
        b = 0.5
        for idx, t_val in enumerate(t_flat):
            if not np.isfinite(t_val):
                p_flat[idx] = np.nan
                continue
            x = df / (df + t_val * t_val)
            p_flat[idx] = regularized_incomplete_beta(a, b, x)
    return p_flat.reshape(np.asarray(t_values).shape).astype(np.float32)


def fdr_bh(p_values: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p_values, dtype=np.float64)
    p_flat = p.ravel()
    q_flat = np.full_like(p_flat, np.nan)
    finite = np.isfinite(p_flat)
    finite_p = p_flat[finite]

    order = np.argsort(finite_p)
    ranked = finite_p[order]
    m = ranked.size
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    finite_indices = np.flatnonzero(finite)
    q_flat[finite_indices[order]] = adjusted
    q = q_flat.reshape(p.shape).astype(np.float32)
    return q, q < alpha


def vertex_table(mask: np.ndarray, beta: np.ndarray, t_values: np.ndarray, p_values: np.ndarray, q_values: np.ndarray) -> pd.DataFrame:
    vertices = np.flatnonzero(mask)
    hemi = np.where(vertices < N_LEFT_VERTICES, "lh", "rh")
    hemi_vertex = np.where(vertices < N_LEFT_VERTICES, vertices, vertices - N_LEFT_VERTICES)
    return pd.DataFrame(
        {
            "vertex_global": vertices.astype(int),
            "hemisphere": hemi,
            "vertex_hemi": hemi_vertex.astype(int),
            "beta": beta[vertices],
            "t_cluster": t_values[vertices],
            "p_cluster": p_values[vertices],
            "q_fdr": q_values[vertices],
        }
    ).sort_values("q_fdr")


def save_figures(out_dir: Path, beta: np.ndarray, t_values: np.ndarray, masks: dict[str, np.ndarray]) -> None:
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for idx, predictor in enumerate(PREDICTORS, start=1):
        axes[idx - 1, 0].hist(beta[idx], bins=80, color="#356A9A", alpha=0.88)
        axes[idx - 1, 0].axvline(0, color="black", linewidth=1)
        axes[idx - 1, 0].set_title(f"{predictor}: beta distribution")
        axes[idx - 1, 0].set_xlabel("cluster-robust beta")
        axes[idx - 1, 0].set_ylabel("vertices")

        axes[idx - 1, 1].hist(t_values[idx], bins=80, color="#8B5A2B", alpha=0.88)
        axes[idx - 1, 1].axvline(0, color="black", linewidth=1)
        axes[idx - 1, 1].set_title(f"{predictor}: t distribution")
        axes[idx - 1, 1].set_xlabel("cluster-robust t")
        axes[idx - 1, 1].set_ylabel("vertices")

    fig.tight_layout()
    fig.savefig(figs_dir / "beta_t_histograms.png", dpi=180)
    plt.close(fig)

    labels = []
    counts = []
    colors = []
    for predictor in PREDICTORS:
        for sign, color in [("positive", "#B33A3A"), ("negative", "#2F6F9F")]:
            labels.append(f"{predictor}\n{sign}")
            counts.append(int(masks[f"{predictor}_{sign}"].sum()))
            colors.append(color)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, counts, color=colors)
    ax.set_ylabel("FDR q < 0.05 vertices")
    ax.set_title("Significant TRIBEv2 surface vertices by predictor and sign")
    for i, count in enumerate(counts):
        ax.text(i, count, str(count), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(figs_dir / "significant_vertex_counts.png", dpi=180)
    plt.close(fig)


def write_readme(out_dir: Path, summary: dict[str, object]) -> None:
    sig = summary["significant_counts"]
    lines = [
        "# TRIBEv2 CK-Video Surface Encoding Regression",
        "",
        "Model: `surface_response_vertex ~ intercept + z(valence) + z(arousal)`",
        "",
        "This is an encoding regression, not a decoding classifier.",
        "",
        "## Data",
        "",
        f"- Labeled videos: `{summary['n_videos']}`",
        f"- Time-point observations: `{summary['n_timepoints']}`",
        f"- Surface vertices: `{summary['n_vertices']}`",
        f"- Surface space: `{summary['surface_space']}`",
        "",
        "## Inference",
        "",
        "- Standard errors: video-clustered robust SE",
        f"- Cluster df: `{summary['cluster_df']}`",
        f"- FDR threshold: `q < {summary['fdr_alpha']}` across vertices, separately for each predictor",
        "- Positive and negative effects are saved separately after FDR.",
        "",
        "## FDR q < 0.05 Counts",
        "",
        "| Predictor | Positive vertices | Negative vertices | Total significant |",
        "|---|---:|---:|---:|",
    ]
    for predictor in PREDICTORS:
        pos = sig[predictor]["positive"]
        neg = sig[predictor]["negative"]
        lines.append(f"| {predictor} | {pos} | {neg} | {pos + neg} |")
    lines.extend(
        [
            "",
            "## Main Outputs",
            "",
            "- `beta_intercept_valence_arousal_vertices.npy`: coefficient maps, shape `3 x n_vertices`",
            "- `t_cluster_intercept_valence_arousal_vertices.npy`: cluster-robust t maps",
            "- `p_cluster_valence_arousal_vertices.npy`: p maps for valence and arousal",
            "- `q_fdr_valence_arousal_vertices.npy`: FDR q maps for valence and arousal",
            "- `fdr05_mask_valence_arousal_vertices.npy`: FDR masks for valence and arousal",
            "- `significant_vertices/*.csv`: positive/negative significant vertices with beta, t, p, q",
            "- `masks/*.npy`: boolean masks for each predictor/sign, including LH/RH split masks",
            "",
            "Caution: because TRIBEv2 predictions are average-subject model outputs, this is not a",
            "human-subject group second level like the IAPS fMRI regression. The inferential clusters",
            "here are CK videos.",
            "",
        ]
    )
    (out_dir / "README_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    run_name = args.run_name or datetime.now().strftime("ckvideo_surface_reg_%Y%m%d_%H%M%S")
    out_dir = args.out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "significant_vertices").mkdir()
    (out_dir / "masks").mkdir()

    metadata = load_metadata(args.metadata_csv)
    video_table = scan_response_files(args.responses_dir, metadata)
    video_table.to_csv(out_dir / "design_video_table.csv", index=False)

    print(f"Videos: {len(video_table)}")
    print(f"Time points: {int(video_table['n_timepoints'].sum())}")
    print(f"Vertices: {int(video_table['n_vertices'].iloc[0])}")
    print(f"Outputs: {out_dir}")

    xtx, xty, n_rows, n_vertices = build_xtx_xty(video_table)
    xtx_inv = np.linalg.inv(xtx)
    beta = xtx_inv @ xty
    se, t_values, sigma2 = clustered_inference(video_table, beta, xtx_inv, n_rows)

    predictor_t = t_values[1:3]
    predictor_beta = beta[1:3]
    p_values = t_to_two_sided_p(predictor_t, df=len(video_table) - 1)
    q_values = np.empty_like(p_values)
    fdr_masks = np.empty(p_values.shape, dtype=bool)
    for idx, predictor in enumerate(PREDICTORS):
        q_values[idx], fdr_masks[idx] = fdr_bh(p_values[idx], alpha=args.fdr_alpha)
        print(f"{predictor}: {int(fdr_masks[idx].sum())} vertices survive FDR q < {args.fdr_alpha}")

    np.save(out_dir / "beta_intercept_valence_arousal_vertices.npy", beta.astype(np.float32))
    np.save(out_dir / "se_cluster_intercept_valence_arousal_vertices.npy", se.astype(np.float32))
    np.save(out_dir / "t_cluster_intercept_valence_arousal_vertices.npy", t_values.astype(np.float32))
    np.save(out_dir / "p_cluster_valence_arousal_vertices.npy", p_values.astype(np.float32))
    np.save(out_dir / "q_fdr_valence_arousal_vertices.npy", q_values.astype(np.float32))
    np.save(out_dir / "fdr05_mask_valence_arousal_vertices.npy", fdr_masks)
    np.save(out_dir / "residual_sigma2_vertices.npy", sigma2.astype(np.float32))
    np.save(out_dir / "xtx.npy", xtx.astype(np.float64))
    np.save(out_dir / "xtx_inv.npy", xtx_inv.astype(np.float64))

    masks: dict[str, np.ndarray] = {}
    significant_counts: dict[str, dict[str, int]] = {}
    for idx, predictor in enumerate(PREDICTORS):
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

    corr = float(np.corrcoef(video_table["valence_z_video"], video_table["arousal_z_video"])[0, 1])
    vif = 1.0 / (1.0 - corr**2)
    summary = {
        "analysis": "TRIBEv2 CK-video vertexwise multiple encoding regression",
        "model": "surface_response_vertex ~ intercept + z_video(valence) + z_video(arousal)",
        "dependent_variable": "TRIBEv2 predicted average-subject fsaverage5 cortical response",
        "independent_variables": list(PREDICTORS),
        "surface_space": "fsaverage5, 10242 LH + 10242 RH vertices",
        "n_videos": int(len(video_table)),
        "n_timepoints": int(n_rows),
        "n_vertices": int(n_vertices),
        "cluster_unit": "video_id",
        "cluster_df": int(len(video_table) - 1),
        "fdr_alpha": float(args.fdr_alpha),
        "predictor_correlation_valence_arousal": corr,
        "predictor_vif_each_predictor": vif,
        "significant_counts": significant_counts,
        "beta_range": {
            predictor: [
                float(np.nanmin(predictor_beta[idx])),
                float(np.nanmax(predictor_beta[idx])),
            ]
            for idx, predictor in enumerate(PREDICTORS)
        },
        "t_range": {
            predictor: [
                float(np.nanmin(predictor_t[idx])),
                float(np.nanmax(predictor_t[idx])),
            ]
            for idx, predictor in enumerate(PREDICTORS)
        },
    }
    (out_dir / "regression_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(out_dir, summary)
    save_figures(out_dir, beta, t_values, masks)

    print("Done.")
    print(out_dir)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses-dir", type=Path, default=DEFAULT_RESPONSES_DIR)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--fdr-alpha", type=float, default=FDR_ALPHA)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
