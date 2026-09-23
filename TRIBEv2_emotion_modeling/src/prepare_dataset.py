from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import HPG_METADATA, HPG_RESPONSES, RANDOM_SEED


def class_from_normalized(values: pd.Series) -> pd.Series:
    bins = [-np.inf, 1 / 3, 2 / 3, np.inf]
    return pd.cut(values, bins=bins, labels=[0, 1, 2], right=False).astype(int)


def minmax(series: pd.Series) -> pd.Series:
    lo = float(series.min())
    hi = float(series.max())
    if hi == lo:
        raise ValueError(f"Cannot min-max normalize constant column {series.name}")
    return (series - lo) / (hi - lo)


def load_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path)
    required = ["Filename", "valence", "arousal"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")
    meta = meta[required].copy()
    meta["video_id"] = meta["Filename"].str.replace(".mp4", "", regex=False)
    for col in ["valence", "arousal"]:
        meta[col] = pd.to_numeric(meta[col], errors="coerce")
        meta[f"{col}_norm"] = minmax(meta[col])
        meta[f"{col}_class"] = class_from_normalized(meta[f"{col}_norm"])
    return meta


def scan_responses(responses_dir: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    meta_ids = set(metadata["video_id"])
    for folder in sorted(p for p in responses_dir.iterdir() if p.is_dir()):
        video_id = folder.name
        pred_path = folder / "predictions.npy"
        summary_path = folder / "summary.json"
        if video_id not in meta_ids or not pred_path.exists() or not summary_path.exists():
            continue
        arr = np.load(pred_path, mmap_mode="r")
        if arr.ndim != 2:
            continue
        rows.append(
            {
                "video_id": video_id,
                "predictions_path": str(pred_path),
                "n_timepoints": int(arr.shape[0]),
                "n_vertices": int(arr.shape[1]),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError(f"No usable response files found in {responses_dir}")
    return table.merge(metadata, on="video_id", how="inner")


def assign_splits(video_table: pd.DataFrame, seed: int) -> pd.DataFrame:
    table = video_table.copy()
    strat = table["valence_class"].astype(str) + "_" + table["arousal_class"].astype(str)
    train_idx, temp_idx = train_test_split(
        np.arange(len(table)),
        test_size=0.30,
        random_state=seed,
        stratify=strat,
    )
    temp = table.iloc[temp_idx]
    temp_strat = temp["valence_class"].astype(str) + "_" + temp["arousal_class"].astype(str)
    val_rel, test_rel = train_test_split(
        np.arange(len(temp)),
        test_size=0.50,
        random_state=seed,
        stratify=temp_strat,
    )
    split = np.array([""] * len(table), dtype=object)
    split[train_idx] = "train"
    split[temp_idx[val_rel]] = "val"
    split[temp_idx[test_rel]] = "test"
    table["split"] = split
    return table


def build_arrays(video_table: pd.DataFrame, out_dir: Path, sample_mode: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_features = int(video_table["n_vertices"].iloc[0])
    if sample_mode == "video_mean":
        n_samples = int(len(video_table))
    elif sample_mode == "timepoint":
        n_samples = int(video_table["n_timepoints"].sum())
    else:
        raise ValueError(f"Unknown sample_mode: {sample_mode}")
    x_path = out_dir / "X_float32.npy"
    x = np.lib.format.open_memmap(
        x_path, mode="w+", dtype=np.float32, shape=(n_samples, n_features)
    )
    sample_rows = []
    y_valence = np.empty(n_samples, dtype=np.int64)
    y_arousal = np.empty(n_samples, dtype=np.int64)
    offset = 0
    for row in video_table.itertuples(index=False):
        arr = np.load(row.predictions_path, mmap_mode="r").astype(np.float32)
        if sample_mode == "video_mean":
            rows_to_write = arr.mean(axis=0, keepdims=True)
            timepoints = [-1]
        else:
            rows_to_write = arr
            timepoints = list(range(arr.shape[0]))
        n = rows_to_write.shape[0]
        x[offset : offset + n] = rows_to_write
        y_valence[offset : offset + n] = int(row.valence_class)
        y_arousal[offset : offset + n] = int(row.arousal_class)
        for j, t in enumerate(timepoints):
            sample_rows.append(
                {
                    "sample_index": offset + j,
                    "video_id": row.video_id,
                    "timepoint": t,
                    "n_original_timepoints": int(row.n_timepoints),
                    "split": row.split,
                    "valence": row.valence,
                    "arousal": row.arousal,
                    "valence_norm": row.valence_norm,
                    "arousal_norm": row.arousal_norm,
                    "valence_class": int(row.valence_class),
                    "arousal_class": int(row.arousal_class),
                }
            )
        offset += n
    x.flush()
    np.save(out_dir / "y_valence.npy", y_valence)
    np.save(out_dir / "y_arousal.npy", y_arousal)
    pd.DataFrame(sample_rows).to_csv(out_dir / "sample_index.csv", index=False)
    video_table.to_csv(out_dir / "video_table.csv", index=False)
    meta = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_videos": int(len(video_table)),
        "sample_mode": sample_mode,
        "class_encoding": {"low": 0, "mid": 1, "high": 2},
        "split_counts_samples": pd.DataFrame(sample_rows)["split"].value_counts().to_dict(),
        "split_counts_videos": video_table["split"].value_counts().to_dict(),
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses-dir", type=Path, default=HPG_RESPONSES)
    parser.add_argument("--metadata-csv", type=Path, default=HPG_METADATA)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--sample-mode",
        choices=["video_mean", "timepoint"],
        default="timepoint",
        help="video_mean creates one sample per video; timepoint creates one sample per TR.",
    )
    args = parser.parse_args()

    metadata = load_metadata(args.metadata_csv)
    video_table = scan_responses(args.responses_dir, metadata)
    video_table = assign_splits(video_table, args.seed)
    build_arrays(video_table, args.out_dir, args.sample_mode)
    print((args.out_dir / "dataset_meta.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
