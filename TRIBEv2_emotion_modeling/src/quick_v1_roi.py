from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ["low", "mid", "high"]
HEMI_VERTICES = 10242
RANDOM_SEED = 42


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


def read_freesurfer_label(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 1:
                continue
            vertices.append(int(parts[0]))
    out = np.asarray(vertices, dtype=np.int64)
    if out.size == 0:
        raise ValueError(f"No vertices found in {path}")
    return out


def load_v1_indices(lh_label: Path, rh_label: Path) -> np.ndarray:
    left = read_freesurfer_label(lh_label)
    right = read_freesurfer_label(rh_label) + HEMI_VERTICES
    indices = np.unique(np.concatenate([left, right]))
    if indices.min() < 0 or indices.max() >= HEMI_VERTICES * 2:
        raise ValueError(
            f"V1 label vertices do not fit fsaverage5: min={indices.min()} max={indices.max()}"
        )
    return indices


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
        if arr.ndim != 2 or arr.shape[1] != HEMI_VERTICES * 2:
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
        np.arange(len(table)), test_size=0.30, random_state=seed, stratify=strat
    )
    temp = table.iloc[temp_idx]
    temp_strat = temp["valence_class"].astype(str) + "_" + temp["arousal_class"].astype(str)
    val_rel, test_rel = train_test_split(
        np.arange(len(temp)), test_size=0.50, random_state=seed, stratify=temp_strat
    )
    split = np.array([""] * len(table), dtype=object)
    split[train_idx] = "train"
    split[temp_idx[val_rel]] = "val"
    split[temp_idx[test_rel]] = "test"
    table["split"] = split
    return table


def build_v1_dataset(video_table: pd.DataFrame, roi_indices: np.ndarray, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_samples = int(video_table["n_timepoints"].sum())
    n_features = int(len(roi_indices))
    x = np.empty((n_samples, n_features), dtype=np.float32)
    y_valence = np.empty(n_samples, dtype=np.int64)
    y_arousal = np.empty(n_samples, dtype=np.int64)
    sample_rows = []
    offset = 0
    for row in video_table.itertuples(index=False):
        arr = np.load(row.predictions_path, mmap_mode="r")
        roi_arr = np.asarray(arr[:, roi_indices], dtype=np.float32)
        n = int(roi_arr.shape[0])
        x[offset : offset + n] = roi_arr
        y_valence[offset : offset + n] = int(row.valence_class)
        y_arousal[offset : offset + n] = int(row.arousal_class)
        for t in range(n):
            sample_rows.append(
                {
                    "sample_index": offset + t,
                    "video_id": row.video_id,
                    "timepoint": t,
                    "split": row.split,
                    "valence_class": int(row.valence_class),
                    "arousal_class": int(row.arousal_class),
                }
            )
        offset += n
    np.save(out_dir / "X_v1_float32.npy", x)
    np.save(out_dir / "y_valence.npy", y_valence)
    np.save(out_dir / "y_arousal.npy", y_arousal)
    np.save(out_dir / "v1_vertex_indices.npy", roi_indices)
    sample_index = pd.DataFrame(sample_rows)
    sample_index.to_csv(out_dir / "sample_index.csv", index=False)
    video_table.to_csv(out_dir / "video_table.csv", index=False)
    meta = {
        "space": "fsaverage5 cortical surface",
        "roi": "bilateral V1_exvivo labels in fsaverage5, used as quick surface-compatible V1 ROI",
        "note": "TRIBE v2 predictions are surface vertices, not MNI152 voxels.",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_videos": int(len(video_table)),
        "roi_vertices_left": int(np.sum(roi_indices < HEMI_VERTICES)),
        "roi_vertices_right": int(np.sum(roi_indices >= HEMI_VERTICES)),
        "class_encoding": {"low": 0, "mid": 1, "high": 2},
        "split_counts_samples": sample_index["split"].value_counts().to_dict(),
        "split_counts_videos": video_table["split"].value_counts().to_dict(),
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for record in df[columns].to_dict(orient="records"):
        vals = []
        for col in columns:
            value = record[col]
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_report(path: Path, y_true, y_pred) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def classical_models(seed: int, n_features: int) -> list[tuple[str, object]]:
    pca128 = min(128, n_features)
    return [
        (
            "ridge_alpha10",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", RidgeClassifier(alpha=10.0, class_weight="balanced")),
                ]
            ),
        ),
        (
            "linear_logreg_C0.1",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=0.1,
                            max_iter=600,
                            solver="saga",
                            n_jobs=-1,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
        (
            f"pca{pca128}_rbf_svm_C1",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=pca128, random_state=seed)),
                    ("clf", SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced")),
                ]
            ),
        ),
        (
            "extra_trees_300",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        ExtraTreesClassifier(
                            n_estimators=300,
                            max_features="sqrt",
                            class_weight="balanced",
                            n_jobs=-1,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
        (
            "hist_gradient_boost",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            max_iter=120,
                            learning_rate=0.06,
                            l2_regularization=0.01,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
    ]


def train_classical(x, y, idx, out_dir: Path, seed: int) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    x_train = x[idx["train"]]
    x_val = x[idx["val"]]
    x_test = x[idx["test"]]
    for target, yy in y.items():
        y_train, y_val, y_test = yy[idx["train"]], yy[idx["val"]], yy[idx["test"]]
        for name, model in classical_models(seed, x.shape[1]):
            model_dir = out_dir / target / name
            model_dir.mkdir(parents=True, exist_ok=True)
            print(f"Training classical {target} / {name}", flush=True)
            start = time.time()
            model.fit(x_train, y_train)
            elapsed = time.time() - start
            joblib.dump(model, model_dir / "model.joblib")
            for split, xx, yt in [
                ("train", x_train, y_train),
                ("val", x_val, y_val),
                ("test", x_test, y_test),
            ]:
                yp = model.predict(xx)
                row = metrics(yt, yp)
                row.update(
                    {
                        "family": "classical",
                        "target": target,
                        "model": name,
                        "split": split,
                        "fit_seconds": elapsed,
                        "n_samples": int(len(yt)),
                    }
                )
                rows.append(row)
                write_report(model_dir / f"{split}_report.json", yt, yp)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "classical_metrics.csv", index=False)
    return df


@dataclass
class NNConfig:
    name: str
    hidden: tuple[int, ...]
    dropout: float
    lr: float
    weight_decay: float
    batch_size: int = 512
    epochs: int = 35
    patience: int = 6


def nn_configs() -> list[NNConfig]:
    return [
        NNConfig("mlp_128_do0.2_lr1e-3", (128,), 0.2, 1e-3, 1e-4),
        NNConfig("mlp_256_128_do0.3_lr1e-3", (256, 128), 0.3, 1e-3, 1e-4),
        NNConfig("mlp_512_128_do0.4_lr3e-4", (512, 128), 0.4, 3e-4, 3e-4),
    ]


class V1Dataset(Dataset):
    def __init__(self, x, y_valence, y_arousal, indices, mean, std):
        self.x = x
        self.y_valence = y_valence
        self.y_arousal = y_arousal
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item):
        i = self.indices[item]
        x = (self.x[i].astype(np.float32) - self.mean) / self.std
        return (
            torch.from_numpy(x),
            torch.tensor(self.y_valence[i], dtype=torch.long),
            torch.tensor(self.y_arousal[i], dtype=torch.long),
        )


class MultiTaskMLP(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        layers = []
        prev = n_features
        for width in hidden:
            layers.extend(
                [
                    nn.Linear(prev, width),
                    nn.BatchNorm1d(width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = width
        self.backbone = nn.Sequential(*layers)
        self.valence_head = nn.Linear(prev, 3)
        self.arousal_head = nn.Linear(prev, 3)

    def forward(self, x):
        z = self.backbone(x)
        return self.valence_head(z), self.arousal_head(z)


def class_weights(y, train_idx, device):
    counts = np.bincount(y[train_idx], minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def eval_nn(model, loader, device, loss_v, loss_a) -> dict:
    model.eval()
    total_loss = 0.0
    n = 0
    yv_true, yv_pred, ya_true, ya_pred = [], [], [], []
    for xb, yv, ya in loader:
        xb, yv, ya = xb.to(device), yv.to(device), ya.to(device)
        out_v, out_a = model(xb)
        loss = loss_v(out_v, yv) + loss_a(out_a, ya)
        total_loss += float(loss.item()) * len(xb)
        n += len(xb)
        yv_true.append(yv.cpu().numpy())
        ya_true.append(ya.cpu().numpy())
        yv_pred.append(out_v.argmax(1).cpu().numpy())
        ya_pred.append(out_a.argmax(1).cpu().numpy())
    return {
        "loss": total_loss / max(n, 1),
        "valence_true": np.concatenate(yv_true),
        "valence_pred": np.concatenate(yv_pred),
        "arousal_true": np.concatenate(ya_true),
        "arousal_pred": np.concatenate(ya_pred),
    }


def train_nn(x, yv, ya, idx, out_dir: Path, seed: int) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("NN device:", device, flush=True)
    mean = x[idx["train"]].mean(axis=0).astype(np.float32)
    std = x[idx["train"]].std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = V1Dataset(x, yv, ya, idx[split], mean, std)
        loaders[split] = DataLoader(ds, batch_size=512, shuffle=(split == "train"), num_workers=2)
    final_rows = []
    epoch_rows = []
    for cfg in nn_configs():
        cfg_dir = out_dir / cfg.name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
        print(f"Training neural {cfg.name}", flush=True)
        model = MultiTaskMLP(x.shape[1], cfg.hidden, cfg.dropout).to(device)
        loss_v = nn.CrossEntropyLoss(weight=class_weights(yv, idx["train"], device))
        loss_a = nn.CrossEntropyLoss(weight=class_weights(ya, idx["train"], device))
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        best_val_loss = float("inf")
        best_epoch = -1
        bad_epochs = 0
        start = time.time()
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            total = 0.0
            n = 0
            for xb, yvb, yab in loaders["train"]:
                xb, yvb, yab = xb.to(device), yvb.to(device), yab.to(device)
                opt.zero_grad(set_to_none=True)
                out_v, out_a = model(xb)
                loss = loss_v(out_v, yvb) + loss_a(out_a, yab)
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(xb)
                n += len(xb)
            val_ev = eval_nn(model, loaders["val"], device, loss_v, loss_a)
            row = {
                "model": cfg.name,
                "epoch": epoch,
                "train_loss": total / max(n, 1),
                "val_loss": val_ev["loss"],
                "valence_val_balanced_accuracy": float(
                    balanced_accuracy_score(val_ev["valence_true"], val_ev["valence_pred"])
                ),
                "arousal_val_balanced_accuracy": float(
                    balanced_accuracy_score(val_ev["arousal_true"], val_ev["arousal_pred"])
                ),
            }
            print(row, flush=True)
            epoch_rows.append(row)
            pd.DataFrame(epoch_rows).to_csv(out_dir / "nn_epoch_metrics.csv", index=False)
            if val_ev["loss"] < best_val_loss:
                best_val_loss = val_ev["loss"]
                best_epoch = epoch
                bad_epochs = 0
                torch.save(model.state_dict(), cfg_dir / "best_model.pt")
            else:
                bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break
        model.load_state_dict(torch.load(cfg_dir / "best_model.pt", map_location=device))
        elapsed = time.time() - start
        for split in ["train", "val", "test"]:
            ev = eval_nn(model, loaders[split], device, loss_v, loss_a)
            for target in ["valence", "arousal"]:
                row = metrics(ev[f"{target}_true"], ev[f"{target}_pred"])
                row.update(
                    {
                        "family": "neural",
                        "target": target,
                        "model": cfg.name,
                        "split": split,
                        "fit_seconds": elapsed,
                        "n_samples": int(len(ev[f"{target}_true"])),
                        "best_epoch": best_epoch,
                    }
                )
                final_rows.append(row)
                write_report(cfg_dir / f"{split}_{target}_report.json", ev[f"{target}_true"], ev[f"{target}_pred"])
    df = pd.DataFrame(final_rows)
    df.to_csv(out_dir / "nn_final_metrics.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses-dir", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--lh-v1-label", type=Path, required=True)
    parser.add_argument("--rh-v1-label", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.metadata_csv)
    roi_indices = load_v1_indices(args.lh_v1_label, args.rh_v1_label)
    video_table = scan_responses(args.responses_dir, metadata)
    video_table = assign_splits(video_table, args.seed)
    dataset_dir = args.out_dir / "dataset"
    meta = build_v1_dataset(video_table, roi_indices, dataset_dir)
    print(json.dumps(meta, indent=2), flush=True)

    x = np.load(dataset_dir / "X_v1_float32.npy")
    y = {
        "valence": np.load(dataset_dir / "y_valence.npy"),
        "arousal": np.load(dataset_dir / "y_arousal.npy"),
    }
    sample_index = pd.read_csv(dataset_dir / "sample_index.csv")
    idx = {
        split: sample_index.index[sample_index["split"] == split].to_numpy()
        for split in ["train", "val", "test"]
    }
    classical = train_classical(x, y, idx, args.out_dir / "classical", args.seed)
    neural = train_nn(x, y["valence"], y["arousal"], idx, args.out_dir / "neural", args.seed)
    all_metrics = pd.concat([classical, neural], ignore_index=True)
    all_metrics.to_csv(args.out_dir / "all_model_metrics.csv", index=False)
    best_val = (
        all_metrics[all_metrics["split"] == "val"]
        .sort_values(["target", "balanced_accuracy"], ascending=[True, False])
        .groupby("target")
        .head(8)
    )
    best_test = (
        all_metrics[all_metrics["split"] == "test"]
        .sort_values(["target", "balanced_accuracy"], ascending=[True, False])
        .groupby("target")
        .head(8)
    )
    best_val.to_csv(args.out_dir / "best_validation_models.csv", index=False)
    best_test.to_csv(args.out_dir / "best_test_models.csv", index=False)
    lines = [
        "# Quick V1 ROI Emotion Modeling",
        "",
        "TRIBE v2 predictions are fsaverage5 cortical surface vertices, not MNI152 voxels.",
        "This quick test used bilateral fsaverage5 V1_exvivo surface labels as a V1 ROI.",
        "",
        f"- Samples: {meta['n_samples']}",
        f"- Videos: {meta['n_videos']}",
        f"- V1 features: {meta['n_features']} ({meta['roi_vertices_left']} LH, {meta['roi_vertices_right']} RH)",
        "",
        "## Best Validation Models",
        "",
        markdown_table(best_val, ["target", "family", "model", "accuracy", "balanced_accuracy", "macro_f1"]),
        "",
        "## Best Test Models",
        "",
        markdown_table(best_test, ["target", "family", "model", "accuracy", "balanced_accuracy", "macro_f1"]),
        "",
    ]
    (args.out_dir / "README_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("DONE", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
