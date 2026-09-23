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
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, Dataset


TARGET_FILES = {"valence": "y_valence.npy", "arousal": "y_arousal.npy"}
CLASS_NAMES = ["low", "mid", "high"]


def load_dataset(dataset_dir: Path):
    x = np.load(dataset_dir / "X_v1_float32.npy")
    sample = pd.read_csv(dataset_dir / "sample_index.csv")
    idx = {
        split: sample.index[sample["split"] == split].to_numpy()
        for split in ["train", "val", "test"]
    }
    y = {target: np.load(dataset_dir / filename) for target, filename in TARGET_FILES.items()}
    return x.astype(np.float32, copy=False), y, idx


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


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


def classical_models(seed: int, n_features: int) -> list[tuple[str, object]]:
    pca64 = min(64, n_features)
    pca128 = min(128, n_features)
    pca256 = min(256, n_features)
    models: list[tuple[str, object]] = []

    for alpha in [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]:
        models.append(
            (
                f"ridge_alpha{alpha:g}",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", RidgeClassifier(alpha=alpha, class_weight="balanced")),
                    ]
                ),
            )
        )

    for c in [0.03, 0.1, 0.3, 1.0]:
        models.append(
            (
                f"logreg_saga_C{c:g}",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            LogisticRegression(
                                C=c,
                                max_iter=1200,
                                solver="saga",
                                class_weight="balanced",
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            )
        )

    for loss in ["log_loss", "modified_huber", "hinge"]:
        for alpha in [1e-4, 1e-3]:
            models.append(
                (
                    f"sgd_{loss}_alpha{alpha:g}",
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            (
                                "clf",
                                SGDClassifier(
                                    loss=loss,
                                    alpha=alpha,
                                    max_iter=2500,
                                    early_stopping=True,
                                    class_weight="balanced",
                                    random_state=seed,
                                ),
                            ),
                        ]
                    ),
                )
            )

    for n_comp in [pca64, pca128, pca256]:
        for c in [0.3, 1.0, 3.0]:
            models.append(
                (
                    f"pca{n_comp}_rbf_svm_C{c:g}",
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("pca", PCA(n_components=n_comp, random_state=seed)),
                            (
                                "clf",
                                SVC(
                                    C=c,
                                    kernel="rbf",
                                    gamma="scale",
                                    class_weight="balanced",
                                    cache_size=2000,
                                    random_state=seed,
                                ),
                            ),
                        ]
                    ),
                )
            )

    models.extend(
        [
            (
                "pca128_poly_svm_C1_d2",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=pca128, random_state=seed)),
                        (
                            "clf",
                            SVC(
                                C=1.0,
                                kernel="poly",
                                degree=2,
                                gamma="scale",
                                class_weight="balanced",
                                cache_size=2000,
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "knn15_distance",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=pca128, random_state=seed)),
                        ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance")),
                    ]
                ),
            ),
            (
                "gaussian_nb",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=pca128, random_state=seed)),
                        ("clf", GaussianNB()),
                    ]
                ),
            ),
        ]
    )

    for name, clf in [
        (
            "extra_trees_500",
            ExtraTreesClassifier(
                n_estimators=500,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        (
            "extra_trees_500_depth12",
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=12,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        (
            "random_forest_500",
            RandomForestClassifier(
                n_estimators=500,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
        ),
    ]:
        models.append((name, Pipeline([("scaler", StandardScaler()), ("clf", clf)])))

    for lr in [0.03, 0.06, 0.1]:
        models.append(
            (
                f"hist_gradient_lr{lr:g}",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            HistGradientBoostingClassifier(
                                max_iter=180,
                                learning_rate=lr,
                                l2_regularization=0.01,
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            )
        )

    return models


def train_classical(x, y_all, idx, out_dir: Path, seed: int) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    x_train, x_val, x_test = x[idx["train"]], x[idx["val"]], x[idx["test"]]
    for target, y in y_all.items():
        y_train, y_val, y_test = y[idx["train"]], y[idx["val"]], y[idx["test"]]
        for model_name, model in classical_models(seed, x.shape[1]):
            model_dir = out_dir / target / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            print(f"Training classical {target} / {model_name}", flush=True)
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
                        "model": model_name,
                        "split": split,
                        "fit_seconds": elapsed,
                        "n_samples": int(len(yt)),
                    }
                )
                rows.append(row)
                write_report(model_dir / f"{split}_report.json", yt, yp)
            pd.DataFrame(rows).to_csv(out_dir / "classical_metrics_partial.csv", index=False)
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
    epochs: int = 60
    patience: int = 8


def nn_configs() -> list[NNConfig]:
    return [
        NNConfig("mlp_64_do0.1_lr1e-3", (64,), 0.1, 1e-3, 1e-4),
        NNConfig("mlp_128_do0.2_lr1e-3", (128,), 0.2, 1e-3, 1e-4),
        NNConfig("mlp_256_128_do0.2_lr1e-3", (256, 128), 0.2, 1e-3, 1e-4),
        NNConfig("mlp_256_128_do0.4_lr3e-4", (256, 128), 0.4, 3e-4, 3e-4),
        NNConfig("mlp_512_256_do0.3_lr3e-4", (512, 256), 0.3, 3e-4, 3e-4),
        NNConfig("mlp_512_256_128_do0.4_lr1e-4", (512, 256, 128), 0.4, 1e-4, 1e-3),
    ]


class SingleTargetDataset(Dataset):
    def __init__(self, x, y, indices, mean, std):
        self.x = x
        self.y = y
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item):
        i = self.indices[item]
        x = (self.x[i].astype(np.float32) - self.mean) / self.std
        return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.long)


class MLP(nn.Module):
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
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def class_weights(y, train_idx, device):
    counts = np.bincount(y[train_idx], minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def eval_nn(model, loader, device, loss_fn) -> dict:
    model.eval()
    total_loss = 0.0
    n = 0
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        total_loss += float(loss.item()) * len(xb)
        n += len(xb)
        y_true.append(yb.cpu().numpy())
        y_pred.append(logits.argmax(1).cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return {"loss": total_loss / max(n, 1), "y_true": y_true, "y_pred": y_pred}


def train_neural(x, y_all, idx, out_dir: Path, seed: int) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("NN device:", device, flush=True)

    mean = x[idx["train"]].mean(axis=0).astype(np.float32)
    std = x[idx["train"]].std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    np.save(out_dir / "train_feature_mean.npy", mean)
    np.save(out_dir / "train_feature_std.npy", std)

    rows = []
    epoch_rows = []
    for target, y in y_all.items():
        datasets = {
            split: SingleTargetDataset(x, y, idx[split], mean, std)
            for split in ["train", "val", "test"]
        }
        loaders = {
            split: DataLoader(
                ds,
                batch_size=512,
                shuffle=(split == "train"),
                num_workers=2,
            )
            for split, ds in datasets.items()
        }
        loss_fn = nn.CrossEntropyLoss(weight=class_weights(y, idx["train"], device))
        for cfg in nn_configs():
            model_name = f"{target}_{cfg.name}"
            model_dir = out_dir / target / cfg.name
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
            print(f"Training neural {target} / {cfg.name}", flush=True)
            model = MLP(x.shape[1], cfg.hidden, cfg.dropout).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            best_score = -1.0
            best_epoch = -1
            bad_epochs = 0
            start = time.time()
            for epoch in range(1, cfg.epochs + 1):
                model.train()
                total = 0.0
                n = 0
                for xb, yb in loaders["train"]:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad(set_to_none=True)
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    loss.backward()
                    opt.step()
                    total += float(loss.item()) * len(xb)
                    n += len(xb)
                val_ev = eval_nn(model, loaders["val"], device, loss_fn)
                val_metrics = metrics(val_ev["y_true"], val_ev["y_pred"])
                row = {
                    "target": target,
                    "model": cfg.name,
                    "epoch": epoch,
                    "train_loss": total / max(n, 1),
                    "val_loss": val_ev["loss"],
                    "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                }
                print(row, flush=True)
                epoch_rows.append(row)
                pd.DataFrame(epoch_rows).to_csv(out_dir / "nn_epoch_metrics.csv", index=False)
                if val_metrics["balanced_accuracy"] > best_score:
                    best_score = val_metrics["balanced_accuracy"]
                    best_epoch = epoch
                    bad_epochs = 0
                    torch.save(model.state_dict(), model_dir / "best_model.pt")
                else:
                    bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    break
            model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
            elapsed = time.time() - start
            for split, loader in loaders.items():
                ev = eval_nn(model, loader, device, loss_fn)
                row = metrics(ev["y_true"], ev["y_pred"])
                row.update(
                    {
                        "family": "neural_single_target",
                        "target": target,
                        "model": model_name,
                        "split": split,
                        "fit_seconds": elapsed,
                        "n_samples": int(len(ev["y_true"])),
                        "best_epoch": best_epoch,
                    }
                )
                rows.append(row)
                write_report(model_dir / f"{split}_report.json", ev["y_true"], ev["y_pred"])
            pd.DataFrame(rows).to_csv(out_dir / "nn_final_metrics_partial.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "nn_final_metrics.csv", index=False)
    return df


def write_summary(out_dir: Path, all_metrics: pd.DataFrame) -> None:
    all_metrics.to_csv(out_dir / "all_model_metrics.csv", index=False)
    best_val = (
        all_metrics[all_metrics["split"] == "val"]
        .sort_values(["target", "balanced_accuracy"], ascending=[True, False])
        .groupby("target")
        .head(12)
    )
    best_test = (
        all_metrics[all_metrics["split"] == "test"]
        .sort_values(["target", "balanced_accuracy"], ascending=[True, False])
        .groupby("target")
        .head(12)
    )
    best_val.to_csv(out_dir / "best_validation_models.csv", index=False)
    best_test.to_csv(out_dir / "best_test_models.csv", index=False)
    columns = ["target", "family", "model", "accuracy", "balanced_accuracy", "macro_f1", "fit_seconds"]
    lines = [
        "# Expanded V1 ROI Model Sweep",
        "",
        "Input: TRIBEv2 fsaverage5 V1-only surface features.",
        "Targets: separate low/mid/high valence and arousal classifiers.",
        "",
        "## Best Validation Models",
        "",
        markdown_table(best_val, columns),
        "",
        "## Best Test Models",
        "",
        markdown_table(best_test, columns),
        "",
    ]
    (out_dir / "README_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument("--skip-neural", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    x, y_all, idx = load_dataset(args.dataset_dir)
    print(f"Loaded X={x.shape} from {args.dataset_dir}", flush=True)
    frames = []
    if not args.skip_classical:
        frames.append(train_classical(x, y_all, idx, args.out_dir / "classical", args.seed))
    if not args.skip_neural:
        frames.append(train_neural(x, y_all, idx, args.out_dir / "neural", args.seed))
    all_metrics = pd.concat(frames, ignore_index=True)
    write_summary(args.out_dir, all_metrics)
    print("DONE", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
