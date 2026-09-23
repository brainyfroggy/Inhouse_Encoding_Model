from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .metrics import classification_metrics, write_report


TARGETS = {
    "valence": "y_valence.npy",
    "arousal": "y_arousal.npy",
}


def load_dataset(dataset_dir: Path):
    x = np.load(dataset_dir / "X_float32.npy", mmap_mode="r")
    sample_index = pd.read_csv(dataset_dir / "sample_index.csv")
    idx = {
        split: sample_index.index[sample_index["split"] == split].to_numpy()
        for split in ["train", "val", "test"]
    }
    y = {name: np.load(dataset_dir / filename) for name, filename in TARGETS.items()}
    return x, y, idx


def model_configs(random_state: int) -> list[tuple[str, object]]:
    return [
        (
            "linear_logreg_C0.1",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=0.1,
                            max_iter=1000,
                            solver="saga",
                            n_jobs=-1,
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        (
            "linear_logreg_C1",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=1.0,
                            max_iter=1000,
                            solver="saga",
                            n_jobs=-1,
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        (
            "ridge_alpha1",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", RidgeClassifier(alpha=1.0, class_weight="balanced")),
                ]
            ),
        ),
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
            "sgd_log_alpha1e-4",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        SGDClassifier(
                            loss="log_loss",
                            alpha=1e-4,
                            max_iter=2000,
                            class_weight="balanced",
                            early_stopping=True,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        (
            "sgd_hinge_alpha1e-4",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        SGDClassifier(
                            loss="hinge",
                            alpha=1e-4,
                            max_iter=2000,
                            class_weight="balanced",
                            early_stopping=True,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        (
            "pca256_logreg_C1",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=256, random_state=random_state)),
                    (
                        "clf",
                        LogisticRegression(
                            C=1.0,
                            max_iter=1000,
                            solver="lbfgs",
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        (
            "pca256_rbf_svm_C1",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=256, random_state=random_state)),
                    (
                        "clf",
                        SVC(
                            C=1.0,
                            kernel="rbf",
                            gamma="scale",
                            class_weight="balanced",
                            probability=True,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        (
            "pca256_random_forest",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=256, random_state=random_state)),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=500,
                            max_features="sqrt",
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        (
            "pca256_extra_trees",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=256, random_state=random_state)),
                    (
                        "clf",
                        ExtraTreesClassifier(
                            n_estimators=500,
                            max_features="sqrt",
                            class_weight="balanced",
                            n_jobs=-1,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        (
            "pca128_hist_gradient_boost",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=128, random_state=random_state)),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            max_iter=300,
                            learning_rate=0.05,
                            l2_regularization=0.01,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
    ]


def predict_proba_if_available(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    x, y_all, idx = load_dataset(args.dataset_dir)
    x_train = np.asarray(x[idx["train"]], dtype=np.float32)
    x_val = np.asarray(x[idx["val"]], dtype=np.float32)
    x_test = np.asarray(x[idx["test"]], dtype=np.float32)

    rows = []
    for target, y in y_all.items():
        y_train = y[idx["train"]]
        y_val = y[idx["val"]]
        y_test = y[idx["test"]]
        for model_name, model in model_configs(args.seed):
            model_dir = args.out_dir / target / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            print(f"Training {target} / {model_name}", flush=True)
            start = time.time()
            model.fit(x_train, y_train)
            fit_seconds = time.time() - start
            joblib.dump(model, model_dir / "model.joblib")

            for split, x_split, y_split in [
                ("train", x_train, y_train),
                ("val", x_val, y_val),
                ("test", x_test, y_test),
            ]:
                pred = model.predict(x_split)
                prob = predict_proba_if_available(model, x_split)
                m = classification_metrics(y_split, pred, prob)
                m.update(
                    {
                        "target": target,
                        "model": model_name,
                        "split": split,
                        "fit_seconds": fit_seconds,
                        "n_samples": int(len(y_split)),
                    }
                )
                rows.append(m)
                write_report(model_dir / f"{split}_report.json", y_split, pred)
                pd.DataFrame(
                    {"y_true": y_split, "y_pred": pred}
                ).to_csv(model_dir / f"{split}_predictions.csv", index=False)

            config = {
                "target": target,
                "model": model_name,
                "fit_seconds": fit_seconds,
                "classes": ["low", "mid", "high"],
                "sklearn_report_val": classification_report(
                    y_val, model.predict(x_val), zero_division=0
                ),
            }
            (model_dir / "model_info.json").write_text(
                json.dumps(config, indent=2), encoding="utf-8"
            )

    results = pd.DataFrame(rows)
    results.to_csv(args.out_dir / "classical_metrics.csv", index=False)
    print(results.sort_values(["target", "split", "macro_f1"], ascending=[True, True, False]))


if __name__ == "__main__":
    main()
