from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)


def classification_metrics(y_true, y_pred, y_prob=None, labels=(0, 1, 2)) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }
    if y_prob is not None:
        try:
            out["log_loss"] = float(log_loss(y_true, y_prob, labels=list(labels)))
        except ValueError:
            out["log_loss"] = np.nan
    else:
        out["log_loss"] = np.nan
    return out


def write_report(path: Path, y_true, y_pred, target_names=("low", "mid", "high")):
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=list(target_names),
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
