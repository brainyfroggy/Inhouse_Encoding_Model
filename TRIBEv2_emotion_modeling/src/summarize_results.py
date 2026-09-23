from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    summary_dir = args.run_dir / "summary"
    summary_dir.mkdir(exist_ok=True)

    tables = []
    classical = args.run_dir / "classical" / "classical_metrics.csv"
    neural = args.run_dir / "neural" / "nn_final_metrics.csv"
    if classical.exists():
        df = pd.read_csv(classical)
        df["family"] = "classical"
        tables.append(df)
    if neural.exists():
        df = pd.read_csv(neural)
        df["family"] = "neural"
        tables.append(df)
    if not tables:
        raise RuntimeError("No metrics files found")

    all_metrics = pd.concat(tables, ignore_index=True, sort=False)
    all_metrics.to_csv(summary_dir / "all_model_metrics.csv", index=False)

    val = all_metrics[all_metrics["split"] == "val"].copy()
    best = (
        val.sort_values(["target", "macro_f1", "balanced_accuracy"], ascending=[True, False, False])
        .groupby("target")
        .head(10)
        .reset_index(drop=True)
    )
    best.to_csv(summary_dir / "best_validation_models_by_target.csv", index=False)

    pivot = (
        val.pivot_table(
            index=["family", "model"],
            columns="target",
            values=["accuracy", "balanced_accuracy", "macro_f1", "log_loss"],
            aggfunc="first",
        )
        .reset_index()
    )
    pivot.to_csv(summary_dir / "validation_metrics_pivot.csv", index=False)

    print("Wrote:")
    for path in sorted(summary_dir.iterdir()):
        print(path)
    print("\nBest validation models:")
    print(best)


if __name__ == "__main__":
    main()
