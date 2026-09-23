from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"N:\Experimental_Data\yujunchen\projects\Inhouse_encoding_model")
DEFAULT_TRIBEV2_DIR = PROJECT_ROOT / "TRIBEv2"
DEFAULT_OUTPUT_DIR = Path(r"N:\Experimental_Data\yujunchen\projects\data\TRIBEv2\images\outputs_image")
DEFAULT_CACHE_DIR = DEFAULT_TRIBEV2_DIR / "cache"
EXPECTED_VERTICES = 20484


def safe_run_name(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[<>:\"/\\|?*\x00-\x1f]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip("._")
    if not text:
        raise ValueError("Image id cannot be empty after sanitization.")
    return text


def resolve_image_path(path_value: object, manifest_csv: Path) -> Path:
    path = Path(str(path_value))
    if not path.is_absolute():
        path = manifest_csv.parent / path
    return path.resolve()


def load_manifest(args: argparse.Namespace) -> pd.DataFrame:
    manifest = pd.read_csv(args.manifest_csv)
    if args.path_column not in manifest.columns:
        raise ValueError(f"Missing image path column `{args.path_column}` in {args.manifest_csv}")
    if args.id_column not in manifest.columns:
        manifest[args.id_column] = manifest[args.path_column].map(lambda value: Path(str(value)).stem)

    manifest = manifest.copy()
    manifest[args.id_column] = manifest[args.id_column].map(safe_run_name)
    manifest["image_path_resolved"] = manifest[args.path_column].map(
        lambda value: str(resolve_image_path(value, args.manifest_csv))
    )

    missing = [path for path in manifest["image_path_resolved"] if not Path(path).exists()]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} image paths do not exist. First missing paths:\n{preview}")

    if manifest[args.id_column].duplicated().any():
        dupes = sorted(manifest.loc[manifest[args.id_column].duplicated(), args.id_column].unique())
        raise ValueError(f"Image ids must be unique after sanitization. Duplicates: {dupes[:20]}")

    if args.limit is not None:
        manifest = manifest.head(args.limit).copy()
    return manifest.reset_index(drop=True)


def prediction_status(predictions_path: Path) -> tuple[str, int | None, int | None]:
    if not predictions_path.exists():
        return "missing", None, None
    try:
        arr = np.load(predictions_path, mmap_mode="r")
    except Exception as exc:  # noqa: BLE001 - status should capture bad local files.
        return f"unreadable:{exc}", None, None
    if arr.ndim != 2:
        return f"bad_ndim:{arr.ndim}", int(arr.shape[0]) if arr.ndim else None, None
    return "ok", int(arr.shape[0]), int(arr.shape[1])


def run_one_image(args: argparse.Namespace, image_id: str, image_path: Path, run_dir: Path) -> dict[str, object]:
    predictions_path = run_dir / "predictions.npy"
    status, n_timepoints, n_vertices = prediction_status(predictions_path)
    if status == "ok" and n_vertices == EXPECTED_VERTICES and not args.overwrite:
        return {
            "status": "skipped_existing",
            "n_timepoints": n_timepoints,
            "n_vertices": n_vertices,
            "returncode": 0,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    command = [
        sys.executable,
        str(args.tribev2_dir / "scripts" / "predict_local.py"),
        str(image_path),
        "--input-type",
        "image",
        "--checkpoint",
        args.checkpoint,
        "--checkpoint-name",
        args.checkpoint_name,
        "--cache-dir",
        str(args.cache_dir),
        "--output-dir",
        str(args.output_dir),
        "--run-name",
        image_id,
        "--device",
        args.device,
        "--image-duration",
        str(args.image_duration),
        "--fps",
        str(args.fps),
        "--max-side",
        str(args.max_side),
    ]
    if args.quiet:
        command.append("--quiet")

    if args.dry_run:
        return {
            "status": "dry_run",
            "n_timepoints": n_timepoints,
            "n_vertices": n_vertices,
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": "",
            "command": " ".join(command),
        }

    completed = subprocess.run(
        command,
        cwd=str(args.tribev2_dir),
        text=True,
        capture_output=True,
        check=False,
    )
    status, n_timepoints, n_vertices = prediction_status(predictions_path)
    if completed.returncode != 0:
        status = f"failed:{status}"
    elif status == "ok" and n_vertices != EXPECTED_VERTICES:
        status = f"bad_vertices:{n_vertices}"

    return {
        "status": status,
        "n_timepoints": n_timepoints,
        "n_vertices": n_vertices,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def run(args: argparse.Namespace) -> Path:
    args.manifest_csv = args.manifest_csv.resolve()
    args.tribev2_dir = args.tribev2_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args)
    rows: list[dict[str, object]] = []

    for index, row in enumerate(manifest.itertuples(index=False), start=1):
        image_id = getattr(row, args.id_column)
        image_path = Path(getattr(row, "image_path_resolved"))
        run_dir = args.output_dir / image_id
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(manifest)}] {image_id}")

        result = run_one_image(args, image_id, image_path, run_dir)
        record = row._asdict()
        record.update(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "run_dir": str(run_dir),
                "predictions_path": str(run_dir / "predictions.npy"),
                "events_path": str(run_dir / "events.csv"),
                "segments_path": str(run_dir / "segments.csv"),
                **result,
            }
        )
        rows.append(record)
        print(f"  {record['status']} shape=({record['n_timepoints']}, {record['n_vertices']})")

    out_csv = args.output_dir / "image_prediction_manifest.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest_csv": str(args.manifest_csv),
        "output_dir": str(args.output_dir),
        "n_images": int(len(rows)),
        "n_ok": int(sum(str(row["status"]) in {"ok", "skipped_existing"} for row in rows)),
        "expected_vertices": EXPECTED_VERTICES,
        "device": args.device,
        "checkpoint": args.checkpoint,
        "image_duration": args.image_duration,
        "fps": args.fps,
        "max_side": args.max_side,
        "prediction_manifest_csv": str(out_csv),
        "note": "Each predictions.npy is a TRIBEv2 fsaverage5 cortical surface array shaped timepoints x 20484.",
    }
    (args.output_dir / "image_prediction_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote image prediction manifest: {out_csv}")
    return out_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a CSV image dataset through local TRIBEv2 still-image inference."
    )
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--id-column", default="image_id")
    parser.add_argument("--path-column", default="image_path")
    parser.add_argument("--tribev2-dir", type=Path, default=DEFAULT_TRIBEV2_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--checkpoint", default="facebook/tribev2")
    parser.add_argument("--checkpoint-name", default="best.ckpt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-duration", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
