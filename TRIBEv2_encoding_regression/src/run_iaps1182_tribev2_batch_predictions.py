from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(r"N:\Experimental_Data\yujunchen\projects\Inhouse_encoding_model")
DEFAULT_IAPS_DIR = Path(r"N:\Experimental_Data\yujunchen\projects\data\IAPS1182")
DEFAULT_TRIBEV2_DIR = PROJECT_ROOT / "TRIBEv2"
DEFAULT_OUTPUT_DIR = DEFAULT_IAPS_DIR / "predictions" / "tribev2_fsaverage5"
DEFAULT_CACHE_DIR = DEFAULT_TRIBEV2_DIR / "cache"
EXPECTED_VERTICES = 20484


def safe_id(value: object) -> str:
    if pd.notna(value):
        try:
            numeric = float(value)
            if numeric.is_integer():
                value = int(numeric)
        except (TypeError, ValueError):
            pass
    text = str(value).strip()
    text = re.sub(r"[<>:\"/\\|?*\x00-\x1f]+", "_", text)
    text = re.sub(r"\s+", "_", text).strip("._")
    if not text:
        raise ValueError("Empty IAPS image id after sanitization.")
    return text


def import_tribev2_helpers(tribev2_dir: Path) -> tuple[object, object, object]:
    sys.path.insert(0, str(tribev2_dir))
    sys.path.insert(0, str(tribev2_dir / "scripts"))
    from predict_local import create_video_from_images, segment_rows, visual_events
    from tribev2 import TribeModel

    return TribeModel, create_video_from_images, visual_events, segment_rows


def load_iaps_manifest(iaps_dir: Path, metadata_csv: Path | None, limit: int | None) -> pd.DataFrame:
    metadata_csv = metadata_csv or (iaps_dir / "IAPS1182_VA.csv")
    image_dir = iaps_dir / "IAPS1182"
    meta = pd.read_csv(metadata_csv)
    meta.columns = [col.strip() for col in meta.columns]
    required = ["image", "type", "arousal", "valence"]
    missing = [col for col in required if col not in meta.columns]
    if missing:
        raise ValueError(f"Missing columns in {metadata_csv}: {missing}")

    meta = meta[required].copy()
    meta["image_id"] = meta["image"].map(safe_id)
    meta["type"] = meta["type"].astype(str).str.strip().str.lower().str.lstrip(".")
    meta["image_path"] = [
        str((image_dir / f"{image_id}.{ext}").resolve())
        for image_id, ext in zip(meta["image_id"], meta["type"], strict=True)
    ]
    for col in ("arousal", "valence"):
        meta[col] = pd.to_numeric(meta[col], errors="coerce")
    meta = meta.dropna(subset=["image_id", "image_path", "arousal", "valence"]).reset_index(drop=True)

    missing_files = [path for path in meta["image_path"] if not Path(path).exists()]
    if missing_files:
        preview = "\n".join(missing_files[:10])
        raise FileNotFoundError(f"{len(missing_files)} IAPS images are missing. First missing paths:\n{preview}")
    if meta["image_id"].duplicated().any():
        dupes = sorted(meta.loc[meta["image_id"].duplicated(), "image_id"].unique())
        raise ValueError(f"IAPS image ids must be unique. Duplicates: {dupes[:20]}")
    if limit is not None:
        meta = meta.head(limit).copy()
    return meta


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


def run(args: argparse.Namespace) -> Path:
    args.iaps_dir = args.iaps_dir.resolve()
    args.tribev2_dir = args.tribev2_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_iaps_manifest(args.iaps_dir, args.metadata_csv, args.limit)
    manifest_csv = args.output_dir / "iaps1182_tribev2_manifest.csv"
    manifest.to_csv(manifest_csv, index=False)

    if args.dry_run:
        print(f"Prepared manifest only: {manifest_csv}")
        return manifest_csv

    TribeModel, create_video_from_images, visual_events, segment_rows = import_tribev2_helpers(args.tribev2_dir)
    feature_device = args.feature_device or args.device
    if feature_device == "auto":
        feature_device = "cuda" if torch.cuda.is_available() else "cpu"
    config_update = {
        "data.text_feature.device": feature_device,
        "data.audio_feature.device": feature_device,
        "data.image_feature.image.device": feature_device,
        "data.video_feature.image.device": feature_device,
    }

    model = TribeModel.from_pretrained(
        args.checkpoint,
        checkpoint_name=args.checkpoint_name,
        cache_folder=args.cache_dir,
        device=args.device,
        config_update=config_update,
    )

    rows: list[dict[str, object]] = []
    for index, row in enumerate(manifest.itertuples(index=False), start=1):
        image_id = str(row.image_id)
        image_path = Path(str(row.image_path))
        run_dir = args.output_dir / image_id
        work_dir = run_dir / "work"
        run_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        predictions_path = run_dir / "predictions.npy"
        status, n_timepoints, n_vertices = prediction_status(predictions_path)
        if status == "ok" and n_vertices == EXPECTED_VERTICES and not args.overwrite:
            print(f"[{index}/{len(manifest)}] {image_id}: skipped existing ({n_timepoints}, {n_vertices})")
            rows.append(
                {
                    **row._asdict(),
                    "run_dir": str(run_dir),
                    "predictions_path": str(predictions_path),
                    "status": "skipped_existing",
                    "n_timepoints": n_timepoints,
                    "n_vertices": n_vertices,
                }
            )
            continue

        print(f"[{index}/{len(manifest)}] {image_id}: predicting")
        stimulus_path = create_video_from_images(
            source=image_path,
            work_dir=work_dir,
            duration=args.image_duration,
            fps=args.fps,
            max_side=args.max_side,
        )
        events = visual_events(stimulus_path)
        events_path = run_dir / "events.csv"
        events.to_csv(events_path, index=False)

        predictions, segments = model.predict(events=events, verbose=not args.quiet)
        np.save(predictions_path, predictions)
        segments_path = run_dir / "segments.csv"
        pd.DataFrame(segment_rows(segments)).to_csv(segments_path, index=False)

        status, n_timepoints, n_vertices = prediction_status(predictions_path)
        if status == "ok" and n_vertices != EXPECTED_VERTICES:
            status = f"bad_vertices:{n_vertices}"
        summary = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": str(image_path),
            "input_type": "image",
            "stimulus_path": str(stimulus_path),
            "checkpoint": args.checkpoint,
            "device": args.device,
            "vision_only": True,
            "prediction_shape": list(predictions.shape),
            "events_csv": str(events_path),
            "segments_csv": str(segments_path),
            "predictions_npy": str(predictions_path),
            "valence": float(row.valence),
            "arousal": float(row.arousal),
            "note": "Predictions are average-subject fsaverage5 cortical responses with TRIBEv2's hemodynamic offset.",
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        rows.append(
            {
                **row._asdict(),
                "run_dir": str(run_dir),
                "predictions_path": str(predictions_path),
                "events_path": str(events_path),
                "segments_path": str(segments_path),
                "status": status,
                "n_timepoints": n_timepoints,
                "n_vertices": n_vertices,
            }
        )
        print(f"  {status} shape=({n_timepoints}, {n_vertices})")

    prediction_manifest = args.output_dir / "image_prediction_manifest.csv"
    pd.DataFrame(rows).to_csv(prediction_manifest, index=False)
    run_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "IAPS1182",
        "source_metadata_csv": str(args.metadata_csv or (args.iaps_dir / "IAPS1182_VA.csv")),
        "source_image_dir": str(args.iaps_dir / "IAPS1182"),
        "output_dir": str(args.output_dir),
        "n_images": int(len(rows)),
        "n_ok": int(sum(str(item["status"]) in {"ok", "skipped_existing"} for item in rows)),
        "expected_vertices": EXPECTED_VERTICES,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "feature_device": feature_device,
        "image_duration": args.image_duration,
        "fps": args.fps,
        "max_side": args.max_side,
        "manifest_csv": str(manifest_csv),
        "prediction_manifest_csv": str(prediction_manifest),
    }
    (args.output_dir / "image_prediction_summary.json").write_text(
        json.dumps(run_summary, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote prediction manifest: {prediction_manifest}")
    return prediction_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run IAPS1182 images through TRIBEv2 with one model load.")
    parser.add_argument("--iaps-dir", type=Path, default=DEFAULT_IAPS_DIR)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--tribev2-dir", type=Path, default=DEFAULT_TRIBEV2_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--checkpoint", default="facebook/tribev2")
    parser.add_argument("--checkpoint-name", default="best.ckpt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-device", default=None)
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
