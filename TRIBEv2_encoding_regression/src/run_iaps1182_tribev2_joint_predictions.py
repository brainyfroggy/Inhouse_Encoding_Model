from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_iaps1182_tribev2_batch_predictions import (
    DEFAULT_CACHE_DIR,
    DEFAULT_IAPS_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRIBEV2_DIR,
    EXPECTED_VERTICES,
    import_tribev2_helpers,
    load_iaps_manifest,
    prediction_status,
)


def video_event(video_path: Path, image_id: str) -> dict[str, object]:
    return {
        "type": "Video",
        "filepath": str(video_path.resolve()),
        "start": 0,
        "timeline": image_id,
        "subject": "default",
    }


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

    TribeModel, create_video_from_images, _visual_events, segment_rows = import_tribev2_helpers(args.tribev2_dir)
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
    events_rows: list[dict[str, object]] = []
    pending_ids: set[str] = set()
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

        print(f"[{index}/{len(manifest)}] {image_id}: preparing still video")
        stimulus_path = create_video_from_images(
            source=image_path,
            work_dir=work_dir,
            duration=args.image_duration,
            fps=args.fps,
            max_side=args.max_side,
        )
        event = video_event(stimulus_path, image_id)
        pd.DataFrame([event]).to_csv(run_dir / "events.csv", index=False)
        events_rows.append(event)
        pending_ids.add(image_id)
        rows.append(
            {
                **row._asdict(),
                "run_dir": str(run_dir),
                "predictions_path": str(predictions_path),
                "events_path": str(run_dir / "events.csv"),
                "segments_path": str(run_dir / "segments.csv"),
                "status": "pending",
                "n_timepoints": None,
                "n_vertices": None,
            }
        )

    if events_rows:
        events = pd.DataFrame(events_rows)
        events.to_csv(args.output_dir / "combined_events.csv", index=False)
        print(f"Predicting {len(events_rows)} IAPS image timelines in one TRIBEv2 call.")
        predictions, segments = model.predict(events=events, verbose=not args.quiet)
        segment_table = pd.DataFrame(segment_rows(segments))
        if len(segment_table) != predictions.shape[0]:
            raise RuntimeError(
                f"Segment table rows ({len(segment_table)}) do not match predictions rows ({predictions.shape[0]})."
            )
        np.save(args.output_dir / "combined_predictions.npy", predictions)
        segment_table.to_csv(args.output_dir / "combined_segments.csv", index=False)

        manifest_by_id = manifest.set_index(manifest["image_id"].astype(str), drop=False)
        event_by_id = events.set_index(events["timeline"].astype(str), drop=False)
        for record in rows:
            if record["status"] != "pending":
                continue
            image_id = str(record["image_id"])
            run_dir = Path(str(record["run_dir"]))
            image_indices = segment_table.index[segment_table["timeline"].astype(str) == image_id].to_numpy()
            if image_indices.size == 0:
                record["status"] = "missing_segments"
                continue

            image_predictions = predictions[image_indices]
            predictions_path = Path(str(record["predictions_path"]))
            np.save(predictions_path, image_predictions)
            image_segments = segment_table.loc[image_indices].copy()
            image_segments.to_csv(run_dir / "segments.csv", index=False)
            status, n_timepoints, n_vertices = prediction_status(predictions_path)
            if status == "ok" and n_vertices != EXPECTED_VERTICES:
                status = f"bad_vertices:{n_vertices}"
            record["status"] = status
            record["n_timepoints"] = n_timepoints
            record["n_vertices"] = n_vertices

            source_row = manifest_by_id.loc[image_id]
            summary = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": str(source_row.image_path),
                "input_type": "image",
                "stimulus_path": str(event_by_id.loc[image_id].filepath),
                "checkpoint": args.checkpoint,
                "device": args.device,
                "feature_device": feature_device,
                "vision_only": True,
                "prediction_shape": list(image_predictions.shape),
                "events_csv": str(run_dir / "events.csv"),
                "segments_csv": str(run_dir / "segments.csv"),
                "predictions_npy": str(predictions_path),
                "valence": float(source_row.valence),
                "arousal": float(source_row.arousal),
                "note": "Predictions are average-subject fsaverage5 cortical responses with TRIBEv2's hemodynamic offset.",
            }
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"  {image_id}: {status} shape=({n_timepoints}, {n_vertices})")

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
        "combined_events_csv": str(args.output_dir / "combined_events.csv"),
        "combined_segments_csv": str(args.output_dir / "combined_segments.csv"),
        "combined_predictions_npy": str(args.output_dir / "combined_predictions.npy"),
    }
    (args.output_dir / "image_prediction_summary.json").write_text(
        json.dumps(run_summary, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote prediction manifest: {prediction_manifest}")
    return prediction_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IAPS1182 images through TRIBEv2 in one joint prediction call.")
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
