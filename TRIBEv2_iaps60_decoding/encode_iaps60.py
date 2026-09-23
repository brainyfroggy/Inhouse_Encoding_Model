from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

from iaps60_common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CONDITIONS_CSV,
    DEFAULT_IMAGE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRIBEV2_DIR,
    EXPECTED_TIMEPOINTS,
    EXPECTED_VERTICES,
    build_manifest,
    save_manifest_audit,
    sha256_file,
    sha256_json,
    utc_now,
    write_json,
)


def git_revision(repo: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def checkpoint_provenance(checkpoint: str, checkpoint_name: str) -> dict[str, object]:
    info: dict[str, object] = {"repo_id_or_path": checkpoint, "filename": checkpoint_name}
    try:
        checkpoint_root = Path(checkpoint)
        if checkpoint_root.exists():
            ckpt_path = checkpoint_root / checkpoint_name
            config_path = checkpoint_root / "config.yaml"
        else:
            from huggingface_hub import hf_hub_download

            ckpt_path = Path(hf_hub_download(checkpoint, checkpoint_name))
            config_path = Path(hf_hub_download(checkpoint, "config.yaml"))
        info.update(
            {
                "checkpoint_path": str(ckpt_path.resolve()),
                "checkpoint_sha256": sha256_file(ckpt_path),
                "config_path": str(config_path.resolve()),
                "config_sha256": sha256_file(config_path),
            }
        )
    except Exception as exc:
        info["provenance_warning"] = str(exc)
    return info


def inference_code_provenance(tribev2_dir: Path) -> dict[str, object]:
    files = [
        tribev2_dir / "tribev2" / "demo_utils.py",
        tribev2_dir / "tribev2" / "model.py",
        tribev2_dir / "scripts" / "predict_local.py",
    ]
    hashes = {str(path.resolve()): sha256_file(path) for path in files if path.is_file()}
    package_versions = {}
    for package in ("neuralset", "neuraltrain", "transformers", "moviepy"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    return {"file_hashes": hashes, "package_versions": package_versions}


def import_tribev2(tribev2_dir: Path):
    scripts_dir = tribev2_dir / "scripts"
    for path in (str(tribev2_dir), str(scripts_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from predict_local import create_video_from_images, segment_rows
    from tribev2 import TribeModel

    return TribeModel, create_video_from_images, segment_rows


def make_stimulus(source: Path, destination: Path, grayscale: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)
        if grayscale:
            image = image.convert("L").convert("RGB")
        else:
            image = image.convert("RGB")
        image.save(destination, format="PNG", optimize=True)


def video_event(video_path: Path, timeline: str) -> dict[str, object]:
    return {
        "type": "Video",
        "filepath": str(video_path.resolve()),
        "start": 0,
        "timeline": timeline,
        "subject": "default",
    }


def valid_prediction(
    path: Path,
    protocol_hash: str,
    summary_path: Path,
    image_id: str,
    source_sha256: str,
) -> bool:
    if not path.is_file() or not summary_path.is_file():
        return False
    try:
        array = np.load(path, mmap_mode="r")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        array.shape == (EXPECTED_TIMEPOINTS, EXPECTED_VERTICES)
        and np.isfinite(array).all()
        and summary.get("protocol_hash") == protocol_hash
        and str(summary.get("image_id")) == image_id
        and summary.get("source_sha256") == source_sha256
    )


def resample_native_predictions(
    combined: np.ndarray,
    source_tr_seconds: float,
    target_tr_seconds: float,
) -> np.ndarray:
    source_times = np.arange(combined.shape[1], dtype=np.float64) * source_tr_seconds
    target_times = np.arange(0.0, source_times[-1] + 1e-9, target_tr_seconds)
    resampled = np.empty((combined.shape[0], len(target_times), combined.shape[2]), dtype=np.float32)
    for target_index, target_time in enumerate(target_times):
        right = int(np.searchsorted(source_times, target_time, side="left"))
        if right == 0:
            resampled[:, target_index] = combined[:, 0]
        elif right >= len(source_times):
            resampled[:, target_index] = combined[:, -1]
        else:
            left = right - 1
            weight = float((target_time - source_times[left]) / (source_times[right] - source_times[left]))
            resampled[:, target_index] = (1.0 - weight) * combined[:, left] + weight * combined[:, right]
    return resampled


def combine_predictions(
    manifest: pd.DataFrame,
    encoding_dir: Path,
    target_tr_seconds: float,
) -> tuple[Path, Path, Path]:
    arrays: list[np.ndarray] = []
    for row in manifest.itertuples(index=False):
        path = encoding_dir / "images" / str(row.image_id) / "predictions_timeseries.npy"
        array = np.load(path)
        if array.shape != (EXPECTED_TIMEPOINTS, EXPECTED_VERTICES):
            raise ValueError(f"Unexpected prediction shape for {row.image_id}: {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite predictions for {row.image_id}")
        arrays.append(array.astype(np.float32, copy=False))
    combined = np.stack(arrays, axis=0)
    timeseries_path = encoding_dir / "iaps60_tribev2_timeseries.npy"
    resampled_path = encoding_dir / "iaps60_tribev2_resampled_tr1p98.npy"
    first_tr_path = encoding_dir / "iaps60_tribev2_first_tr.npy"
    np.save(timeseries_path, combined)
    resampled = resample_native_predictions(combined, source_tr_seconds=1.0, target_tr_seconds=target_tr_seconds)
    if not np.array_equal(resampled[:, 0], combined[:, 0]):
        raise RuntimeError("Origin-anchored resampling changed the prespecified first response sample.")
    np.save(resampled_path, resampled)
    np.save(first_tr_path, resampled[:, 0, :])
    return timeseries_path, resampled_path, first_tr_path


def run(args: argparse.Namespace) -> Path:
    conditions_csv = args.conditions_csv.resolve()
    image_dir = args.image_dir.resolve()
    tribev2_dir = args.tribev2_dir.resolve()
    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    encoding_dir = output_dir / "encoding"
    encoding_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(conditions_csv, image_dir)
    if args.limit is not None:
        if args.limit < 1 or args.limit > len(manifest):
            raise ValueError("--limit must be between 1 and 60.")
        manifest = manifest.head(args.limit).copy()
    manifest_path, audit_path = save_manifest_audit(manifest, output_dir, conditions_csv, image_dir)

    repo = git_revision(tribev2_dir)
    checkpoint_info = checkpoint_provenance(args.checkpoint, args.checkpoint_name)
    code_info = inference_code_provenance(tribev2_dir)
    protocol = {
        "name": "META-inspired static-image TRIBE-v2 protocol adapted to Bo et al. IAPS",
        "checkpoint": args.checkpoint,
        "checkpoint_name": args.checkpoint_name,
        "checkpoint_sha256": checkpoint_info.get("checkpoint_sha256"),
        "checkpoint_config_sha256": checkpoint_info.get("config_sha256"),
        "tribev2_git_commit": repo["commit"],
        "tribev2_dirty_worktree": repo["dirty"],
        "inference_code_file_hashes": code_info["file_hashes"],
        "inference_package_versions": code_info["package_versions"],
        "image_duration_seconds": args.image_duration,
        "video_fps": args.fps,
        "video_codec": "libx264 via local predict_local.create_video_from_images",
        "max_input_side_pixels": args.max_side,
        "grayscale_to_match_bo_stimuli": args.grayscale,
        "vision_only": True,
        "native_prediction_tr_seconds": 1.0,
        "model_hemodynamic_alignment_seconds": 5.0,
        "target_dataset_tr_seconds": args.target_tr,
        "temporal_resampling": "origin-anchored linear interpolation from 1.0 s to target TR",
        "retained_response_index": 0,
        "retained_response_interpretation": (
            "first target-grid sample, beginning 5 s after stimulus onset; identical to native row 0 "
            "under origin-anchored interpolation"
        ),
        "expected_timepoints_per_image": EXPECTED_TIMEPOINTS,
        "expected_vertices": EXPECTED_VERTICES,
        "surface_space": "fsaverage5; LH 10242 then RH 10242",
        "population_average_subject": True,
    }
    if args.image_duration != 3.0:
        raise ValueError("The fixed META protocol requires --image-duration 3.0 seconds.")
    if not np.isclose(args.target_tr, 1.98):
        raise ValueError("This fixed Bo et al. comparison requires --target-tr 1.98 seconds.")
    protocol_hash = sha256_json(protocol)
    protocol["protocol_hash"] = protocol_hash
    write_json(encoding_dir / "protocol.json", protocol)
    if args.dry_run:
        print(f"Validated manifest: {manifest_path}")
        print(f"Data audit: {audit_path}")
        return manifest_path

    TribeModel, create_video_from_images, segment_rows = import_tribev2(tribev2_dir)
    feature_device = args.feature_device or args.device
    if feature_device == "auto":
        feature_device = "cuda" if torch.cuda.is_available() else "cpu"
    model_device = args.device
    if model_device == "auto":
        model_device = "cuda" if torch.cuda.is_available() else "cpu"
    config_update = {
        "data.text_feature.device": feature_device,
        "data.audio_feature.device": feature_device,
        "data.image_feature.image.device": feature_device,
        "data.video_feature.image.device": feature_device,
        "data.num_workers": 0,
        "data.batch_size": args.batch_size,
        "data.video_feature.image.batch_size": args.video_feature_batch_size,
    }

    events_rows: list[dict[str, object]] = []
    pending: set[str] = set()
    output_rows: list[dict[str, object]] = []
    for index, row in enumerate(manifest.itertuples(index=False), start=1):
        image_id = str(row.image_id)
        timeline = f"iaps_{image_id}"
        image_run_dir = encoding_dir / "images" / image_id
        summary_path = image_run_dir / "summary.json"
        prediction_path = image_run_dir / "predictions_timeseries.npy"
        if valid_prediction(
            prediction_path,
            protocol_hash,
            summary_path,
            image_id=image_id,
            source_sha256=str(row.image_sha256),
        ) and not args.overwrite:
            print(f"[{index}/{len(manifest)}] {image_id}: reusing valid 3-TR prediction")
            status = "reused"
        else:
            image_run_dir.mkdir(parents=True, exist_ok=True)
            stimulus_kind = "grayscale" if args.grayscale else "rgb"
            content_key = f"{protocol_hash[:12]}_{str(row.image_sha256)[:12]}"
            stimulus_path = image_run_dir / f"stimulus_{stimulus_kind}_{content_key}.png"
            work_dir = image_run_dir / f"work_{content_key}"
            video_path = work_dir / f"{stimulus_path.stem}_still_video.mp4"
            if args.overwrite or not stimulus_path.is_file() or not video_path.is_file() or video_path.stat().st_size == 0:
                make_stimulus(Path(row.image_path), stimulus_path, grayscale=args.grayscale)
                video_path = create_video_from_images(
                    source=stimulus_path,
                    work_dir=work_dir,
                    duration=args.image_duration,
                    fps=args.fps,
                    max_side=args.max_side,
                )
            event = video_event(video_path, timeline)
            pd.DataFrame([event]).to_csv(image_run_dir / "events.csv", index=False)
            events_rows.append(event)
            pending.add(image_id)
            status = "pending"
            print(f"[{index}/{len(manifest)}] {image_id}: prepared 3-second still clip")
        output_rows.append(
            {
                "image_id": image_id,
                "emotion_label": row.emotion_label,
                "timeline": timeline,
                "prediction_path": str(prediction_path.resolve()),
                "status": status,
            }
        )

    if pending:
        print(f"Loading TRIBE v2 once; {len(pending)} image timelines require prediction.")
        model = TribeModel.from_pretrained(
            args.checkpoint,
            checkpoint_name=args.checkpoint_name,
            cache_folder=cache_dir,
            device=model_device,
            config_update=config_update,
        )
        actual_native_tr = float(model.data.TR)
        actual_hemodynamic_offset = float(model.data.neuro.offset)
        if not np.isclose(actual_native_tr, 1.0):
            raise RuntimeError(f"Checkpoint native TR is {actual_native_tr}, expected 1.0 s.")
        if not np.isclose(actual_hemodynamic_offset, 5.0):
            raise RuntimeError(
                f"Checkpoint hemodynamic offset is {actual_hemodynamic_offset}, expected 5.0 s."
            )
        events = pd.DataFrame(events_rows)
        combined_events_path = encoding_dir / "pending_events.csv"
        events.to_csv(combined_events_path, index=False)
        predictions, segments = model.predict(events=events, verbose=not args.quiet)
        segment_table = pd.DataFrame(segment_rows(segments))
        if len(segment_table) != predictions.shape[0]:
            raise RuntimeError("Segment metadata and prediction row counts differ.")
        segment_table.to_csv(encoding_dir / "pending_segments.csv", index=False)

        for row in manifest.itertuples(index=False):
            image_id = str(row.image_id)
            if image_id not in pending:
                continue
            timeline = f"iaps_{image_id}"
            image_run_dir = encoding_dir / "images" / image_id
            indices = segment_table.index[segment_table["timeline"].astype(str) == timeline].to_numpy()
            if indices.size != EXPECTED_TIMEPOINTS:
                raise RuntimeError(
                    f"{image_id} produced {indices.size} retained timepoints; expected {EXPECTED_TIMEPOINTS}."
                )
            image_segments = segment_table.loc[indices].sort_values("start")
            starts = image_segments["start"].to_numpy(dtype=float)
            durations = image_segments["duration"].to_numpy(dtype=float)
            if not np.allclose(starts, [0.0, 1.0, 2.0]) or not np.allclose(durations, 1.0):
                raise RuntimeError(
                    f"{image_id} segment timing is starts={starts.tolist()}, durations={durations.tolist()}"
                )
            if not image_segments["event_types"].astype(str).str.contains("Video").all():
                raise RuntimeError(f"{image_id} has a retained segment without a Video event.")
            ordered_indices = image_segments.index.to_numpy()
            image_predictions = predictions[ordered_indices].astype(np.float32, copy=False)
            if image_predictions.shape != (EXPECTED_TIMEPOINTS, EXPECTED_VERTICES):
                raise RuntimeError(f"{image_id} prediction shape is {image_predictions.shape}")
            if not np.isfinite(image_predictions).all():
                raise RuntimeError(f"{image_id} prediction contains non-finite values.")
            prediction_path = image_run_dir / "predictions_timeseries.npy"
            np.save(prediction_path, image_predictions)
            np.save(image_run_dir / "prediction_first_tr.npy", image_predictions[0])
            image_segments.to_csv(image_run_dir / "segments.csv", index=False)
            stimulus_kind = "grayscale" if args.grayscale else "rgb"
            content_key = f"{protocol_hash[:12]}_{str(row.image_sha256)[:12]}"
            stimulus_name = f"stimulus_{stimulus_kind}_{content_key}.png"
            summary = {
                "created_at": utc_now(),
                "image_id": image_id,
                "emotion_label": row.emotion_label,
                "source_image": row.image_path,
                "source_sha256": row.image_sha256,
                "derived_stimulus": str((image_run_dir / stimulus_name).resolve()),
                "derived_stimulus_sha256": sha256_file(image_run_dir / stimulus_name),
                "prediction_shape": list(image_predictions.shape),
                "protocol_hash": protocol_hash,
                "primary_row_index": 0,
                "primary_row_alignment_seconds_after_onset": 5.0,
                "predictions_timeseries": str(prediction_path.resolve()),
                "prediction_first_tr": str((image_run_dir / "prediction_first_tr.npy").resolve()),
            }
            write_json(image_run_dir / "summary.json", summary)
            print(f"  {image_id}: saved {tuple(image_predictions.shape)}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    timeseries_path, resampled_path, first_tr_path = combine_predictions(
        manifest,
        encoding_dir,
        target_tr_seconds=args.target_tr,
    )
    for item in output_rows:
        item["status"] = "ok"
    prediction_manifest = encoding_dir / "prediction_manifest.csv"
    pd.DataFrame(output_rows).to_csv(prediction_manifest, index=False)

    summary = {
        "created_at": utc_now(),
        "status": "complete" if len(manifest) == 60 else "test_subset_complete",
        "n_images": int(len(manifest)),
        "timeseries_shape": [int(len(manifest)), EXPECTED_TIMEPOINTS, EXPECTED_VERTICES],
        "primary_response_shape": [int(len(manifest)), EXPECTED_VERTICES],
        "manifest_csv": str(manifest_path.resolve()),
        "data_audit_json": str(audit_path.resolve()),
        "protocol_json": str((encoding_dir / "protocol.json").resolve()),
        "prediction_manifest_csv": str(prediction_manifest.resolve()),
        "timeseries_npy": str(timeseries_path.resolve()),
        "resampled_target_tr_npy": str(resampled_path.resolve()),
        "primary_first_tr_npy": str(first_tr_path.resolve()),
        "target_dataset_tr_seconds": args.target_tr,
        "resampled_timeseries_shape": list(np.load(resampled_path, mmap_mode="r").shape),
        "protocol_hash": protocol_hash,
        "checkpoint": checkpoint_info,
        "tribev2_repo": {"path": str(tribev2_dir), **repo},
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "model_device": model_device,
            "feature_device": feature_device,
            "data_batch_size": args.batch_size,
            "num_workers": 0,
        },
    }
    write_json(encoding_dir / "encoding_summary.json", summary)
    print(f"Primary response matrix: {first_tr_path}")

    if args.remove_work_videos:
        for work_dir in (encoding_dir / "images").glob("*/work*"):
            if work_dir.is_dir():
                shutil.rmtree(work_dir)
        print("Removed derived work videos after successful prediction; grayscale PNG stimuli were retained.")
    return first_tr_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 60 fsaverage5 TRIBE-v2 responses with the META 3-second static-image protocol."
    )
    parser.add_argument("--conditions-csv", type=Path, default=DEFAULT_CONDITIONS_CSV)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--tribev2-dir", type=Path, default=DEFAULT_TRIBEV2_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--checkpoint", default="facebook/tribev2")
    parser.add_argument("--checkpoint-name", default="best.ckpt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--video-feature-batch-size", type=int, default=1)
    parser.add_argument("--image-duration", type=float, default=3.0)
    parser.add_argument("--target-tr", type=float, default=1.98, help="Ke Bo dataset acquisition TR in seconds.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--grayscale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--remove-work-videos", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Testing only; decoding requires all 60 images.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
