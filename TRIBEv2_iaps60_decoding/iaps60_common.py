from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = Path(__file__).resolve().parent

DEFAULT_CONDITIONS_CSV = Path(
    r"N:\Experimental_Data\yujunchen\projects\IAPS_Searchlight\fmri_conditions_trial_ordered.csv"
)
DEFAULT_IMAGE_DIR = Path(
    r"N:\Experimental_Data\yujunchen\projects\IAPS_Searchlight\data\raw_iaps"
)
DEFAULT_TRIBEV2_DIR = PROJECT_ROOT / "TRIBEv2"
# exca embeds a long extractor configuration in its cache directory name.  A
# short local root avoids Win32/UNC path-length failures on this network share.
DEFAULT_CACHE_DIR = Path(r"C:\t2c") if os.name == "nt" else DEFAULT_TRIBEV2_DIR / "cache"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "iaps60_meta_static_3s"
DEFAULT_SURFACE_ATLAS = (
    PROJECT_ROOT
    / "TRIBEv2_encoding_regression"
    / "outputs"
    / "ckvideo_surface_reg_20260709_local"
    / "atlases_fsaverage5"
    / "kastner2015_labels_fsaverage5.npy"
)
DEFAULT_SURFACE_ATLAS_SUMMARY = DEFAULT_SURFACE_ATLAS.parent / "atlas_resampling_summary.json"
DEFAULT_KEBO_DIR = Path(r"N:\Experimental_Data\yujunchen\projects\AI_IAPS\Decoding")
DEFAULT_KEBO_PLEASANT = DEFAULT_KEBO_DIR / "kebo_roi_decoding_pleasant_vs_neutral_summary.csv"
DEFAULT_KEBO_UNPLEASANT = DEFAULT_KEBO_DIR / "kebo_roi_decoding_unpleasant_vs_neutral_summary.csv"

EXPECTED_IMAGES = 60
EXPECTED_PER_CLASS = 20
EXPECTED_REPETITIONS = 5
EXPECTED_VERTICES = 20_484
EXPECTED_TIMEPOINTS = 3

TYPE_TO_LABEL = {"Nt": "neutral", "Pl": "pleasant", "Up": "unpleasant"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

ROI_SPECS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("V1v", (1,)),
    ("V1d", (2,)),
    ("V2v", (3,)),
    ("V2d", (4,)),
    ("V3v", (5,)),
    ("V3d", (6,)),
    ("hV4", (7,)),
    ("VO1", (8,)),
    ("VO2", (9,)),
    ("PHC1", (10,)),
    ("PHC2", (11,)),
    ("hMT", (13,)),
    ("LO1", (15,)),
    ("LO2", (14,)),
    ("V3a", (17,)),
    ("V3b", (16,)),
    ("IPS", (18, 19, 20, 21, 22, 23)),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")


def normalize_iaps_id(value: object) -> str:
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    if not re.fullmatch(r"\d+", text):
        raise ValueError(f"Invalid IAPS id: {value!r}")
    return text


def _one_value(group: pd.DataFrame, column: str, image_id: str) -> object:
    values = group[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"IAPS {image_id} has conflicting {column} values: {values.tolist()}")
    return values.iloc[0]


def _image_lookup(image_dir: Path) -> dict[str, Path]:
    images = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    lookup: dict[str, Path] = {}
    for path in images:
        stem = normalize_iaps_id(path.stem)
        if stem in lookup:
            raise ValueError(f"Duplicate image stem {stem}: {lookup[stem]} and {path}")
        lookup[stem] = path.resolve()
    return lookup


def build_manifest(conditions_csv: Path, image_dir: Path) -> pd.DataFrame:
    conditions_csv = conditions_csv.resolve()
    image_dir = image_dir.resolve()
    if not conditions_csv.is_file():
        raise FileNotFoundError(conditions_csv)
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)

    trials = pd.read_csv(
        conditions_csv,
        dtype={"iaps_id": "string", "type": "string", "category": "string"},
    )
    required = {
        "trial_id",
        "iaps_id",
        "run",
        "type",
        "stim_order",
        "category",
        "Mean_Valence",
        "Mean_Arousal",
    }
    missing = sorted(required.difference(trials.columns))
    if missing:
        raise ValueError(f"Missing condition columns: {missing}")
    if trials[list(required)].isna().any().any():
        raise ValueError("The condition table contains missing required values.")

    trials = trials.copy()
    trials["iaps_id"] = trials["iaps_id"].map(normalize_iaps_id)
    trials["type"] = trials["type"].str.strip()
    unknown_types = sorted(set(trials["type"]).difference(TYPE_TO_LABEL))
    if unknown_types:
        raise ValueError(f"Unknown emotion codes: {unknown_types}")
    if len(trials) != EXPECTED_IMAGES * EXPECTED_REPETITIONS:
        raise ValueError(f"Expected 300 trials, found {len(trials)}")
    if trials["trial_id"].nunique() != len(trials):
        raise ValueError("trial_id must be unique.")

    lookup = _image_lookup(image_dir)
    rows: list[dict[str, object]] = []
    for image_id, group in trials.groupby("iaps_id", sort=False):
        if len(group) != EXPECTED_REPETITIONS:
            raise ValueError(f"IAPS {image_id} appears {len(group)} times, expected 5.")
        if sorted(pd.to_numeric(group["run"]).astype(int).tolist()) != [1, 2, 3, 4, 5]:
            raise ValueError(f"IAPS {image_id} does not occur once in every run.")
        if image_id not in lookup:
            raise FileNotFoundError(f"No image file matches IAPS id {image_id} in {image_dir}")

        image_path = lookup[image_id]
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            source_mode = image.mode
        type_code = str(_one_value(group, "type", image_id))
        rows.append(
            {
                "image_id": image_id,
                "type": type_code,
                "emotion_label": TYPE_TO_LABEL[type_code],
                "category": str(_one_value(group, "category", image_id)),
                "stim_order_within_type": int(_one_value(group, "stim_order", image_id)),
                "mean_valence": float(_one_value(group, "Mean_Valence", image_id)),
                "mean_arousal": float(_one_value(group, "Mean_Arousal", image_id)),
                "n_repetitions": len(group),
                "runs": ";".join(map(str, sorted(pd.to_numeric(group["run"]).astype(int)))),
                "trial_ids": ";".join(map(str, sorted(pd.to_numeric(group["trial_id"]).astype(int)))),
                "image_path": str(image_path),
                "image_sha256": sha256_file(image_path),
                "source_width": width,
                "source_height": height,
                "source_mode": source_mode,
            }
        )

    manifest = pd.DataFrame(rows)
    manifest["_sort"] = pd.to_numeric(manifest["image_id"])
    manifest = manifest.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

    if len(manifest) != EXPECTED_IMAGES or manifest["image_id"].nunique() != EXPECTED_IMAGES:
        raise ValueError(f"Expected 60 unique images, found {len(manifest)}")
    counts = manifest["emotion_label"].value_counts().to_dict()
    expected_counts = {label: EXPECTED_PER_CLASS for label in TYPE_TO_LABEL.values()}
    if counts != expected_counts:
        raise ValueError(f"Expected 20 images/class, found {counts}")
    extras = sorted(set(lookup).difference(manifest["image_id"]))
    if extras:
        raise ValueError(f"Image directory contains IDs not present in the condition table: {extras}")
    return manifest


def save_manifest_audit(
    manifest: pd.DataFrame,
    output_dir: Path,
    conditions_csv: Path,
    image_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "iaps60_unique_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    audit_path = output_dir / "iaps60_data_audit.json"
    audit = {
        "created_at": utc_now(),
        "conditions_csv": str(conditions_csv.resolve()),
        "conditions_sha256": sha256_file(conditions_csv.resolve()),
        "image_dir": str(image_dir.resolve()),
        "n_unique_images": int(len(manifest)),
        "n_trials": int(manifest["n_repetitions"].sum()),
        "class_counts_unique_images": {
            key: int(value) for key, value in manifest["emotion_label"].value_counts().sort_index().items()
        },
        "repetitions_per_image": sorted(manifest["n_repetitions"].unique().astype(int).tolist()),
        "all_source_images_rgb": bool((manifest["source_mode"] == "RGB").all()),
        "source_dimensions": (
            manifest.groupby(["source_width", "source_height"]).size().rename("count").reset_index().to_dict("records")
        ),
        "manifest_csv": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "validation": "PASS: 60 filename-matched unique images, 20/class, five consistent repetitions/image.",
    }
    write_json(audit_path, audit)
    return manifest_path, audit_path


def load_surface_rois(atlas_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    atlas_path = atlas_path.resolve()
    labels = np.load(atlas_path)
    if labels.shape != (EXPECTED_VERTICES,):
        raise ValueError(f"Expected fsaverage5 atlas shape ({EXPECTED_VERTICES},), found {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.allclose(labels, np.rint(labels)):
            raise ValueError("Surface atlas contains non-integer labels.")
        labels = np.rint(labels).astype(np.int16)

    rois: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    occupied = np.zeros(EXPECTED_VERTICES, dtype=bool)
    for name, label_ids in ROI_SPECS:
        mask = np.isin(labels, label_ids)
        indices = np.flatnonzero(mask).astype(np.int32)
        if not indices.size:
            raise ValueError(f"ROI {name} has no vertices in {atlas_path}")
        if np.any(occupied[indices]):
            raise ValueError(f"ROI {name} overlaps an earlier ROI.")
        occupied[indices] = True
        rois[name] = indices
        rows.append(
            {
                "roi": name,
                "kastner_label_ids": ";".join(map(str, label_ids)),
                "n_vertices": int(indices.size),
                "hemispheres": "bilateral",
                "surface_order": "LH 10242 vertices, then RH 10242 vertices",
            }
        )
    return labels, rois, pd.DataFrame(rows)
