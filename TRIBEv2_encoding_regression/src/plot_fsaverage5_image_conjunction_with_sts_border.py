from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from plot_fsaverage5_tmap_with_sts_mt_borders import (
    DEFAULT_FSAVERAGE5_DIR,
    DEFAULT_JULIAN_DIR,
    load_surface_pairs,
    load_surfaces,
    plot_tmap_with_borders,
    sample_volume_mask_to_surface,
)


DEFAULT_RUN_DIR = Path(
    r"N:\Experimental_Data\yujunchen\projects\Inhouse_encoding_model"
    r"\TRIBEv2_encoding_regression\outputs\iaps1182_surface_reg_valence_arousal_20260712_local"
)


def load_image_conjunction(run_dir: Path) -> tuple[np.ndarray, dict[str, object]]:
    with open(run_dir / "regression_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    predictors = list(summary["predictors"])
    for required in ("valence", "arousal"):
        if required not in predictors:
            raise ValueError(f"`{required}` is required for this conjunction plot; predictors={predictors}")

    files = summary["output_files"]
    t_values = np.load(run_dir / files["t"])
    fdr_masks = np.load(run_dir / files["fdr_mask"])

    valence_idx = predictors.index("valence")
    arousal_idx = predictors.index("arousal")
    valence_t = t_values[valence_idx + 1]
    arousal_t = t_values[arousal_idx + 1]
    both_fdr = fdr_masks[valence_idx] & fdr_masks[arousal_idx]
    min_abs_t = np.where(both_fdr, np.minimum(np.abs(valence_t), np.abs(arousal_t)), 0.0).astype(np.float32)

    out_data = run_dir / "conjunction_valence_arousal"
    out_data.mkdir(exist_ok=True)
    np.save(out_data / "valence_arousal_both_fdr05_mask_vertices.npy", both_fdr)
    np.save(out_data / "valence_arousal_both_fdr05_min_abs_t_vertices.npy", min_abs_t)
    conjunction_summary = {
        "total": int(both_fdr.sum()),
        "stat_map": "min(abs(t_valence), abs(t_arousal))",
        "threshold": "both valence and arousal FDR q < 0.05",
        "t_file": files["t"],
        "fdr_mask_file": files["fdr_mask"],
    }
    (out_data / "conjunction_summary.json").write_text(json.dumps(conjunction_summary, indent=2), encoding="utf-8")
    return min_abs_t, conjunction_summary


def run(
    run_dir: Path,
    fsaverage5_dir: Path,
    julian_dir: Path,
    surface: str,
    ribbon_steps: int,
    radius_vox: int,
) -> Path:
    out_dir = run_dir / "figures_fsaverage5_roi_borders"
    data_dir = run_dir / "roi_borders_fsaverage5"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    surfaces = load_surfaces(fsaverage5_dir, surface)
    surface_pairs = load_surface_pairs(fsaverage5_dir)
    sts_l = sample_volume_mask_to_surface(
        julian_dir / "lSTS.img",
        surface_pairs["lh"][0],
        surface_pairs["lh"][1],
        ribbon_steps=ribbon_steps,
        radius_vox=radius_vox,
    )
    sts_r = sample_volume_mask_to_surface(
        julian_dir / "rSTS.img",
        surface_pairs["rh"][0],
        surface_pairs["rh"][1],
        ribbon_steps=ribbon_steps,
        radius_vox=radius_vox,
    )
    sts_mask = np.concatenate([sts_l, sts_r])
    t_map, conjunction_summary = load_image_conjunction(run_dir)

    np.save(data_dir / "julian2012_sts_projected_fsaverage5_mask.npy", sts_mask)
    border_summary = {
        "space": "fsaverage5",
        "surface": surface,
        "stat_map": "conjunction_valence_arousal/valence_arousal_both_fdr05_min_abs_t_vertices.npy",
        "julian_lsts_source": str(julian_dir / "lSTS.img"),
        "julian_rsts_source": str(julian_dir / "rSTS.img"),
        "julian_projection": f"sampled through white-pial ribbon, {ribbon_steps} depths, radius {radius_vox} voxel",
        "sts_vertices": int(sts_mask.sum()),
        "conjunction": conjunction_summary,
    }
    (data_dir / "sts_border_summary.json").write_text(json.dumps(border_summary, indent=2), encoding="utf-8")

    out_path = out_dir / "fsaverage5_min_t_with_julian_sts_border.png"
    plot_tmap_with_borders(out_path, surfaces, t_map, sts_mask)
    (out_dir / "README_STS_BORDER.md").write_text(
        "# fsaverage5 Conjunction t-map with Julian2012 STS Border\n\n"
        "This figure matches the CK-video conjunction style: the background map is "
        "`min(abs(t_valence), abs(t_arousal))` and is shown only for vertices where "
        "both image-level valence and arousal are FDR-significant.\n\n"
        "The cyan outline is Julian2012 lSTS/rSTS projected from volume masks to "
        "the fsaverage5 surface.\n",
        encoding="utf-8",
    )
    print(json.dumps(border_summary, indent=2))
    print(f"Wrote {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot image-regression valence/arousal conjunction with Julian2012 STS border.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--fsaverage5-dir", type=Path, default=DEFAULT_FSAVERAGE5_DIR)
    parser.add_argument("--julian-dir", type=Path, default=DEFAULT_JULIAN_DIR)
    parser.add_argument("--surface", choices=["inflated", "pial", "white"], default="inflated")
    parser.add_argument("--ribbon-steps", type=int, default=7)
    parser.add_argument("--radius-vox", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.run_dir, args.fsaverage5_dir, args.julian_dir, args.surface, args.ribbon_steps, args.radius_vox)


if __name__ == "__main__":
    main()
