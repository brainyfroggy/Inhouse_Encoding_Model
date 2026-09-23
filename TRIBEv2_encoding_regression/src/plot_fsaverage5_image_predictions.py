from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

from plot_fsaverage5_surface_results import N_LEFT_VERTICES, load_surfaces, plot_hemi_view


DEFAULT_RESPONSES_DIR = Path(r"N:\Experimental_Data\yujunchen\projects\data\TRIBEv2\images\outputs_image")
DEFAULT_FSAVERAGE5_DIR = Path(
    r"N:\Experimental_Data\yujunchen\projects\data\NSD_atlas\nsddata\freesurfer\fsaverage5"
)
EXPECTED_VERTICES = 20484


def aggregate_prediction(predictions: np.ndarray, method: str, timepoint: int | None) -> np.ndarray:
    if predictions.ndim != 2:
        raise ValueError(f"Expected predictions shaped timepoints x vertices, got {predictions.shape}")
    if predictions.shape[1] != EXPECTED_VERTICES:
        raise ValueError(f"Expected {EXPECTED_VERTICES} fsaverage5 vertices, got {predictions.shape[1]}")

    if method == "mean":
        return predictions.mean(axis=0)
    if method == "first":
        return predictions[0]
    if method == "last":
        return predictions[-1]
    if method == "timepoint":
        if timepoint is None:
            raise ValueError("--timepoint is required when --aggregate timepoint is used.")
        if timepoint < 0 or timepoint >= predictions.shape[0]:
            raise IndexError(f"Timepoint {timepoint} is outside 0..{predictions.shape[0] - 1}")
        return predictions[timepoint]
    raise ValueError(f"Unknown aggregation method: {method}")


def robust_symmetric_norm(values: np.ndarray, percentile: float) -> colors.Normalize:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    vmax = float(np.percentile(np.abs(finite), percentile))
    vmax = max(vmax, float(np.nanmax(np.abs(finite))), 1.0e-6) if vmax == 0 else max(vmax, 1.0e-6)
    return colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def read_summary(run_dir: Path) -> dict[str, object]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_prediction(
    run_dir: Path,
    surfaces: dict[str, tuple[np.ndarray, np.ndarray]],
    aggregate: str,
    timepoint: int | None,
    percentile: float,
    surface: str,
) -> Path:
    predictions_path = run_dir / "predictions.npy"
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)

    predictions = np.asarray(np.load(predictions_path, mmap_mode="r"), dtype=np.float64)
    vertex_values = aggregate_prediction(predictions, aggregate, timepoint)
    summary = read_summary(run_dir)
    source = Path(str(summary.get("source", run_dir.name))).name

    out_dir = run_dir / "figures_fsaverage5_prediction"
    out_dir.mkdir(parents=True, exist_ok=True)

    label = f"{aggregate}_{timepoint}" if aggregate == "timepoint" else aggregate
    out_path = out_dir / f"fsaverage5_prediction_{label}.png"
    norm = robust_symmetric_norm(vertex_values, percentile)

    fig = plt.figure(figsize=(15, 8.5), constrained_layout=False)
    views = [("lh", "lateral"), ("lh", "medial"), ("rh", "medial"), ("rh", "lateral")]
    for col, (hemi, view) in enumerate(views):
        ax = fig.add_subplot(1, 4, col + 1, projection="3d")
        coords, faces = surfaces[hemi]
        hemi_values = vertex_values[:N_LEFT_VERTICES] if hemi == "lh" else vertex_values[N_LEFT_VERTICES:]
        plot_hemi_view(ax, coords, faces, hemi_values, norm, hemi, view, f"{hemi.upper()} {view}")

    fig.suptitle(
        f"TRIBEv2 image prediction on fsaverage5: {run_dir.name} ({source}), {aggregate}",
        fontsize=14,
    )
    mappable = cm.ScalarMappable(norm=norm, cmap="coolwarm")
    cbar_ax = fig.add_axes([0.28, 0.08, 0.44, 0.025])
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Predicted cortical response")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.13, wspace=0.00)
    fig.savefig(out_path, dpi=260, facecolor="white")
    plt.close(fig)

    notes = f"""# fsaverage5 Image Prediction Figure

This figure shows TRIBEv2's predicted cortical response for one image on the
fsaverage5 `{surface}` surface.

- Image id: `{run_dir.name}`
- Source: `{summary.get("source", "")}`
- Predictions: `{predictions_path}`
- Prediction shape: `{list(predictions.shape)}`
- Aggregation: `{aggregate}`
- Output: `{out_path.name}`

Values are TRIBEv2 predicted average-subject cortical responses, not regression
statistics and not MNI voxels.
"""
    (out_dir / f"README_prediction_{label}.md").write_text(notes, encoding="utf-8")
    return out_path


def resolve_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir:
        return [path.resolve() for path in args.run_dir]

    if args.image_id:
        return [(args.responses_dir / image_id).resolve() for image_id in args.image_id]

    run_dirs = sorted(p.resolve() for p in args.responses_dir.iterdir() if (p / "predictions.npy").exists())
    if args.limit is not None:
        run_dirs = run_dirs[: args.limit]
    return run_dirs


def run(args: argparse.Namespace) -> list[Path]:
    surfaces = load_surfaces(args.fsaverage5_dir, args.surface)
    run_dirs = resolve_run_dirs(args)
    if not run_dirs:
        raise RuntimeError("No image prediction run folders found.")

    outputs = []
    for index, run_dir in enumerate(run_dirs, start=1):
        print(f"[{index}/{len(run_dirs)}] {run_dir}")
        outputs.append(
            plot_prediction(
                run_dir=run_dir,
                surfaces=surfaces,
                aggregate=args.aggregate,
                timepoint=args.timepoint,
                percentile=args.percentile,
                surface=args.surface,
            )
        )
        print(f"  wrote {outputs[-1]}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TRIBEv2 per-image predictions on fsaverage5.")
    parser.add_argument("--run-dir", type=Path, action="append", default=None, help="Image output folder with predictions.npy.")
    parser.add_argument("--responses-dir", type=Path, default=DEFAULT_RESPONSES_DIR)
    parser.add_argument("--image-id", nargs="*", default=None, help="Image ids under --responses-dir.")
    parser.add_argument("--fsaverage5-dir", type=Path, default=DEFAULT_FSAVERAGE5_DIR)
    parser.add_argument("--surface", choices=["inflated", "pial", "white"], default="inflated")
    parser.add_argument("--aggregate", choices=["mean", "first", "last", "timepoint"], default="mean")
    parser.add_argument("--timepoint", type=int, default=None)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of folders when plotting all runs.")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
