from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from rsa_searchlight_va import (
    N_HEMI_VERTICES,
    N_IMAGES,
    expand_to_surface,
    export_nifti_maps,
    rank_unit,
)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    ordered = p_values[order]
    ranks = np.arange(1, p_values.size + 1, dtype=np.float64)
    adjusted = ordered * p_values.size / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q_values = np.empty_like(adjusted)
    q_values[order] = np.clip(adjusted, 0.0, 1.0)
    return q_values


def delete_one_image_correlations(
    neural_rank_units: np.ndarray,
    behavioral_rank_unit: np.ndarray,
    full_rho: np.ndarray,
) -> np.ndarray:
    """Delete all RDM pairs involving each image and correlate retained full ranks.

    Full-sample ranks are retained when a stimulus is deleted. This makes the
    jackknife influence calculation deterministic and avoids introducing a
    different rank scale in every leave-one-image-out sample.
    """
    pair_left, pair_right = np.triu_indices(N_IMAGES, 1)
    n_pairs = pair_left.size
    leave_one_rho = np.empty((full_rho.size, N_IMAGES), dtype=np.float32)
    for image_index in range(N_IMAGES):
        removed = np.flatnonzero((pair_left == image_index) | (pair_right == image_index))
        n_retained = n_pairs - removed.size
        removed_x = np.asarray(neural_rank_units[:, removed], dtype=np.float64)
        removed_y = np.asarray(behavioral_rank_unit[removed], dtype=np.float64)

        # Full rank-unit vectors have sum 0 and sum of squares 1.
        sum_x = -removed_x.sum(axis=1)
        sum_xx = 1.0 - np.square(removed_x).sum(axis=1)
        sum_y = -removed_y.sum()
        sum_yy = 1.0 - np.square(removed_y).sum()
        sum_xy = full_rho - removed_x @ removed_y
        covariance = sum_xy - (sum_x * sum_y / n_retained)
        variance_x = sum_xx - np.square(sum_x) / n_retained
        variance_y = sum_yy - (sum_y * sum_y / n_retained)
        denominator = np.sqrt(np.maximum(variance_x, 0.0) * max(variance_y, 0.0))
        leave_one_rho[:, image_index] = np.clip(covariance / denominator, -0.999999, 0.999999)
    return leave_one_rho


def jackknife_second_level(
    neural_rank_units: np.ndarray,
    behavioral_rdm: np.ndarray,
) -> dict[str, np.ndarray]:
    triangle = np.triu_indices(N_IMAGES, 1)
    behavioral_rank_unit = rank_unit(behavioral_rdm[triangle])
    full_rho = np.asarray(neural_rank_units @ behavioral_rank_unit, dtype=np.float64)
    full_fisher_z = np.arctanh(np.clip(full_rho, -0.999999, 0.999999))
    leave_one_rho = delete_one_image_correlations(neural_rank_units, behavioral_rank_unit, full_rho)
    leave_one_z = np.arctanh(leave_one_rho.astype(np.float64))

    # Delete-one jackknife pseudovalues put the 60 stimulus influence estimates
    # on a common Fisher-z scale, analogous to subject-level Fisher-z maps.
    pseudovalues = N_IMAGES * full_fisher_z[:, None] - (N_IMAGES - 1) * leave_one_z
    jackknife_mean_z = pseudovalues.mean(axis=1)
    jackknife_se = pseudovalues.std(axis=1, ddof=1) / np.sqrt(N_IMAGES)
    t_value = np.divide(
        jackknife_mean_z,
        jackknife_se,
        out=np.zeros_like(jackknife_mean_z),
        where=jackknife_se > 0,
    )
    degrees_of_freedom = N_IMAGES - 1
    p_positive = student_t.sf(t_value, df=degrees_of_freedom)
    p_two_sided = student_t.sf(np.abs(t_value), df=degrees_of_freedom) * 2.0
    q_positive = benjamini_hochberg(p_positive)
    q_two_sided = benjamini_hochberg(p_two_sided)
    return {
        "full_spearman_rho": full_rho.astype(np.float32),
        "full_fisher_z": full_fisher_z.astype(np.float32),
        "jackknife_mean_fisher_z": jackknife_mean_z.astype(np.float32),
        "jackknife_se_fisher_z": jackknife_se.astype(np.float32),
        "t_value": t_value.astype(np.float32),
        "p_positive": p_positive.astype(np.float32),
        "p_two_sided": p_two_sided.astype(np.float32),
        "q_fdr_positive": q_positive.astype(np.float32),
        "q_fdr_two_sided": q_two_sided.astype(np.float32),
    }


def run(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    neighborhoods = np.load(args.searchlight_dir / "surface_searchlight_neighborhoods.npz")
    centers = neighborhoods["centers"]
    neighborhood_sizes = np.diff(neighborhoods["offsets"])
    neural_rank_units = np.load(args.searchlight_dir / "neural_rdm_rank_units.npy", mmap_mode="r")
    behavioral_rdm = np.load(args.searchlight_dir / "behavioral_va_rdm.npy")
    expected_shape = (centers.size, N_IMAGES * (N_IMAGES - 1) // 2)
    if neural_rank_units.shape != expected_shape:
        raise ValueError(f"Expected neural rank shape {expected_shape}, found {neural_rank_units.shape}")

    statistics = jackknife_second_level(neural_rank_units, behavioral_rdm)
    significant_positive = (statistics["t_value"] > 0) & (statistics["q_fdr_positive"] <= args.alpha)
    significant_two_sided = statistics["q_fdr_two_sided"] <= args.alpha
    exploratory_p001 = (statistics["t_value"] > 0) & (statistics["p_positive"] < 0.001)

    surface_statistics = {key: expand_to_surface(value, centers) for key, value in statistics.items()}
    surface_statistics["significant_fdr_positive"] = expand_to_surface(
        significant_positive.astype(np.float32), centers, fill=0.0
    )
    surface_statistics["significant_fdr_two_sided"] = expand_to_surface(
        significant_two_sided.astype(np.float32), centers, fill=0.0
    )
    surface_statistics["exploratory_uncorrected_p001_positive"] = expand_to_surface(
        exploratory_p001.astype(np.float32), centers, fill=0.0
    )
    np.savez_compressed(args.output_dir / "rsa_jackknife_t_surface_statistics.npz", centers=centers, **surface_statistics)

    table = pd.DataFrame(
        {
            "surface_vertex": centers,
            "hemisphere": np.where(centers < N_HEMI_VERTICES, "lh", "rh"),
            "hemisphere_vertex": centers % N_HEMI_VERTICES,
            "neighborhood_vertices": neighborhood_sizes,
            **statistics,
            "significant_fdr05_positive": significant_positive,
            "significant_fdr05_two_sided": significant_two_sided,
            "exploratory_uncorrected_p001_positive": exploratory_p001,
        }
    )
    table.to_csv(args.output_dir / "rsa_jackknife_t_all_vertices.csv", index=False)
    table[significant_positive].to_csv(args.output_dir / "significant_vertices_fdr05_positive.csv", index=False)

    t_surface = surface_statistics["t_value"]
    q_surface = surface_statistics["q_fdr_positive"]
    p_surface = surface_statistics["p_positive"]
    nifti_maps = {
        "rsa_jackknife_t_unthresholded_mni152_surface.nii.gz": t_surface,
        "rsa_jackknife_t_fdr05_positive_mni152_surface.nii.gz": np.where(
            surface_statistics["significant_fdr_positive"] > 0, t_surface, 0.0
        ),
        "rsa_jackknife_t_fdr05_twosided_mni152_surface.nii.gz": np.where(
            surface_statistics["significant_fdr_two_sided"] > 0, t_surface, 0.0
        ),
        "rsa_jackknife_t_uncorrected_p001_positive_mni152_surface.nii.gz": np.where(
            surface_statistics["exploratory_uncorrected_p001_positive"] > 0, t_surface, 0.0
        ),
        "rsa_jackknife_neglog10_p_positive_mni152_surface.nii.gz": -np.log10(
            np.clip(p_surface, np.finfo(np.float32).tiny, 1.0)
        ),
        "rsa_jackknife_neglog10_q_fdr_positive_mni152_surface.nii.gz": -np.log10(
            np.clip(q_surface, np.finfo(np.float32).tiny, 1.0)
        ),
        "rsa_jackknife_fdr05_positive_mask_mni152_surface.nii.gz": surface_statistics[
            "significant_fdr_positive"
        ],
    }
    written_niftis, projection = export_nifti_maps(
        args.output_dir,
        args.fsaverage5_dir,
        args.template,
        nifti_maps,
        args.ribbon_steps,
        args.radius_vox,
    )

    peak_index = int(np.argmax(statistics["t_value"]))
    summary = {
        "status": "complete",
        "analysis": "stimulus-jackknife second-level analogue for deterministic TRIBE v2 searchlight RSA",
        "first_level": "Spearman correlation between neural and valence/arousal RDMs; Fisher z transformed",
        "second_level_analogue": "delete-one-image jackknife Fisher-z pseudovalues; one-sample t over 60 stimulus pseudovalues",
        "rank_note": "Leave-one-image correlations retain the full-sample pair ranks.",
        "degrees_of_freedom": N_IMAGES - 1,
        "tail": "positive one-sided primary; two-sided also saved",
        "multiple_comparison_control": "Benjamini-Hochberg FDR across 18,715 cortical searchlight centers",
        "alpha": args.alpha,
        "n_significant_fdr05_positive": int(significant_positive.sum()),
        "n_significant_fdr05_two_sided": int(significant_two_sided.sum()),
        "n_exploratory_uncorrected_p001_positive": int(exploratory_p001.sum()),
        "peak": {
            "surface_vertex": int(centers[peak_index]),
            "hemisphere": "lh" if centers[peak_index] < N_HEMI_VERTICES else "rh",
            "hemisphere_vertex": int(centers[peak_index] % N_HEMI_VERTICES),
            "t_value": float(statistics["t_value"][peak_index]),
            "p_positive": float(statistics["p_positive"][peak_index]),
            "q_fdr_positive": float(statistics["q_fdr_positive"][peak_index]),
            "full_spearman_rho": float(statistics["full_spearman_rho"][peak_index]),
        },
        "checkpoint_constraint": (
            "The downloaded TRIBE v2 checkpoint has one averaged subject layer (subject dimension=1), "
            "so these are stimulus pseudovalues, not independent human-subject maps."
        ),
        "nifti_files": written_niftis,
        "nifti_projection": projection,
    }
    (args.output_dir / "rsa_jackknife_t_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme = f"""# TRIBE v2 valence–arousal RSA: jackknife t maps

This reproduces the presentation logic of the notebook—Fisher-z first-level RSA
values followed by a one-sample t statistic—but adapts the second level to the
data actually available. The checkpoint contains one average-subject prediction,
not independent subjects. The 60 second-level observations are delete-one-image
jackknife pseudovalues on the Fisher-z scale (df=59), so the inferential population
is the stimulus set, not people.

Positive one-sided p-values were BH-FDR corrected across all {centers.size}
cortical searchlight centers at q<={args.alpha:g}.

Significant positive centers: {int(significant_positive.sum())}

Peak: t={float(statistics['t_value'][peak_index]):.4f}, p={float(statistics['p_positive'][peak_index]):.6g},
q={float(statistics['q_fdr_positive'][peak_index]):.6g}, full rho={float(statistics['full_spearman_rho'][peak_index]):.4f}.

For MRIcroGL, use `rsa_jackknife_t_fdr05_positive_mni152_surface.nii.gz`.
If it is empty, no searchlight survived FDR. The unthresholded t map is
`rsa_jackknife_t_unthresholded_mni152_surface.nii.gz`. The p<.001 map is explicitly
uncorrected and exploratory; it must not be described as multiple-comparison-corrected.

All NIfTIs are surface values rasterized into the MRIcroGL MNI152 display grid,
not a volumetric reanalysis.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    return args.output_dir


def parse_args() -> argparse.Namespace:
    project = Path(r"N:\Experimental_Data\yujunchen\projects\Inhouse_Encoding_Model")
    output_root = project / "outputs" / "iaps60_meta_static_3s"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--searchlight-dir", type=Path, default=output_root / "rsa_searchlight_va")
    parser.add_argument("--output-dir", type=Path, default=output_root / "rsa_searchlight_va_jackknife_t")
    parser.add_argument(
        "--fsaverage5-dir",
        type=Path,
        default=Path(r"N:\Experimental_Data\yujunchen\projects\data\NSD_atlas\nsddata\freesurfer\fsaverage5"),
    )
    parser.add_argument("--template", type=Path, default=Path(r"C:\MRIcroGL\Resources\standard\mni152.nii.gz"))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--ribbon-steps", type=int, default=7)
    parser.add_argument("--radius-vox", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
