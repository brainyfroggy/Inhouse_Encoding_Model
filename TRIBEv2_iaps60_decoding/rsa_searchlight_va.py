from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.freesurfer import read_geometry
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import rankdata

EXPORT_SOURCE = Path(__file__).resolve().parents[1] / "TRIBEv2_encoding_regression" / "src"
sys.path.insert(0, str(EXPORT_SOURCE))
from export_surface_to_mni152_nifti import (
    make_projection_coords,
    rasterize_vertices,
    read_nifti_header,
    save_like_template,
    vertex_voxels,
)


N_HEMI_VERTICES = 10242
N_IMAGES = 60
CANONICAL_TYPES = ("Pl", "Nt", "Up")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_unit(vector: np.ndarray) -> np.ndarray:
    ranks = rankdata(np.asarray(vector, dtype=np.float64), method="average")
    ranks -= ranks.mean()
    norm = np.linalg.norm(ranks)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Cannot rank-normalize a constant or non-finite RDM.")
    return (ranks / norm).astype(np.float32)


def read_cortex_vertices(path: Path) -> np.ndarray:
    vertices = np.loadtxt(path, skiprows=2, usecols=0, dtype=np.int64)
    if vertices.ndim != 1 or vertices.size == 0:
        raise ValueError(f"Could not read cortex vertices from {path}")
    return vertices


def mesh_graph(coords: np.ndarray, faces: np.ndarray) -> coo_matrix:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    lengths = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    values = np.concatenate((lengths, lengths))
    return coo_matrix((values, (rows, cols)), shape=(coords.shape[0], coords.shape[0])).tocsr()


def build_surface_searchlights(
    fsaverage5_dir: Path,
    radius_mm: float,
    dijkstra_batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers_all: list[np.ndarray] = []
    neighborhoods: list[np.ndarray] = []
    for hemi_index, hemi in enumerate(("lh", "rh")):
        white, faces = read_geometry(fsaverage5_dir / "surf" / f"{hemi}.white")
        pial, pial_faces = read_geometry(fsaverage5_dir / "surf" / f"{hemi}.pial")
        if not np.array_equal(faces, pial_faces):
            raise ValueError(f"{hemi} white and pial meshes do not share faces")
        midgray = (white + pial) / 2.0
        graph = mesh_graph(midgray, faces)
        cortex = read_cortex_vertices(fsaverage5_dir / "label" / f"{hemi}.cortex.label")
        cortex_mask = np.zeros(midgray.shape[0], dtype=bool)
        cortex_mask[cortex] = True
        offset = hemi_index * N_HEMI_VERTICES

        for start in range(0, cortex.size, dijkstra_batch):
            batch_centers = cortex[start : start + dijkstra_batch]
            distances = dijkstra(graph, indices=batch_centers, directed=False, limit=radius_mm)
            for row_index, center in enumerate(batch_centers):
                local = np.flatnonzero(np.isfinite(distances[row_index]) & cortex_mask)
                if local.size < 3:
                    raise ValueError(f"Searchlight at {hemi}:{center} contains only {local.size} vertices")
                centers_all.append(np.asarray(center + offset, dtype=np.int32))
                neighborhoods.append((local + offset).astype(np.int32))

    centers = np.asarray(centers_all, dtype=np.int32)
    offsets = np.zeros(centers.size + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([neighbor.size for neighbor in neighborhoods])
    indices = np.concatenate(neighborhoods).astype(np.int32)
    return centers, offsets, indices


def load_and_validate_inputs(
    response_path: Path,
    manifest_path: Path,
    behavioral_rdm_path: Path,
    validate_manifest_rdm: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    responses = np.load(response_path)
    manifest = pd.read_csv(manifest_path)
    if responses.shape != (N_IMAGES, 2 * N_HEMI_VERTICES):
        raise ValueError(f"Expected response shape (60, 20484), found {responses.shape}")
    if manifest.shape[0] != N_IMAGES:
        raise ValueError(f"Expected 60 manifest rows, found {manifest.shape[0]}")
    required = {"image_id", "type", "stim_order_within_type", "mean_valence", "mean_arousal"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    canonical_rows = []
    for condition in CANONICAL_TYPES:
        group = manifest[manifest["type"] == condition].sort_values("stim_order_within_type")
        if group.shape[0] != 20 or group["stim_order_within_type"].tolist() != list(range(1, 21)):
            raise ValueError(f"Condition {condition} does not contain exactly stim_order 1..20")
        canonical_rows.extend(group.index.tolist())
    canonical_rows_array = np.asarray(canonical_rows, dtype=np.int64)
    canonical_manifest = manifest.loc[canonical_rows_array].reset_index(drop=True)
    canonical_responses = np.asarray(responses[canonical_rows_array], dtype=np.float32)

    saved_behavioral = pd.read_csv(behavioral_rdm_path, index_col=0).to_numpy(dtype=np.float64)
    if saved_behavioral.shape != (N_IMAGES, N_IMAGES):
        raise ValueError(f"Expected a 60x60 behavioral RDM, found {saved_behavioral.shape}")
    if not np.all(np.isfinite(saved_behavioral)):
        raise ValueError(f"Behavioral RDM contains non-finite values: {behavioral_rdm_path}")
    if not np.allclose(saved_behavioral, saved_behavioral.T, atol=1e-8):
        raise ValueError(f"Behavioral RDM is not symmetric: {behavioral_rdm_path}")
    if not np.allclose(np.diag(saved_behavioral), 0.0, atol=1e-8):
        raise ValueError(f"Behavioral RDM diagonal is not zero: {behavioral_rdm_path}")
    if validate_manifest_rdm:
        va = canonical_manifest[["mean_valence", "mean_arousal"]].to_numpy(dtype=np.float64)
        delta = va[:, None, :] - va[None, :, :]
        reconstructed = np.sqrt(np.sum(delta * delta, axis=2))
        reconstructed /= reconstructed.max()
        max_error = float(np.max(np.abs(reconstructed - saved_behavioral)))
        if max_error > 1e-8:
            raise ValueError(f"Manifest-derived valence/arousal RDM does not match saved RDM; max error={max_error}")
    return canonical_responses, saved_behavioral, canonical_manifest, canonical_rows_array


def compute_searchlight_ranks(
    responses: np.ndarray,
    centers: np.ndarray,
    offsets: np.ndarray,
    neighbor_indices: np.ndarray,
    output_path: Path,
) -> np.ndarray:
    triangle = np.triu_indices(N_IMAGES, 1)
    n_pairs = triangle[0].size
    ranks = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(centers.size, n_pairs),
    )
    for searchlight_index in range(centers.size):
        neighborhood = neighbor_indices[offsets[searchlight_index] : offsets[searchlight_index + 1]]
        patterns = responses[:, neighborhood].astype(np.float64, copy=False)
        centered = patterns - patterns.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1)
        if np.any(norms <= np.finfo(np.float64).eps):
            raise ValueError(f"Constant image pattern in searchlight centered at vertex {centers[searchlight_index]}")
        normalized = centered / norms[:, None]
        correlation = np.clip(normalized @ normalized.T, -1.0, 1.0)
        neural_rdm = 1.0 - correlation
        ranks[searchlight_index] = rank_unit(neural_rdm[triangle])
        if (searchlight_index + 1) % 1000 == 0 or searchlight_index + 1 == centers.size:
            print(f"Computed neural RDM ranks: {searchlight_index + 1}/{centers.size}", flush=True)
    ranks.flush()
    return np.load(output_path, mmap_mode="r")


def permutation_inference(
    neural_rank_units: np.ndarray,
    behavioral_rdm: np.ndarray,
    n_permutations: int,
    seed: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    triangle = np.triu_indices(N_IMAGES, 1)
    model_unit = rank_unit(behavioral_rdm[triangle])
    observed = np.asarray(neural_rank_units @ model_unit, dtype=np.float64)
    rng = np.random.default_rng(seed)

    pointwise_positive_count = np.zeros(observed.size, dtype=np.int64)
    pointwise_two_sided_count = np.zeros(observed.size, dtype=np.int64)
    fwer_positive_count = np.zeros(observed.size, dtype=np.int64)
    fwer_two_sided_count = np.zeros(observed.size, dtype=np.int64)
    max_positive_null = np.empty(n_permutations, dtype=np.float32)
    max_absolute_null = np.empty(n_permutations, dtype=np.float32)

    completed = 0
    while completed < n_permutations:
        current = min(batch_size, n_permutations - completed)
        permuted_models = np.empty((triangle[0].size, current), dtype=np.float32)
        for column in range(current):
            permutation = rng.permutation(N_IMAGES)
            permuted = behavioral_rdm[np.ix_(permutation, permutation)]
            permuted_models[:, column] = rank_unit(permuted[triangle])
        scores = np.asarray(neural_rank_units @ permuted_models, dtype=np.float32)
        batch_max_positive = scores.max(axis=0)
        batch_max_absolute = np.abs(scores).max(axis=0)
        max_positive_null[completed : completed + current] = batch_max_positive
        max_absolute_null[completed : completed + current] = batch_max_absolute

        pointwise_positive_count += np.sum(scores >= observed[:, None], axis=1)
        pointwise_two_sided_count += np.sum(np.abs(scores) >= np.abs(observed[:, None]), axis=1)
        fwer_positive_count += np.sum(batch_max_positive[None, :] >= observed[:, None], axis=1)
        fwer_two_sided_count += np.sum(batch_max_absolute[None, :] >= np.abs(observed[:, None]), axis=1)
        completed += current
        print(f"Permutation inference: {completed}/{n_permutations}", flush=True)

    denominator = float(n_permutations + 1)
    return {
        "rho": observed.astype(np.float32),
        "fisher_z": np.arctanh(np.clip(observed, -0.999999, 0.999999)).astype(np.float32),
        "p_uncorrected_positive": ((pointwise_positive_count + 1) / denominator).astype(np.float32),
        "p_uncorrected_two_sided": ((pointwise_two_sided_count + 1) / denominator).astype(np.float32),
        "p_fwer_positive": ((fwer_positive_count + 1) / denominator).astype(np.float32),
        "p_fwer_two_sided": ((fwer_two_sided_count + 1) / denominator).astype(np.float32),
        "max_positive_null": max_positive_null,
        "max_absolute_null": max_absolute_null,
    }


def expand_to_surface(center_values: np.ndarray, centers: np.ndarray, fill: float = np.nan) -> np.ndarray:
    surface = np.full(2 * N_HEMI_VERTICES, fill, dtype=np.float32)
    surface[centers] = np.asarray(center_values, dtype=np.float32)
    return surface


def export_nifti_maps(
    output_dir: Path,
    fsaverage5_dir: Path,
    template_path: Path,
    surface_maps: dict[str, np.ndarray],
    ribbon_steps: int,
    radius_vox: int,
) -> tuple[list[str], dict[str, object]]:
    template_header, template_shape, affine, _ = read_nifti_header(template_path)
    coords, projection_vertex_indices = make_projection_coords(
        fsaverage5_dir,
        mode="ribbon",
        ribbon_steps=ribbon_steps,
    )
    ijk, in_bounds = vertex_voxels(coords, affine, template_shape)
    written = []
    for filename, surface_values in surface_maps.items():
        projection_values = np.nan_to_num(surface_values[projection_vertex_indices], nan=0.0)
        volume = rasterize_vertices(projection_values, ijk, in_bounds, template_shape, radius_vox=radius_vox)
        save_like_template(volume, output_dir / filename, template_header, filename.replace(".nii.gz", ""))
        written.append(filename)
    projection = {
        "template": str(template_path.resolve()),
        "template_shape": list(template_shape),
        "template_affine": affine.tolist(),
        "projection_mode": "white-to-pial ribbon",
        "ribbon_steps": ribbon_steps,
        "rasterization_radius_vox": radius_vox,
        "projection_points": int(coords.shape[0]),
        "projection_points_in_bounds": int(in_bounds.sum()),
        "note": "Surface statistics rasterized into MNI152 solely for MRIcroGL display; not a volumetric voxelwise analysis.",
    }
    return written, projection


def run(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    responses, behavioral_rdm, manifest, response_reorder = load_and_validate_inputs(
        args.response,
        args.manifest,
        args.behavioral_rdm,
        not args.skip_manifest_rdm_check,
    )
    manifest.to_csv(args.output_dir / "iaps60_rsa_order.csv", index=False)
    np.save(args.output_dir / "response_row_reorder.npy", response_reorder)
    np.save(args.output_dir / "behavioral_va_rdm.npy", behavioral_rdm.astype(np.float32))

    centers_path = args.output_dir / "surface_searchlight_neighborhoods.npz"
    if centers_path.exists() and not args.rebuild_neighborhoods:
        saved = np.load(centers_path)
        centers = saved["centers"]
        offsets = saved["offsets"]
        neighbor_indices = saved["neighbor_indices"]
        saved_radius = float(saved["radius_mm"])
        if saved_radius != args.radius_mm:
            raise ValueError(f"Saved searchlight radius is {saved_radius}, requested {args.radius_mm}")
    else:
        centers, offsets, neighbor_indices = build_surface_searchlights(
            args.fsaverage5_dir,
            args.radius_mm,
            args.dijkstra_batch,
        )
        np.savez_compressed(
            centers_path,
            centers=centers,
            offsets=offsets,
            neighbor_indices=neighbor_indices,
            radius_mm=np.asarray(args.radius_mm),
        )
    neighborhood_sizes = np.diff(offsets)
    print(
        f"Searchlights: {centers.size}; neighborhood vertices min/mean/max "
        f"{neighborhood_sizes.min()}/{neighborhood_sizes.mean():.1f}/{neighborhood_sizes.max()}",
        flush=True,
    )

    neural_rank_path = args.output_dir / "neural_rdm_rank_units.npy"
    neural_rank_complete = args.output_dir / "neural_rdm_rank_units.complete"
    if neural_rank_path.exists() and neural_rank_complete.exists() and not args.recompute_rdms:
        neural_rank_units = np.load(neural_rank_path, mmap_mode="r")
        expected_shape = (centers.size, N_IMAGES * (N_IMAGES - 1) // 2)
        if neural_rank_units.shape != expected_shape:
            raise ValueError(f"Saved neural rank array has shape {neural_rank_units.shape}, expected {expected_shape}")
    else:
        neural_rank_units = compute_searchlight_ranks(
            responses,
            centers,
            offsets,
            neighbor_indices,
            neural_rank_path,
        )
        neural_rank_complete.write_text("complete\n", encoding="ascii")

    statistics = permutation_inference(
        neural_rank_units,
        behavioral_rdm,
        args.n_permutations,
        args.seed,
        args.permutation_batch,
    )
    alpha = args.alpha
    significant_positive = (statistics["rho"] > 0) & (statistics["p_fwer_positive"] <= alpha)
    significant_two_sided = statistics["p_fwer_two_sided"] <= alpha
    significant_vertices = centers[significant_positive]
    pd.DataFrame(
        {
            "surface_vertex": significant_vertices,
            "hemisphere": np.where(significant_vertices < N_HEMI_VERTICES, "lh", "rh"),
            "hemisphere_vertex": significant_vertices % N_HEMI_VERTICES,
            "neighborhood_vertices": neighborhood_sizes[significant_positive],
            "spearman_rho": statistics["rho"][significant_positive],
            "fisher_z": statistics["fisher_z"][significant_positive],
            "p_uncorrected_positive": statistics["p_uncorrected_positive"][significant_positive],
            "p_fwer_positive": statistics["p_fwer_positive"][significant_positive],
        }
    ).to_csv(args.output_dir / "significant_vertices_fwer05_positive.csv", index=False)

    surface_statistics = {
        key: expand_to_surface(value, centers)
        for key, value in statistics.items()
        if not key.startswith("max_")
    }
    surface_statistics["significant_fwer_positive"] = expand_to_surface(significant_positive.astype(np.float32), centers, fill=0.0)
    surface_statistics["significant_fwer_two_sided"] = expand_to_surface(significant_two_sided.astype(np.float32), centers, fill=0.0)
    np.savez_compressed(args.output_dir / "rsa_surface_statistics.npz", centers=centers, **surface_statistics)
    np.savez_compressed(
        args.output_dir / "permutation_max_null.npz",
        max_positive=statistics["max_positive_null"],
        max_absolute=statistics["max_absolute_null"],
    )

    rho_surface = surface_statistics["rho"]
    z_surface = surface_statistics["fisher_z"]
    p_positive_surface = surface_statistics["p_fwer_positive"]
    nifti_maps = {
        "rsa_spearman_r_mni152_surface.nii.gz": rho_surface,
        "rsa_fisher_z_mni152_surface.nii.gz": z_surface,
        "rsa_fisher_z_fwer05_positive_mni152_surface.nii.gz": np.where(
            surface_statistics["significant_fwer_positive"] > 0,
            z_surface,
            0.0,
        ),
        "rsa_fisher_z_fwer05_twosided_mni152_surface.nii.gz": np.where(
            surface_statistics["significant_fwer_two_sided"] > 0,
            z_surface,
            0.0,
        ),
        "rsa_neglog10_p_fwer_positive_mni152_surface.nii.gz": -np.log10(
            np.clip(p_positive_surface, 1.0 / (args.n_permutations + 1), 1.0)
        ),
        "rsa_fwer05_positive_mask_mni152_surface.nii.gz": surface_statistics["significant_fwer_positive"],
    }
    written_niftis, projection = export_nifti_maps(
        args.output_dir,
        args.fsaverage5_dir,
        args.template,
        nifti_maps,
        args.ribbon_steps,
        args.radius_vox,
    )

    peak_index = int(np.nanargmax(statistics["rho"]))
    summary = {
        "status": "complete",
        "analysis": "fixed-effect surface searchlight RSA of deterministic TRIBE v2 predictions",
        "behavioral_model": args.behavioral_model_label,
        "response": str(args.response.resolve()),
        "response_sha256": sha256_file(args.response),
        "manifest": str(args.manifest.resolve()),
        "behavioral_rdm": str(args.behavioral_rdm.resolve()),
        "surface_space": "fsaverage5; LH 10242 followed by RH 10242",
        "searchlight": "edge-weighted surface-geodesic neighborhood",
        "radius_mm": args.radius_mm,
        "n_cortical_centers": int(centers.size),
        "neighborhood_vertices_min_mean_max": [
            int(neighborhood_sizes.min()),
            float(neighborhood_sizes.mean()),
            int(neighborhood_sizes.max()),
        ],
        "inference": {
            "unit": "image identity (60 unique IAPS images)",
            "method": "joint row/column image-label permutation of the behavioral RDM",
            "n_permutations": args.n_permutations,
            "seed": args.seed,
            "primary_tail": "positive one-sided",
            "multiple_comparison_control": "maximum Spearman rho across all cortical searchlight centers; FWER",
            "alpha": alpha,
            "second_level_note": "No subject-level second-level model is valid because TRIBE provides one deterministic population-average map per image.",
        },
        "peak": {
            "surface_vertex": int(centers[peak_index]),
            "hemisphere": "lh" if centers[peak_index] < N_HEMI_VERTICES else "rh",
            "hemisphere_vertex": int(centers[peak_index] % N_HEMI_VERTICES),
            "spearman_rho": float(statistics["rho"][peak_index]),
            "p_fwer_positive": float(statistics["p_fwer_positive"][peak_index]),
        },
        "n_significant_positive_fwer": int(significant_positive.sum()),
        "n_significant_two_sided_fwer": int(significant_two_sided.sum()),
        "nifti_files": written_niftis,
        "nifti_projection": projection,
    }
    (args.output_dir / "rsa_searchlight_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme = f"""# Synthetic TRIBE v2 valence-arousal searchlight RSA

The neural data are the 60 deterministic TRIBE v2 fsaverage5 response maps. The
behavioral model is: {args.behavioral_model_label}. Searchlights are {args.radius_mm:g}-mm edge-weighted geodesic neighborhoods
on the fsaverage5 midgray surface; neural dissimilarity is correlation distance;
RSA similarity is Spearman rho.

## Statistical inference

A human-subject second-level t test is not available for these synthetic data:
there is only one population-average prediction per image. Treating vertices,
time samples, or resampled predictions as subjects would be pseudoreplication.
Instead, the null distribution jointly permutes the behavioral RDM's rows and
columns across the 60 image identities ({args.n_permutations} permutations). The
primary corrected p-value uses each permutation's maximum positive rho over all
{centers.size} cortical centers, controlling whole-surface FWER at alpha={alpha:g}.
A two-sided maximum-absolute-rho result is also saved.

## MRIcroGL

Start with `rsa_fisher_z_fwer05_positive_mni152_surface.nii.gz`. Empty output is a
valid result when no vertex survives whole-surface FWER. The NIfTIs rasterize the
surface map through the fsaverage5 white–pial ribbon into MRIcroGL's MNI152 grid;
they are display volumes, not a volumetric statistical reanalysis.

`significant_vertices_fwer05_positive.csv` contains the corresponding fsaverage5
vertex IDs, effect sizes, and corrected p-values.

Significant positive vertices: {int(significant_positive.sum())}

Peak rho: {float(statistics['rho'][peak_index]):.4f}, positive FWER p={float(statistics['p_fwer_positive'][peak_index]):.6f},
{summary['peak']['hemisphere']} vertex {summary['peak']['hemisphere_vertex']}.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    return args.output_dir


def parse_args() -> argparse.Namespace:
    project = Path(r"N:\Experimental_Data\yujunchen\projects\Inhouse_Encoding_Model")
    iaps_searchlight = Path(r"N:\Experimental_Data\yujunchen\projects\IAPS_Searchlight")
    fsaverage5 = Path(r"N:\Experimental_Data\yujunchen\projects\data\NSD_atlas\nsddata\freesurfer\fsaverage5")
    output_root = project / "outputs" / "iaps60_meta_static_3s"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, default=output_root / "encoding" / "iaps60_tribev2_first_tr.npy")
    parser.add_argument("--manifest", type=Path, default=output_root / "iaps60_unique_manifest.csv")
    parser.add_argument("--behavioral-rdm", type=Path, default=iaps_searchlight / "outputs" / "rdms" / "rdm_behavioral_normed.csv")
    parser.add_argument(
        "--behavioral-model-label",
        default="Euclidean distance in 2D mean-valence/mean-arousal space; Spearman RSA",
    )
    parser.add_argument("--fsaverage5-dir", type=Path, default=fsaverage5)
    parser.add_argument("--template", type=Path, default=Path(r"C:\MRIcroGL\Resources\standard\mni152.nii.gz"))
    parser.add_argument("--output-dir", type=Path, default=output_root / "rsa_searchlight_va")
    parser.add_argument("--radius-mm", type=float, default=15.0)
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--dijkstra-batch", type=int, default=128)
    parser.add_argument("--permutation-batch", type=int, default=100)
    parser.add_argument("--ribbon-steps", type=int, default=7)
    parser.add_argument("--radius-vox", type=int, default=1)
    parser.add_argument("--rebuild-neighborhoods", action="store_true")
    parser.add_argument("--recompute-rdms", action="store_true")
    parser.add_argument(
        "--skip-manifest-rdm-check",
        action="store_true",
        help="Allow externally prepared 60x60 RDMs, e.g. z-scored valence/arousal, after basic matrix validation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
