from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from nibabel.freesurfer import read_geometry, read_morph_data, write_morph_data
from nilearn import datasets, surface
from scipy.stats import t as student_t


N_HEMI_VERTICES = 10242
N_VERTICES = 2 * N_HEMI_VERTICES
PREDICTORS = ("valence", "arousal")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    adjusted = p_values[order] * p_values.size / np.arange(1, p_values.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q_values = np.empty_like(adjusted)
    q_values[order] = np.clip(adjusted, 0.0, 1.0)
    return q_values


def load_cortex_mask(fsaverage5_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    mask = np.zeros(N_VERTICES, dtype=bool)
    hemispheres: dict[str, np.ndarray] = {}
    for hemi, offset in (("lh", 0), ("rh", N_HEMI_VERTICES)):
        vertices = np.loadtxt(
            fsaverage5_dir / "label" / f"{hemi}.cortex.label",
            skiprows=2,
            usecols=0,
            dtype=np.int64,
        )
        hemispheres[hemi] = vertices
        mask[vertices + offset] = True
    return mask, hemispheres


def fit_regression(
    responses: np.ndarray,
    manifest: pd.DataFrame,
    cortex_mask: np.ndarray,
    alpha: float,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, object]]:
    ratings = manifest[["mean_valence", "mean_arousal"]].to_numpy(dtype=np.float64)
    rating_mean = ratings.mean(axis=0)
    rating_sd = ratings.std(axis=0, ddof=1)
    standardized = (ratings - rating_mean) / rating_sd
    design = np.column_stack((np.ones(responses.shape[0]), standardized))
    coefficient_names = ("intercept",) + PREDICTORS
    xtx_inverse = np.linalg.inv(design.T @ design)
    beta = xtx_inverse @ design.T @ responses
    residuals = responses - design @ beta
    residual_df = responses.shape[0] - design.shape[1]
    residual_variance = np.sum(np.square(residuals), axis=0) / residual_df
    se = np.sqrt(np.diag(xtx_inverse)[:, None] * residual_variance[None, :])
    t_value = beta / se
    p_value = 2.0 * student_t.sf(np.abs(t_value[1:]), df=residual_df)

    q_value = np.ones_like(p_value)
    fdr_mask = np.zeros_like(p_value, dtype=bool)
    for predictor_index in range(len(PREDICTORS)):
        q_value[predictor_index, cortex_mask] = bh_fdr(p_value[predictor_index, cortex_mask])
        fdr_mask[predictor_index, cortex_mask] = q_value[predictor_index, cortex_mask] <= alpha

    design_table = manifest[
        ["image_id", "type", "stim_order_within_type", "mean_valence", "mean_arousal"]
    ].copy()
    design_table["valence_z"] = standardized[:, 0]
    design_table["arousal_z"] = standardized[:, 1]

    significant_counts = {}
    for predictor_index, predictor in enumerate(PREDICTORS):
        significant = fdr_mask[predictor_index]
        values = t_value[predictor_index + 1]
        significant_counts[predictor] = {
            "positive": int(np.sum(significant & (values > 0))),
            "negative": int(np.sum(significant & (values < 0))),
            "total": int(significant.sum()),
        }
    predictor_r = float(np.corrcoef(standardized.T)[0, 1])
    summary = {
        "analysis": "IAPS-60 TRIBE v2 vertexwise valence/arousal multiple regression",
        "model": "surface_response_vertex ~ intercept + z(mean_valence) + z(mean_arousal)",
        "dependent_variable": "deterministic average-subject TRIBE v2 first-TR fsaverage5 response",
        "n_images": int(responses.shape[0]),
        "n_vertices": int(responses.shape[1]),
        "n_cortical_vertices_tested": int(cortex_mask.sum()),
        "residual_df": int(residual_df),
        "standard_errors": "ordinary least squares across 60 unique images",
        "p_values": "two-sided Student t",
        "multiple_comparison_control": "Benjamini-Hochberg FDR separately for valence and arousal across cortical vertices",
        "fdr_alpha": float(alpha),
        "predictor_correlation": predictor_r,
        "predictor_vif_each": float(1.0 / (1.0 - predictor_r * predictor_r)),
        "rating_mean": dict(zip(PREDICTORS, rating_mean.tolist(), strict=True)),
        "rating_sd": dict(zip(PREDICTORS, rating_sd.tolist(), strict=True)),
        "coefficient_names": list(coefficient_names),
        "significant_counts": significant_counts,
        "t_range_cortex": {
            predictor: [
                float(t_value[index + 1, cortex_mask].min()),
                float(t_value[index + 1, cortex_mask].max()),
            ]
            for index, predictor in enumerate(PREDICTORS)
        },
    }
    arrays = {
        "beta": beta.astype(np.float32),
        "se": se.astype(np.float32),
        "t": t_value.astype(np.float32),
        "p": p_value.astype(np.float32),
        "q": q_value.astype(np.float32),
        "fdr_mask": fdr_mask,
        "residual_variance": residual_variance.astype(np.float32),
    }
    return arrays, design_table, summary


def inflated_face_colors(
    faces: np.ndarray,
    sulc: np.ndarray,
    signed_t: np.ndarray,
    norm: colors.Normalize,
) -> np.ndarray:
    face_sulc = sulc[faces].mean(axis=1)
    rgba = np.empty((faces.shape[0], 4), dtype=np.float64)
    rgba[face_sulc <= 0] = (0.78, 0.78, 0.78, 1.0)
    rgba[face_sulc > 0] = (0.56, 0.56, 0.56, 1.0)
    vertex_values = np.where(signed_t != 0, signed_t, np.nan)
    count = np.sum(np.isfinite(vertex_values[faces]), axis=1)
    mean_value = np.divide(
        np.nansum(vertex_values[faces], axis=1),
        count,
        out=np.full(faces.shape[0], np.nan),
        where=count > 0,
    )
    active = count > 0
    rgba[active] = plt.get_cmap("coolwarm")(norm(mean_value[active]))
    return rgba


def plot_inflated_view(axis, coords, faces, facecolors, hemi: str, view: str, title: str) -> None:
    collection = Poly3DCollection(
        coords[faces],
        facecolors=facecolors,
        edgecolors="none",
        linewidths=0,
        antialiased=True,
    )
    axis.add_collection3d(collection)
    xyz_min, xyz_max = coords.min(axis=0), coords.max(axis=0)
    center = (xyz_min + xyz_max) / 2.0
    extent = float(np.max(xyz_max - xyz_min)) / 2.0
    axis.set_xlim(center[0] - extent, center[0] + extent)
    axis.set_ylim(center[1] - extent, center[1] + extent)
    axis.set_zlim(center[2] - extent, center[2] + extent)
    angles = {
        ("lh", "lateral"): (0, 180),
        ("lh", "medial"): (0, 0),
        ("rh", "medial"): (0, 180),
        ("rh", "lateral"): (0, 0),
    }
    axis.view_init(*angles[(hemi, view)])
    axis.set_title(title, fontsize=11, weight="bold", pad=0)
    axis.set_axis_off()
    try:
        axis.set_box_aspect((1, 1, 1), zoom=1.55)
    except TypeError:
        axis.set_box_aspect((1, 1, 1))


def plot_inflated(
    output_path: Path,
    fsaverage5_dir: Path,
    signed_t: np.ndarray,
    vmax: float,
    alpha: float,
) -> None:
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    surfaces = {}
    for hemi, offset in (("lh", 0), ("rh", N_HEMI_VERTICES)):
        coords, faces = read_geometry(fsaverage5_dir / "surf" / f"{hemi}.inflated")
        sulc = read_morph_data(fsaverage5_dir / "surf" / f"{hemi}.sulc")
        surfaces[hemi] = (coords, faces, sulc, offset)
    views = (("lh", "lateral"), ("lh", "medial"), ("rh", "medial"), ("rh", "lateral"))
    fig = plt.figure(figsize=(19, 12.5), facecolor="white")
    for row, predictor in enumerate(PREDICTORS):
        for column, (hemi, view) in enumerate(views):
            axis = fig.add_subplot(2, 4, row * 4 + column + 1, projection="3d")
            coords, faces, sulc, offset = surfaces[hemi]
            hemi_t = signed_t[row, offset : offset + N_HEMI_VERTICES]
            facecolors = inflated_face_colors(faces, sulc, hemi_t, norm)
            plot_inflated_view(axis, coords, faces, facecolors, hemi, view, f"{predictor.capitalize()} · {hemi.upper()} {view}")
    mappable = cm.ScalarMappable(norm=norm, cmap="coolwarm")
    mappable.set_array([])
    colorbar_axis = fig.add_axes([0.27, 0.045, 0.46, 0.025])
    colorbar = fig.colorbar(mappable, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Signed partial-regression t value", fontsize=12)
    fig.suptitle(
        f"IAPS-60 TRIBE v2 multiple regression on FreeSurfer fsaverage5 · BH-FDR q ≤ {alpha:g}",
        fontsize=17,
        weight="bold",
        y=0.975,
    )
    fig.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.08, wspace=-0.08, hspace=-0.16)
    fig.savefig(output_path, dpi=250, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_flat_hemi(
    axis,
    mesh_path: str,
    sulc_path: str,
    cortex_vertices: np.ndarray,
    signed_t: np.ndarray,
    norm: colors.Normalize,
    label: str,
) -> None:
    mesh = surface.load_surf_mesh(mesh_path)
    coords, faces = np.asarray(mesh.coordinates), np.asarray(mesh.faces)
    sulc = np.asarray(surface.load_surf_data(sulc_path))
    cortex_mask = np.zeros(N_HEMI_VERTICES, dtype=bool)
    cortex_mask[cortex_vertices] = True
    base_tri = mtri.Triangulation(coords[:, 0], coords[:, 1], faces)
    base_tri.set_mask(~np.all(cortex_mask[faces], axis=1))
    axis.tripcolor(
        base_tri,
        (sulc > 0).astype(float),
        shading="flat",
        cmap=ListedColormap(["#929292", "#D0D0D0"]),
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    significant = signed_t != 0
    heat_tri = mtri.Triangulation(coords[:, 0], coords[:, 1], faces)
    heat_tri.set_mask((~np.all(cortex_mask[faces], axis=1)) | (~np.any(significant[faces], axis=1)))
    axis.tripcolor(
        heat_tri,
        signed_t,
        shading="gouraud",
        cmap="coolwarm",
        norm=norm,
        rasterized=True,
    )
    axis.set_aspect("equal")
    axis.set_xlim(coords[:, 0].min(), coords[:, 0].max())
    axis.set_ylim(coords[:, 1].min(), coords[:, 1].max())
    axis.set_axis_off()
    axis.text(0.02, 0.98, label, transform=axis.transAxes, ha="left", va="top", fontsize=13, weight="bold")


def plot_flatmap(
    output_path: Path,
    signed_t: np.ndarray,
    cortex_by_hemi: dict[str, np.ndarray],
    vmax: float,
    alpha: float,
) -> None:
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fsaverage5 = datasets.fetch_surf_fsaverage("fsaverage5")
    fig, axes = plt.subplots(2, 2, figsize=(17, 15), facecolor="white")
    for row, predictor in enumerate(PREDICTORS):
        render_flat_hemi(
            axes[row, 0], fsaverage5.flat_left, fsaverage5.sulc_left, cortex_by_hemi["lh"],
            signed_t[row, :N_HEMI_VERTICES], norm, f"{predictor.capitalize()} · LH"
        )
        render_flat_hemi(
            axes[row, 1], fsaverage5.flat_right, fsaverage5.sulc_right, cortex_by_hemi["rh"],
            signed_t[row, N_HEMI_VERTICES:], norm, f"{predictor.capitalize()} · RH"
        )
    mappable = cm.ScalarMappable(norm=norm, cmap="coolwarm")
    mappable.set_array([])
    colorbar = fig.colorbar(mappable, ax=axes, orientation="horizontal", fraction=0.025, pad=0.025, aspect=55)
    colorbar.set_label("Signed partial-regression t value", fontsize=12)
    fig.suptitle(
        f"IAPS-60 TRIBE v2 valence–arousal multiple regression · FreeSurfer fsaverage5 flatmap · BH-FDR q ≤ {alpha:g}",
        fontsize=17,
        weight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.08, top=0.94, wspace=0.025, hspace=0.015)
    fig.savefig(output_path, dpi=250, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_positive_flat_hemi(
    axis,
    mesh_path: str,
    sulc_path: str,
    cortex_vertices: np.ndarray,
    conjunction_t: np.ndarray,
    norm: colors.Normalize,
    label: str,
) -> None:
    mesh = surface.load_surf_mesh(mesh_path)
    coords, faces = np.asarray(mesh.coordinates), np.asarray(mesh.faces)
    sulc = np.asarray(surface.load_surf_data(sulc_path))
    cortex_mask = np.zeros(N_HEMI_VERTICES, dtype=bool)
    cortex_mask[cortex_vertices] = True
    base_tri = mtri.Triangulation(coords[:, 0], coords[:, 1], faces)
    base_tri.set_mask(~np.all(cortex_mask[faces], axis=1))
    axis.tripcolor(
        base_tri,
        (sulc > 0).astype(float),
        shading="flat",
        cmap=ListedColormap(["#929292", "#D0D0D0"]),
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    active = conjunction_t > 0
    heat_tri = mtri.Triangulation(coords[:, 0], coords[:, 1], faces)
    heat_tri.set_mask((~np.all(cortex_mask[faces], axis=1)) | (~np.any(active[faces], axis=1)))
    axis.tripcolor(
        heat_tri,
        conjunction_t,
        shading="gouraud",
        cmap="inferno",
        norm=norm,
        rasterized=True,
    )
    axis.set_aspect("equal")
    axis.set_xlim(coords[:, 0].min(), coords[:, 0].max())
    axis.set_ylim(coords[:, 1].min(), coords[:, 1].max())
    axis.set_axis_off()
    axis.text(0.02, 0.98, label, transform=axis.transAxes, ha="left", va="top", fontsize=13, weight="bold")


def plot_positive_conjunction_flatmap(
    output_path: Path,
    conjunction_t: np.ndarray,
    cortex_by_hemi: dict[str, np.ndarray],
    vmax: float,
    alpha: float,
) -> None:
    norm = colors.Normalize(vmin=0.0, vmax=vmax)
    fsaverage5 = datasets.fetch_surf_fsaverage("fsaverage5")
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.8), facecolor="white")
    render_positive_flat_hemi(
        axes[0], fsaverage5.flat_left, fsaverage5.sulc_left, cortex_by_hemi["lh"],
        conjunction_t[:N_HEMI_VERTICES], norm, "Positive valence ∩ positive arousal · LH"
    )
    render_positive_flat_hemi(
        axes[1], fsaverage5.flat_right, fsaverage5.sulc_right, cortex_by_hemi["rh"],
        conjunction_t[N_HEMI_VERTICES:], norm, "Positive valence ∩ positive arousal · RH"
    )
    if np.any(conjunction_t > 0):
        mappable = cm.ScalarMappable(norm=norm, cmap="inferno")
        mappable.set_array([])
        colorbar = fig.colorbar(mappable, ax=axes, orientation="horizontal", fraction=0.045, pad=0.035, aspect=55)
        colorbar.set_label("Minimum partial-regression t value: min(t valence, t arousal)", fontsize=12)
    else:
        fig.text(
            0.5,
            0.075,
            "No vertices meet the positive conjunction criterion",
            ha="center",
            va="center",
            fontsize=13,
            weight="bold",
        )
    fig.suptitle(
        f"IAPS-60 TRIBE v2 positive valence–arousal conjunction · FreeSurfer fsaverage5 flatmap · both BH-FDR q ≤ {alpha:g}",
        fontsize=17,
        weight="bold",
        y=0.97,
    )
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.13, top=0.91, wspace=0.025)
    fig.savefig(output_path, dpi=250, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    response_path = args.response.resolve()
    manifest_path = args.manifest.resolve()
    responses = np.load(response_path).astype(np.float64)
    manifest = pd.read_csv(manifest_path)
    if responses.shape != (60, N_VERTICES):
        raise ValueError(f"Expected response shape (60, 20484), found {responses.shape}")
    required = {"image_id", "type", "stim_order_within_type", "mean_valence", "mean_arousal"}
    missing = required.difference(manifest.columns)
    if missing or manifest.shape[0] != 60:
        raise ValueError(f"Invalid manifest; missing={sorted(missing)}, rows={manifest.shape[0]}")
    cortex_mask, cortex_by_hemi = load_cortex_mask(args.fsaverage5_dir)
    arrays, design_table, summary = fit_regression(responses, manifest, cortex_mask, args.fdr_alpha)
    summary["response_path"] = str(response_path)
    summary["response_sha256"] = sha256_file(response_path)
    summary["manifest_path"] = str(manifest_path)

    design_table.to_csv(args.output_dir / "design_iaps60_valence_arousal.csv", index=False)
    np.save(args.output_dir / "beta_intercept_valence_arousal_vertices.npy", arrays["beta"])
    np.save(args.output_dir / "se_intercept_valence_arousal_vertices.npy", arrays["se"])
    np.save(args.output_dir / "t_intercept_valence_arousal_vertices.npy", arrays["t"])
    np.save(args.output_dir / "p_valence_arousal_vertices.npy", arrays["p"])
    np.save(args.output_dir / "q_fdr_valence_arousal_vertices.npy", arrays["q"])
    np.save(args.output_dir / "fdr05_mask_valence_arousal_vertices.npy", arrays["fdr_mask"])
    np.save(args.output_dir / "residual_variance_vertices.npy", arrays["residual_variance"])

    signed_t = np.where(arrays["fdr_mask"], arrays["t"][1:], 0.0).astype(np.float32)
    positive_conjunction_mask = (
        arrays["fdr_mask"][0]
        & arrays["fdr_mask"][1]
        & (arrays["t"][1] > 0)
        & (arrays["t"][2] > 0)
    )
    positive_conjunction_t = np.where(
        positive_conjunction_mask,
        np.minimum(arrays["t"][1], arrays["t"][2]),
        0.0,
    ).astype(np.float32)
    np.save(args.output_dir / "positive_valence_arousal_conjunction_mask_vertices.npy", positive_conjunction_mask)
    np.save(args.output_dir / "positive_valence_arousal_conjunction_min_t_vertices.npy", positive_conjunction_t)
    summary["positive_valence_arousal_conjunction"] = {
        "definition": "q_valence <= alpha and q_arousal <= alpha and t_valence > 0 and t_arousal > 0",
        "display_value": "min(t_valence, t_arousal)",
        "total_vertices": int(positive_conjunction_mask.sum()),
        "lh_vertices": int(positive_conjunction_mask[:N_HEMI_VERTICES].sum()),
        "rh_vertices": int(positive_conjunction_mask[N_HEMI_VERTICES:].sum()),
    }
    overlay_dir = args.output_dir / "freesurfer_overlays"
    overlay_dir.mkdir(exist_ok=True)
    for predictor_index, predictor in enumerate(PREDICTORS):
        for hemi, offset in (("lh", 0), ("rh", N_HEMI_VERTICES)):
            values = signed_t[predictor_index, offset : offset + N_HEMI_VERTICES]
            write_morph_data(overlay_dir / f"{hemi}.{predictor}_t_fdr05.curv", values)

        mask = arrays["fdr_mask"][predictor_index]
        table = pd.DataFrame(
            {
                "surface_vertex": np.flatnonzero(mask),
                "hemisphere": np.where(np.flatnonzero(mask) < N_HEMI_VERTICES, "lh", "rh"),
                "hemisphere_vertex": np.flatnonzero(mask) % N_HEMI_VERTICES,
                "beta": arrays["beta"][predictor_index + 1, mask],
                "t": arrays["t"][predictor_index + 1, mask],
                "p": arrays["p"][predictor_index, mask],
                "q": arrays["q"][predictor_index, mask],
            }
        )
        table.to_csv(args.output_dir / f"significant_{predictor}_vertices_fdr05.csv", index=False)

    for hemi, offset in (("lh", 0), ("rh", N_HEMI_VERTICES)):
        values = positive_conjunction_t[offset : offset + N_HEMI_VERTICES]
        write_morph_data(overlay_dir / f"{hemi}.positive_valence_arousal_conjunction_min_t_fdr05.curv", values)
    conjunction_vertices = np.flatnonzero(positive_conjunction_mask)
    pd.DataFrame(
        {
            "surface_vertex": conjunction_vertices,
            "hemisphere": np.where(conjunction_vertices < N_HEMI_VERTICES, "lh", "rh"),
            "hemisphere_vertex": conjunction_vertices % N_HEMI_VERTICES,
            "beta_valence": arrays["beta"][1, positive_conjunction_mask],
            "beta_arousal": arrays["beta"][2, positive_conjunction_mask],
            "t_valence": arrays["t"][1, positive_conjunction_mask],
            "t_arousal": arrays["t"][2, positive_conjunction_mask],
            "min_t": positive_conjunction_t[positive_conjunction_mask],
            "q_valence": arrays["q"][0, positive_conjunction_mask],
            "q_arousal": arrays["q"][1, positive_conjunction_mask],
        }
    ).to_csv(args.output_dir / "positive_valence_arousal_conjunction_vertices_fdr05.csv", index=False)

    significant_values = np.abs(signed_t[signed_t != 0])
    vmax = float(np.quantile(significant_values, 0.995))
    plot_inflated(
        args.output_dir / "iaps60_va_multiple_regression_fsaverage5_inflated.png",
        args.fsaverage5_dir,
        signed_t,
        vmax,
        args.fdr_alpha,
    )
    plot_flatmap(
        args.output_dir / "iaps60_va_multiple_regression_fsaverage5_flatmap.png",
        signed_t,
        cortex_by_hemi,
        vmax,
        args.fdr_alpha,
    )
    conjunction_values = positive_conjunction_t[positive_conjunction_t > 0]
    conjunction_vmax = float(np.quantile(conjunction_values, 0.995)) if conjunction_values.size else 1.0
    plot_positive_conjunction_flatmap(
        args.output_dir / "iaps60_positive_valence_arousal_conjunction_min_t_fsaverage5_flatmap.png",
        positive_conjunction_t,
        cortex_by_hemi,
        conjunction_vmax,
        args.fdr_alpha,
    )
    (args.output_dir / "regression_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme = f"""# IAPS-60 TRIBE v2 valence/arousal multiple regression

Model: `surface response ~ intercept + z(mean valence) + z(mean arousal)`.

The 60 unique images are the observations. Tests use OLS with {summary['residual_df']}
residual degrees of freedom and two-sided t p-values. BH-FDR q<={args.fdr_alpha:g}
is applied separately to valence and arousal across {summary['n_cortical_vertices_tested']}
fsaverage5 cortex vertices. These are image-level model-characterization statistics,
not human-subject second-level statistics.

Significant vertices:

- Valence: {summary['significant_counts']['valence']['total']} total
- Arousal: {summary['significant_counts']['arousal']['total']} total
- Positive valence/arousal conjunction: {summary['positive_valence_arousal_conjunction']['total_vertices']} total

Figures display signed partial-regression t values after FDR.
The conjunction figure retains vertices where both predictor t values are positive
and both predictor-specific BH-FDR q values are <={args.fdr_alpha:g}, then displays
`min(t_valence, t_arousal)`.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    return args.output_dir


def parse_args() -> argparse.Namespace:
    project = Path(r"N:\Experimental_Data\yujunchen\projects\Inhouse_Encoding_Model")
    output_root = project / "outputs" / "iaps60_meta_static_3s"
    parser = argparse.ArgumentParser()
    parser.add_argument("--response", type=Path, default=output_root / "encoding" / "iaps60_tribev2_first_tr.npy")
    parser.add_argument("--manifest", type=Path, default=output_root / "iaps60_unique_manifest.csv")
    parser.add_argument(
        "--fsaverage5-dir",
        type=Path,
        default=Path(r"N:\Experimental_Data\yujunchen\projects\data\NSD_atlas\nsddata\freesurfer\fsaverage5"),
    )
    parser.add_argument("--output-dir", type=Path, default=output_root / "multiple_regression_valence_arousal")
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
