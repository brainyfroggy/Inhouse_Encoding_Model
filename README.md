# Inhouse Encoding Model

Research code applying Meta AI's **TRIBEv2** multimodal brain-encoding model to
affective-neuroscience stimuli (IAPS images and Cowen-Keltner emotional
videos). This repository contains only the user's own wrapper/analysis code
around TRIBEv2 — no data, model weights, or vendored third-party repositories
are included (see **Dependencies** below).

## Purpose

TRIBEv2 predicts cortical (fsaverage5 surface) fMRI responses to arbitrary
image/video stimuli. This repo builds three affective-neuroscience analyses
on top of those predicted responses:

1. **`TRIBEv2_emotion_modeling/`** — trains classical and neural classifiers
   (ridge/logistic regression, SVM, gradient boosting, MLPs) to predict
   low/mid/high valence and arousal classes for Cowen-Keltner videos from
   TRIBEv2's predicted fMRI response time points.
2. **`TRIBEv2_encoding_regression/`** — fits vertex-wise encoding regressions
   (`surface_response ~ intercept + z(valence) + z(arousal)`) relating
   stimulus affect ratings to TRIBEv2 predicted fsaverage5 surface activity,
   for both CK videos and IAPS-style still images.
3. **`TRIBEv2_iaps60_decoding/`** — a fixed, reproducible screening analysis
   that decodes valence/arousal (pleasant/unpleasant vs. neutral) from
   TRIBEv2 predicted responses to the 60 unique Bo et al. IAPS images, in 17
   bilateral Kastner/Wang ROIs, with RSA searchlight and vertex-wise decoding
   variants.

Together these ask: how well do TRIBEv2's predicted (not measured) cortical
responses carry stimulus affect information, and where on the cortical
surface does that information concentrate?

## Contents

- `AFFECTIVE_ENCODING_RESEARCH_ROADMAP.md` — project roadmap / research plan
  (affect theory, EEG, LPP, and the path to an emotion-capable
  stimulus-to-brain model).
- `IMAGE_TRIBE_AFFECTIVE_FMRI_BLUEPRINT.md` — blueprint for a newer,
  image-only fMRI foundation-encoder build (dataset census and training
  design); referenced from the roadmap doc above.
- `TRIBEv2_emotion_modeling/` — `src/` training/evaluation code and `hpg/`
  HiPerGator SLURM batch scripts. See its own
  [README](TRIBEv2_emotion_modeling/README.md) for data layout, label
  definitions, and run instructions.
- `TRIBEv2_encoding_regression/` — `src/` regression-fitting and
  surface-plotting/export scripts, plus `.ps1` pipeline drivers. See its own
  [README](TRIBEv2_encoding_regression/README.md).
- `TRIBEv2_iaps60_decoding/` — encoding/decoding scripts, RSA searchlight
  variants, and a `run_pipeline.ps1` driver. See its own
  [README](TRIBEv2_iaps60_decoding/README.md).
- `outputs/tribev2_literature_audit/build_tribev2_literature_audit.py` — a
  standalone script that builds a literature-audit spreadsheet; kept because
  it is source code, even though the rest of `outputs/` (result data) is
  excluded.

Only the three sub-package READMEs above carry full parameter-level detail
(exact CLI flags, data paths, output file layouts). This top-level README
summarizes what each does and how they relate; consult the sub-README before
running anything.

## How to Use

### 1. Install TRIBEv2 first (external dependency, not vendored here)

Nothing in this repo runs standalone. Every script here calls into a local
TRIBEv2 installation for stimulus-to-response prediction. TRIBEv2 is **not
included** in this repo — clone it separately from Meta AI's official
repository and follow its own README/license for installation. See
**Dependencies** below for why it was left out and what license applies.

Once TRIBEv2 is installed and you can run its own prediction script (e.g.
`TRIBEv2/scripts/predict_local.py`) against your stimuli, the three
sub-packages below consume its predicted responses.

### 2. Run the sub-package that matches your question

- **Want classifiers on predicted responses (valence/arousal class
  prediction)?** Use `TRIBEv2_emotion_modeling/`. It expects TRIBEv2's
  predicted-response `.npy` files for the Cowen-Keltner video set plus the
  CK emotion metadata CSV (paths are configurable via environment variables
  in the `hpg/*.sbatch` scripts). Run order: `src.prepare_dataset` →
  `src.train_classical` / `src.train_nn` → `src.summarize_results` (or submit
  `hpg/train_b200.sbatch`, which runs that full sequence as one SLURM job).
  Output: a timestamped `runs/<timestamp>/` directory with `dataset/`,
  `classical/`, `neural/`, and `summary/` subfolders. Full detail in
  [TRIBEv2_emotion_modeling/README.md](TRIBEv2_emotion_modeling/README.md).

- **Want vertex-wise encoding regression (does valence/arousal predict
  surface activity)?** Use `TRIBEv2_encoding_regression/`. It expects either
  CK-video TRIBEv2 predictions (`run_ckvideo_surface_regression.py`) or an
  image metadata CSV (`image_id,image_path,valence,arousal,...`) that you
  first run through `run_image_dataset_tribev2_predictions.py` to generate
  per-image predictions, then `run_image_surface_regression.py` to fit the
  regression, then `plot_fsaverage5_image_surface_results.py` to render
  surface figures. The two `.ps1` scripts
  (`run_iaps1182_full_pipeline.ps1`, `run_iaps1182_postprocess_after_predictions.ps1`)
  chain these steps for the specific IAPS1182 image set used in this
  project's own runs. Output: a `outputs/<run-name>/` directory with
  regression statistics (BH-FDR significant vertices) and figures. Full
  detail in
  [TRIBEv2_encoding_regression/README.md](TRIBEv2_encoding_regression/README.md).

- **Want to decode valence/arousal from predicted responses to a fixed IAPS
  image set?** Use `TRIBEv2_iaps60_decoding/`. It expects the 60 unique Bo et
  al. IAPS source images plus their condition table; running
  `run_pipeline.ps1` (or `encode_iaps60.py` then `decode_iaps60.py`
  separately) generates TRIBEv2 responses, then decodes pleasant/unpleasant
  vs. neutral in 17 bilateral ROIs with a linear SVM, with vertex-wise
  (`decode_iaps60_vertexwise.py`) and RSA-searchlight
  (`rsa_searchlight_va.py`, `rsa_searchlight_va_jackknife_t.py`) variants
  available separately. Output: an `outputs/iaps60_meta_static_3s/` directory
  ending in a generated `RESULTS.md` report. Full detail in
  [TRIBEv2_iaps60_decoding/README.md](TRIBEv2_iaps60_decoding/README.md).

None of the three sub-packages depend on each other — pick the one matching
your question and follow its own README for exact commands.

## Dependencies

This project is built on top of two external repositories that are
**intentionally not included** in this repo:

- **TRIBEv2** (Meta AI's brain-encoding model). The original project vendored
  a full clone of Meta AI's TRIBEv2 repository (its own `.git` history) under
  `TRIBEv2/`. That clone is excluded here because it is not this repo's code
  and its own README carries a **CC BY-NC 4.0 (non-commercial)** license
  badge. To reproduce this code, clone TRIBEv2 separately from Meta AI's
  official repository and consult its own license/README for terms of use
  and installation instructions.
- **V-JEPA2 / exca** and related TRIBE runtime dependencies referenced by the
  scripts (see each subfolder's own `README.md` / `requirements.txt` for
  details on the local Python/conda environment expected).

Python package dependencies for `TRIBEv2_emotion_modeling/` are pinned in its
`requirements.txt` (numpy, pandas, scikit-learn, torch, matplotlib, tqdm,
joblib). `TRIBEv2_encoding_regression/` and `TRIBEv2_iaps60_decoding/` add
plotting/surface dependencies (matplotlib, nibabel/FreeSurfer-surface
tooling) noted in their own READMEs — install alongside whatever environment
TRIBEv2 itself requires, since these scripts share a Python process with
TRIBEv2's inference code.

## Results notes (folded in from excluded run/output folders)

The `runs*/` and `outputs/` subfolders under `TRIBEv2_emotion_modeling/` and
`TRIBEv2_encoding_regression/` hold pure result data (metrics, arrays,
figures) and were excluded from this repo. Their narrative README content is
reproduced below for reference.

### TRIBEv2_emotion_modeling — full run (`runs/20260624_172020`)

Models use TRIBEv2 predicted fMRI responses (20,484-vertex fsaverage5 vectors,
one per video time point) as `X`, with CK video-level valence/arousal
annotations min-max normalized and binned into low/mid/high thirds as
targets. Splits are assigned by video (2185 labeled videos, 15,533 time-point
samples; 10,854 train / 2,401 validation / 2,278 test).

Best validation balanced accuracy: valence ≈ 0.490 (`ridge_alpha10`), arousal
≈ 0.510 (`pca256_rbf_svm_C1`). Best test balanced accuracy: valence ≈ 0.493
(`mlp_512_256_do0.3_lr1e-4`), arousal ≈ 0.472
(`mlp_1024_512_do0.2_lr1e-3`). Tree/boosting models showed strong
train/validation overfitting; validation and test scores are the trustworthy
numbers. Overall, performance for both valence and arousal classification
from TRIBEv2 predicted responses was modest (all balanced accuracies well
under 0.6 for a 3-class problem where chance is 0.33).

### TRIBEv2_emotion_modeling — expanded V1-only sweep (`runs_v1_more_models`)

Using only fsaverage5 V1_exvivo surface vertices (671 features) as input,
best validation balanced accuracy was ≈ 0.512 (arousal,
`arousal_mlp_512_256_128_do0.4_lr1e-4`) and ≈ 0.482 (valence, `ridge_alpha1`).
Best test balanced accuracy was ≈ 0.464 (arousal) and ≈ 0.495 (valence,
`logreg_saga_C0.1`).

### TRIBEv2_emotion_modeling — quick V1-only sweep (`runs_v1_quick`)

Same V1 ROI setup (671 features, 15,533 samples, 2,185 videos). Best
validation balanced accuracy ≈ 0.483 (arousal, `mlp_256_128_do0.3_lr1e-3`),
≈ 0.474 (valence, `ridge_alpha10`). Note tree-based classical models
(`hist_gradient_boost`, `extra_trees_300`) showed high raw accuracy but much
lower balanced accuracy, indicating class-imbalance-driven overfitting rather
than genuine performance.

### TRIBEv2_encoding_regression — CK-video surface regression (`outputs/ckvideo_surface_reg_20260709_local`)

Encoding regression `surface_response_vertex ~ intercept + z(valence) +
z(arousal)` fit on 2,185 labeled CK videos / 15,533 time-point observations
across 20,484 fsaverage5 vertices, with video-clustered robust standard
errors (cluster df = 2,184) and BH-FDR q < 0.05 per predictor. At FDR q <
0.05: valence had 5,910 positive / 3,716 negative significant vertices
(9,626 total); arousal had 7,666 positive / 7,309 negative (14,975 total).
Caution noted in the original results: because TRIBEv2 predictions are
average-subject model outputs, this is not a human-subject group second
level — the inferential clusters here are CK videos, not participants.

### TRIBEv2_encoding_regression — IAPS image surface regression (`outputs/iaps1182_surface_reg_valence_arousal_20260712_local`)

Same regression form fit image-wise (not clustered) on 1,182 images across
20,484 fsaverage5 vertices (OLS, residual df = 1,179), FDR q < 0.05 per
predictor. Valence: 6,112 positive / 2,795 negative significant vertices
(8,907 total). Arousal: 13,258 positive / 2,267 negative (15,525 total). Same
caution applies: TRIBEv2 outputs are average-subject fsaverage5 surface
predictions, not measured human-subject data.

## Excluded from this repository

Per data-safety policy, the following were left out of this repo:

- `TRIBEv2/` — vendored clone of Meta AI's TRIBEv2 repo (own `.git`, CC
  BY-NC 4.0 license) — see **Dependencies** above.
- All `outputs/` and `runs*/` result-data directories everywhere (arrays,
  figures, metrics JSON/CSV) except the one code file noted above and the
  narrative results folded into this README.
- `_dry_run_output/` under `TRIBEv2_iaps60_decoding/`.
- `hpg_ssh_debug.log`, `hpg_ssh_debug_port22.log`, `hpg_ssh_tty_duo.log` —
  excluded as a security precaution; they contained a local Windows
  username, local file paths, and an HPC hostname/IP.
- `TRIBEv2_local_additions.zip` — redundant once `TRIBEv2/` itself is
  excluded.
- `image_fmri_dataset_census.csv`, `_codex_tmp/`, `.codex/`,
  `.codex_research_tmp/`, `.agents/`, and `__pycache__/` directories.
