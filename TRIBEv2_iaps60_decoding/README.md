# TRIBE v2 encoding and emotion decoding for the 60 Bo et al. IAPS images

This folder implements a fixed, reproducible screening analysis:

1. Audit and deduplicate the 300-row condition table to 60 unique images (20 `Nt`, 20 `Pl`, 20 `Up`).
2. Convert each source JPEG to grayscale RGB, because the Bo et al. experiment presented grayscale stimuli.
3. Make a 3-second silent video with identical frames, following the static-image protocol in the attached META/TRIBE paper.
4. Generate a deterministic, population-average TRIBE v2 response on `fsaverage5` (20,484 vertices), resample from native 1 Hz to Ke Bo's 1.98-second acquisition TR with an origin-anchored linear grid, and retain the prespecified first target sample. It is exactly native row 0 and begins 5 seconds after stimulus onset.
5. Decode pleasant-vs-neutral and unpleasant-vs-neutral in 17 bilateral Kastner/Wang ROIs with a linear SVM.
6. Compare the results descriptively with the saved local Ke Bo reproduction and with the publication's 54% group threshold.

## Run

From PowerShell:

```powershell
& "N:\Experimental_Data\yujunchen\projects\Inhouse_Encoding_Model\TRIBEv2_iaps60_decoding\run_pipeline.ps1"
```

The script uses the `neuro_161` conda environment and the local NVIDIA GPU. Encoding is resumable: a saved image response is reused only when it has the expected `(3, 20484)` shape and the same protocol hash.

On Windows, extracted V-JEPA2 features are cached under `C:\t2c`. The short local path is intentional: TRIBE/exca creates long configuration-derived folder names that exceed the Win32 path limit when rooted under the project network share. This cache is temporary/rebuildable; final outputs stay in the project.

To validate the image/label manifest without loading TRIBE:

```powershell
& "C:\Users\yujunchen\AppData\Local\miniconda3\envs\neuro_161\python.exe" `
  "N:\Experimental_Data\yujunchen\projects\Inhouse_Encoding_Model\TRIBEv2_iaps60_decoding\encode_iaps60.py" `
  --dry-run
```

## Primary analysis choices

- Unit of cross-validation: unique image, not trial repetition.
- Samples per binary contrast: 40 (20/class).
- Response: first of three TRIBE rows; no post-hoc timepoint selection or temporal averaging.
- Surface ordering: 10,242 left-hemisphere vertices followed by 10,242 right-hemisphere vertices.
- ROIs: `V1v,V1d,V2v,V2d,V3v,V3d,hV4,VO1,VO2,PHC1,PHC2,hMT,LO1,LO2,V3a,V3b,IPS`.
- Atlas mapping: existing Kastner2015 surface labels resampled from `fsaverage` to `fsaverage5` by nearest-neighbor `sphere.reg`; hMT is label 13 and IPS is labels 18–23.
- Primary decoder: MATLAB-style per-image pattern z-scoring across ROI vertices plus `SVC(kernel="linear", C=1)`.
- Sensitivity decoder: training-fold `StandardScaler` plus `LinearSVC(C=1)`, matching the newer local reproduction's preprocessing/classifier while retaining unique-image folds.
- Accuracy estimate: 100 repeated stratified 10-fold partitions.
- Inference: a fixed stratified 10-fold test with 999 label permutations per ROI/contrast, followed by BH-FDR over all 34 tests.

The volumetric `kastner_dict.npy` is deliberately not applied to TRIBE surface vectors. Its NIfTI masks are useful for volumetric measured fMRI, but they do not share the coordinate system or indexing of `fsaverage5`.

## Outputs

Default output root:

```text
N:\Experimental_Data\yujunchen\projects\Inhouse_Encoding_Model\outputs\iaps60_meta_static_3s
```

Important files:

- `iaps60_unique_manifest.csv`: one row per image, labels, ratings, trials, file hashes, and source dimensions.
- `iaps60_data_audit.json`: balance and matching validation.
- `encoding/protocol.json`: frozen stimulus/model/timepoint protocol and protocol hash.
- `encoding/iaps60_tribev2_timeseries.npy`: `(60, 3, 20484)` full synthetic series.
- `encoding/iaps60_tribev2_resampled_tr1p98.npy`: `(60, 2, 20484)` explicit target-TR series.
- `encoding/iaps60_tribev2_first_tr.npy`: `(60, 20484)` primary response matrix.
- `decoding/kastner17_fsaverage5_mapping.csv`: ROI labels and vertex counts.
- `decoding/roi_decoding_results.csv`: all accuracy, split-sensitivity, permutation, FDR, and comparison values.
- `decoding/roi_decoding_comparison.png`: two-panel comparison figure.
- `RESULTS.md`: generated methods/results report and interpretation cautions.

## Interpretation boundary

This is not a literal replication of the Bo et al. group analysis. Bo et al. had 20 participants and five noisy measured responses per image; TRIBE supplies one deterministic population response per image. Repeating a synthetic map five times would leak the same feature vector across random folds, so the pipeline never does that.

The paper's 54% cutoff is a group-level permutation threshold and is only a descriptive reference for the synthetic test. Above-chance synthetic decoding can also reflect scene content and semantic structure already present in V-JEPA2. The strongest follow-up is a matched image-level analysis of measured data, followed by real-to-synthetic cross-decoding or representational correspondence.
