# TRIBEv2 Surface Encoding Regression

This folder runs encoding-style multiple regression on local TRIBEv2
predictions.

The CK-video analysis uses:

```text
TRIBEv2 surface response at each vertex = intercept + z(valence) + z(arousal)
```

Important notes:

- This is not decoding. Valence and arousal are independent variables.
- The dependent variable is the TRIBEv2 predicted cortical response for each
  fsaverage5 surface vertex.
- TRIBEv2 outputs are surface vertices, not MNI volumetric voxels.
- Each CK-video time point is included as an observation, while inference uses
  video-clustered standard errors so repeated time points from the same video do
  not act like independent videos.
- FDR is Benjamini-Hochberg q < 0.05 across vertices, separately for valence and
  arousal.
- Significant positive and negative vertices are saved separately.

Run locally from `N:\Experimental_Data\yujunchen\projects\Inhouse_encoding_model`:

```powershell
$py = "C:\Users\yujunchen\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py TRIBEv2_encoding_regression\src\run_ckvideo_surface_regression.py
```

## Image Dataset Workflow

TRIBEv2 image predictions are still fsaverage5 cortical surface vertices:
`20484 = 10242 LH + 10242 RH`. They are not MNI voxels.

Prepare an image metadata CSV with at least:

```text
image_id,image_path,valence,arousal
img001,C:\path\to\img001.jpg,4.2,5.1
```

Run each image through local TRIBEv2 still-image inference:

```powershell
cd N:\Experimental_Data\yujunchen\projects\Inhouse_encoding_model
$tribePy = "C:\path\to\python-with-tribev2-installed.exe"
& $tribePy TRIBEv2_encoding_regression\src\run_image_dataset_tribev2_predictions.py `
  --manifest-csv C:\path\to\image_metadata.csv `
  --device cuda `
  --quiet
```

Use the TRIBEv2 environment described in
`N:\Experimental_Data\yujunchen\projects\Inhouse_encoding_model\TRIBEv2\README_LOCAL.md`.
The dataset runner calls `TRIBEv2\scripts\predict_local.py` once per image.
For a fresh environment that will also run the regression and plotting scripts,
install the local package with plotting support plus pandas:

```powershell
python -m pip install -e "N:\Experimental_Data\yujunchen\projects\Inhouse_encoding_model\TRIBEv2[plotting]" pandas
```

Default image prediction outputs go to:

```text
N:\Experimental_Data\yujunchen\projects\data\TRIBEv2\images\outputs_image
```

Each image gets a folder containing `predictions.npy`, `events.csv`,
`segments.csv`, and `summary.json`. The dataset runner also writes
`image_prediction_manifest.csv`.

### View One Image Prediction

To see how TRIBEv2 predicts the cortical response to one image, plot that
image's `predictions.npy` directly on fsaverage5:

```powershell
& $tribePy TRIBEv2_encoding_regression\src\plot_fsaverage5_image_predictions.py `
  --image-id img001
```

Or pass an explicit output folder:

```powershell
& $tribePy TRIBEv2_encoding_regression\src\plot_fsaverage5_image_predictions.py `
  --run-dir N:\Experimental_Data\yujunchen\projects\data\TRIBEv2\images\outputs_image\img001
```

The default view averages the short still-image TRIBEv2 time series before
plotting. You can instead inspect a specific predicted time sample:

```powershell
& $tribePy TRIBEv2_encoding_regression\src\plot_fsaverage5_image_predictions.py `
  --image-id img001 `
  --aggregate timepoint `
  --timepoint 0
```

This produces:

```text
figures_fsaverage5_prediction\fsaverage5_prediction_mean.png
```

These maps are raw TRIBEv2 predicted cortical responses for individual images,
not regression statistics.

Fit an image-level fsaverage5 regression:

```powershell
& $tribePy TRIBEv2_encoding_regression\src\run_image_surface_regression.py `
  --metadata-csv C:\path\to\image_metadata.csv `
  --predictors valence arousal `
  --run-name image_surface_reg_valence_arousal_local
```

For still images, the regression script averages each image's TRIBEv2
time-sample predictions by default, then fits:

```text
surface_response_vertex ~ intercept + z_image(valence) + z_image(arousal)
```

Render fsaverage5 surface figures:

```powershell
& $tribePy TRIBEv2_encoding_regression\src\plot_fsaverage5_image_surface_results.py `
  --run-dir TRIBEv2_encoding_regression\outputs\image_surface_reg_valence_arousal_local
```

Use `--predictors` to swap in any numeric image-level metadata columns. The
visualization script reads predictor names and output filenames from
`regression_summary.json`.
