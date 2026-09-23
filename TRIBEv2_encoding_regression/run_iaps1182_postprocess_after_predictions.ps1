param(
  [int]$PredictionPid = 0
)

$ErrorActionPreference = "Stop"

$ProjectRoot = "N:\Experimental_Data\yujunchen\projects\Inhouse_encoding_model"
$Python = "C:\Users\yujunchen\AppData\Local\miniconda3\envs\neuro_161\python.exe"
$PredDir = "N:\Experimental_Data\yujunchen\projects\data\IAPS1182\predictions\tribev2_fsaverage5"
$RunName = "iaps1182_surface_reg_valence_arousal_20260712_local"
$RegDir = Join-Path $ProjectRoot "TRIBEv2_encoding_regression\outputs\$RunName"
$LogDir = Join-Path $PredDir "logs"

New-Item -ItemType Directory -Force $LogDir | Out-Null
Set-Location $ProjectRoot

if ($PredictionPid -gt 0) {
  "[$(Get-Date -Format s)] Waiting for prediction PID $PredictionPid" | Out-File -FilePath (Join-Path $LogDir "00_postprocess_watcher.log") -Append
  Wait-Process -Id $PredictionPid
}

$predictionCount = (Get-ChildItem -Recurse -File $PredDir -Filter predictions.npy | Measure-Object).Count
"[$(Get-Date -Format s)] Found $predictionCount prediction files" | Out-File -FilePath (Join-Path $LogDir "00_postprocess_watcher.log") -Append
if ($predictionCount -lt 1182) {
  throw "Expected 1182 prediction files before regression, found $predictionCount."
}

if (Test-Path $RegDir) {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  Rename-Item -LiteralPath $RegDir -NewName "$RunName.previous_$stamp"
}

& $Python -u TRIBEv2_encoding_regression\src\run_image_surface_regression.py `
  --responses-dir $PredDir `
  --metadata-csv (Join-Path $PredDir "iaps1182_tribev2_manifest.csv") `
  --image-id-column image_id `
  --predictors valence arousal `
  --aggregate mean `
  --run-name $RunName `
  *> (Join-Path $LogDir "02_surface_regression.log")

& $Python -u TRIBEv2_encoding_regression\src\plot_fsaverage5_image_surface_results.py `
  --run-dir $RegDir `
  *> (Join-Path $LogDir "03_fsaverage5_figures.log")

& $Python -u TRIBEv2_encoding_regression\src\plot_fsaverage5_image_conjunction_with_sts_border.py `
  --run-dir $RegDir `
  *> (Join-Path $LogDir "04_conjunction_sts_border.log")

"[$(Get-Date -Format s)] Postprocess complete: $RegDir" | Out-File -FilePath (Join-Path $LogDir "00_postprocess_watcher.log") -Append
