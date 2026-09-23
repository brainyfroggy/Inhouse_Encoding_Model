$ErrorActionPreference = "Stop"

$ProjectRoot = "N:\Experimental_Data\yujunchen\projects\Inhouse_encoding_model"
$Python = "C:\Users\yujunchen\AppData\Local\miniconda3\envs\neuro_161\python.exe"
$PredDir = "N:\Experimental_Data\yujunchen\projects\data\IAPS1182\predictions\tribev2_fsaverage5"
$CacheDir = "C:\tribev2_cache"
$RunName = "iaps1182_surface_reg_valence_arousal_20260712_local"
$RegDir = Join-Path $ProjectRoot "TRIBEv2_encoding_regression\outputs\$RunName"
$LogDir = Join-Path $PredDir "logs"

New-Item -ItemType Directory -Force $PredDir, $CacheDir, $LogDir | Out-Null
Set-Location $ProjectRoot

Write-Host "[$(Get-Date -Format s)] Starting IAPS1182 TRIBEv2 predictions"
$predLog = Join-Path $LogDir "01_tribev2_predictions.log"
& $Python -u TRIBEv2_encoding_regression\src\run_iaps1182_tribev2_joint_predictions.py `
  --device cuda `
  --feature-device cuda `
  --cache-dir $CacheDir `
  --quiet `
  *> $predLog

Write-Host "[$(Get-Date -Format s)] Starting IAPS1182 surface regression"
if (Test-Path $RegDir) {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  Rename-Item -LiteralPath $RegDir -NewName "$RunName.previous_$stamp"
}
$regLog = Join-Path $LogDir "02_surface_regression.log"
& $Python -u TRIBEv2_encoding_regression\src\run_image_surface_regression.py `
  --responses-dir $PredDir `
  --metadata-csv (Join-Path $PredDir "iaps1182_tribev2_manifest.csv") `
  --image-id-column image_id `
  --predictors valence arousal `
  --aggregate mean `
  --run-name $RunName `
  *> $regLog

Write-Host "[$(Get-Date -Format s)] Rendering fsaverage5 figures"
& $Python -u TRIBEv2_encoding_regression\src\plot_fsaverage5_image_surface_results.py `
  --run-dir $RegDir `
  *> (Join-Path $LogDir "03_fsaverage5_figures.log")

Write-Host "[$(Get-Date -Format s)] Done"
