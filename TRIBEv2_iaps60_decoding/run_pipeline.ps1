param(
    [string]$Python = "C:\Users\yujunchen\AppData\Local\miniconda3\envs\neuro_161\python.exe",
    [string]$OutputDir = "N:\Experimental_Data\yujunchen\projects\Inhouse_Encoding_Model\outputs\iaps60_meta_static_3s",
    [int]$Jobs = -1,
    [int]$Permutations = 999,
    [switch]$OverwriteEncoding,
    [switch]$KeepWorkVideos
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python interpreter not found: $Python"
}

$encodeArgs = @(
    (Join-Path $ScriptDir "encode_iaps60.py"),
    "--output-dir", $OutputDir,
    "--cache-dir", "C:\t2c",
    "--device", "cuda",
    "--feature-device", "cuda",
    "--batch-size", "1",
    "--video-feature-batch-size", "1",
    "--image-duration", "3",
    "--fps", "10",
    "--max-side", "1024",
    "--grayscale"
)
if ($OverwriteEncoding) {
    $encodeArgs += "--overwrite"
}
if (-not $KeepWorkVideos) {
    $encodeArgs += "--remove-work-videos"
}

& $Python @encodeArgs
if ($LASTEXITCODE -ne 0) {
    throw "TRIBE v2 encoding failed with exit code $LASTEXITCODE"
}

& $Python `
    (Join-Path $ScriptDir "decode_iaps60.py") `
    --output-dir $OutputDir `
    --n-repeats 100 `
    --n-permutations $Permutations `
    --jobs $Jobs
if ($LASTEXITCODE -ne 0) {
    throw "ROI decoding failed with exit code $LASTEXITCODE"
}

Write-Host "Pipeline complete: $(Join-Path $OutputDir 'RESULTS.md')"
