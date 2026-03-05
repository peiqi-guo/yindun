param(
    [Parameter(Mandatory = $true)]
    [string]$onnx_path,

    [Parameter(Mandatory = $true)]
    [string]$onnx2ncnn_path,

    [Parameter(Mandatory = $false)]
    [string]$ncnnoptimize_path,

    [Parameter(Mandatory = $true)]
    [string]$out_dir,

    [Parameter(Mandatory = $true)]
    [string]$android_assets_dir
)

$ErrorActionPreference = "Stop"

function Resolve-AbsPath {
    param([string]$PathValue)
    $resolved = Resolve-Path -Path $PathValue -ErrorAction SilentlyContinue
    if ($resolved) {
        return $resolved.Path
    }
    return [System.IO.Path]::GetFullPath($PathValue)
}

function Assert-FileExists {
    param(
        [string]$PathValue,
        [string]$Name
    )
    if (-not (Test-Path -LiteralPath $PathValue -PathType Leaf)) {
        throw "[ERROR] $Name not found: $PathValue"
    }
}

$onnx_path = Resolve-AbsPath $onnx_path
$onnx2ncnn_path = Resolve-AbsPath $onnx2ncnn_path
$out_dir = Resolve-AbsPath $out_dir
$android_assets_dir = Resolve-AbsPath $android_assets_dir

if ($ncnnoptimize_path) {
    $ncnnoptimize_path = Resolve-AbsPath $ncnnoptimize_path
}

Assert-FileExists -PathValue $onnx_path -Name "ONNX file"
Assert-FileExists -PathValue $onnx2ncnn_path -Name "onnx2ncnn executable"
if ($ncnnoptimize_path) {
    Assert-FileExists -PathValue $ncnnoptimize_path -Name "ncnnoptimize executable"
}

New-Item -ItemType Directory -Force -Path $out_dir | Out-Null
New-Item -ItemType Directory -Force -Path $android_assets_dir | Out-Null

$tmp_param = Join-Path $out_dir "mobilenetv3_small_raw.param"
$tmp_bin = Join-Path $out_dir "mobilenetv3_small_raw.bin"

$final_param = Join-Path $out_dir "mobilenetv3_small.param"
$final_bin = Join-Path $out_dir "mobilenetv3_small.bin"

$opt_param = Join-Path $out_dir "mobilenetv3_small_opt.param"
$opt_bin = Join-Path $out_dir "mobilenetv3_small_opt.bin"

Write-Host "[INFO] Converting ONNX -> NCNN"
Write-Host "[INFO] onnx2ncnn: $onnx2ncnn_path"
& $onnx2ncnn_path $onnx_path $tmp_param $tmp_bin
if ($LASTEXITCODE -ne 0) {
    throw "[ERROR] onnx2ncnn failed with exit code: $LASTEXITCODE"
}

if ($ncnnoptimize_path) {
    Write-Host "[INFO] Optimizing NCNN model"
    Write-Host "[INFO] ncnnoptimize: $ncnnoptimize_path"
    & $ncnnoptimize_path $tmp_param $tmp_bin $opt_param $opt_bin 0
    if ($LASTEXITCODE -ne 0) {
        throw "[ERROR] ncnnoptimize failed with exit code: $LASTEXITCODE"
    }

    Copy-Item -Force -LiteralPath $opt_param -Destination $final_param
    Copy-Item -Force -LiteralPath $opt_bin -Destination $final_bin
}
else {
    Copy-Item -Force -LiteralPath $tmp_param -Destination $final_param
    Copy-Item -Force -LiteralPath $tmp_bin -Destination $final_bin
}

$assets_ncnn_dir = Join-Path $android_assets_dir "ncnn"
New-Item -ItemType Directory -Force -Path $assets_ncnn_dir | Out-Null

Copy-Item -Force -LiteralPath $final_param -Destination (Join-Path $assets_ncnn_dir "mobilenetv3_small.param")
Copy-Item -Force -LiteralPath $final_bin -Destination (Join-Path $assets_ncnn_dir "mobilenetv3_small.bin")

Write-Host "[OK] NCNN artifacts ready:"
Write-Host "      $final_param"
Write-Host "      $final_bin"
Write-Host "[OK] Copied to Android assets:"
Write-Host "      $(Join-Path $assets_ncnn_dir 'mobilenetv3_small.param')"
Write-Host "      $(Join-Path $assets_ncnn_dir 'mobilenetv3_small.bin')"
