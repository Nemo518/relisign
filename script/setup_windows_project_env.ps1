param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonExe = "C:\Users\Nemo518\.conda\envs\torch13\python.exe"
)

$resolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$pathsConfigPath = Join-Path $resolvedRoot "paths.local.json"
$pathsConfig = $null
if (Test-Path $pathsConfigPath) {
    $pathsConfig = Get-Content -Path $pathsConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
}

$env:UNI_SIGN_ROOT = $resolvedRoot
$env:CE_CSL_ROOT = if ($pathsConfig -and $pathsConfig.ce_csl_root) { $pathsConfig.ce_csl_root } else { Join-Path $resolvedRoot "data\CE_CSL\CE-CSL" }
$env:UNISIGN_DATA_ROOT = if ($pathsConfig -and $pathsConfig.data_root) { $pathsConfig.data_root } else { Join-Path $resolvedRoot "data" }
$env:UNISIGN_OUTPUT_ROOT = if ($pathsConfig -and $pathsConfig.output_root) { $pathsConfig.output_root } else { Join-Path $resolvedRoot "out" }
$env:UNISIGN_PRETRAINED_ROOT = if ($pathsConfig -and $pathsConfig.pretrained_root) { $pathsConfig.pretrained_root } else { Join-Path $resolvedRoot "pretrained_weight" }
$env:UNISIGN_CACHE_ROOT = if ($pathsConfig -and $pathsConfig.cache_root) { $pathsConfig.cache_root } else { Join-Path $resolvedRoot ".cache" }

if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $env:PYTHONPATH = $resolvedRoot
} elseif (-not ($env:PYTHONPATH -split ';' | Where-Object { $_ -eq $resolvedRoot })) {
    $env:PYTHONPATH = "$resolvedRoot;$($env:PYTHONPATH)"
}

Write-Output "UNI_SIGN_ROOT=$env:UNI_SIGN_ROOT"
Write-Output "CE_CSL_ROOT=$env:CE_CSL_ROOT"
Write-Output "UNISIGN_DATA_ROOT=$env:UNISIGN_DATA_ROOT"
Write-Output "UNISIGN_OUTPUT_ROOT=$env:UNISIGN_OUTPUT_ROOT"
Write-Output "UNISIGN_PRETRAINED_ROOT=$env:UNISIGN_PRETRAINED_ROOT"
Write-Output "UNISIGN_CACHE_ROOT=$env:UNISIGN_CACHE_ROOT"
Write-Output "PYTHONPATH=$env:PYTHONPATH"
Write-Output "PYTHON_EXE=$PythonExe"
Write-Output ""
Write-Output "Quick check:"
Write-Output "  $PythonExe .\script\check_project_resources.py --check_checkpoint_load --sample_limit 2"
