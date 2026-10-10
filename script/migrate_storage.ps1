param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$TargetRoot = ""
)

$ErrorActionPreference = "Stop"

$project = (Resolve-Path $ProjectRoot).Path
if ([string]::IsNullOrWhiteSpace($TargetRoot)) {
    $TargetRoot = Join-Path (Split-Path -Parent $project) ((Split-Path -Leaf $project) + "-storage")
}

if (-not (Test-Path $TargetRoot)) {
    New-Item -ItemType Directory -Path $TargetRoot | Out-Null
}

$mapping = @(
    @{ Name = "data"; Source = Join-Path $project "data"; Target = Join-Path $TargetRoot "data" },
    @{ Name = "out"; Source = Join-Path $project "out"; Target = Join-Path $TargetRoot "out" },
    @{ Name = "pretrained_weight"; Source = Join-Path $project "pretrained_weight"; Target = Join-Path $TargetRoot "pretrained_weight" },
    @{ Name = ".cache"; Source = Join-Path $project ".cache"; Target = Join-Path $TargetRoot ".cache" }
)

$results = @()
foreach ($entry in $mapping) {
    if (-not (Test-Path $entry.Source)) {
        $results += [PSCustomObject]@{ Name = $entry.Name; Action = "skip"; Detail = "source missing" }
        continue
    }

    if (Test-Path $entry.Target) {
        $results += [PSCustomObject]@{ Name = $entry.Name; Action = "skip"; Detail = "target already exists" }
        continue
    }

    Move-Item -LiteralPath $entry.Source -Destination $entry.Target
    $results += [PSCustomObject]@{ Name = $entry.Name; Action = "moved"; Detail = $entry.Target }
}

$pathsConfig = [ordered]@{
    storage_root    = $TargetRoot
    data_root       = (Join-Path $TargetRoot "data")
    output_root     = (Join-Path $TargetRoot "out")
    pretrained_root = (Join-Path $TargetRoot "pretrained_weight")
    cache_root      = (Join-Path $TargetRoot ".cache")
    mt5_root        = (Join-Path $TargetRoot "pretrained_weight\mt5-base")
    ce_csl_root     = (Join-Path $TargetRoot "data\CE_CSL\CE-CSL")
}

$jsonPath = Join-Path $project "paths.local.json"
$pathsConfig | ConvertTo-Json | Set-Content -Path $jsonPath -Encoding UTF8

Write-Host "Storage migration summary:"
$results | Format-Table -AutoSize
Write-Host ""
Write-Host "Wrote local path config:" $jsonPath
Write-Host "External storage root:" $TargetRoot
