param(
    [ValidateSet("doctor", "forward", "smoke", "train", "eval-dev", "eval-test", "demo", "rgb-smoke", "rgb-train", "rgb-train-unfreeze", "rgb-eval-dev", "rgb-eval-test")]
    [string]$Action = "doctor",
    [string]$PythonExe = "",
    [string]$Checkpoint = ".\out\cecsl_pose_only_round4\best_checkpoint.pth",
    [string]$Split = "dev",
    [int]$SampleIndex = 0,
    [int]$SampleCount = 5,
    [string]$Device = "cuda",
    [string]$DType = "bf16",
    [switch]$FreezeMt5,
    [switch]$NoWindow,
    [int]$CameraId = 0
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$defaultPython = "C:\Users\Nemo518\.conda\envs\torch13\python.exe"
$pathsConfigPath = Join-Path $repoRoot "paths.local.json"
$pathsConfig = $null
if (Test-Path $pathsConfigPath) {
    $pathsConfig = Get-Content -Path $pathsConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Resolve-UniSignStoragePath {
    param([string]$RawPath)

    if ([string]::IsNullOrWhiteSpace($RawPath)) {
        return $RawPath
    }

    if ([System.IO.Path]::IsPathRooted($RawPath)) {
        return $RawPath
    }

    $normalized = $RawPath.Replace('/', '\').TrimStart('.').TrimStart('\')
    if ($pathsConfig) {
        if ($normalized -eq 'out' -and $pathsConfig.output_root) { return $pathsConfig.output_root }
        if ($normalized.StartsWith('out\') -and $pathsConfig.output_root) {
            return Join-Path $pathsConfig.output_root $normalized.Substring(4)
        }
        if ($normalized -eq 'pretrained_weight' -and $pathsConfig.pretrained_root) { return $pathsConfig.pretrained_root }
        if ($normalized.StartsWith('pretrained_weight\') -and $pathsConfig.pretrained_root) {
            return Join-Path $pathsConfig.pretrained_root $normalized.Substring(18)
        }
    }

    return Join-Path $repoRoot $normalized
}

if (-not $PythonExe) {
    if ($env:UNI_SIGN_PYTHON) {
        $PythonExe = $env:UNI_SIGN_PYTHON
    }
    elseif (Test-Path $defaultPython) {
        $PythonExe = $defaultPython
    }
    else {
        $PythonExe = "python"
    }
}

$defaultCeCslRoot = if ($pathsConfig -and $pathsConfig.ce_csl_root) { $pathsConfig.ce_csl_root } else { Join-Path $repoRoot "data\CE_CSL\CE-CSL" }
$defaultLabelRoot = if ($pathsConfig -and $pathsConfig.data_root) { Join-Path $pathsConfig.data_root "CE_CSL" } else { Join-Path $repoRoot "data\CE_CSL" }
if (-not $env:CE_CSL_ROOT) {
    $env:CE_CSL_ROOT = $defaultCeCslRoot
}

$legacyDefaultCheckpoint = ".\out\cecsl_pose_only_round4\best_checkpoint.pth"
if ($Checkpoint -eq $legacyDefaultCheckpoint) {
    $Checkpoint = Resolve-UniSignStoragePath ".\out\cecsl_pose_only_round4\best_checkpoint.pth"
} else {
    $Checkpoint = Resolve-UniSignStoragePath $Checkpoint
}

function Invoke-UniSignPython {
    param(
        [string[]]$ScriptArgs
    )

    Write-Host ""
    Write-Host ">> $PythonExe $($ScriptArgs -join ' ')"
    & $PythonExe @ScriptArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Push-Location $repoRoot

try {
    switch ($Action) {
        "doctor" {
            Invoke-UniSignPython -ScriptArgs @(
                ".\script\check_project_resources.py",
                "--cecsl_root", $env:CE_CSL_ROOT,
                "--check_checkpoint_load",
                "--sample_limit", "2"
            )
            Invoke-UniSignPython -ScriptArgs @(
                ".\script\check_cecsl_dataset.py",
                "--cecsl_root", $env:CE_CSL_ROOT,
                "--label_root", $defaultLabelRoot
            )
        }

        "forward" {
            Invoke-UniSignPython -ScriptArgs @(
                ".\script\test_cecsl_forward.py",
                "--split", $Split,
                "--sample_index", "$SampleIndex",
                "--device", $Device,
                "--dtype", $DType,
                "--checkpoint", $Checkpoint
            )
        }

        "smoke" {
            $args = @(
                ".\local_sign_tools\train_cecsl_smoke.py",
                "--checkpoint", $Checkpoint,
                "--device", $Device,
                "--dtype", $DType,
                "--sample_limit", "2",
                "--max_steps", "1",
                "--batch_size", "1"
            )
            if ($FreezeMt5) {
                $args += "--freeze_mt5"
            }
            Invoke-UniSignPython -ScriptArgs $args
        }

        "train" {
            $outputDir = Resolve-UniSignStoragePath ".\out\cecsl_pose_only_round5_unfreeze"
            $args = @(
                ".\local_sign_tools\finetune_cecsl_pose_only.py",
                "--checkpoint", $Checkpoint,
                "--output_dir", $outputDir,
                "--device", $Device,
                "--dtype", $DType,
                "--epochs", "3",
                "--batch_size", "1",
                "--num_workers", "2",
                "--lr", "3e-6",
                "--gradient_accumulation_steps", "8",
                "--eval_split", "dev"
            )
            if ($FreezeMt5) {
                $args += "--freeze_mt5"
            }
            Invoke-UniSignPython -ScriptArgs $args
        }

        "eval-dev" {
            Invoke-UniSignPython -ScriptArgs @(
                ".\script\eval_cecsl_samples.py",
                "--split", "dev",
                "--checkpoint", $Checkpoint,
                "--sample_count", "$SampleCount",
                "--sample_mode", "random",
                "--seed", "42",
                "--device", $Device,
                "--dtype", $DType
            )
        }

        "eval-test" {
            Invoke-UniSignPython -ScriptArgs @(
                ".\script\eval_cecsl_samples.py",
                "--split", "test",
                "--checkpoint", $Checkpoint,
                "--sample_count", "$SampleCount",
                "--sample_mode", "random",
                "--seed", "42",
                "--device", $Device,
                "--dtype", $DType
            )
        }

        "demo" {
            $args = @(
                ".\local_sign_tools\webcam_demo.py",
                "--finetune", $Checkpoint,
                "--camera_id", "$CameraId",
                "--device", $Device,
                "--dtype", $DType,
                "--flip"
            )
            if ($NoWindow) {
                $args += "--no_window"
            }
            Invoke-UniSignPython -ScriptArgs $args
        }

        "rgb-smoke" {
            $args = @(
                ".\local_sign_tools\finetune_cecsl_rgb_support.py",
                "--checkpoint", (Resolve-UniSignStoragePath ".\out\stage3_finetuning\best_checkpoint.pth"),
                "--output_dir", (Resolve-UniSignStoragePath ".\out\cecsl_rgb_support_smoke"),
                "--device", $Device,
                "--dtype", $DType,
                "--epochs", "1",
                "--batch_size", "1",
                "--num_workers", "0",
                "--lr", "1e-6",
                "--weight_decay", "1e-4",
                "--gradient_accumulation_steps", "1",
                "--eval_split", "dev",
                "--train_sample_limit", "16",
                "--eval_sample_limit", "16"
            )
            if ($FreezeMt5) {
                $args += "--freeze_mt5"
            }
            Invoke-UniSignPython -ScriptArgs $args
        }

        "rgb-train" {
            $baseCheckpoint = Resolve-UniSignStoragePath ".\out\stage3_finetuning\best_checkpoint.pth"
            if ($Checkpoint -ne (Resolve-UniSignStoragePath ".\out\cecsl_pose_only_round4\best_checkpoint.pth")) {
                $baseCheckpoint = $Checkpoint
            }

            $args = @(
                ".\local_sign_tools\finetune_cecsl_rgb_support.py",
                "--checkpoint", $baseCheckpoint,
                "--output_dir", (Resolve-UniSignStoragePath ".\out\cecsl_rgb_support_round1"),
                "--device", $Device,
                "--dtype", $DType,
                "--epochs", "3",
                "--batch_size", "1",
                "--num_workers", "2",
                "--lr", "1e-6",
                "--weight_decay", "1e-4",
                "--gradient_accumulation_steps", "4",
                "--eval_split", "dev",
                "--save_epoch_checkpoints"
            )
            if ($FreezeMt5) {
                $args += "--freeze_mt5"
            }
            Invoke-UniSignPython -ScriptArgs $args
        }

        "rgb-train-unfreeze" {
            $baseCheckpoint = Resolve-UniSignStoragePath ".\out\cecsl_rgb_support_round1\best_checkpoint.pth"
            if ($Checkpoint -ne (Resolve-UniSignStoragePath ".\out\cecsl_pose_only_round4\best_checkpoint.pth")) {
                $baseCheckpoint = $Checkpoint
            }

            $args = @(
                ".\local_sign_tools\finetune_cecsl_rgb_support.py",
                "--checkpoint", $baseCheckpoint,
                "--output_dir", (Resolve-UniSignStoragePath ".\out\cecsl_rgb_support_round2_unfreeze"),
                "--device", $Device,
                "--dtype", $DType,
                "--epochs", "3",
                "--batch_size", "1",
                "--num_workers", "2",
                "--lr", "3e-7",
                "--weight_decay", "1e-4",
                "--gradient_accumulation_steps", "4",
                "--eval_split", "dev",
                "--save_epoch_checkpoints"
            )
            Invoke-UniSignPython -ScriptArgs $args
        }

        "rgb-eval-dev" {
            $baseCheckpoint = Resolve-UniSignStoragePath ".\out\cecsl_rgb_support_round1\best_checkpoint.pth"
            if ($Checkpoint -ne (Resolve-UniSignStoragePath ".\out\cecsl_pose_only_round4\best_checkpoint.pth")) {
                $baseCheckpoint = $Checkpoint
            }

            Invoke-UniSignPython -ScriptArgs @(
                ".\script\eval_cecsl_samples.py",
                "--split", "dev",
                "--checkpoint", $baseCheckpoint,
                "--rgb_support",
                "--sample_count", "$SampleCount",
                "--sample_mode", "random",
                "--seed", "42",
                "--device", $Device,
                "--dtype", $DType
            )
        }

        "rgb-eval-test" {
            $baseCheckpoint = Resolve-UniSignStoragePath ".\out\cecsl_rgb_support_round1\best_checkpoint.pth"
            if ($Checkpoint -ne (Resolve-UniSignStoragePath ".\out\cecsl_pose_only_round4\best_checkpoint.pth")) {
                $baseCheckpoint = $Checkpoint
            }

            Invoke-UniSignPython -ScriptArgs @(
                ".\script\eval_cecsl_samples.py",
                "--split", "test",
                "--checkpoint", $baseCheckpoint,
                "--rgb_support",
                "--sample_count", "$SampleCount",
                "--sample_mode", "random",
                "--seed", "42",
                "--device", $Device,
                "--dtype", $DType
            )
        }
    }
}
finally {
    Pop-Location
}
