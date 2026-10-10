$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$python = "C:\Users\Nemo518\.conda\envs\torch13\python.exe"
if (-not (Test-Path $python)) {
    throw "torch13 Python not found: $python"
}

$env:UNISIGN_VIEW_API = "http://127.0.0.1:8000"
& $python -m streamlit run view/app.py
