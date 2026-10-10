$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$python = "C:\Users\Nemo518\.conda\envs\torch13\python.exe"
if (-not (Test-Path $python)) {
    throw "torch13 Python not found: $python"
}

& $python -m uvicorn view.api_server:app --host 127.0.0.1 --port 8000
