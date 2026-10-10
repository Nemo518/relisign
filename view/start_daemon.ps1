# ReliSign 守护启动 — 崩溃自动重启
$python = "C:\Users\Nemo518\.conda\envs\torch13\python.exe"
$script = "view\web_app\app.py"

while ($true) {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  ReliSign 服务启动 $(Get-Date)" -ForegroundColor Green
    Write-Host "  访问: http://localhost:8001" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green

    $proc = Start-Process -FilePath $python -ArgumentList $script -PassThru -NoNewWindow
    $proc.WaitForExit()

    $code = $proc.ExitCode
    Write-Host "进程退出 (code=$code), 3秒后自动重启..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
}
