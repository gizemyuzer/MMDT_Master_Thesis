$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = 'python'

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE`: $($Arguments -join ' ')"
    }
}

Start-Transcript -Path (Join-Path $PSScriptRoot 'coverage_age_run.log') -Append
try {
Invoke-CheckedPython -Arguments @('-c', 'import torch; print("CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"); assert torch.cuda.is_available()')
Invoke-CheckedPython -Arguments @('run_coverage_age_control.py', '--self-test-only')
Invoke-CheckedPython -Arguments @('run_coverage_age_control.py', '--mode', 'smoke', '--seed', '42', '--device', 'cuda')
foreach ($seed in 42..52) {
    Write-Host "Starting coverage control seed $seed at $(Get-Date -Format o)"
    Invoke-CheckedPython -Arguments @('run_coverage_age_control.py', '--mode', 'train', '--seed', "$seed", '--device', 'cuda', '--allow-candidate-linkage')
    Write-Host "Completed coverage control seed $seed at $(Get-Date -Format o)"
}
Invoke-CheckedPython -Arguments @('evaluate_coverage_age_control.py')
} finally {
    Stop-Transcript
}
