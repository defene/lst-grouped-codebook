param(
    [ValidateSet('smoke','train','inspect')][string]$Mode = 'smoke',
    [string]$Config = 'experiments/paper1/configs/prithvi_vq11.json',
    [string]$Output = '',
    [int]$Steps = 4
)
$ErrorActionPreference = 'Stop'
$environmentPython = Join-Path $PSScriptRoot '.venv-paper1\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $environmentPython)) { throw 'Run setup_paper1.ps1 first' }
Push-Location $PSScriptRoot
try {
    if ($Mode -eq 'inspect') {
        & $environmentPython -m eo_denoise inspect --config $Config
    } else {
        if (-not $Output) { $Output = "runs/paper1/$Mode-$(Get-Date -Format 'yyyyMMdd-HHmmss')" }
        $arguments = @('-m','eo_denoise',$Mode,'--config',$Config,'--output',$Output)
        if ($Mode -eq 'smoke') { $arguments += @('--steps',"$Steps") }
        & $environmentPython @arguments
    }
    if ($LASTEXITCODE -ne 0) { throw 'Paper 1 command failed' }
} finally { Pop-Location }
