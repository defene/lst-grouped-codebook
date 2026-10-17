param([string]$Python = "C:\ProgramData\anaconda3\python.exe", [switch]$Cpu)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$environmentDir = Join-Path $projectRoot '.venv-paper1'
$environmentPython = Join-Path $environmentDir 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $environmentPython)) {
    & $Python -m venv --system-site-packages $environmentDir
    if ($LASTEXITCODE -ne 0) { throw 'Could not create training environment' }
}
$torchIndex = if ($Cpu) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu128' }
& $environmentPython -m pip install torch==2.11.0 torchvision==0.26.0 --index-url $torchIndex
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $environmentPython -m pip install -r (Join-Path $projectRoot 'experiments\paper1\requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Training dependencies failed' }
& $environmentPython -c "import torch; print('torch:',torch.__version__,'cuda:',torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw 'PyTorch import check failed' }
