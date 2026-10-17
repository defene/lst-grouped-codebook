param([int]$Workers = 4)
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    python -m eo_data prepare --dataset all --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw 'Data preparation failed; rerun the same command to resume.' }
    python -m eo_data verify --dataset all
    if ($LASTEXITCODE -ne 0) { throw 'Data verification failed.' }
} finally {
    Pop-Location
}
