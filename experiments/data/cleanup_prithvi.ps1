$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..')).TrimEnd('\')
if ($taskRoot -ne 'C:\Users\rog\Desktop\iclr') { throw 'Unexpected workspace' }
$taskArchive = Join-Path $taskRoot 'artifacts\prithvi_data_v1_20260909.zip'
$taskVerification = Get-Content -LiteralPath ($taskArchive + '.verification.json') -Raw | ConvertFrom-Json
$taskAudit = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'original_prithvi_cleanup_audit.json') -Raw | ConvertFrom-Json
$taskValidation = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'materialization_validation.json') -Raw | ConvertFrom-Json
$taskStorage = Get-Content -LiteralPath (Join-Path $taskRoot 'processed_data\v1\prithvi\storage.json') -Raw | ConvertFrom-Json
if ($taskVerification.status -ne 'passed' -or $taskAudit.status -ne 'passed' -or $taskValidation.status -ne 'passed') { throw 'Required verification is missing' }
if ($taskVerification.sha256 -ne $taskStorage.archive_sha256) { throw 'Archive identity mismatch' }
if ((Get-Item -LiteralPath $taskArchive).Length -ne $taskVerification.size) { throw 'Archive size changed' }
if ($taskAudit.checked_files -ne 969665 -or $taskAudit.source_changed) { throw 'Original source audit mismatch' }
$taskTargets = @(
    [pscustomobject]@{ Path=(Join-Path $taskRoot 'prithvi_data.zip'); Recursive=$false },
    [pscustomobject]@{ Path=(Join-Path $taskRoot 'processed_data\v1\prithvi\packs'); Recursive=$true },
    [pscustomobject]@{ Path=(Join-Path $taskRoot 'prithvi_data'); Recursive=$true },
    [pscustomobject]@{ Path=(Join-Path $taskRoot 'processed_data\development'); Recursive=$true },
    [pscustomobject]@{ Path=(Join-Path $taskRoot 'processed_data\restore_check'); Recursive=$true }
)
foreach ($taskTarget in $taskTargets) {
    $taskAbsolute = [IO.Path]::GetFullPath($taskTarget.Path).TrimEnd('\')
    if (-not $taskAbsolute.StartsWith($taskRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Deletion target escapes workspace' }
    if ($taskArchive -eq $taskAbsolute -or $taskArchive.StartsWith($taskAbsolute + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Deletion would include latest archive' }
    if (Test-Path -LiteralPath $taskAbsolute) {
        $taskItem = Get-Item -LiteralPath $taskAbsolute -Force
        if ($taskItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Unexpected reparse point at deletion target' }
        if ((Resolve-Path -LiteralPath $taskAbsolute).Path.TrimEnd('\') -ne $taskAbsolute) { throw 'Resolved deletion path mismatch' }
    }
}
$taskOldZip = Join-Path $taskRoot 'prithvi_data.zip'
if ((Test-Path -LiteralPath $taskOldZip) -and (Get-Item -LiteralPath $taskOldZip).Length -ne $taskAudit.old_zip_bytes) { throw 'Old ZIP changed' }
$taskReportPath = Join-Path $PSScriptRoot 'storage_cleanup.json'
$taskReport = [ordered]@{
    status='running'; authorization='全部删除，并给我一份最新的压缩文件';
    started_utc=(Get-Date).ToUniversalTime().ToString('o');
    archive=$taskArchive; archive_sha256=$taskVerification.sha256;
    original_bytes=$taskAudit.original_bytes; old_zip_bytes=$taskAudit.old_zip_bytes;
    free_bytes_before=[long](Get-PSDrive -Name C).Free; deleted=@()
}
$taskReport | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $taskReportPath -Encoding utf8
# Hold the verified replacement open with read sharing while deleting old copies.
$taskArchiveGuard = [IO.File]::Open($taskArchive,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read)
try {
    foreach ($taskTarget in $taskTargets) {
        if (Test-Path -LiteralPath $taskTarget.Path) {
            Write-Output ('Removing verified obsolete copy: ' + $taskTarget.Path)
            if ($taskTarget.Recursive) { Remove-Item -LiteralPath $taskTarget.Path -Recurse -Force }
            else { Remove-Item -LiteralPath $taskTarget.Path -Force }
        }
        if (Test-Path -LiteralPath $taskTarget.Path) { throw ('Deletion incomplete: ' + $taskTarget.Path) }
        $taskReport.deleted += $taskTarget.Path
        $taskReport | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $taskReportPath -Encoding utf8
    }
    $taskReport.status = 'complete'
    $taskReport.free_bytes_after = [long](Get-PSDrive -Name C).Free
    $taskReport.freed_bytes_during_cleanup = $taskReport.free_bytes_after - $taskReport.free_bytes_before
    $taskReport.finished_utc = (Get-Date).ToUniversalTime().ToString('o')
    $taskReport | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $taskReportPath -Encoding utf8
    $taskReport | ConvertTo-Json -Depth 8
} catch {
    $taskReport.status = 'failed'
    $taskReport.error = $_.Exception.Message
    $taskReport | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $taskReportPath -Encoding utf8
    throw
} finally {
    $taskArchiveGuard.Dispose()
}
