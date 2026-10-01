<#
.SYNOPSIS
    Wait for VeltronLM training to finish, verify a checkpoint, optionally shut down.

.DESCRIPTION
    Default behaviour is to WAIT AND REPORT ONLY. It never shuts the machine down unless
    -ShutdownWhenDone is passed, so running it by mistake cannot end the session early.

    Order of operations when shutting down:
      1. wait for the trainer process to disappear (or stop it gracefully),
      2. run the repository's checkpoint verifier,
      3. print exactly which checkpoint will be preserved,
      4. only then call shutdown /s /t 60.

    A 60-second shutdown delay is deliberate: it is a visible, cancellable window
    (shutdown /a), so an accidental invocation can still be aborted.

.PARAMETER ShutdownWhenDone
    Actually shut Windows down. Omit this and the script only reports.

.PARAMETER Hours
    Give up waiting after this many hours. 0 waits indefinitely.

.PARAMETER Restart
    Restart instead of shutting down.

.EXAMPLE
    .\shutdown_after_training.ps1                       # wait and report only
    .\shutdown_after_training.ps1 -ShutdownWhenDone      # then shut down
    .\shutdown_after_training.ps1 -ShutdownWhenDone -Restart
#>
[CmdletBinding()]
param(
    [string]$RunName = 'mini-pretrain',
    [switch]$ShutdownWhenDone,
    [switch]$Restart,
    [double]$Hours = 0,
    [int]$ShutdownDelaySeconds = 60
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'veltron_common.ps1')

$run = Get-VeltronRun -RunName $RunName

Write-Host ''
Write-Host '=== VeltronLM: wait for training, then act ===' -ForegroundColor Cyan
Write-Host "  shutdown on completion : $(if ($ShutdownWhenDone) { 'YES' } else { 'NO (report only)' })"
if ($ShutdownWhenDone) {
    Write-Host "  action                 : $(if ($Restart) { 'restart' } else { 'shut down' }) in $ShutdownDelaySeconds s (cancel with: shutdown /a)"
}

$deadline = if ($Hours -gt 0) { (Get-Date).AddHours($Hours) } else { [datetime]::MaxValue }
$lastReport = Get-Date

# --------------------------------------------------------------------- wait
while ($true) {
    $procs = @(Get-VeltronTrainProcess -Run $run)
    if ($procs.Count -eq 0) {
        Write-Host '  training has finished.' -ForegroundColor Green
        break
    }
    if ((Get-Date) -ge $deadline) {
        Write-Host "  gave up waiting after $Hours h; training is still running." -ForegroundColor Yellow
        Write-Host '  Nothing was stopped and nothing was shut down.'
        exit 8
    }
    if (((Get-Date) - $lastReport).TotalSeconds -ge 60) {
        $lastReport = Get-Date
        Write-Host ("  [{0}] still training, PID {1}" -f (Get-Date -Format HH:mm:ss), ($procs.ProcessId -join ','))
    }
    Start-Sleep -Seconds 10
}

# ------------------------------------------------------------------- verify
$py = Get-VeltronPython
$state = Get-VeltronState -Run $run

Write-Host ''
Write-Host '  verifying checkpoints' -ForegroundColor Cyan
$v = & $py (Join-Path $script:VeltronRepo 'scripts\verify_checkpoints.py') $run.RunDir 2>&1
$v | Select-Object -Last 8 | ForEach-Object { "    $_" }

$invalid = 0
if ($v -match 'invalid:\s*(\d+)') { $invalid = [int]$Matches[1] }
$latest = if ($state) { $state.latest_valid } else { $null }

Write-Host ''
Write-Host '  CHECKPOINT TO BE PRESERVED' -ForegroundColor Cyan
if ($latest) {
    Write-Host "    $(Get-VeltronCheckpointLabel $state)"
    Write-Host "    tokens  : $($state.tokens_seen)"
    Write-Host "    val loss: $($state.val_loss)"
} else {
    Write-Host '    NONE -- there is nothing resumable.' -ForegroundColor Red
}
if ($invalid -gt 0) {
    Write-Host "    invalid checkpoints (kept, not deleted): $($state.invalid -join ', ')" -ForegroundColor Yellow
}

if (-not $ShutdownWhenDone) {
    Write-Host ''
    Write-Host '  REPORT ONLY -- Windows was NOT shut down.' -ForegroundColor Yellow
    Write-Host '  Re-run with -ShutdownWhenDone to actually shut down.'
    exit 0
}

# ------------------------------------------------------------------ shutdown
if (-not $latest) {
    Write-Host ''
    Write-Host '  REFUSING to shut down: no valid checkpoint exists.' -ForegroundColor Red
    exit 9
}

Write-Host ''
Write-Host "  shutting down in $ShutdownDelaySeconds seconds. Cancel with: shutdown /a" -ForegroundColor Yellow
if ($Restart) {
    & shutdown.exe /r /t $ShutdownDelaySeconds /c "VeltronLM training finished"
} else {
    & shutdown.exe /s /t $ShutdownDelaySeconds /c "VeltronLM training finished"
}
exit 0