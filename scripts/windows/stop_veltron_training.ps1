<#
.SYNOPSIS
    Safely stop VeltronLM training without losing progress.

.DESCRIPTION
    Never kills the Python process. Instead it writes a cooperative stop flag which the
    trainer polls once per optimiser step. The trainer then:

      1. finishes the optimiser step that is in flight,
      2. runs the evaluation and checkpoint write if one is due at this step,
      3. breaks out of the loop and performs its final evaluation,
      4. writes a final checkpoint, rotates, and writes summary.json.

    That is the same clean-exit path the built-in `max_hours` budget uses, so nothing
    about the checkpoint format or training state changes.

    IMPORTANT: the stop file only takes effect if the running trainer was started with
    `stop_file` configured. A trainer started before this feature existed -- or started
    without it -- is already past the point where it can be asked politely, so the script
    falls back to waiting for the next checkpoint to be written and verifying it, then
    reports exactly what a hard stop would cost.

.PARAMETER RunName
    Checkpoint run directory name. Defaults to mini-pretrain.

.PARAMETER Force
    Do not ask. After the next checkpoint is confirmed valid, terminate the process.
    Only offered when the graceful path is unavailable, and only after verification.

.EXAMPLE
    .\stop_veltron_training.ps1
    .\stop_veltron_training.ps1 -Force
#>
[CmdletBinding()]
param(
    [string]$RunName = 'mini-pretrain',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'veltron_common.ps1')

$run = Get-VeltronRun -RunName $RunName

Write-Host ''
Write-Host '=== VeltronLM: safe stop ===' -ForegroundColor Cyan
Write-Host "  repository : $($script:VeltronRepo)"
Write-Host "  run        : $RunName"

# ---------------------------------------------------------------- is it running?
$procs = @(Get-VeltronTrainProcess -Run $run)
$state = Get-VeltronState -Run $run

if ($procs.Count -eq 0) {
    Write-Host '  training   : NOT RUNNING' -ForegroundColor Yellow
    if ($state -and $state.latest_valid) {
        Write-Host "  latest valid checkpoint : $($state.latest_valid) (step $($state.latest_valid_step))"
    } else {
        Write-Host '  latest valid checkpoint : NONE -- nothing to resume from' -ForegroundColor Red
    }
    # Clear a stale stop file so a later resume is not immediately cancelled.
    if (Test-Path $run.StopFile) {
        Clear-VeltronStop -Run $run
        Write-Host "  removed stale stop file : $($run.StopFile)"
    }
    Write-Host ''
    exit 0
}

$pid0 = $procs[0].ProcessId
Write-Host "  training   : RUNNING (PID $pid0)" -ForegroundColor Green
if ($procs.Count -gt 1) {
    Write-Host "  WARNING    : $($procs.Count) trainer processes found: $(($procs.ProcessId) -join ', ')" -ForegroundColor Yellow
}

# ---------------------------------------------------------------- current state
if ($state) {
    Write-Host "  latest valid checkpoint : $(Get-VeltronCheckpointLabel $state)"
    Write-Host "  tokens at that checkpoint : $($state.tokens_seen)"
    if ($state.latest_valid_step -ne $null) {
        Write-Host "  checkpoint size : $(Format-VeltronBytes $state.all_checkpoints[-1].bytes)"
    }
    if ($state.invalid -and $state.invalid.Count -gt 0) {
        Write-Host "  invalid checkpoints (never deleted): $($state.invalid -join ', ')" -ForegroundColor Yellow
    }
}

# ---------------------------------------------------------------- decide the path
$gracefulAvailable = $false
if ($state -and $state.summary) { }
try {
    $logTail = ''
    if (Test-Path $state.log_path) {
        $logTail = (Get-Content -LiteralPath $state.log_path -Tail 400 -ErrorAction SilentlyContinue) -join "`n"
    }
    # The trainer logs the stop-file path in its "training start" banner when enabled.
    if ($logTail -match 'stop_file') { $gracefulAvailable = $true }
} catch { }

Write-Host ''
if (-not $gracefulAvailable) {
    Write-Host '  MODE: hard stop' -ForegroundColor Yellow
    Write-Host '  This trainer was started WITHOUT stop_file configured, so it cannot be' -ForegroundColor Yellow
    Write-Host '  asked to finish the current step. Checkpoints are written atomically' -ForegroundColor Yellow
    Write-Host '  (staging dir + rename + COMPLETE marker), so terminating now can only' -ForegroundColor Yellow
    Write-Host '  lose work since the last successful checkpoint -- never a checkpoint itself.' -ForegroundColor Yellow
    Write-Host ''

    if (-not $Force) {
        Write-Host '  Nothing has been done. The training run is untouched.' -ForegroundColor Cyan
        Write-Host '  Re-run with -Force to wait for the next checkpoint, verify it, then stop.'
        Write-Host ''
        exit 3
    }

    # Wait until a checkpoint newer than the one we already have is present AND valid.
    $known = if ($state) { $state.latest_valid_step } else { 0 }
    Write-Host "  waiting for a checkpoint newer than step $known ..." -ForegroundColor Cyan
    $deadline = (Get-Date).AddMinutes(90)
    $seen = $known
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-VeltronTrainProcess -Run $run)) {
            Write-Host '  training exited on its own.' -ForegroundColor Cyan
            break
        }
        $s = Get-VeltronState -Run $run
        if ($s -and $s.latest_valid_step -and $s.latest_valid_step -gt $seen) {
            Write-Host "  new checkpoint: $(Get-VeltronCheckpointLabel $s)" -ForegroundColor Green
            $seen = $s.latest_valid_step
            break
        }
        Start-Sleep -Seconds 20
    }

    if (-not $seen -or $seen -eq $known) {
        Write-Host '  no new checkpoint appeared before the deadline; nothing was stopped.' -ForegroundColor Red
        exit 4
    }

    # Verify before terminating. If verification fails we do NOT kill anything.
    $py = Get-VeltronPython
    $verify = & $py (Join-Path $script:VeltronRepo 'scripts\verify_checkpoints.py') $run.RunDir 2>&1
    $verifyOk = ($verify | Select-String -Pattern 'invalid:\s*0') -ne $null
    Write-Host "  verification: $(if ($verifyOk) { 'PASSED' } else { 'FAILED' })" `
        -ForegroundColor $(if ($verifyOk) { 'Green' } else { 'Red' })
    if (-not $verifyOk) {
        Write-Host '  refusing to terminate while the checkpoint does not verify.' -ForegroundColor Red
        $verify | Select-Object -Last 20
        exit 5
    }

    Write-Host "  terminating PID $pid0 ..." -ForegroundColor Yellow
    Stop-Process -Id $pid0 -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
} else {
    Write-Host '  MODE: graceful stop' -ForegroundColor Green
    $flag = Request-VeltronStop -Run $run
    Write-Host "  stop request issued : $flag" -ForegroundColor Green
    Write-Host '  The trainer will finish the current step, write any due checkpoint,'
    Write-Host '  run its final evaluation and exit. This takes up to one step'
    Write-Host '  interval (~21 s at the current settings).'
}

# ---------------------------------------------------------------- confirm
Write-Host ''
$final = Get-VeltronState -Run $run
$stillRunning = @(Get-VeltronTrainProcess -Run $run).Count -gt 0

if ($stillRunning) {
    Write-Host '  training process is still present; give it a few seconds.' -ForegroundColor Yellow
    Start-Sleep -Seconds 20
    $stillRunning = @(Get-VeltronTrainProcess -Run $run).Count -gt 0
}

Write-Host '  RESULT' -ForegroundColor Cyan
Write-Host "    training stopped   : $(-not $stillRunning)"
if ($final) {
    Write-Host "    checkpoint kept    : $(Get-VeltronCheckpointLabel $final)"
    Write-Host "    tokens preserved   : $($final.tokens_seen)"
    if ($final.all_checkpoints.Count -gt 0) {
        Write-Host "    all checkpoints    : $((($final.all_checkpoints | ForEach-Object { $_.name }) -join ', '))"
    }
}
Write-Host "    stop flag          : $($run.StopFile)"
Write-Host ''
Write-Host '  Resume with:' -ForegroundColor Cyan
Write-Host "    .\scripts\windows\start_veltron_training.ps1 -RunName $RunName"
Write-Host ''
exit $(if ($stillRunning) { 6 } else { 0 })