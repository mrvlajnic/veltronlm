<#
.SYNOPSIS
    Run VeltronLM training for a bounded window of time, then stop cleanly.

.DESCRIPTION
    Starts (or resumes) training, watches the wall clock, and when the window closes asks
    the trainer to stop. The trainer finishes its in-flight optimiser step, writes any
    due checkpoint, performs its final evaluation and exits through its normal path.

    The time limit is enforced twice, deliberately:
      * `max_hours` is passed to the trainer, which stops at a step boundary on its own;
      * this script also watches the clock and requests a stop if the trainer has not
        exited within the grace period.

    So if the trainer's own budget works, nothing external is needed. If it somehow hangs,
    the grace period escalates -- but only after a checkpoint has been confirmed.

.PARAMETER Hours
    Window length in hours. Fractional values work, so 0.05 is a 3-minute smoke test.

.PARAMETER GraceMinutes
    How long to wait after the window closes before considering a hard stop.

.PARAMETER HardStop
    Permit a forced termination after the grace period. Off by default: without it the
    script waits indefinitely and reports, rather than risking a checkpoint.

.EXAMPLE
    .\run_veltron_training_window.ps1 -Hours 4
    .\run_veltron_training_window.ps1 -Hours 0.05 -DryRun
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [double]$Hours,

    [string]$RunName = 'mini-pretrain',
    [string]$Config  = 'configs/pretrain_mini_evening.yaml',
    [double]$GraceMinutes = 10,
    [switch]$HardStop,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'veltron_common.ps1')

$run = Get-VeltronRun -RunName $RunName -Config $Config

Write-Host ''
Write-Host '=== VeltronLM: bounded training window ===' -ForegroundColor Cyan
Write-Host "  window      : $Hours h ($([math]::Round($Hours*60,1)) min)"
Write-Host "  grace       : $GraceMinutes min"
Write-Host "  hard stop   : $(if ($HardStop) { 'ENABLED' } else { 'disabled (will wait and report)' })"

$before = Get-VeltronState -Run $run
if ($before) {
    Write-Host "  start ckpt  : $(Get-VeltronCheckpointLabel $before)"
} else {
    Write-Host '  start ckpt  : NONE (would train from scratch)'
}

if ($DryRun) {
    Write-Host ''
    Write-Host '  DRY RUN -- nothing started.' -ForegroundColor Cyan
    exit 0
}

# ------------------------------------------------------------------ launch
$startScript = Join-Path $PSScriptRoot 'start_veltron_training.ps1'
# Pass the window through in HOURS, because that is the unit `train.py --max-hours`
# expects. An earlier version multiplied by 3600 and handed the trainer 108 *hours* for a
# 108-second window; the run only stopped because the watcher below caught it, which
# means the trainer's own budget was silently wrong by a factor of 3600.
#
# Both mechanisms are kept on purpose, and the trainer's fires first because it stops at a
# step boundary on its own. This watcher is the backstop.
& $startScript -RunName $RunName -Config $Config -MaxHours $Hours -Background
if ($LASTEXITCODE -ne 0) {
    Write-Host "  start failed or training is already running (exit $LASTEXITCODE)" -ForegroundColor Red
    if ($LASTEXITCODE -eq 10) {
        Write-Host '  (already running is not an error for this script)' -ForegroundColor Yellow
        exit 10
    }
    exit $LASTEXITCODE
}

Start-Sleep -Seconds 5
$proc = @(Get-VeltronTrainProcess -Run $run)
if ($proc.Count -eq 0) {
    Write-Host '  trainer did not start; check the log directory.' -ForegroundColor Red
    exit 7
}
$trainerPid = $proc[0].ProcessId
Write-Host "  trainer PID : $trainerPid"
Write-Host "  watching until $(Get-Date).AddHours($Hours)"
Write-Host ''

# ------------------------------------------------------------------- watch
$started  = Get-Date
$deadline = $started.AddHours($Hours)
$lastReport = Get-Date

while ($true) {
    if (-not (Get-Process -Id $trainerPid -ErrorAction SilentlyContinue)) {
        Write-Host "  trainer exited on its own at $(Get-Date -Format HH:mm:ss)." -ForegroundColor Green
        break
    }
    if ((Get-Date) -ge $deadline) { break }

    # Quiet progress line once a minute; safe to run alongside the trainer because it
    # only reads the JSONL log.
    if (((Get-Date) - $lastReport).TotalSeconds -ge 60) {
        $lastReport = Get-Date
        $s = Get-VeltronState -Run $run
        if ($s) {
            $done = ''
            try {
                $line = Get-Content -LiteralPath $s.log_path -Tail 1 -ErrorAction SilentlyContinue
                if ($line -and $line -match '"step":\s*(\d+)') { $done = "step $($Matches[1])" }
            } catch { }
            $elapsed = (Get-Date) - $started
            $rem = $deadline - (Get-Date)
            Write-Host ("  [{0}] elapsed {1}  remaining {2}  {3}" -f `
                (Get-Date -Format HH:mm:ss), (Format-VeltronDuration $elapsed), `
                (Format-VeltronDuration $rem), $done)
        }
    }
    Start-Sleep -Seconds 10
}

# ------------------------------------------------------------------- close out
$elapsed = (Get-Date) - $started
Write-Host ''
Write-Host '  window closed; asking the trainer to stop cleanly' -ForegroundColor Cyan
Request-VeltronStop -Run $run | Out-Null

$graceDeadline = (Get-Date).AddMinutes($GraceMinutes)
while ((Get-Process -Id $trainerPid -ErrorAction SilentlyContinue) -and (Get-Date) -lt $graceDeadline) {
    Start-Sleep -Seconds 5
}

$stillRunning = $null -ne (Get-Process -Id $trainerPid -ErrorAction SilentlyContinue)
if ($stillRunning) {
    if ($HardStop) {
        Write-Host "  grace period expired; verifying then forcing stop" -ForegroundColor Yellow
        $py = Get-VeltronPython
        $v = & $py (Join-Path $script:VeltronRepo 'scripts\verify_checkpoints.py') $run.RunDir 2>&1
        if (($v | Select-String -Pattern 'invalid:\s*0') -eq $null) {
            Write-Host '  checkpoint does NOT verify; refusing to force-stop.' -ForegroundColor Red
            exit 5
        }
        Stop-Process -Id $trainerPid -Force -ErrorAction SilentlyContinue
        $stillRunning = $false
        Write-Host '  forced stop completed after a verified checkpoint' -ForegroundColor Yellow
    } else {
        Write-Host '  trainer still running after the grace period.' -ForegroundColor Yellow
        Write-Host '  Nothing was forced. Re-run with -HardStop to allow that.'
        exit 6
    }
}

# --------------------------------------------------------------------- report
$after = Get-VeltronState -Run $run
Write-Host ''
Write-Host '=== WINDOW SUMMARY ===' -ForegroundColor Cyan
Write-Host "  starting checkpoint : $(Get-VeltronCheckpointLabel $before)"
Write-Host "  final checkpoint    : $(Get-VeltronCheckpointLabel $after)"
Write-Host "  final step          : $(if ($after) { $after.latest_valid_step } else { 'n/a' })"
Write-Host "  tokens processed    : $(if ($after) { $after.tokens_seen } else { 'n/a' })"
Write-Host "  elapsed             : $(Format-VeltronDuration $elapsed)"
Write-Host "  requested window    : $(Format-VeltronDuration ([timespan]::FromSeconds($Hours*3600)))"
if ($after) {
    Write-Host "  train loss at ckpt  : $($after.train_loss)"
    Write-Host "  val loss at ckpt    : $($after.val_loss)"
    Write-Host "  val perplexity      : $($after.val_perplexity)"
    if ($after.summary) {
        Write-Host "  stop reason         : $($after.summary.stop_reason)"
    }
    Write-Host "  checkpoints on disk : $((($after.all_checkpoints | ForEach-Object { $_.name }) -join ', '))"
    if ($after.invalid.Count -gt 0) {
        Write-Host "  invalid (kept)      : $($after.invalid -join ', ')" -ForegroundColor Yellow
    }
}
$logDir = New-VeltronLogDir
Write-Host "  logs                : $logDir"
Write-Host ''
exit 0