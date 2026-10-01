<#
.SYNOPSIS
    Start or resume VeltronLM training.

.DESCRIPTION
    Uses the repository's existing training command and resume mechanism verbatim:

        python -m veltron.train --config configs/pretrain_mini_evening.yaml

    That config already sets `resume: auto`, which makes the trainer restore from the
    newest checkpoint that passes its own validation. No CLI flag is invented here and the
    command line is not augmented, so there is nothing for this script to get wrong.

    Behaviour:
      * refuses to start if a trainer is already running (no duplicate jobs),
      * clears any stale stop flag, otherwise the trainer would exit immediately,
      * enables the cooperative stop file so a later graceful stop is possible,
      * reports which checkpoint will be resumed, BEFORE launching,
      * logs stdout and stderr to a persistent file,
      * exits with the trainer's own exit code.

.PARAMETER RunName
    Checkpoint run directory name. Defaults to mini-pretrain.

.PARAMETER MaxHours
    Overrides the config's wall-clock budget. The trainer stops cleanly at a step boundary.

.PARAMETER Foreground
    Run in this shell (default) so output is visible. Use -Background to detach.

.PARAMETER DryRun
    Report exactly what would happen and exit without launching anything.

.EXAMPLE
    .\start_veltron_training.ps1
    .\start_veltron_training.ps1 -DryRun
    .\start_veltron_training.ps1 -Background
#>
[CmdletBinding()]
param(
    [string]$RunName = 'mini-pretrain',
    [string]$Config  = 'configs/pretrain_mini_evening.yaml',
    [double]$MaxHours = 0,
    [switch]$Background,
    [switch]$DryRun,
    [switch]$ForceConcurrent
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'veltron_common.ps1')

$run = Get-VeltronRun -RunName $RunName -Config $Config

Write-Host ''
Write-Host '=== VeltronLM: start / resume ===' -ForegroundColor Cyan
Write-Host "  repository : $($script:VeltronRepo)"
Write-Host "  run        : $RunName"
Write-Host "  config     : $Config"

# --------------------------------------------------- duplicate-process protection
$existing = @(Get-VeltronTrainProcess -Run $run)
$state = Get-VeltronState -Run $run

if ($existing.Count -gt 0) {
    Write-Host ''
    Write-Host '  ALREADY RUNNING -- not starting a second trainer.' -ForegroundColor Yellow
    foreach ($p in $existing) {
        Write-Host "    PID $($p.ProcessId)  $($p.CommandLine)"
    }
    if ($state) {
        Write-Host "    latest checkpoint : $(Get-VeltronCheckpointLabel $state)"
    }
    Write-Host '  Use status_veltron_training.ps1 for live progress.' -ForegroundColor Yellow
    Write-Host ''
    exit 10
}

# A trainer on a DIFFERENT config is still a trainer: on one GPU the two will exhaust
# VRAM between them. This check exists because per-config detection alone allowed exactly
# that to happen once already.
$anyTrainer = @(Get-AnyVeltronTrainProcess)
if ($anyTrainer.Count -gt 0) {
    Write-Host ''
    Write-Host '  ANOTHER VELTRON TRAINER IS RUNNING (different config).' -ForegroundColor Red
    foreach ($p in $anyTrainer) {
        Write-Host "    PID $($p.ProcessId)  $($p.CommandLine)" -ForegroundColor Yellow
    }
    Write-Host ''
    Write-Host '  On a single GPU two trainers exhaust VRAM between them; the last one to' -ForegroundColor Red
    Write-Host '  start dies with "There is not enough GPU video memory available!".' -ForegroundColor Red
    Write-Host '  Stop that run first:' -ForegroundColor Red
    Write-Host '    .\scripts\windows\stop_veltron_training.ps1 -RunName <its run name>' -ForegroundColor Yellow
    Write-Host ''
    Write-Host '  Overriding is possible with -ForceConcurrent but it will likely OOM.' -ForegroundColor DarkGray
    if (-not $ForceConcurrent) { exit 11 }
}

# ------------------------------------------------------------------- preflight
$configPath = Join-Path $script:VeltronRepo $Config
if (-not (Test-Path $configPath)) {
    Write-Host "  config not found: $configPath" -ForegroundColor Red
    exit 2
}

$resumeFrom = '<none -- this will start from scratch>'
if ($state -and $state.latest_valid) {
    $resumeFrom = "$(Get-VeltronCheckpointLabel $state)"
    if ($state.invalid -and $state.invalid.Count -gt 0) {
        Write-Host ''
        Write-Host "  note: ignoring invalid checkpoint(s): $($state.invalid -join ', ')" -ForegroundColor Yellow
        Write-Host '        (not deleted, and not used for resume)' -ForegroundColor Yellow
    }
} else {
    Write-Host ''
    Write-Host '  WARNING: no valid checkpoint found. Training would start from step 0.' -ForegroundColor Red
}

$py = Get-VeltronPython
Write-Host ''
Write-Host "  python      : $py"
Write-Host "  resuming    : $resumeFrom"
Write-Host "  checkpoint dir: $($run.RunDir)"

# Build the argument list. This is the repository's documented command, unchanged.
$argList = @('-m', 'veltron.train', '--config', $Config)
if ($MaxHours -gt 0) { $argList += @('--max-hours', "$MaxHours") }
$cmdLine = "python $($argList -join ' ')"

# The stop flag is enabled via --set, which is an existing train.py argument for config
# overrides. It is the only addition, and it changes no training hyperparameter.
$argList += @('--set', "stop_file=$($run.StopFile)")

Write-Host "  command     : $cmdLine --set stop_file=<stopfile>"
Write-Host "  stop file   : $($run.StopFile)"
if ($MaxHours -gt 0) { Write-Host "  time budget : $MaxHours h (stops at a step boundary)" }

if ($DryRun) {
    Write-Host ''
    Write-Host '  DRY RUN -- nothing was started.' -ForegroundColor Cyan
    exit 0
}

# ------------------------------------------------------- environment + stop flag
Set-VeltronEnvironment | Out-Null

# A stop flag left over from a previous session would make the trainer exit immediately.
if (Test-Path $run.StopFile) {
    Clear-VeltronStop -Run $run
    Write-Host '  cleared a stale stop flag from a previous session' -ForegroundColor Yellow
}

if (-not (Test-Path $run.RunDir)) {
    New-Item -ItemType Directory -Path $run.RunDir -Force | Out-Null
}
$logDir = New-VeltronLogDir
$logFile = Join-Path $logDir "mini_train_$(Get-Date -Format 'yyyyMMdd-HHmmss').log"

Write-Host "  log file    : $logFile"
Write-Host ''
Write-Host '=== training output ===' -ForegroundColor Cyan

if ($Background) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName  = $py
    $psi.Arguments = ($argList | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '
    $psi.WorkingDirectory = $script:VeltronRepo
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.CreateNoWindow = $true

    $proc = [System.Diagnostics.Process]::Start($psi)
    # Tee both streams to the log file. Background streams must be drained concurrently
    # or the pipe buffer fills and the trainer blocks.
    $writer = [System.IO.StreamWriter]::new($logFile, $false)
    $writer.AutoFlush = $true
    Register-ObjectEvent -InputObject $proc -EventName OutputDataReceived -Action {
        if ($EventArgs.Data) { Add-Content -LiteralPath $using:logFile -Value $EventArgs.Data }
    } | Out-Null
    Register-ObjectEvent -InputObject $proc -EventName ErrorDataReceived -Action {
        if ($EventArgs.Data) { Add-Content -LiteralPath $using:logFile -Value $EventArgs.Data }
    } | Out-Null
    $proc.BeginOutputReadLine()
    $proc.BeginErrorReadLine()

    Set-Content -LiteralPath (Join-Path $run.RunDir 'AUTORUN_PID') -Value $proc.Id -Encoding ascii
    Write-Host "  background PID $($proc.Id), logging to $logFile"
    Write-Host ''
    exit 0
}

# ------------------------------------------------------------------ foreground
# stderr must NOT be merged into the success stream here. PyTorch writes deprecation
# notices (e.g. the pynvml warning) to stderr on every import, and under
# ErrorActionPreference='Stop' a merged stderr stream turns a harmless warning into a
# terminating NativeCommandError -- the script then reports failure for a run that
# actually succeeded. Instead stderr goes to its own file and the preference is relaxed
# only for the duration of the call.
$errFile = "$logFile.err"
$prevPref = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    & $py @argList 2> $errFile | Tee-Object -FilePath $logFile
    $code = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $prevPref
}

Write-Host ''
Write-Host "  trainer exited with code $code"
if ((Test-Path $errFile) -and (Get-Item $errFile).Length -gt 0) {
    Write-Host '  --- stderr (last 10 lines) ---'
    Get-Content -LiteralPath $errFile -Tail 10 -ErrorAction SilentlyContinue |
        ForEach-Object { Write-Host "    $_" }
}
$after = Get-VeltronState -Run $run
if ($after) {
    Write-Host "  checkpoint kept : $(Get-VeltronCheckpointLabel $after)"
    if ($after.summary) {
        Write-Host "  stop reason     : $($after.summary.stop_reason)"
        Write-Host "  final val loss  : $($after.summary.final_val_loss)"
    }
}
exit $code