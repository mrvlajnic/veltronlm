<#
.SYNOPSIS
    Create a Task Scheduler task that resumes VeltronLM training at a scheduled time.

.DESCRIPTION
    Creates "VeltronLM Auto Resume", which runs the bounded-window script so the run ends
    cleanly and always leaves a valid checkpoint.

    The script first REPORTS the machine's real sleep capability, because whether Windows
    can wake the PC at a scheduled time depends entirely on the hardware and firmware:

      * Standby (S3)  -> possible with the Windows "wake timers" power setting, which is
                         OFF by default on many machines. Enabled by -EnableWakeTimers.
      * Hibernate     -> Task Scheduler can wake from a *scheduled* timer only if the BIOS
                         exposes an RTC alarm. Windows cannot verify this; it is reported
                         as unverifiable rather than assumed.
      * Shutdown       -> the machine must be powered on. Task Scheduler cannot start a PC
                         that is fully powered off without a BIOS RTC alarm or Wake-on-LAN.

    Nothing here touches BIOS/UEFI settings.

    Duplicate protection is inside the window script, so a task that fires while training
    is already running will report the existing PID and exit rather than starting a second
    trainer.

.PARAMETER Time
    Start time, HH:mm. Default 08:00.

.PARAMETER Hours
    Window length in hours. Default 6.

.PARAMETER EnableWakeTimers
    Turn on the Windows wake-timer power setting (AC). Needed to wake from S3 sleep.

.PARAMETER TaskName
    Task name. Default "VeltronLM Auto Resume".

.PARAMETER EveryMinutes
    Re-check on this interval instead of once a day. The task is a SAFETY NET: it runs
    `start_veltron_training.ps1`, which refuses to start a trainer when one is already
    running. So with -EveryMinutes 30 the task fires often, does nothing while training is
    healthy, and restarts it within 30 minutes if it dies. With the default daily trigger a
    crash would cost the rest of the day.

.PARAMETER MaxHours
    Window handed to the trainer when the task does start it. This is what bounds a single
    unattended run.

.EXAMPLE
    .\install_scheduled_task.ps1 -Time 08:00 -Hours 6 -EnableWakeTimers
    .\install_scheduled_task.ps1 -EveryMinutes 30 -MaxHours 8 -TaskName "VeltronLM Keepalive"
    .\install_scheduled_task.ps1 -Remove
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$Time = '08:00',
    [double]$Hours = 6,
    [string]$TaskName = 'VeltronLM Auto Resume',
    [switch]$EnableWakeTimers,
    [switch]$Remove,
    [int]$EveryMinutes = 0,
    [double]$MaxHours = 0
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'veltron_common.ps1')

Write-Host ''
Write-Host '=== VeltronLM: scheduled auto-resume ===' -ForegroundColor Cyan
Write-Host "  task name : $TaskName"
Write-Host "  start     : $Time"
Write-Host "  window    : $Hours h"

# ------------------------------------------------------------------- inspect
Write-Host ''
Write-Host '  machine sleep capability (reported, not assumed)' -ForegroundColor Cyan
$sleepAvail = (powercfg /a 2>&1) -join "`n"
$hasS0    = $sleepAvail -match 'S0 Low Power Idle'
$hasS3    = $sleepAvail -match 'Standby \(S3\)'
$hasHiber = $sleepAvail -match '(?m)^\s*Hibernate\s*$'
$hasS0ix  = $sleepAvail -match 'Modern Standby'

Write-Host "    Modern Standby (S0 low power idle) : $(if ($hasS0) { 'present' } else { 'absent' })"
Write-Host "    Standby S3                        : $(if ($hasS3) { 'available' } else { 'not available' })"
Write-Host "    Hibernate                         : $(if ($hasHiber) { 'available' } else { 'not available' })"

# Current wake-timer setting for the active scheme.
$schemeLine = (powercfg /getactivescheme 2>&1) -join ''
$schemeGuid = if ($schemeLine -match '([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})') { $Matches[1] } else { '' }
$wakeIndex = $null
if ($schemeGuid) {
    $q = (powercfg /query $schemeGuid SUB_SLEEP STANDBYIDLE 2>&1) -join "`n"
    if ($q -match 'Current AC Power Setting Index:\s*0x([0-9a-fA-F]+)') {
        $wakeIndex = [Convert]::ToInt32($Matches[1], 16)
    }
    $schemeName = if ($schemeLine -match '\((.+?)\)\s*$') { $Matches[1] } else { 'unknown' }
}
Write-Host "    active power scheme                 : $schemeName"
if ($null -ne $wakeIndex) {
    Write-Host "    wake timers (AC)                   : $(if ($wakeIndex -eq 1) { 'ENABLED' } else { 'DISABLED' })" `
        -ForegroundColor $(if ($wakeIndex -eq 1) { 'Green' } else { 'Yellow' })
} else {
    Write-Host '    wake timers (AC)                   : could not read (setting not exposed)'
}

# ------------------------------------------------------------------- remove
if ($Remove) {
    Write-Host ''
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "  removed scheduled task: $TaskName" -ForegroundColor Yellow
    } else {
        Write-Host "  no scheduled task named '$TaskName'"
    }
    exit 0
}

# -------------------------------------------------------------- enable wake
if ($EnableWakeTimers) {
    if (-not $schemeGuid) { Write-Host '  could not resolve the active power scheme.' -ForegroundColor Red; exit 3 }
    Write-Host ''
    Write-Host "  enabling wake timers on scheme $schemeGuid (AC only)..." -ForegroundColor Yellow
    if ($PSCmdlet.ShouldProcess('power scheme wake timers', 'set to enabled')) {
        powercfg /setacvalueindex $schemeGuid SUB_SLEEP STANDBYIDLE 1 | Out-Null
        powercfg /setactive $schemeGuid | Out-Null
        Write-Host '  wake timers enabled.' -ForegroundColor Green
    } else {
        Write-Host '  (what-if: not applied)'
    }
}

# ------------------------------------------------------------------- create
$windowScript = Join-Path $PSScriptRoot 'run_veltron_training_window.ps1'
$startScript  = Join-Path $PSScriptRoot 'start_veltron_training.ps1'
if (-not (Test-Path $windowScript)) {
    Write-Host "  missing $windowScript" -ForegroundColor Red; exit 2
}

$psExe = (Get-Command powershell.exe).Source

# Two shapes of task:
#   EveryMinutes > 0  -> a keepalive safety net. It calls start_veltron_training.ps1,
#                        which exits 10/11 when a trainer is already running, so repeated
#                        firings are harmless and a crashed run is restarted automatically.
#   otherwise          -> a daily window that runs the bounded-window script.
if ($EveryMinutes -gt 0) {
    $bh = if ($MaxHours -gt 0) { $MaxHours } else { $Hours }
    $arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$startScript`" -RunName 'mini-pretrain' -MaxHours $bh -Background"
} else {
    $arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$windowScript`" -Hours $Hours -RunName 'mini-pretrain' -GraceMinutes 10"
}

Write-Host ''
Write-Host '  creating scheduled task...' -ForegroundColor Cyan

if ($PSCmdlet.ShouldProcess($TaskName, 'create scheduled task')) {
    $action = New-ScheduledTaskAction -Execute $psExe -Argument $arguments -WorkingDirectory $script:VeltronRepo
    if ($EveryMinutes -gt 0) {
        # RepetitionInterval/RepetitionDuration are the supported cmdlet parameters.
        # Assigning to $trigger.Repetition.Interval fails: the Repetition object returned
        # by New-ScheduledTaskTrigger has no settable Interval property.
        $trigger = New-ScheduledTaskTrigger `
            -Once `
            -At (Get-Date).AddMinutes(2) `
            -RepetitionInterval (New-TimeSpan -Minutes $EveryMinutes) `
            -RepetitionDuration (New-TimeSpan -Days 1)
    } else {
        $trigger = New-ScheduledTaskTrigger -Daily -At ([datetime]::ParseExact($Time, 'HH:mm', $null))
    }
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Hours ($Hours + 1)) `
        -MultipleInstances IgnoreNew `
        -RestartCount 2 `
        -RestartInterval (New-TimeSpan -Minutes 5)
    # Ask Windows to leave sleep at the trigger time. No-op when the machine is awake,
    # which is the normal case here since sleep is disabled.
    $settings.WakeToRun  = $true
    $settings.Hidden     = $false

    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Principal $principal -Settings $settings `
        -Description "VeltronLM training safety net. Refuses to start when a trainer is already running." `
        -Force | Out-Null

    Write-Host "  created: $TaskName" -ForegroundColor Green
} else {
    Write-Host '  (what-if: task not created)'
    Write-Host "  would run: $psExe $arguments"
    exit 0
}

# -------------------------------------------------------------------- report
Write-Host ''
Write-Host '  TASK SETTINGS' -ForegroundColor Cyan
Write-Host "    name              : $TaskName"
Write-Host "    runs at           : daily at $Time"
Write-Host "    command           : $psExe $arguments"
Write-Host "    working directory : $($script:VeltronRepo)"
Write-Host "    wake to run       : enabled (requires wake timers on)"
Write-Host "    multiple instances: IgnoreNew (a second trigger will not start a duplicate)"
Write-Host "    logs              : $($script:VeltronLogs)"
Write-Host ''
Write-Host '  MANAGE' -ForegroundColor Cyan
Write-Host "    run now       : Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "    status        : Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
Write-Host "    disable       : Disable-ScheduledTask -TaskName '$TaskName'"
Write-Host "    cancel/delete : .\install_scheduled_task.ps1 -Remove"
Write-Host ''
Write-Host '  WAKE BEHAVIOUR -- read this before relying on it' -ForegroundColor Yellow
Write-Host "    * Waking from SLEEP (S3): works once the wake-timer setting is on."
if ($null -ne $wakeIndex -and $wakeIndex -ne 1) {
        Write-Host "      Currently DISABLED. Re-run this script with -EnableWakeTimers." -ForegroundColor Yellow
    } else {
        Write-Host "      Currently enabled." -ForegroundColor Green
    }
Write-Host "    * Waking from HIBERNATION: needs an RTC alarm in BIOS/UEFI." -ForegroundColor Yellow
Write-Host "      Windows cannot verify this. Enable 'Wake on RTC alarm' / 'Resume by" -ForegroundColor Yellow
Write-Host "      RTC' / 'Wake on date' in your firmware if the machine must wake from" -ForegroundColor Yellow
Write-Host "      hibernation. Windows does not touch firmware settings." -ForegroundColor Yellow
Write-Host "    * Waking from FULL SHUTDOWN: not possible via Task Scheduler alone." -ForegroundColor Yellow
Write-Host "      Needs a BIOS RTC alarm or Wake-on-LAN with a second always-on device." -ForegroundColor Yellow
Write-Host ''
exit 0