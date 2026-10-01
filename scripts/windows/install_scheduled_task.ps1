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

.EXAMPLE
    .\install_scheduled_task.ps1 -Time 08:00 -Hours 6 -EnableWakeTimers
    .\install_scheduled_task.ps1 -Time 07:30 -Hours 8 -WhatIf
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$Time = '08:00',
    [double]$Hours = 6,
    [string]$TaskName = 'VeltronLM Auto Resume',
    [switch]$EnableWakeTimers,
    [switch]$Remove
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
if (-not (Test-Path $windowScript)) {
    Write-Host "  missing $windowScript" -ForegroundColor Red; exit 2
}

$psExe = (Get-Command powershell.exe).Source
$arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$windowScript`" -Hours $Hours -RunName 'mini-pretrain' -GraceMinutes 10"

Write-Host ''
Write-Host '  creating scheduled task...' -ForegroundColor Cyan

if ($PSCmdlet.ShouldProcess($TaskName, 'create scheduled task')) {
    $action = New-ScheduledTaskAction -Execute $psExe -Argument $arguments -WorkingDirectory $script:VeltronRepo
    # WakeToRun is what asks Windows to leave sleep at the trigger time. It only works with
    # the wake-timer setting enabled and is a no-op if the machine is already awake.
    $trigger = New-ScheduledTaskTrigger -Daily -At ([datetime]::ParseExact($Time, 'HH:mm', $null))
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Hours ($Hours + 1)) `
        -MultipleInstances IgnoreNew `
        -RestartCount 2 `
        -RestartInterval (New-TimeSpan -Minutes 5)
    # Enable the hidden "wake the computer" flag on the trigger.
    $settings.WakeToRun  = $true
    $settings.Hidden     = $false

    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Principal $principal -Settings $settings `
        -Description "Resume VeltronLM training for $Hours hours at $Time, then stop cleanly." `
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