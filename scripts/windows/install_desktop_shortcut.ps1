<#
.SYNOPSIS
    Create desktop shortcuts for the VeltronLM training workflow.

.DESCRIPTION
    Creates shortcuts on the current user's Desktop:

        VeltronLM - Resume Training   runs a bounded training window and stops cleanly
        VeltronLM - Stop Training     safe stop, never a raw process kill
        VeltronLM - Status            read-only dashboard, safe while training runs

    Shortcuts are per-user (the Desktop path can be redirected by OneDrive), so they are
    created where this user actually sees them and are not committed to the repository.

.PARAMETER Hours
    Window length baked into the Resume shortcut. Edit the .cmd to change it later.

.EXAMPLE
    .\install_desktop_shortcut.ps1
    .\install_desktop_shortcut.ps1 -Hours 2
#>
[CmdletBinding()]
param(
    [double]$Hours = 8,
    [string]$RunName = 'mini-pretrain'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'veltron_common.ps1')

$desktop = [Environment]::GetFolderPath('Desktop')
if (-not $desktop -or -not (Test-Path $desktop)) {
    Write-Host "ERROR: could not resolve the Desktop folder for this user." -ForegroundColor Red
    exit 2
}

Write-Host ''
Write-Host '=== VeltronLM: desktop shortcuts ===' -ForegroundColor Cyan
Write-Host "  desktop   : $desktop"
Write-Host "  repository: $($script:VeltronRepo)"
Write-Host "  window    : $Hours h"
Write-Host ''

$shell = New-Object -ComObject WScript.Shell

function New-VeltronShortcut {
    param(
        [string]$Name,
        [string]$Target,
        [string]$Arguments,
        [string]$IconPath,
        [int]$IconIndex = 0
    )
    $path = Join-Path $desktop $Name
    $sc = $shell.CreateShortcut($path)
    $sc.TargetPath       = $Target
    $sc.Arguments        = $Arguments
    $sc.WorkingDirectory = $script:VeltronRepo
    $sc.WindowStyle      = 1
    $sc.Description      = 'VeltronLM training automation'
    if ($IconPath -and (Test-Path $IconPath)) { $sc.IconLocation = "$IconPath,$IconIndex" }
    $sc.Save()
    return $path
}

$psExe = (Get-Command powershell.exe).Source
$cmdBat = Join-Path $PSScriptRoot 'VeltronLM Resume Training.cmd'

# 1. Resume: a .cmd launcher, so the window stays open with a summary at the end.
$resumePath = Join-Path $desktop 'VeltronLM - Resume Training.lnk'
if (Test-Path $cmdBat) {
    $sc = $shell.CreateShortcut($resumePath)
    $sc.TargetPath       = $cmdBat
    $sc.WorkingDirectory = $script:VeltronRepo
    $sc.WindowStyle      = 1
    $sc.Description      = "VeltronLM - resume training for $Hours hours"
    $sc.Save()
    Write-Host "  created : $resumePath"
} else {
    # Fall back to PowerShell if the .cmd is missing.
    New-VeltronShortcut -Name 'VeltronLM - Resume Training.lnk' `
        -Target $psExe `
        -Arguments "-NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $PSScriptRoot 'run_veltron_training_window.ps1')`" -Hours $Hours -RunName $RunName" | Out-Null
    Write-Host "  created (PowerShell fallback) : $resumePath"
}

# 2. Stop.
$stopPath = New-VeltronShortcut -Name 'VeltronLM - Stop Training.lnk' `
    -Target $psExe `
    -Arguments "-NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $PSScriptRoot 'stop_veltron_training.ps1')`" -RunName $RunName -Force"
Write-Host "  created : $stopPath"

# 3. Status.
$statusPath = New-VeltronShortcut -Name 'VeltronLM - Status.lnk' `
    -Target $psExe `
    -Arguments "-NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $PSScriptRoot 'status_veltron_training.ps1')`" -RunName $RunName -Watch"
Write-Host "  created : $statusPath"

Write-Host ''
Write-Host '  Resume double-clicks a bounded window and stops it cleanly at the end.' -ForegroundColor Gray
Write-Host '  Stop waits for the next checkpoint, verifies it, then stops.' -ForegroundColor Gray
Write-Host '  Status is read-only and safe to run during training.' -ForegroundColor Gray
Write-Host ''
exit 0