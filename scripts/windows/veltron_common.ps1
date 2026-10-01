<#
.SYNOPSIS
    Shared helpers for the VeltronLM Windows training automation.

.DESCRIPTION
    Single place for paths, process detection, checkpoint discovery and logging so the
    individual scripts cannot drift apart. Every script dot-sources this file.

    Process detection deliberately uses CIM (Win32_Process) rather than the process name,
    because `python` is the name of every Python interpreter on the machine and matching on
    it alone would report unrelated Python work as VeltronLM training.
#>

Set-StrictMode -Version Latest

# Force invariant formatting. The machine's locale renders 1063.7 as "1.063,7", which
# reads as a perplexity of 1.06 rather than 1063. Loss and perplexity numbers must be
# unambiguous.
$script:VeltronCulture = [System.Globalization.CultureInfo]::InvariantCulture

# ------------------------------------------------------------------ paths

$script:VeltronRepo   = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$script:VeltronPython = $null
$script:VeltronLogs    = Join-Path $script:VeltronRepo 'logs'

function Get-VeltronPython {
    <#  Locate the interpreter that has torch installed.
        Honours VELTRON_PYTHON, then a sibling .venv, then whatever `python` resolves to. #>
    if ($script:VeltronPython) { return $script:VeltronPython }

    $candidates = @()
    if ($env:VELTRON_PYTHON) { $candidates += $env:VELTRON_PYTHON }

    $venv = Join-Path $script:VeltronRepo '.venv\Scripts\python.exe'
    if (Test-Path $venv) { $candidates += $venv }

    $candidates += 'python'

    foreach ($c in $candidates) {
        try {
            $null = & $c -c "import torch" 2>&1
            if ($LASTEXITCODE -eq 0) { $script:VeltronPython = $c; return $c }
        } catch { }
    }
    # Fall back to `python` and let the caller discover the failure.
    $script:VeltronPython = 'python'
    return $script:VeltronPython
}

function New-VeltronLogDir {
    if (-not (Test-Path $script:VeltronLogs)) {
        New-Item -ItemType Directory -Path $script:VeltronLogs -Force | Out-Null
    }
    return $script:VeltronLogs
}

# --------------------------------------------------------------- run config

function Get-VeltronRun {
    <#  Defaults mirror configs/pretrain_mini_evening.yaml exactly. Every value can be
        overridden from the command line; none of them are invented. #>
    param(
        [string]$RunName = 'mini-pretrain',
        [string]$Config  = 'configs/pretrain_mini_evening.yaml'
    )
    $cfgFile = Split-Path -Leaf $Config
    return [pscustomobject]@{
        RunName  = $RunName
        Config   = $Config
        # The trainer is launched as `--config <file>`, so the config filename is what
        # actually appears in the command line -- the run name does not. Matching on
        # RunName silently found nothing, which read as "not running" while the trainer
        # was demonstrably using 10.8 GiB of VRAM.
        ProcessMarker = $cfgFile
        RunDir   = Join-Path $script:VeltronRepo "checkpoints\$RunName"
        StopFile = Join-Path $script:VeltronRepo "checkpoints\$RunName\STOP_REQUEST"
        LogFile  = Join-Path $script:VeltronRepo "logs\mini_train.log"
    }
}

# --------------------------------------------------------- process detection

function Get-VeltronTrainProcess {
    <#
      Returns every running `python` whose command line contains BOTH `veltron.train` and
      the configured config filename. Both conditions are required, so a different
      VeltronLM run or an unrelated Python job is never mistaken for this one.

      Matching the *config file* rather than the run name matters: the trainer is invoked
      as `--config configs/pretrain_mini_evening.yaml`, so the run name "mini-pretrain"
      never appears on the command line.

      Safe to call while training is active: it only queries WMI.
    #>
    param([Parameter(Mandatory)] $Run)

    $needle  = 'veltron.train'
    $marker  = $Run.ProcessMarker
    $found   = @()

    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue
    foreach ($p in $procs) {
        if (-not $p.CommandLine) { continue }
        if ($p.CommandLine -like "*$needle*" -and $p.CommandLine -like "*$marker*") {
            $found += [pscustomobject]@{
                ProcessId = $p.ProcessId
                CommandLine = $p.CommandLine
                Started    = $p.CreationDate
            }
        }
    }
    return $found
}

function Get-AnyVeltronTrainProcess {
    <#
      Every running `veltron.train`, whatever config it uses.

      This exists because of a real incident: per-config duplicate detection correctly
      allowed a second trainer to start on a DIFFERENT config, and the two fought over a
      12 GiB card until the real run died with
      "Could not allocate tensor with 134217728 bytes". The atomic checkpoint design meant
      nothing was lost, but the run stopped. On a single-GPU machine any two trainers are
      mutually exclusive regardless of which checkpoint directory they use, so callers must
      check this too.
    #>
    $needle = 'veltron.train'
    $found  = @()
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue
    foreach ($p in $procs) {
        if (-not $p.CommandLine) { continue }
        if ($p.CommandLine -like "*$needle*") {
            $found += [pscustomobject]@{
                ProcessId = $p.ProcessId
                CommandLine = $p.CommandLine
            }
        }
    }
    return $found
}

function Test-VeltronTrainingRunning {
    param([Parameter(Mandatory)] $Run)
    $p = Get-VeltronTrainProcess -Run $Run
    return (@($p).Count -gt 0)
}

# ------------------------------------------------------- checkpoint discovery

function Get-VeltronState {
    <#  Delegates to scripts/veltron_state.py, which reuses the trainer's own
        CheckpointManager. PowerShell therefore cannot disagree with the trainer about
        which checkpoint is resumable. #>
    param([Parameter(Mandatory)] $Run)

    $py = Get-VeltronPython
    $stateScript = Join-Path $script:VeltronRepo 'scripts\veltron_state.py'

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName  = $py
    $psi.Arguments = "scripts\veltron_state.py --run `"$($Run.RunDir)`" --compact"
    $psi.WorkingDirectory = $script:VeltronRepo
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.CreateNoWindow = $true

    $proc = [System.Diagnostics.Process]::Start($psi)
    $stdout = $proc.StandardOutput.ReadToEnd()
    $proc.WaitForExit()

    if (-not $stdout.Trim()) { return $null }
    try { return $stdout | ConvertFrom-Json } catch { return $null }
}

function Get-VeltronLatestValidCheckpoint {
    param([Parameter(Mandatory)] $Run)
    $s = Get-VeltronState -Run $Run
    if ($s -and $s.latest_valid) { return $s.latest_valid }
    return $null
}

function Get-VeltronStopFile {
    param([Parameter(Mandatory)] $Run)
    return $Run.StopFile
}

# ------------------------------------------------------------ stop requests

function Request-VeltronStop {
    <#  Creates the cooperative stop flag. A trainer started with `stop_file` configured
        sees it at the next step boundary, finishes the in-flight step, writes any due
        checkpoint, then exits through its normal final-save path. #>
    param([Parameter(Mandatory)] $Run)

    $dir = Split-Path -Parent $Run.StopFile
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    Set-Content -LiteralPath $Run.StopFile -Value (Get-Date).ToString('o') -Encoding utf8
    return $Run.StopFile
}

function Clear-VeltronStop {
    param([Parameter(Mandatory)] $Run)
    if (Test-Path $Run.StopFile) { Remove-Item -LiteralPath $Run.StopFile -Force }
}

# -------------------------------------------------------------- environment

function Set-VeltronEnvironment {
    <#  The repo is used via PYTHONPATH rather than an installed package, and Serbian
        output crashes the cp1252 console, hence PYTHONIOENCODING. #>
    param([int]$RunLogLevel = 1)

    $env:PYTHONPATH = $script:VeltronRepo
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONUNBUFFERED = '1'
    $env:VELTRON_LOG_LEVEL = 'INFO'
    return $env:PYTHONPATH
}

# --------------------------------------------------------------- formatting

function Format-VeltronBytes {
    param([double]$Bytes)
    if ($Bytes -le 0) { return '0 B' }
    $units = @('B', 'KiB', 'MiB', 'GiB', 'TiB')
    $i = 0
    $v = $Bytes
    while ($v -ge 1024 -and $i -lt $units.Count - 1) { $v /= 1024; $i++ }
    return ('{0:N2} {1}' -f $v, $units[$i], $script:VeltronCulture)
}

function Get-VeltronLeaf {
    <#  Directory name of a path, tolerating $null.
        `Split-Path -Leaf $null` raises a parameter-binding error under StrictMode, which
        is what happens whenever a run directory does not exist yet -- a normal state for
        a first run, and exactly the case the automation must survive. #>
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return '<none>' }
    return (Split-Path -Leaf $Path)
}

function Get-VeltronCheckpointLabel {
    <#  "step-00000500 (step 500)" or "<none>" when nothing is resumable yet. #>
    param($State)
    if (-not $State -or -not $State.latest_valid) { return '<none>' }
    return ('{0} (step {1})' -f (Get-VeltronLeaf $State.latest_valid), $State.latest_valid_step)
}

function Format-VeltronDuration {
    param([timespan]$Span)
    $h = [math]::Floor($Span.TotalHours)
    $m = [math]::Floor($Span.TotalMinutes % 60)
    $s = [math]::Floor($Span.TotalSeconds % 60)
    return ('{0}h {1}m {2}s' -f $h, $m, $s)
}