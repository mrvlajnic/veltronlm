<#
.SYNOPSIS
    Show VeltronLM training status. Safe to run while training is active.

.DESCRIPTION
    Reads only. It queries WMI for the process, reads the JSONL training log, and calls
    the repository's own state helper. It never writes to the checkpoint directory and
    never opens the checkpoint files for writing, so running it during training cannot
    disturb the run.

.EXAMPLE
    .\status_veltron_training.ps1
    .\status_veltron_training.ps1 -RunName micro-pretrain
    .\status_veltron_training.ps1 -Watch
#>
[CmdletBinding()]
param(
    [string]$RunName = 'mini-pretrain',
    [switch]$Watch,
    [int]$WatchSeconds = 20
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'veltron_common.ps1')

do {
    $run = Get-VeltronRun -RunName $RunName
    Clear-Host -ErrorAction SilentlyContinue
    Write-Host ''
    Write-Host '=== VeltronLM status ===' -ForegroundColor Cyan
    Write-Host "  repository : $($script:VeltronRepo)"
    Write-Host "  run        : $RunName"
    Write-Host "  time       : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host ''

    # ------------------------------------------------------------- process
    $procs = @(Get-VeltronTrainProcess -Run $run)
    if ($procs.Count -gt 0) {
        $p = $procs[0]
        Write-Host 'PROCESS' -ForegroundColor Cyan
        Write-Host "  training   : RUNNING" -ForegroundColor Green
        Write-Host "  PID        : $($p.ProcessId)"
        try {
            $gp = Get-Process -Id $p.ProcessId
            $mins = [math]::Round($gp.CPU / 60, 1)
            Write-Host "  CPU time   : $mins min"
            Write-Host "  RAM        : $(Format-VeltronBytes $gp.WorkingSet64)"
            $minutes = [math]::Round(((Get-Date) - $gp.StartTime).TotalMinutes, 1)
            Write-Host "  uptime     : $minutes min"
        } catch { }
    } else {
        Write-Host 'PROCESS' -ForegroundColor Cyan
        Write-Host '  training   : not running' -ForegroundColor Yellow
    }

    # -------------------------------------------------------------- GPU
    try {
        $gpu = Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory -ErrorAction SilentlyContinue |
               Where-Object { $_.DedicatedUsage -gt 50MB } | Sort-Object DedicatedUsage -Descending
        if ($gpu) {
            Write-Host ''
            Write-Host 'GPU (dedicated VRAM)' -ForegroundColor Cyan
            foreach ($g in $gpu) {
                $pid_ = [int]($g.Name -replace '^pid_(\d+).*', '$1')
                $nm = try { (Get-Process -Id $pid_ -ErrorAction Stop).ProcessName } catch { '<gone>' }
                Write-Host ("  {0,-22} {1,7:N2} GiB" -f $nm, ($g.DedicatedUsage / 1GB), $script:VeltronCulture)
            }
        }
    } catch { }

    # -------------------------------------------------------- checkpoints
    $state = Get-VeltronState -Run $run
    Write-Host ''
    Write-Host 'CHECKPOINTS' -ForegroundColor Cyan
    if (-not $state -or -not $state.exists) {
        Write-Host "  directory not found: $($run.RunDir)" -ForegroundColor Red
    } elseif (-not $state.latest_valid) {
        Write-Host '  latest valid : NONE' -ForegroundColor Red
    } else {
        Write-Host "  latest valid : $(Get-VeltronCheckpointLabel $state)"
        Write-Host "  step         : $($state.latest_valid_step)"
        Write-Host "  tokens seen  : $($state.tokens_seen)"
        Write-Host "  train loss   : $($state.train_loss)"
        Write-Host "  val loss     : $($state.val_loss)"
        Write-Host "  val perplex. : $($state.val_perplexity)"
        if ($state.model)       { Write-Host "  model        : $($state.model) ($($state.parameters) params)" }
        if ($state.dataset_version) { Write-Host "  dataset      : $($state.dataset_version)" }
        if ($state.best)        { Write-Host "  best         : $(Split-Path -Leaf $state.best)" }
        if ($state.invalid.Count -gt 0) {
            Write-Host "  INVALID (kept, not deleted): $($state.invalid -join ', ')" -ForegroundColor Yellow
        }
        $ck = $state.all_checkpoints | Sort-Object step
        $totalBytes = ($ck | Measure-Object -Property bytes -Sum).Sum
        Write-Host "  on disk      : $($ck.Count) checkpoint(s), $(Format-VeltronBytes $totalBytes)"
        foreach ($c in ($ck | Select-Object -Last 6)) {
            $flag = if ($c.valid) { 'ok  ' } else { 'BAD ' }
            Write-Host ("    {0} {1,-18} step {2,-6} val {3}" -f $flag, $c.name, $c.step, $c.val_loss)
        }
        if ($state.disk_free_gib) {
            $warn = if ($state.disk_free_gib -lt 20) { ' <-- LOW, reduce keep_last' } else { '' }
            Write-Host "  free disk    : $($state.disk_free_gib) GiB$warn" -ForegroundColor $(if ($state.disk_free_gib -lt 20) { 'Yellow' } else { 'Gray' })
        }
    }

    # -------------------------------------------------------------- log
    Write-Host ''
    Write-Host 'LOG' -ForegroundColor Cyan
    if ($state -and $state.log_path -and (Test-Path $state.log_path)) {
        Write-Host "  training log : $($state.log_path)"
        $lines = @(Get-Content -LiteralPath $state.log_path -Tail 400 -ErrorAction SilentlyContinue)
        $evals = @($lines | Where-Object { $_ -match '"eval"' })
        if ($evals.Count -gt 0) {
            Write-Host ''
            Write-Host '  evaluations' -ForegroundColor Cyan
            foreach ($e in ($evals | Select-Object -Last 5)) {
                try {
                    $o = $e | ConvertFrom-Json
                    Write-Host ("    step {0,-6} val_loss {1,8:N4}  ppl {2,10:N1}" -f `
                        $o.step, $o.val_loss, $o.val_perplexity, $script:VeltronCulture)
                } catch { }
            }
        }
    }
    $logDir = New-VeltronLogDir
    Write-Host "  session logs : $logDir"

    if ($state -and $state.summary_path) {
        Write-Host ''
        Write-Host "  summary      : $($state.summary_path)"
        if ($state.summary -and $state.summary.stop_reason) {
            Write-Host "  stop reason  : $($state.summary.stop_reason)"
        }
    }

    if ($Watch) {
        Write-Host ''
        Write-Host "  (watching, Ctrl-C to exit)" -ForegroundColor DarkGray
        Start-Sleep -Seconds $WatchSeconds
    }
} while ($Watch)

Write-Host ''
exit 0