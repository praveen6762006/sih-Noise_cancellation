# Live status of the downloads. Refreshes every 3 seconds. Ctrl+C to stop.
#   powershell -ExecutionPolicy Bypass -File scripts\status.ps1        (live)
#   powershell -ExecutionPolicy Bypass -File scripts\status.ps1 -Once  (print once)
param([switch]$Once)
$root = Split-Path -Parent $PSScriptRoot
$dl = Join-Path $root "data\downloads"

function Bar($frac, $w = 30) {
    $frac = [Math]::Max(0, [Math]::Min(1, $frac)); $n = [int]($frac * $w)
    return "[" + ("#" * $n) + ("." * ($w - $n)) + "]"
}
function Row($label, $bytes, $total, $note = "") {
    $pct = if ($total -gt 0) { $bytes / $total } else { 0 }
    "{0,-26} {1} {2,5:P0}  {3,8:N0} / {4,-8:N0} MB {5}" -f $label, (Bar $pct), $pct, ($bytes / 1MB), ($total / 1MB), $note
}

$prev = $null; $prevT = $null
while ($true) {
    $now = Get-Date
    $rx = (Get-NetAdapterStatistics | Where-Object { $_.ReceivedBytes -gt 1e8 } | Measure-Object ReceivedBytes -Sum).Sum
    $speed = if ($prev) { ($rx - $prev) / 1MB / ($now - $prevT).TotalSeconds } else { 0 }
    $prev = $rx; $prevT = $now

    $torch = Get-ChildItem $env:TEMP -Recurse -Filter "torch-*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
    $log = if (Test-Path "$dl\download.log") { Get-Content "$dl\download.log" -Raw -ErrorAction SilentlyContinue } else { "" }
    $torchOk = (& python -c "import torch;print('ok' if torch.cuda.is_available() else 'cpu-only')" 2>$null)

    if (-not $Once) { Clear-Host }
    "Download status   {0:HH:mm:ss}     network: {1:N1} MB/s" -f $now, $speed
    "-" * 78
    if ($torchOk -eq "ok") { "{0,-26} INSTALLED, CUDA available" -f "PyTorch (CUDA 12.4)" }
    elseif ($torchOk -eq "cpu-only") { "{0,-26} installed but CUDA NOT available" -f "PyTorch" }
    elseif ($torch) { Row "PyTorch wheel (~2.5 GB est)" $torch.Length 2539MB "then unpack+install" }
    else { "{0,-26} unpacking / installing (no wheel on disk)" -f "PyTorch" }

    $items = @(
        @("LibriSpeech dev-clean",  "dev-clean.tar.gz",        337926286,  "LIBRISPEECH dev-clean done"),
        @("LibriSpeech test-clean", "test-clean.tar.gz",       346663984,  "LIBRISPEECH test-clean done"),
        @("LibriSpeech train-100",  "train-clean-100.tar.gz",  6387309499, "LIBRISPEECH train-clean-100 done"),
        @("Gunshot dataset (zip)",  "edge-collected-gunshot-audio.zip", 1568000000, "GUNSHOT done"))
    foreach ($i in $items) {
        $f = Join-Path $dl $i[1]
        if ($log -match [regex]::Escape($i[3])) { "{0,-26} DONE and extracted" -f $i[0] }
        elseif (Test-Path $f) { Row $i[0] (Get-Item $f).Length $i[2] }
        else { "{0,-26} waiting to start" -f $i[0] }
    }
    "-" * 78
    "Disk free on C: {0:N0} GB        (Ctrl+C to close this view)" -f ((Get-PSDrive C).Free / 1GB)
    if ($Once) { break }
    Start-Sleep -Seconds 3
}
