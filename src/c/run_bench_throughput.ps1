# Levanta feature_store_server.exe, corre el stress test de Python contra
# los tres mecanismos IPC/FFI del repo, y apaga el servidor -- en un .ps1
# dedicado en vez de inline en el Makefile: GNU Make en Windows corre las
# recetas a traves de sh.exe (no cmd.exe), y sh.exe expande cualquier `$var`
# suyo propio dentro de un `-Command "..."` de PowerShell embebido en una
# linea de receta -- incluso escapando `$$` para Make, sh sigue viendo
# `$p`/`$exitCode` como SUS variables (vacias) y corrompe el script antes
# de que PowerShell lo vea. Un archivo .ps1 real evita ese problema de raiz.

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$serverExe = Join-Path $repoRoot "outputs\models\feature_store_server.exe"
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $serverExe)) {
    throw "$serverExe no existe. Corre primero: make build-c"
}

$serverProcess = Start-Process -PassThru -NoNewWindow -FilePath $serverExe
Start-Sleep -Seconds 1

try {
    & $python -m src.python.benchmark_throughput
    $exitCode = $LASTEXITCODE
} finally {
    if (-not $serverProcess.HasExited) {
        Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue
    }
}

exit $exitCode
