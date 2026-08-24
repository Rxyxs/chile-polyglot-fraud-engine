# Compila fraud_core con MSVC (cl.exe): una DLL para que Python la cargue
# via ctypes, un ejecutable de benchmark, y un ejecutable de tests.
# Requiere Visual Studio Build Tools / Visual Studio con el workload de
# C++ instalado.
#
# Ejecutar desde la raiz del repo:  powershell -File src\c\build.ps1

$ErrorActionPreference = "Stop"

$vcvarsCandidates = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)
$vcvars = $vcvarsCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $vcvars) {
    throw "No se encontro vcvars64.bat. Instala Visual Studio Build Tools (workload 'Desktop development with C++')."
}

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$cDir = Join-Path $repoRoot "src\c"
$outDir = Join-Path $repoRoot "outputs\models"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Write-Host "Usando vcvars64.bat: $vcvars"

$testFile = Join-Path $repoRoot "tests\c\test_fraud_core.c"

$buildCmd = "call `"$vcvars`" && cd /d `"$cDir`" && " +
    "cl /nologo /O2 /LD /Fe:`"$outDir\fraud_core.dll`" fraud_core.c && " +
    "cl /nologo /O2 /Fe:`"$outDir\fraud_core_bench.exe`" fraud_core.c bench_main.c && " +
    "cl /nologo /O2 /Fe:`"$outDir\fraud_core_test.exe`" fraud_core.c `"$testFile`""

cmd /c $buildCmd
if ($LASTEXITCODE -ne 0) {
    throw "Compilacion fallida (codigo $LASTEXITCODE)"
}

Write-Host "Compilado OK -> $outDir\fraud_core.dll, $outDir\fraud_core_bench.exe, $outDir\fraud_core_test.exe"
