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
    # Cualquier otra edicion de Visual Studio (Enterprise, Professional, una version nueva):
    # vswhere sabe donde esta la que tiene el toolset de C++ instalado.
    $vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($installPath) {
            $candidate = Join-Path $installPath "VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path $candidate) { $vcvars = $candidate }
        }
    }
}
if (-not $vcvars) {
    throw "No se encontro vcvars64.bat. Instala Visual Studio Build Tools (workload 'Desktop development with C++')."
}

# vcvars64.bat internamente llama a vswhere.exe; en algunas maquinas (esta
# incluida) el directorio del VS Installer no esta en PATH por defecto, lo
# que hace fallar vcvars64.bat con "vswhere.exe no se reconoce" aunque la
# ruta de vcvars64.bat en si se haya encontrado bien. Se agrega ese
# directorio a PATH antes de invocarlo, en vez de asumir que ya esta.
$vsInstallerDir = "C:\Program Files (x86)\Microsoft Visual Studio\Installer"
if ((Test-Path $vsInstallerDir) -and ($env:Path -notlike "*$vsInstallerDir*")) {
    $env:Path = "$vsInstallerDir;$env:Path"
}

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$cDir = Join-Path $repoRoot "src\c"
$outDir = Join-Path $repoRoot "outputs\models"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Write-Host "Usando vcvars64.bat: $vcvars"

$testFile = Join-Path $repoRoot "tests\c\test_fraud_core.c"

# Se compila desde un .bat temporal y no con una cadena inline para `cmd /c`: con rutas con
# espacios y varias comillas, cmd puede no ejecutar nada y devolver codigo 0 (paso en CI).
$batPath = Join-Path ([System.IO.Path]::GetTempPath()) "build_fraud_core.bat"
$batLines = @(
    "@echo off",
    "call `"$vcvars`" || exit /b 1",
    "cd /d `"$cDir`" || exit /b 1",
    "cl /nologo /O2 /LD /Fe:`"$outDir\fraud_core.dll`" fraud_core.c || exit /b 1",
    "cl /nologo /O2 /Fe:`"$outDir\fraud_core_bench.exe`" fraud_core.c bench_main.c || exit /b 1",
    "cl /nologo /O2 /Fe:`"$outDir\fraud_core_test.exe`" fraud_core.c `"$testFile`" || exit /b 1",
    "cl /nologo /O2 /Fe:`"$outDir\feature_store_server.exe`" fraud_core.c feature_store_server.c || exit /b 1"
)
Set-Content -Path $batPath -Value $batLines -Encoding ASCII

& $batPath
if ($LASTEXITCODE -ne 0) {
    throw "Compilacion fallida (codigo $LASTEXITCODE)"
}

# No basta el codigo de salida: se comprueba que los cuatro productos existan.
foreach ($product in @("fraud_core.dll", "fraud_core_bench.exe", "fraud_core_test.exe", "feature_store_server.exe")) {
    if (-not (Test-Path (Join-Path $outDir $product))) {
        throw "La compilacion termino sin errores pero no se genero $product en $outDir"
    }
}

Write-Host "Compilado OK -> $outDir\fraud_core.dll, $outDir\fraud_core_bench.exe, $outDir\fraud_core_test.exe, $outDir\feature_store_server.exe"
