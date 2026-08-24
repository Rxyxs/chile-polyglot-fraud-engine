# Compiles fraud_core.c into libfraudcore (fraud_core.dll on Windows via
# MSVC), plus its benchmark and test executables. The actual MSVC
# environment setup (vcvars64.bat) lives in build.ps1, not here, since
# GNU Make's shell has no clean way to source a .bat file's env vars.
#
# Usage (from repo root): make -C src/c build | test | bench

MODELS_DIR := ../../outputs/models

.PHONY: build test bench clean

build:
	powershell -NoProfile -ExecutionPolicy Bypass -File build.ps1

test: build
	$(MODELS_DIR)/fraud_core_test.exe

bench: build
	$(MODELS_DIR)/fraud_core_bench.exe

clean:
	powershell -NoProfile -Command "Remove-Item -Force -ErrorAction SilentlyContinue '$(MODELS_DIR)/fraud_core.dll', '$(MODELS_DIR)/fraud_core_bench.exe', '$(MODELS_DIR)/fraud_core_test.exe', '$(MODELS_DIR)/*.obj', '$(MODELS_DIR)/*.exp', '$(MODELS_DIR)/*.lib', '*.obj'"
