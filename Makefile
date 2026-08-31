# Orchestrates the C / Ruby / Python build+test+run cycle for the polyglot
# fraud engine. Requires: MSVC (Visual Studio Build Tools, C++ workload) for
# the C layer, Ruby 3.2+ with Bundler for the rules engine, Python 3.10+
# with a virtualenv at .venv for the ML/API/dashboard layer.
#
# Usage: make <target>   (see targets below)

PYTHON := .venv/Scripts/python.exe
RUBY_DIR := src/ruby

.PHONY: help build-c test-c bench-c install-ruby test-ruby generate-data train test-python test run-api run-dashboard all clean run-feature-store bench-throughput observability-up

help:
	@echo "Targets:"
	@echo "  build-c            Compile fraud_core.dll + benchmark + test exe + feature_store_server.exe (MSVC)"
	@echo "  test-c             Run the C assert-based test harness"
	@echo "  bench-c            Run the C benchmark (validates < 1ms per call)"
	@echo "  install-ruby       bundle install for the Ruby rules engine"
	@echo "  test-ruby          Run RSpec against the Ruby DSL"
	@echo "  generate-data      Generate synthetic Chilean bank transactions"
	@echo "  train              Build features (via C+Ruby) and train the ML layer"
	@echo "  train-extra        Train Logistic Regression baseline + PyTorch MLP (Focal Loss, ReLU/GELU/Swish) and persist comparison to DuckDB"
	@echo "  test-python        Run pytest (includes API + latency + mmap feature-store tests)"
	@echo "  test               Run all three languages' test suites"
	@echo "  all                build-c + install-ruby + generate-data + train"
	@echo "  run-api            Start the FastAPI server (uvicorn), exposes /metrics"
	@echo "  run-dashboard      Start the Streamlit dashboard"
	@echo "  run-feature-store  Start feature_store_server.exe (shared-memory IPC feature store)"
	@echo "  bench-throughput   Stress test: req/s for ctypes DLL / mmap IPC / Ruby pipe IPC"
	@echo "  observability-up   docker compose up for Prometheus + Grafana (see observability/)"
	@echo "  clean              Remove compiled C artifacts"

build-c:
	$(MAKE) -C src/c build

test-c:
	$(MAKE) -C src/c test

bench-c:
	$(MAKE) -C src/c bench

install-ruby:
	cd $(RUBY_DIR) && bundle config set --local path vendor/bundle && bundle install

test-ruby:
	cd $(RUBY_DIR) && bundle exec rspec ../../tests/ruby/rules_engine_spec.rb --format documentation

generate-data:
	$(PYTHON) -m src.python.generate_data

train: build-c
	$(PYTHON) -m src.python.train_model

train-extra: train
	$(PYTHON) -m src.python.train_extra_models

test-python:
	$(PYTHON) -m pytest -v

test: test-c test-ruby test-python

all: build-c install-ruby generate-data train

run-api: build-c
	$(PYTHON) -m uvicorn src.python.api:app --reload

run-dashboard:
	$(PYTHON) -m streamlit run src/app/dashboard.py

run-feature-store: build-c
	outputs/models/feature_store_server.exe

# Corre el stress test contra el motor C (ctypes), el feature store (mmap
# IPC) y el motor de reglas Ruby (pipe IPC) -- levanta y apaga
# feature_store_server.exe automaticamente, no requiere tenerlo corriendo
# de antemano. Ver src/c/run_bench_throughput.ps1 para por que esto es un
# script .ps1 real y no una linea de PowerShell inline en esta receta.
bench-throughput: build-c
	powershell -NoProfile -ExecutionPolicy Bypass -File src/c/run_bench_throughput.ps1

observability-up:
	docker compose -f observability/docker-compose.yml up

clean:
	$(MAKE) -C src/c clean
