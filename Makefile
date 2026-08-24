# Orchestrates the C / Ruby / Python build+test+run cycle for the polyglot
# fraud engine. Requires: MSVC (Visual Studio Build Tools, C++ workload) for
# the C layer, Ruby 3.2+ with Bundler for the rules engine, Python 3.10+
# with a virtualenv at .venv for the ML/API/dashboard layer.
#
# Usage: make <target>   (see targets below)

PYTHON := .venv/Scripts/python.exe
RUBY_DIR := src/ruby

.PHONY: help build-c test-c bench-c install-ruby test-ruby generate-data train test-python test run-api run-dashboard all clean

help:
	@echo "Targets:"
	@echo "  build-c          Compile fraud_core.dll + benchmark + test exe (MSVC)"
	@echo "  test-c           Run the C assert-based test harness"
	@echo "  bench-c          Run the C benchmark (validates < 1ms per call)"
	@echo "  install-ruby     bundle install for the Ruby rules engine"
	@echo "  test-ruby        Run RSpec against the Ruby DSL"
	@echo "  generate-data    Generate synthetic Chilean bank transactions"
	@echo "  train            Build features (via C+Ruby) and train the ML layer"
	@echo "  test-python      Run pytest (includes API + latency tests)"
	@echo "  test             Run all three languages' test suites"
	@echo "  all              build-c + install-ruby + generate-data + train"
	@echo "  run-api          Start the FastAPI server (uvicorn)"
	@echo "  run-dashboard    Start the Streamlit dashboard"
	@echo "  clean            Remove compiled C artifacts"

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

test-python:
	$(PYTHON) -m pytest -v

test: test-c test-ruby test-python

all: build-c install-ruby generate-data train

run-api: build-c
	$(PYTHON) -m uvicorn src.python.api:app --reload

run-dashboard:
	$(PYTHON) -m streamlit run src/app/dashboard.py

clean:
	$(MAKE) -C src/c clean
