VENV ?= .venv
PYTHON ?= python3
PY ?= $(VENV)/bin/python
PIP ?= $(PY) -m pip

.PHONY: help venv setup run clean

help:
	@echo "Targets:"
	@echo "  make setup   Create venv + install deps"
	@echo "  make run     Run experiments (writes to project/results/)"
	@echo "  make clean   Remove venv + results output"

venv:
	"$(PYTHON)" -m venv "$(VENV)"
	"$(PY)" -m pip install --upgrade pip

setup: venv
	"$(PIP)" install -r project/requirements.txt

run:
	"$(PY)" -m project.main

clean:
	rm -rf "$(VENV)" project/results
