# Makefile for mtrack experiment tracking

.PHONY: help setup baseline run-baseline run-scan eval export clean

help:
	@echo "mtrack - Lightweight ML Experiment Tracking"
	@echo ""
	@echo "Available targets:"
	@echo "  setup        - Install dependencies and initialize dataset store"
	@echo "  baseline     - Run baseline experiment with default config"
	@echo "  run-baseline - Alias for baseline"
	@echo "  run-scan     - Run hyperparameter scan"
	@echo "  eval         - Evaluate a specific experiment"
	@echo "  export       - Export a trained model"
	@echo "  clean        - Clean up temporary files"
	@echo ""
	@echo "Environment variables:"
	@echo "  DATASET_STORE - Root path for dataset store (default: /ml-data-store)"
	@echo "  EXP_DIR       - Experiment directory for eval/export commands"

DATASET_STORE ?= /ml-data-store

setup:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo ""
	@echo "Setup complete!"
	@echo "Note: You may need to manually create the dataset store directory:"
	@echo "  mkdir -p $(DATASET_STORE)"

baseline: run-baseline

run-baseline:
	@echo "Running baseline experiment..."
	python src/train.py --config configs/baseline.yaml --dataset-store $(DATASET_STORE)

run-scan:
	@echo "Running hyperparameter scan..."
	@echo "Note: Full scan implementation requires additional scan logic."
	@echo "For now, running baseline with scan config..."
	python src/train.py --config configs/scan.yaml --dataset-store $(DATASET_STORE)

eval:
	@if [ -z "$(EXP_DIR)" ]; then \
		echo "Error: EXP_DIR not set. Usage: make eval EXP_DIR=experiments/<folder>"; \
		exit 1; \
	fi
	python src/eval.py --experiment-dir $(EXP_DIR)

export:
	@if [ -z "$(EXP_DIR)" ]; then \
		echo "Error: EXP_DIR not set. Usage: make export EXP_DIR=experiments/<folder>"; \
		exit 1; \
	fi
	python src/export.py --experiment-dir $(EXP_DIR)

clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "Clean complete!"
