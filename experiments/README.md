# Experiments

This directory contains experiment run folders. Each experiment is identified by:
- Experiment ID (E####)
- Timestamp
- Dataset name
- Model name

## Folder Structure

Each experiment folder contains:
- `manifest.json` — Complete provenance (inputs, outputs, git state)
- `config_resolved.yaml` — Concrete configuration used for this run
- `metrics.json` — Performance metrics
- `notes.md` — Decision log and observations
- `artifacts/` — Small output files (plots, checkpoints)
- `pointers/` — References to large artifacts and dataset revisions

## Experiment Index

| ID | Date | Model | Dataset | Primary Metric | Notes |
|----|------|-------|---------|----------------|-------|
| TBD | TBD | resnet_mnist_digits | mnist D0001 | TBD | Baseline run |

## Current Status

No experiments run yet. Run the baseline experiment to get started.
