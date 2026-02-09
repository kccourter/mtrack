# mtrack — Lightweight ML Experiment Tracking

Git-centric experiment tracking framework for early ML R&D.

## Quick Start

1. Set up dataset store (see `mtrack.md` for details)
2. Run baseline experiment: `python src/train.py --config configs/baseline.yaml`
3. Run hyperparameter scan: `python src/train.py --config configs/scan.yaml`

## Repository Structure

- `src/` — Training, evaluation, and utility scripts
- `configs/` — Configuration files (baseline and scan directives)
- `models/baseline/` — Baseline model references and provenance
- `experiments/` — Experiment run folders with manifests, metrics, and artifacts

## Core Principles

- **Git is central**: Every experiment is reproducible from a commit SHA
- **Immutable baselines**: Pin dataset + model revision for each run
- **Artifact stamping**: Every output includes experiment ID + commit SHA + timestamps
- **Decision traceability**: Record why changes were made, not just what changed

See `mtrack.md` for complete documentation.
