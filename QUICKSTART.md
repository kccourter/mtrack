# Quick Start Guide

Get started with mtrack in 5 minutes.

## Prerequisites

- Python 3.8+
- Git
- pip

## Setup

1. **Install dependencies:**
   ```bash
   make setup
   # or manually: pip install -r requirements.txt
   ```

2. **Create dataset store directory:**
   ```bash
   mkdir -p /ml-data-store/mnist
   ```

3. **Verify repository structure:**
   ```bash
   tree -L 2
   ```

## Running Your First Experiment

### Option 1: Using Make (Recommended)

```bash
# Run baseline experiment
make baseline
```

### Option 2: Direct Python Command

```bash
python src/train.py --config configs/baseline.yaml
```

### What Happens During a Run

1. **Experiment folder created** in `experiments/` with format:
   ```
   YYYY-MM-DD__dataset__model__E####/
   ```

2. **Files generated:**
   - `manifest.json` - Complete provenance (git state, model, dataset, config)
   - `config_resolved.yaml` - Exact configuration used
   - `metrics.json` - Performance metrics
   - `notes.md` - Decision log template (fill this out!)
   - `pointers/dataset_pointer.txt` - Dataset revision reference

3. **Results displayed** in terminal

## Next Steps

### 1. Review Results

```bash
# Navigate to experiment directory
cd experiments/<your-experiment-folder>

# View manifest
cat manifest.json

# View metrics
cat metrics.json

# Edit notes
vim notes.md  # or use your preferred editor
```

### 2. Commit Your Experiment

Following Conventional Commits format:

```bash
git add experiments/<your-experiment-folder>
git commit -m "feat: experiment E0001 - mnist baseline with resnet"
```

### 3. Run a Hyperparameter Scan

Edit `configs/scan.yaml` to adjust parameters, then:

```bash
make run-scan
```

### 4. Evaluate an Experiment

```bash
make eval EXP_DIR=experiments/<your-experiment-folder>
```

### 5. Export a Model

```bash
make export EXP_DIR=experiments/<your-experiment-folder>
```

## Understanding the Workflow

### Experiment Lifecycle

```
1. Modify config/code
   ↓
2. Run experiment (generates E####)
   ↓
3. Review results & fill notes.md
   ↓
4. Commit to Git
   ↓
5. Decide next action
```

### Key Concepts

- **Experiment ID (E####):** Unique identifier for each run
- **Dataset Revision (D####):** Immutable dataset version
- **Manifest:** Complete provenance record
- **Config Hash:** Ensures reproducibility

## Common Tasks

### Find Latest Experiment

```bash
ls -lt experiments/ | head -5
```

### View Experiment History

```bash
git log --oneline --grep="experiment"
```

### Compare Two Experiments

```bash
# Compare metrics
diff experiments/<exp1>/metrics.json experiments/<exp2>/metrics.json

# Compare configs
diff experiments/<exp1>/config_resolved.yaml experiments/<exp2>/config_resolved.yaml
```

## Troubleshooting

### "Dataset store not found"

Create the directory:
```bash
mkdir -p /ml-data-store/mnist
```

Or specify a custom location:
```bash
python src/train.py --config configs/baseline.yaml --dataset-store /path/to/your/store
```

### "Git not found" or "dirty worktree"

Ensure you're in a Git repository:
```bash
git init
git add .
git commit -m "feat: initial mtrack setup"
```

### Import Errors

Install dependencies:
```bash
pip install -r requirements.txt
```

## File Structure Reference

```
mtrack/
├── configs/              # Configuration files
│   ├── baseline.yaml     # Baseline run config
│   └── scan.yaml         # Hyperparameter scan directives
├── experiments/          # All experiment runs
│   └── YYYY-MM-DD__dataset__model__E####/
│       ├── manifest.json
│       ├── metrics.json
│       ├── config_resolved.yaml
│       ├── notes.md
│       ├── artifacts/
│       └── pointers/
├── models/baseline/      # Model references
│   ├── model_ref.json
│   └── model_card.md
├── src/                  # Source code
│   ├── train.py          # Training script
│   ├── eval.py           # Evaluation script
│   ├── export.py         # Model export script
│   └── utils/            # Utilities
└── mtrack.md            # Complete documentation
```

## Next: Read the Full Documentation

See `mtrack.md` for:
- Two-tree storage model (experiment repo + dataset store)
- Dataset revisioning workflow
- Artifact store integration
- Team collaboration patterns
- Advanced hyperparameter scanning

## Getting Help

- Review `mtrack.md` for complete documentation
- Check experiment `notes.md` templates for guidance
- Look at generated `manifest.json` to understand provenance
