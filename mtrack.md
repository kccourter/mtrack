# mtrack.md — Lightweight ML Experiment Tracking (Git-Centric)

## Purpose
Create a **lightweight, agile experiment tracking** workflow for early ML R&D where **Git is the system of record** for:
- Model baselines (starting with a Hugging Face model)
- Dataset baselines (starting with MNIST or equivalent canonical dataset)
- Experiment directives (hyperparameter scan configs)
- Results summaries (metrics, artifacts, decision logs)
- Model card provenance (downloaded model card or a generated one)

This is intentionally **not** “enterprise MLOps.” It is a **minimum-viable rigor** framework designed for small teams (1–5) to iterate quickly while staying reproducible and transition-ready.

---

## Non-goals
- A full experiment platform (e.g., hosted tracking server) is not required.
- Real-time dashboards are optional; Markdown summaries + logs are sufficient.
- This framework does not prescribe a particular training stack (PyTorch/TF) beyond common conventions.

---

## Core Principles
1. **Git is central**: every experiment should be reproducible from a commit SHA.
2. **Immutable baselines**: pin a dataset + model revision for each run.
3. **Artifact stamping**: every output artifact includes a unique Experiment ID + commit SHA + timestamps.
4. **Decision traceability**: record why changes were made, not just what changed.
5. **Separation of concerns**:
   - Code & directives live in Git
   - Large artifacts can live in an object store (optional) but must be referenced from Git with hashes/URIs.

---

## Two-Tree Storage Model (Separation of Concerns)
To model the future “distributed/cloud data” reality, use **two distinct trees**:

1) **Experiment Repo (Git)** — code, configs, model refs, run manifests, metrics, notes  
2) **Dataset Store (Local now, S3 later)** — versioned dataset materializations + immutable hashes

You may implement this as:
- **Option A (recommended): two separate Git repos**
  - `mtrack-exp/` (experiments)
  - `mtrack-data/` (dataset metadata + scripts; the *bulk data* stays out of Git)
- **Option B: one Git repo + two top-level trees**
  - `repo/` (exp) and `repo_data/` (data store) with strict rules and independent versioning
- **Option C: Git repo for experiments + Git submodule for dataset metadata**
  - experiment repo pins a specific dataset-meta commit via submodule

Either way, the *experiment repo must be able to reproduce a run* given:
- the experiment commit SHA
- a dataset **revision ID** (and local path / URI) in the manifest

---

## Repository Layout (Suggested)

### 1) Experiment Repo (Git): `mtrack-exp/`
```
mtrack-exp/
  README.md
  mtrack.md                      # this document
  .gitignore

  src/                           # training, eval, utilities
    train.py
    eval.py
    export.py
    utils/
      stamping.py
      hf_io.py
      data_io.py                 # reads dataset manifests + local store paths

  configs/
    baseline.yaml
    scan.yaml                    # hyperparameter scan directives (grid/random/bayes)
    env.yaml                     # environment pins (optional)

  models/
    baseline/
      model_ref.json             # model id + revision
      model_card.md              # downloaded or generated model card baseline
      license.txt                # if applicable

  experiments/
    README.md
    2026-02-08__mnist__hf-baseline__E0001/
      manifest.json
      config_resolved.yaml
      metrics.json
      notes.md
      artifacts/
      pointers/
        dataset_pointer.txt      # points to dataset revision ID + local store path/URI
        artifact_store.txt       # optional: URIs to large binaries stored elsewhere
```

### 2) Dataset Meta Repo (Git): `mtrack-data/`
This repo holds **dataset revision metadata + acquisition/materialization scripts**.
```
mtrack-data/
  README.md
  datasets/
    mnist/
      revisions/
        D0001/
          dataset_manifest.json  # split sizes, hashes, provenance, materialization recipe
          notes.md               # what changed and why
        D0002/
          dataset_manifest.json
          notes.md
      latest.txt                 # optional pointer to latest revision ID

  scripts/
    materialize_mnist.py         # downloads + writes local store, then hashes
    hash_dataset.py
    make_revision.py             # creates D#### folder + manifests
```

### 3) Dataset Store (NOT Git): local filesystem now, S3 later
This is where the actual data files live.
```
/ml-data-store/
  mnist/
    D0001/
      train/...
      test/...
      heldout_new/...
      checksums.json
    D0002/
      train/...
      test/...
      heldout_new/...
      checksums.json
```
**Rule:** the dataset store is *append-only per revision*. Never mutate `D0001/`; create `D0002/`.

---
## Standard Identifiers & Stamping
### Experiment ID
Format:
- `E####` (monotonic per repo) **and**
- a run folder timestamp for ordering

Example folder:
- `2026-02-08__mnist__resnet_mnist_digits__E0007/`

### Required Stamp Fields (every run)
- `experiment_id`
- `created_utc`
- `git_commit`
- `git_branch`
- `dirty_worktree` (true/false)
- `model_id`
- `model_revision` (HF commit/tag)
- `dataset_id`
- `dataset_version` (HF datasets version or custom hash)
- `seed`
- `config_hash` (hash of resolved config file)

---

# Dataset Revisioning (Incremental Growth Workflow)

## Why dataset revisions matter
In early R&D, the dataset evolves as you:
- add new examples (new sensors, new conditions, new labels)
- refine preprocessing/feature pipelines
- correct labeling errors
- introduce harder cases (domain shift)

To keep experiments comparable, treat the dataset as a **versioned artifact** with explicit revisions.

---

## Dataset Revision ID
Use a monotonic ID:
- `D####` (dataset revision)

Examples:
- `D0001` = baseline subset of MNIST materialized locally
- `D0002` = baseline + “new” data added (previously held out)

---

## Incremental Approach (Recommended Starter Plan)

### Iteration 1 — Baseline Dataset from MNIST (subset + holdback)
1) **Materialize MNIST locally** into the dataset store.
2) **Set aside a slice** that will simulate “newly arrived training data” later.
   - Example: hold back 10% of the *training* set as `heldout_new/`
3) Define dataset revision `D0001`:
   - `train/` contains the baseline training subset
   - `test/` stays stable (or pin a stable eval split)
   - `heldout_new/` is present but **not used** in training for D0001
4) Compute checksums and write:
   - `mtrack-data/datasets/mnist/revisions/D0001/dataset_manifest.json`
   - `/ml-data-store/mnist/D0001/checksums.json`

**Goal:** You can run `E0001` pinned to model revision + dataset `D0001`.

### Iteration 2 — Promote “New” Data into Training (revision bump)
1) Treat the previously-held-out `heldout_new/` as **new training data**.
2) Create a new dataset revision `D0002` by materializing a new folder:
   - `/ml-data-store/mnist/D0002/`
3) For `D0002`, training includes baseline + the previously held-out slice.
4) Recompute checksums and record what changed in the manifest and notes.
5) Run `E0002` (or later) pinned to dataset `D0002`.

**Goal:** Measure performance delta due to dataset expansion while keeping provenance clear.

### Iteration 3+ — Repeat for realism
- Add additional “new” data slices or perturbations (noise, domain shift)
- Introduce labeling corrections
- Introduce new features/preprocessing versions

---

## Dataset Manifest (Minimum Fields)
Each dataset revision directory contains `dataset_manifest.json`:

```json
{
  "dataset_id": "mnist",
  "revision_id": "D0002",
  "created_utc": "2026-02-08T17:22:10Z",
  "materialization": {
    "store_type": "local_fs",
    "store_root": "/ml-data-store",
    "relative_path": "mnist/D0002",
    "recipe": "scripts/materialize_mnist.py --holdback 10% --promote-heldout true"
  },
  "splits": {
    "train": {"examples": 60000, "hash": "sha256:..."},
    "test": {"examples": 10000, "hash": "sha256:..."},
    "heldout_new": {"examples": 0, "hash": "sha256:..."}  // or omitted once promoted
  },
  "provenance": {
    "source": "MNIST",
    "provider": "torchvision|huggingface_datasets",
    "downloaded_utc": "2026-02-08T16:10:00Z"
  },
  "notes": "D0002 promotes prior heldout slice into train split."
}
```

---

## How experiments point to dataset revisions
Each experiment run writes a small pointer file in Git:

`experiments/<run>/pointers/dataset_pointer.txt`
```
dataset_id=mnist
dataset_revision=D0002
data_store_root=/ml-data-store
relative_path=mnist/D0002
checksums=sha256:...
dataset_meta_commit=<git sha from mtrack-data>
```

This keeps the experiment repo lightweight while enabling you to swap:
- local filesystem today
- S3/minio tomorrow (same schema, different `store_type` + URI)

---

## Rules of the road
- **Never mutate a revision in place.** New data or preprocessing changes require a new `D####`.
- The dataset-meta repo (Git) records *what* the revision is and *how* to reproduce it.
- The dataset store contains the physical bytes for each revision.
- Experiments must record both:
  - `dataset_revision` (D####)
  - `dataset_meta_commit` (so the manifest and recipe are pinned)
  - checksums for integrity

---

## Baseline Setup (Day 0)
### 1) Choose a starting Hugging Face model
- For this demo, we use **lane99/resnet_mnist_digits** - a ResNet model trained on MNIST digits.
- Pin a **specific revision** (commit SHA or tag).

Record into:
- `models/baseline/model_ref.json`

Example fields:
```json
{
  "provider": "huggingface",
  "model_id": "lane99/resnet_mnist_digits",
  "revision": "main",
  "task": "image-classification",
  "notes": "ResNet baseline for MNIST digit classification - mtrack demo"
}
```

### 2) Acquire model card
- If the model has a model card, **download and store** it as:
  - `models/baseline/model_card.md`
- If it does not, **generate one** using the template below and save it there.

### 3) Choose a baseline dataset (MNIST)
Use one of:
- Hugging Face datasets: `mnist`
- TorchVision MNIST
- A mirrored internal baseline dataset (if needed)

Record into:
- `data/mnist/dataset.json`
- `data/README.md` describing acquisition and hashes

Example fields:
```json
{
  "dataset_id": "mnist",
  "provider": "huggingface_datasets",
  "revision_id": "D0001",
  "splits": ["train", "test"],
  "expected_examples": {"train": 60000, "test": 10000},
  "hashes": {
    "train": "sha256:...",
    "test": "sha256:..."
  }
}
```

---

## Model Card Template (If Missing)
Save as `models/baseline/model_card.md` (or per-run copy if modified):

### Model Details
- **Model name / ID:**
- **Provider:**
- **Revision / commit:**
- **Architecture summary:**
- **Intended task:**
- **Input format:**
- **Output format:**

### Training Data
- **Dataset(s):**
- **Dataset version(s):**
- **Preprocessing:**

### Evaluation
- **Metrics used:**
- **Baseline performance (if known):**
- **Known failure cases:**

### Limitations & Risks
- **Operational constraints:**
- **Bias / fairness considerations (as applicable):**
- **Safety considerations:**

### License & Use
- **License:**
- **Restrictions / attribution:**

### Provenance
- **Downloaded from:**
- **Download date:**
- **Stored in repo at:**

---

## Hyperparameter Scan Directives
Keep scan definitions in `configs/scan.yaml`. This file is a **first-class artifact**: changes must be committed.

### Scan Types (supported patterns)
- Grid search
- Random search
- Bayesian optimization (optional; still driven by YAML directives)

### Example `configs/scan.yaml`
```yaml
scan:
  method: random            # grid | random | bayes
  max_trials: 20
  seed: 1337

  parameters:
    learning_rate:
      distribution: log_uniform
      min: 1.0e-5
      max: 5.0e-3

    batch_size:
      values: [32, 64, 128]

    weight_decay:
      values: [0.0, 1.0e-4, 1.0e-3]

    dropout:
      values: [0.0, 0.1, 0.2]
```

### Resolution Rule
Every run must output a **resolved, concrete config**:
- `experiments/<run>/config_resolved.yaml`
This file is the ground truth of what actually ran.

---

## Experiment Manifest Schema
Each run writes `manifest.json` with both **inputs** and **outputs**:

```json
{
  "experiment_id": "E0007",
  "created_utc": "2026-02-08T17:22:10Z",
  "git": {
    "commit": "abc123...",
    "branch": "main",
    "dirty_worktree": false
  },
  "inputs": {
    "model": {
      "provider": "huggingface",
      "model_id": "lane99/resnet_mnist_digits",
      "revision": "main"
    },
    "dataset": {
      "dataset_id": "mnist",
      "revision_id": "D0002",
      "dataset_meta_commit": "def456...",
      "store": {
        "type": "local_fs",
        "root": "/ml-data-store",
        "relative_path": "mnist/D0002"
      },
      "hashes": {
        "train": "sha256:...",
        "test": "sha256:..."
      }
    },
    "config": {
      "path": "configs/baseline.yaml",
      "resolved_path": "experiments/.../config_resolved.yaml",
      "config_hash": "sha256:..."
    }
  },
  "outputs": {
    "metrics_path": "experiments/.../metrics.json",
    "artifacts_dir": "experiments/.../artifacts",
    "exported_model_ref": "experiments/.../pointers/artifact_store.txt"
  }
}
```

---

## Metrics File (Minimum Set)
Write `metrics.json` per run:
- `primary_metric` (e.g., accuracy)
- `loss`
- `latency_ms` (if measured)
- `throughput` (optional)
- `resource_profile` (CPU/GPU/RAM footprint; optional)

Example:
```json
{
  "primary_metric": {"name": "accuracy", "value": 0.984},
  "loss": 0.041,
  "eval": {
    "accuracy": 0.984,
    "precision_macro": 0.984,
    "recall_macro": 0.983
  },
  "profiling": {
    "latency_ms_p50": 2.1,
    "latency_ms_p95": 3.7,
    "device": "cpu"
  }
}
```

---

## Experiment Flow (Git-Centric)
This is the standard lifecycle for any experiment.

### Step 0 — Start from Known Baselines
- Checkout a known commit SHA (baseline)
- Confirm:
  - model `model_id + revision`
  - dataset `dataset_id + version + hashes`

### Step 1 — Create a New Experiment Folder
- Allocate next Experiment ID (E####)
- Create run directory:
  - `experiments/<timestamp>__<dataset>__<model>__E####/`

### Step 2 — Modify Inputs
Make changes to any of:
- model selection (HF model id / revision)
- dataset pointer/version or preprocessing
- code (training/eval)
- hyperparameter scan directives (`configs/scan.yaml`)
- baseline config (`configs/baseline.yaml`)

### Step 3 — Resolve Directives Into a Concrete Run Config
- Generate `config_resolved.yaml` (filled with actual chosen values)
- Compute and record `config_hash`

### Step 4 — Execute Training/Eval
- Run training
- Run evaluation
- Generate metrics and artifacts
- Export an inference-ready artifact if applicable (optional but recommended early)

### Step 5 — Stamp Everything
Write:
- `manifest.json`
- `metrics.json`
- `notes.md` (brief decision log)
- `config_resolved.yaml`
- any artifacts / pointers

### Step 6 — Commit to Git
Commit includes:
- configs changes
- new experiment folder (minus large binaries)
- updated indices (optional)

**Commit message convention (recommended):**
- `mtrack: E0007 mnist resnet — lr scan + eval metrics`
- Include experiment id and a short intent phrase.

### Step 7 — Tag Milestones (Optional)
For significant results:
- `git tag mtrack-E0007`
- or tag “transition candidates” for handoff readiness

---

## Restamping & Provenance Rules
When you change *anything*, you restamp all relevant artifacts:
- change `configs/scan.yaml` → new `config_resolved.yaml` and new `config_hash`
- change dataset pointer → new dataset hash fields in manifest
- change model revision → update model ref and model card provenance section
- change code affecting results → new run folder (don’t overwrite old run)

**Never overwrite a previous run’s `metrics.json` in place.** Create a new run folder with a new experiment id.

---

## Optional: Artifact Store Integration
If artifacts exceed Git comfort:
- store in S3/minio/Artifactory/etc.
- record pointers in:
  - `experiments/<run>/pointers/artifact_store.txt`
- include checksums in `manifest.json`

Example `artifact_store.txt`:
```
s3://ml-artifacts/mtrack/E0007/model.onnx  sha256:...
s3://ml-artifacts/mtrack/E0007/weights.bin sha256:...
```

---

## Team Workflow
### For 1 researcher (SBIR-like)
- Keep scope tight: one model, one dataset, one metric
- Daily commits encouraged
- Weekly summary in `experiments/README.md` or in a top-level `CHANGELOG.md`

### For 2–5 engineers
- Use PRs for changes to:
  - configs/
  - core training/eval code
- Allow direct commits for new experiment run folders (unless governance requires PR)
- Use a single shared convention for Experiment IDs

---

## Suggested Minimal Tooling (Implementation-Agnostic)
- `make baseline` → fetch model + dataset pointers, write baseline metadata
- `make run` → resolve config, run training/eval, write run folder
- `make report` → generate a markdown summary (optional)

Even without Makefiles, the key is that scripts produce:
- resolved config
- manifest
- metrics
- notes
- artifacts/pointers

---

## “Done” Definition for This mtrack Experiment
This framework experiment is successful when:

1. A Hugging Face model is selected and pinned by revision.
2. The model card is downloaded (or generated) and stored in-repo.
3. MNIST is referenced with reproducible acquisition + hashes.
4. A baseline run is executed producing:
   - `manifest.json`, `metrics.json`, `config_resolved.yaml`, `notes.md`
5. A second run is executed where:
   - hyperparameter scan directives are modified
   - new run artifacts are restamped
   - all changes are committed to Git

---

## Appendix: Minimal Decision Log Template (`notes.md`)
```
# E#### Notes

## Intent
What are we testing and why?

## Change Summary
- Model:
- Data:
- Config / scan:
- Code:

## Results
- Primary metric:
- Secondary metrics:
- Observed failure modes:

## Next Decision
Based on results, what do we do next?
```

---

## Quick Start Checklist
- [ ] Create repo structure
- [ ] Add baseline HF model reference + revision
- [ ] Download or create model card
- [ ] Add MNIST dataset pointer + hashes
- [ ] Add baseline config + scan config
- [ ] Run baseline, stamp artifacts, commit
- [ ] Modify scan directives, rerun, stamp artifacts, commit
