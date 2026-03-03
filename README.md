## Installation
Please refer to [INSTALL.md](readme/INSTALL.md) for background installation notes.

## SURF Research Cloud (Ubuntu 22.04) Step-by-Step
This repository includes a Docker + `uv` + Makefile workflow for running on a SURF Research Cloud VM.

### 0. Prerequisites on the VM
1. Ubuntu 22.04 VM with NVIDIA driver already installed.
2. This repository cloned on the VM.
3. Your MLflow tracking server already running on a separate VM.

### 1. Install Docker + NVIDIA Container Toolkit
Run once on the SURF VM:
~~~bash
make vm-bootstrap-docker
~~~

Then log out/in (or run `newgrp docker`) so your user can run Docker without `sudo`.

### 2. Verify GPU Access from Docker
~~~bash
make verify-docker-gpu
~~~

If this VM has no GPU (CPU-only smoke test), run:
~~~bash
make verify-docker-cpu
~~~

### 3. Build the Project Image (dependencies installed with `uv`)
~~~bash
make docker-build
~~~

The Docker image:
1. Uses CUDA 11.8 + Ubuntu 22.04.
2. Installs Python and dependencies via `uv`.
3. Installs PyTorch/torchvision, project requirements, and Apex.

### 4. Prepare and Ingest the HSPOT Dataset
Expected layout:
1. `datasets/hspot/raw_data/train`
2. `datasets/hspot/raw_data/val`
3. `datasets/hspot/raw_data/test`

Run ingestion in Docker:
~~~bash
make ingest DATASET_PATH=datasets/hspot ANNO_NAME=anno.json MOT17=false DET_OPTIONS=""
~~~

Notes:
1. `DET_OPTIONS=""` ingests all sequence folders and does not require MOT17 detector suffixes.
2. HSPOT is ingested as a single-class MOT-style dataset, so the MOT class column is ignored.
3. `seqinfo.ini` is still used when present, so HSPOT keeps its true frame rate of 1 fps.
4. This project uses dataset key `MOT_HSPOT`.

CPU-only smoke test (no dataset required):
~~~bash
make smoke-cpu
~~~
This validates container startup and core CLI wiring without spending GPU hours.

### 5. Configure MLflow Tracking URI
Point jobs to your separate MLflow VM:
~~~bash
export MLFLOW_TRACKING_URI=http://<MLFLOW_VM_IP_OR_HOST>:5000
~~~

All `make` targets pass this environment variable into the container.

### 6. Add TrackEval (required for HOTA)
~~~bash
make trackeval-add
~~~

Update later:
~~~bash
make trackeval-update
~~~

### 7. Establish Baseline
The default HSPOT config initializes from:
1. `weights/DLA-34-FPN_EMM_crowdhuman_mot17.pth`

Establish a pre-training baseline on `val` before any HSPOT fine-tuning:
~~~bash
make baseline \
  TEST_SET=val \
  EVAL_METRIC=both
~~~

Baseline outputs are written under `artifacts/baseline`.
In MLflow these runs are tagged with `stage=baseline_eval`.

### 8. Fine-tune
Train on `train` split:
~~~bash
make train \
  TRAIN_SPLIT=train
~~~

In MLflow these runs are tagged with `stage=fine_tune`.

Run on CPU (debug/smoke only, much slower):
~~~bash
make train GPU=none DEVICE=cpu TRAIN_SPLIT=train
~~~

### 9. Test
Find your checkpoint:
~~~bash
find artifacts/train -name model_final.pth
~~~

Compare the fine-tuned checkpoint against the pre-training baseline in:
~~~bash
find artifacts/baseline -name inference_metrics.json
~~~

Validation evaluation:
~~~bash
make test-finetune \
  TEST_SET=val \
  EVAL_METRIC=both
~~~

In MLflow these validation runs are tagged with `stage=fine_tune_eval`.

Run evaluation on CPU:
~~~bash
make test-finetune GPU=none DEVICE=cpu TEST_SET=val
~~~

### 10. Hyperparameter Tuning (Optuna, 1 GPU)
`make tune` expects a SiamMOT checkpoint as `BASE_MODEL_FILE`, typically the `model_final.pth`
from a previous HSPOT training run under `artifacts/train/...`.

Run HPO:
~~~bash
make tune \
  BASE_MODEL_FILE=/workspace/artifacts/train/DLA-34-FPN_box_EMM_MOT_HSPOT/model_final.pth \
  EVAL_METRIC=hota \
  N_TRIALS=10 \
  MAX_ITER=1500 \
  PRUNE_CHECKPOINTS=500,1000 \
  HPO_TRAIN_SPLIT=train \
  HPO_VAL_SPLIT=val
~~~

CPU-only HPO smoke run (very small):
~~~bash
make tune \
  GPU=none \
  DEVICE=cpu \
  BASE_MODEL_FILE=/workspace/artifacts/train/DLA-34-FPN_box_EMM_MOT_HSPOT/model_final.pth \
  N_TRIALS=1 \
  MAX_ITER=10 \
  PRUNE_CHECKPOINTS=5
~~~

Behavior:
1. Each trial fine-tunes with `train_net.py`.
2. Each trial evaluates on `val` with `test_net.py`.

These defaults keep HPO aligned with the small HSPOT `val` split by training each trial on
`train`, validating on `val`, and using a shorter schedule than the original generic settings.

Objective metric by `EVAL_METRIC`:
1. `clear` -> `infer/mot/idf1`
2. `hota` -> `infer/mot/hota`
3. `both` -> `infer/mot/hota`

HPO outputs:
1. `artifacts/hpo/best_trial.json`
2. `artifacts/hpo/study_trials.json`
3. `artifacts/hpo/hpo_summary.json`

### 11. Final Training With Best HPO Settings
Train a final HSPOT model with the best hyperparameters found by Optuna:
~~~bash
make train-best-hpo \
  BASE_MODEL_FILE=/workspace/artifacts/train/DLA-34-FPN_box_EMM_MOT_HSPOT/model_final.pth
~~~

This command:
1. reads `artifacts/hpo/best_trial.json`
2. extracts the best trial's `sampled_cfg`
3. launches `train_net.py` on `train` with those hyperparameters

Outputs are written under `artifacts/best_hpo_train`.
In MLflow these runs are tagged with `stage=final_train_best_hpo`.

### 12. Final Evaluation Of The Best HPO Model
Run the final test evaluation explicitly with the dedicated Make target:
~~~bash
make test-best-hpo \
  BEST_HPO_TEST_SET=test \
  EVAL_METRIC=both
~~~

By default this evaluates:
1. `/workspace/artifacts/best_hpo_train/DLA-34-FPN_box_EMM_MOT_HSPOT_best_hpo/model_final.pth`
2. on the HSPOT `test` split
3. with explicit MLflow tags including `stage=final_eval_best_hpo`

Outputs are written under `artifacts/best_hpo_eval`.

## Useful Commands
Show available Make targets:
~~~bash
make help
~~~

Open an interactive shell in the container:
~~~bash
make docker-shell
~~~
