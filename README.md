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
1. `datasets/hspot/train`
2. `datasets/hspot/val`
3. `datasets/hspot/test`

Run ingestion in Docker:
~~~bash
make ingest DATASET_PATH=datasets/hspot ANNO_NAME=anno.json MOT17=true DET_OPTIONS=""
~~~

Notes:
1. `DET_OPTIONS=""` ingests all sequence folders and does not require MOT17 detector suffixes.
2. Use `MOT17=false` if your GT does not include MOT17 class/visibility columns.
3. This project uses dataset key `MOT_HSPOT`.

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

### 7. Fine-tune
Train on `train` split:
~~~bash
make train \
  TRAIN_SPLIT=train
~~~

Fine-tune on `val` split (if desired):
~~~bash
make train TRAIN_SPLIT=val
~~~

Run on CPU (debug/smoke only, much slower):
~~~bash
make train GPU=none DEVICE=cpu TRAIN_SPLIT=train
~~~

### 8. Test
Find your checkpoint:
~~~bash
find artifacts/train -name model_final.pth
~~~

Validation evaluation:
~~~bash
make test \
  MODEL_FILE=/workspace/artifacts/train/<MODEL_NAME>/model_final.pth \
  TEST_SET=val \
  EVAL_METRIC=both
~~~

Final test evaluation:
~~~bash
make test \
  MODEL_FILE=/workspace/artifacts/train/<MODEL_NAME>/model_final.pth \
  TEST_SET=test \
  EVAL_METRIC=both
~~~

Run evaluation on CPU:
~~~bash
make test GPU=none DEVICE=cpu MODEL_FILE=/workspace/artifacts/train/<MODEL_NAME>/model_final.pth TEST_SET=val
~~~

### 9. Hyperparameter Tuning (Optuna, 1 GPU)
Run HPO:
~~~bash
make tune \
  BASE_MODEL_FILE=/workspace/artifacts/train/<MODEL_NAME>/model_final.pth \
  EVAL_METRIC=hota \
  N_TRIALS=20 \
  MAX_ITER=6000 \
  PRUNE_CHECKPOINTS=1000,3000 \
  HPO_TRAIN_SPLIT=val \
  HPO_VAL_SPLIT=val \
  HPO_TEST_SPLIT=test
~~~

CPU-only HPO smoke run (very small):
~~~bash
make tune \
  GPU=none \
  DEVICE=cpu \
  BASE_MODEL_FILE=/workspace/artifacts/train/<MODEL_NAME>/model_final.pth \
  N_TRIALS=1 \
  MAX_ITER=10 \
  PRUNE_CHECKPOINTS=5
~~~

Behavior:
1. Each trial fine-tunes with `train_net.py`.
2. Each trial evaluates on `val` with `test_net.py`.
3. After study completion, one final evaluation runs on `test`.

Objective metric by `EVAL_METRIC`:
1. `clear` -> `infer/mot/idf1`
2. `hota` -> `infer/mot/hota`
3. `both` -> `infer/mot/hota`

HPO outputs:
1. `artifacts/hpo/best_trial.json`
2. `artifacts/hpo/study_trials.json`
3. `artifacts/hpo/final_test_eval/final_test_metrics.json`

## Useful Commands
Show available Make targets:
~~~bash
make help
~~~

Open an interactive shell in the container:
~~~bash
make docker-shell
~~~
