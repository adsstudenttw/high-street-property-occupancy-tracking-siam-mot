## Installation
Please refer to [INSTALL.md](readme/INSTALL.md) for background installation notes.

## SURF Research Cloud (Ubuntu 22.04) Step-by-Step
This repository includes a Docker + uv + Makefile workflow for running on a SURF Research Cloud VM.

### GPU Setup (Main)

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

### 2. Move Docker And containerd Storage To The SURF Volume (Recommended For 20GB Root Disk)
Create Docker and containerd storage paths on the mounted SURF volume:
~~~bash
sudo mkdir -p /data/siammot_storage/docker
sudo mkdir -p /data/siammot_storage/containerd
~~~

Back up Docker daemon config (if present):
~~~bash
sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak.$(date +%F-%H%M%S) 2>/dev/null || true
~~~

Edit `/etc/docker/daemon.json` so it contains:
~~~json
{
  "data-root": "/data/siammot_storage/docker"
}
~~~

If your `/etc/docker/daemon.json` already contains keys (for example NVIDIA runtime config),
add only the `"data-root"` key and keep existing keys intact.

On GPU VMs that already have the NVIDIA runtime configured, the merged file will look like:
~~~json
{
  "data-root": "/data/siammot_storage/docker",
  "runtimes": {
    "nvidia": {
      "args": [],
      "path": "nvidia-container-runtime"
    }
  }
}
~~~

Create `/etc/containerd/config.toml` so containerd persistent storage is also moved off the root disk:
~~~bash
sudo mkdir -p /etc/containerd
sudo tee /etc/containerd/config.toml > /dev/null <<'EOF'
version = 2
root = "/data/siammot_storage/containerd"
state = "/run/containerd"
EOF
~~~

Restart containerd and Docker, then verify:
~~~bash
sudo systemctl restart containerd
sudo systemctl restart docker
make verify-storage-root
~~~

Expected:
1. `DockerRootDir=/data/siammot_storage/docker`
2. `root = "/data/siammot_storage/containerd"`
3. `state = "/run/containerd"`

You can also run `make verify-docker-root` or `make verify-containerd-root` individually.

If you previously used the default `/var/lib/containerd`, you can move the old data aside after the restart:
~~~bash
sudo systemctl stop docker
sudo systemctl stop containerd
sudo mv /var/lib/containerd /var/lib/containerd.bak.$(date +%F-%H%M%S)
sudo mkdir -p /var/lib/containerd
sudo systemctl start containerd
sudo systemctl start docker
~~~

Keep the backup directory until Docker builds and runs succeed on the VM.

### 3. Verify GPU Access from Docker
~~~bash
make verify-docker-gpu
~~~

If your VM has no GPU, follow the separate **CPU-Only Setup** section below.

### 4. Build the Project Image (dependencies installed with uv)
~~~bash
make docker-build
~~~

The Docker image:
1. Uses CUDA 11.0 + Ubuntu 20.04 inside the container.
2. Installs `uv`, then creates a Python 3.8 virtual environment at `/opt/venv`.
3. Installs CUDA 11.0 PyTorch/torchvision wheels, project dependencies from `requirements.txt`, `maskrcnn-benchmark` (with the upstream compatibility patch), and Apex.
4. Runs correctly on a newer Ubuntu 22.04 host as long as the NVIDIA driver is recent enough for CUDA 11.x containers.

Notes:
1. Apex is pinned to `git+https://github.com/NVIDIA/apex.git@da9f5ae` to stay compatible with `torch==1.7.1+cu110`.
2. The image still builds Apex with `APEX_CPP_EXT=1 APEX_CUDA_EXT=1`, matching the upstream `maskrcnn-benchmark` expectation more closely than a Python-only Apex install.
3. The Docker build also patches Apex's newer `torch.library` and `torch.compiler` checks so `import apex` remains compatible with Torch 1.7.1.
4. `cityscapesscripts` is installed with the `maskrcnn-benchmark` extras in the image, rather than as a generic project requirement.
5. NumPy is capped at the validated legacy value range because the MXNet/GluonCV dependency path used by SiamMOT still relies on the removed `np.bool` alias.
6. Pillow is capped below `10` because GluonCV still references legacy constants such as `Image.LINEAR`, which Pillow removed in `10.0.0`.
7. `requirements.txt` now uses upper bounds derived from `requirements_exact.txt` so `uv` can resolve dependencies without drifting past the validated legacy stack.
8. The Docker build preinstalls the legacy `numpy`/`Pillow` values and `cython`, then installs `requirements.txt` with `--no-build-isolation`, so older source builds such as `pycocotools==2.0.2` can complete under `uv`.
9. `pycocotools` is installed from `cocoapi/PythonAPI` in the image, following the upstream `maskrcnn-benchmark` install guide, because the old `pycocotools==2.0.2` sdist does not build reliably under `uv`.
10. `protobuf` is capped at `3.20.3` because older `tensorboard` code in this stack breaks with newer protobuf descriptor changes.
11. `maskrcnn-benchmark` is built with `FORCE_CUDA=1` and `TORCH_CUDA_ARCH_LIST=8.0+PTX`; CUDA 11.0 cannot compile `sm_86` directly, so this keeps the image compatible with Ampere A10 hosts via PTX JIT.

### 5. Put Datasets, Weights, and Artifacts on the SURF Volume
Set the host-side storage root to your mounted SURF volume path:
~~~bash
export HOST_STORAGE_ROOT=/data/siammot_storage/siammot
~~~

Create required host-side root directories:
~~~bash
make ensure-storage-dirs
~~~

Confirm Docker mount mappings:
~~~bash
make print-storage-config
~~~

With this setting, the Make targets will:
1. read datasets from `${HOST_STORAGE_ROOT}/datasets` (host) via `/workspace/datasets` (container)
2. read weights from `${HOST_STORAGE_ROOT}/weights` (host) via `/workspace/weights` (container)
3. write artifacts to `${HOST_STORAGE_ROOT}/artifacts` (host) via `/workspace/artifacts` (container)

Place the pretrained checkpoint at:
1. `${HOST_STORAGE_ROOT}/weights/DLA-34-FPN_EMM_crowdhuman_mot17.pth`

### 6. Prepare and Ingest the HSPOT Dataset
Expected layout:
1. `${HOST_STORAGE_ROOT}/datasets/hspot/raw_data/train`
2. `${HOST_STORAGE_ROOT}/datasets/hspot/raw_data/val`
3. `${HOST_STORAGE_ROOT}/datasets/hspot/raw_data/test`

Run ingestion in Docker:
~~~bash
make ingest DATASET_PATH=/workspace/datasets/hspot ANNO_NAME=anno.json MOT17=false DET_OPTIONS=""
~~~

Notes:
1. `DET_OPTIONS=""` ingests all sequence folders and does not require MOT17 detector suffixes.
2. HSPOT is ingested as a single-class MOT-style dataset, so the MOT class column is ignored.
3. `seqinfo.ini` is still used when present, so HSPOT keeps its true frame rate of 1 fps.
4. This project uses dataset key `MOT_HSPOT`.

### 7. Configure MLflow Tracking URI
Point jobs to your separate MLflow VM:
~~~bash
export MLFLOW_TRACKING_URI=http://ubuntu2204sudo.property-occupa.src.surf-hosted.nl:80
~~~

All `make` targets pass this environment variable into the container. Do not leave it at
`127.0.0.1` for Docker runs unless the MLflow server is actually running inside that same
container, because loopback inside Docker points back to the container itself.

Run the dedicated MLflow smoke test before training if you want to validate remote logging end to end:
~~~bash
make smoke-mlflow
~~~

It creates a short run under experiment `remote-mlflow-smoke-test`, logs one metric and one text artifact, then prints the run ID.

### 8. TrackEval Status
TrackEval is already vendored in this repository under `third_party/TrackEval`.
If you want to update it later:
~~~bash
make trackeval-update
~~~

### 9. Establish Baseline
The default HSPOT config initializes from:
1. `/workspace/weights/DLA-34-FPN_EMM_crowdhuman_mot17.pth` (mapped from `${HOST_STORAGE_ROOT}/weights`)

Establish a pre-training baseline on `val` before any HSPOT fine-tuning:
~~~bash
make baseline \
  TEST_SET=val \
  EVAL_METRIC=hota
~~~

Baseline outputs are written under `${HOST_STORAGE_ROOT}/artifacts/baseline` on the VM host.
In MLflow these runs are tagged with `stage=baseline_eval`.

HOTA duplicate handling:
1. By default, HOTA normalizes duplicate GT and prediction `(frame, track_id)` pairs with `keep_first` so baseline, fine-tuning, HPO, and final evaluation can all complete without extra flags.
2. This is controlled by:
   `INFERENCE.HOTA_DUPLICATE_GT_POLICY`
   `INFERENCE.HOTA_DUPLICATE_PRED_POLICY`
   Supported values: `error`, `keep_first`, `keep_highest_conf`
3. The default baseline command therefore already runs with `keep_first` normalization:
~~~bash
make baseline \
  TEST_SET=val \
  EVAL_METRIC=hota
~~~
4. Use strict mode when you want to diagnose whether the raw tracker output is directly TrackEval-valid:
~~~bash
make baseline \
  TEST_SET=val \
  EVAL_METRIC=hota \
  TEST_EXTRA_OPTS="INFERENCE.HOTA_DUPLICATE_PRED_POLICY error INFERENCE.HOTA_DUPLICATE_GT_POLICY error"
~~~
5. In normalized mode, duplicate predictions are collapsed only in the temporary MOT export used for TrackEval. The in-memory SIAMMOT results are not modified.

### 10. Fine-tune
Train on `train` split:
~~~bash
make train \
  TRAIN_SPLIT=train
~~~

In MLflow these runs are tagged with `stage=fine_tune`.

### 11. Test
Find your checkpoint:
~~~bash
find "${HOST_STORAGE_ROOT}/artifacts/train" -name model_final.pth
~~~

Compare the fine-tuned checkpoint against the pre-training baseline in:
~~~bash
find "${HOST_STORAGE_ROOT}/artifacts/baseline" -name inference_metrics.json
~~~

Validation evaluation:
~~~bash
make test-finetune \
  TEST_SET=val \
  EVAL_METRIC=hota
~~~

In MLflow these validation runs are tagged with `stage=fine_tune_eval`.

### 12. Hyperparameter Tuning (Optuna, 1 GPU)
`make tune` starts each HPO trial from `BASE_MODEL_FILE`, which defaults to the pre-trained
SiamMOT checkpoint `/workspace/weights/DLA-34-FPN_EMM_crowdhuman_mot17.pth`.

Run HPO:
~~~bash
make tune \
  BASE_MODEL_FILE=/workspace/weights/DLA-34-FPN_EMM_crowdhuman_mot17.pth \
  EVAL_METRIC=hota \
  N_TRIALS=15 \
  MAX_ITER=2000 \
  PRUNE_CHECKPOINTS=600,1400 \
  HPO_TRAIN_SPLIT=train \
  HPO_VAL_SPLIT=val
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
1. `${HOST_STORAGE_ROOT}/artifacts/hpo/best_trial.json`
2. `${HOST_STORAGE_ROOT}/artifacts/hpo/study_trials.json`
3. `${HOST_STORAGE_ROOT}/artifacts/hpo/hpo_summary.json`

### 13. Final Evaluation Of The Best HPO Model
Run the final test evaluation explicitly with the dedicated Make target:
~~~bash
make test-best-hpo \
  BEST_HPO_TEST_SET=test \
  EVAL_METRIC=hota
~~~

By default this evaluates:
1. the `user_attrs.final_checkpoint` from `${HOST_STORAGE_ROOT}/artifacts/hpo/best_trial.json`
2. on the HSPOT `test` split
3. with explicit MLflow tags including `stage=final_eval_best_hpo`

Outputs are written under `${HOST_STORAGE_ROOT}/artifacts/best_hpo_eval`.
Optionally set `BEST_HPO_MODEL_FILE=<checkpoint>` to override the checkpoint from `best_trial.json`.

### CPU-Only Setup (Separate Path)
Use this path for CPU-only validation/debug runs. It is much slower than GPU training.

### 0. Prerequisites on the VM
1. Ubuntu 22.04 VM (no GPU required).
2. This repository cloned on the VM.
3. Your MLflow tracking server already running on a separate VM.

### 1. Install Docker (CPU-Only)
~~~bash
make vm-bootstrap-cpu
~~~

Then log out/in (or run `newgrp docker`) so your user can run Docker without `sudo`.

### 2. Move Docker And containerd Storage To The SURF Volume (Recommended For 20GB Root Disk)
Create Docker and containerd storage on the mounted SURF volume:
~~~bash
sudo mkdir -p /data/siammot_storage/docker
sudo mkdir -p /data/siammot_storage/containerd
~~~

Back up Docker daemon config (if present):
~~~bash
sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak.$(date +%F-%H%M%S) 2>/dev/null || true
~~~

Edit `/etc/docker/daemon.json` so it contains:
~~~json
{
  "data-root": "/data/siammot_storage/docker"
}
~~~

Create `/etc/containerd/config.toml` so containerd persistent storage is also moved off the root disk:
~~~bash
sudo mkdir -p /etc/containerd
sudo tee /etc/containerd/config.toml > /dev/null <<'EOF'
version = 2
root = "/data/siammot_storage/containerd"
state = "/run/containerd"
EOF
~~~

Restart containerd and Docker, then verify:
~~~bash
sudo systemctl restart containerd
sudo systemctl restart docker
make verify-storage-root
~~~

Expected:
1. `DockerRootDir=/data/siammot_storage/docker`
2. `root = "/data/siammot_storage/containerd"`
3. `state = "/run/containerd"`

You can also run `make verify-docker-root` or `make verify-containerd-root` individually.

If you previously used the default `/var/lib/containerd`, you can move the old data aside after the restart:
~~~bash
sudo systemctl stop docker
sudo systemctl stop containerd
sudo mv /var/lib/containerd /var/lib/containerd.bak.$(date +%F-%H%M%S)
sudo mkdir -p /var/lib/containerd
sudo systemctl start containerd
sudo systemctl start docker
~~~

### 3. Build the Project Image
~~~bash
make docker-build
~~~

### 4. Configure SURF Volume Storage
~~~bash
export HOST_STORAGE_ROOT=/data/siammot_storage/siammot
make ensure-storage-dirs
make print-storage-config
~~~

### 5. Verify CPU Container Runtime
~~~bash
make verify-docker-cpu
make smoke-cpu
make smoke-mlflow
~~~

### 6. Prepare Dataset and Weights
1. Put dataset under `${HOST_STORAGE_ROOT}/datasets/hspot/raw_data/...`
2. Put pretrained checkpoint at `${HOST_STORAGE_ROOT}/weights/DLA-34-FPN_EMM_crowdhuman_mot17.pth`
3. Ingest dataset:

~~~bash
make ingest DATASET_PATH=/workspace/datasets/hspot ANNO_NAME=anno.json MOT17=false DET_OPTIONS=""
~~~

### 7. CPU Baseline, Train, and Eval Commands
~~~bash
make baseline GPU=none DEVICE=cpu TEST_SET=val EVAL_METRIC=hota
make train GPU=none DEVICE=cpu TRAIN_SPLIT=train
make test-finetune GPU=none DEVICE=cpu TEST_SET=val EVAL_METRIC=hota
~~~

### 8. Optional Tiny CPU HPO Smoke Run
~~~bash
make tune \
  GPU=none \
  DEVICE=cpu \
  BASE_MODEL_FILE=/workspace/weights/DLA-34-FPN_EMM_crowdhuman_mot17.pth \
  N_TRIALS=1 \
  MAX_ITER=10 \
  PRUNE_CHECKPOINTS=5
~~~

## Useful Commands
Show available Make targets:
~~~bash
make help
~~~

Open an interactive shell in the container:
~~~bash
make docker-shell
~~~
