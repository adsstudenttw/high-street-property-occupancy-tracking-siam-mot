SHELL := /bin/bash
.DEFAULT_GOAL := help

IMAGE_NAME ?= siammot
IMAGE_TAG ?= ubuntu22-cu118-uv
IMAGE := $(IMAGE_NAME):$(IMAGE_TAG)

PROJECT_ROOT := $(abspath .)
WORKDIR ?= /workspace

GPU ?= all
SHM_SIZE ?= 16g
MLFLOW_TRACKING_URI ?= http://127.0.0.1:5000
DEVICE ?= cuda

CONFIG_FILE ?= configs/dla/DLA_34_FPN_EMM_HSPOT.yaml
DATASET_KEY ?= MOT_HSPOT

# Host-side storage locations (set HOST_STORAGE_ROOT to your mounted SURF volume).
HOST_STORAGE_ROOT ?= $(PROJECT_ROOT)
HOST_DATASETS_DIR ?= $(HOST_STORAGE_ROOT)/datasets
HOST_WEIGHTS_DIR ?= $(HOST_STORAGE_ROOT)/weights
HOST_ARTIFACT_ROOT ?= $(HOST_STORAGE_ROOT)/artifacts
HOST_TRAIN_ARTIFACT_DIR ?= $(HOST_ARTIFACT_ROOT)/train
HOST_INFER_ARTIFACT_DIR ?= $(HOST_ARTIFACT_ROOT)/infer
HOST_HPO_ARTIFACT_DIR ?= $(HOST_ARTIFACT_ROOT)/hpo
HOST_BASELINE_ARTIFACT_DIR ?= $(HOST_ARTIFACT_ROOT)/baseline
HOST_FINE_TUNE_EVAL_ARTIFACT_DIR ?= $(HOST_ARTIFACT_ROOT)/fine_tune_eval
HOST_BEST_HPO_EVAL_ARTIFACT_DIR ?= $(HOST_ARTIFACT_ROOT)/best_hpo_eval

# Container-side mount points for the large data directories.
CONTAINER_DATASETS_DIR ?= $(WORKDIR)/datasets
CONTAINER_WEIGHTS_DIR ?= $(WORKDIR)/weights
CONTAINER_ARTIFACT_ROOT ?= $(WORKDIR)/artifacts

ARTIFACT_ROOT ?= $(CONTAINER_ARTIFACT_ROOT)
TRAIN_ARTIFACT_DIR ?= $(ARTIFACT_ROOT)/train
INFER_ARTIFACT_DIR ?= $(ARTIFACT_ROOT)/infer
HPO_ARTIFACT_DIR ?= $(ARTIFACT_ROOT)/hpo
BASELINE_ARTIFACT_DIR ?= $(ARTIFACT_ROOT)/baseline
FINE_TUNE_EVAL_ARTIFACT_DIR ?= $(ARTIFACT_ROOT)/fine_tune_eval
BEST_HPO_EVAL_ARTIFACT_DIR ?= $(ARTIFACT_ROOT)/best_hpo_eval

DATASET_PATH ?= $(CONTAINER_DATASETS_DIR)/hspot
ANNO_NAME ?= anno.json
MOT17 ?= false
DET_OPTIONS ?=

TRAIN_SPLIT ?= train
TRAIN_EXTRA_OPTS ?=

TEST_SET ?= val
EVAL_METRIC ?= both
MODEL_FILE ?=
BASELINE_MODEL_FILE ?= $(CONTAINER_WEIGHTS_DIR)/DLA-34-FPN_EMM_crowdhuman_mot17.pth
FINE_TUNE_MODEL_FILE ?= $(TRAIN_ARTIFACT_DIR)/DLA-34-FPN_box_EMM_MOT_HSPOT/model_final.pth
TEST_EXTRA_OPTS ?=

BASE_MODEL_FILE ?= $(CONTAINER_WEIGHTS_DIR)/DLA-34-FPN_EMM_crowdhuman_mot17.pth
BEST_TRIAL_FILE ?= $(HPO_ARTIFACT_DIR)/best_trial.json
BEST_HPO_MODEL_FILE ?=
BEST_HPO_TEST_SET ?= test
STUDY_NAME ?= hspot_hpo
HPO_TRAIN_SPLIT ?= train
HPO_VAL_SPLIT ?= val
N_TRIALS ?= 30
MAX_ITER ?= 1800
PRUNE_CHECKPOINTS ?= 450,1100
TUNE_MLFLOW_FLAG ?= --mlflow-enabled
TUNE_EXTRA_OPTS ?= --timeout-sec 360000

ifeq ($(GPU),none)
DOCKER_GPU_ARGS :=
else
DOCKER_GPU_ARGS := --gpus $(GPU)
endif

COMMON_DEVICE_OPTS := MODEL.DEVICE $(DEVICE)
ifeq ($(DEVICE),cpu)
COMMON_DEVICE_OPTS += DTYPE float32
endif

DOCKER_RUN_BASE := docker run --rm $(DOCKER_GPU_ARGS) --shm-size=$(SHM_SIZE) \
	-e MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
	-e PYTHONPATH=$(WORKDIR) \
	-v "$(PROJECT_ROOT):$(WORKDIR)" \
	-v "$(HOST_DATASETS_DIR):$(CONTAINER_DATASETS_DIR)" \
	-v "$(HOST_WEIGHTS_DIR):$(CONTAINER_WEIGHTS_DIR)" \
	-v "$(HOST_ARTIFACT_ROOT):$(CONTAINER_ARTIFACT_ROOT)" \
	-w $(WORKDIR)

.PHONY: help vm-bootstrap-cpu vm-bootstrap-docker verify-docker-gpu docker-build docker-shell \
	verify-docker-cpu smoke-cpu ingest baseline train test-finetune test test-best-hpo tune trackeval-add trackeval-update \
	ensure-storage-dirs print-storage-config

help: ## Show available targets.
	@grep -E '^[a-zA-Z0-9_.-]+:.*## ' Makefile | awk 'BEGIN {FS=":.*## "}; {printf "%-24s %s\n", $$1, $$2}'

ensure-storage-dirs: ## Ensure host-side storage directories exist.
	mkdir -p "$(HOST_DATASETS_DIR)" "$(HOST_WEIGHTS_DIR)" "$(HOST_ARTIFACT_ROOT)"

print-storage-config: ## Print host/container storage mappings used by Docker runs.
	@echo "HOST_STORAGE_ROOT=$(HOST_STORAGE_ROOT)"
	@echo "HOST_DATASETS_DIR=$(HOST_DATASETS_DIR) -> $(CONTAINER_DATASETS_DIR)"
	@echo "HOST_WEIGHTS_DIR=$(HOST_WEIGHTS_DIR) -> $(CONTAINER_WEIGHTS_DIR)"
	@echo "HOST_ARTIFACT_ROOT=$(HOST_ARTIFACT_ROOT) -> $(CONTAINER_ARTIFACT_ROOT)"

vm-bootstrap-docker: ## Install Docker + NVIDIA container toolkit on Ubuntu 22.04.
	sudo apt-get update
	sudo apt-get install -y ca-certificates curl gnupg lsb-release
	sudo install -m 0755 -d /etc/apt/keyrings
	curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
	sudo chmod a+r /etc/apt/keyrings/docker.gpg
	echo "deb [arch=$$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $$(. /etc/os-release && echo $$VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
	sudo apt-get update
	sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
	sudo usermod -aG docker $$USER
	curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
	. /etc/os-release; curl -s -L "https://nvidia.github.io/libnvidia-container/$${ID}$${VERSION_ID}/libnvidia-container.list" | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
	sudo apt-get update
	sudo apt-get install -y nvidia-container-toolkit
	sudo nvidia-ctk runtime configure --runtime=docker
	sudo systemctl restart docker
	@echo ""
	@echo "Bootstrap complete. Log out/in (or run 'newgrp docker') before using Docker without sudo."

vm-bootstrap-cpu: ## Install Docker only on Ubuntu 22.04 (no NVIDIA toolkit).
	sudo apt-get update
	sudo apt-get install -y ca-certificates curl gnupg lsb-release
	sudo install -m 0755 -d /etc/apt/keyrings
	curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
	sudo chmod a+r /etc/apt/keyrings/docker.gpg
	echo "deb [arch=$$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $$(. /etc/os-release && echo $$VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
	sudo apt-get update
	sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
	sudo usermod -aG docker $$USER
	sudo systemctl restart docker
	@echo ""
	@echo "CPU bootstrap complete. Log out/in (or run 'newgrp docker') before using Docker without sudo."

verify-docker-gpu: ## Validate Docker + GPU runtime.
	docker --version
	docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

verify-docker-cpu: GPU=none
verify-docker-cpu: ensure-storage-dirs ## Validate container startup without GPU access.
	docker --version
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python -c 'import platform; print(platform.platform())'"

docker-build: ## Build the project image (dependencies installed with uv).
	docker build -t $(IMAGE) -f Dockerfile .

docker-shell: ensure-storage-dirs ## Open an interactive shell in the project container.
	$(DOCKER_RUN_BASE) -it $(IMAGE) bash

smoke-cpu: GPU=none
smoke-cpu: DEVICE=cpu
smoke-cpu: ensure-storage-dirs ## Run a CPU-only smoke test (no dataset required).
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python -c 'import torch; print(\"torch\", torch.__version__); print(\"cuda_available\", torch.cuda.is_available())'"
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python tools/train_net.py --help > /tmp/train_help.txt && python tools/test_net.py --help > /tmp/test_help.txt && python tools/tune_optuna.py --help > /tmp/tune_help.txt && echo smoke_ok"

ingest: ensure-storage-dirs ## Ingest custom MOT dataset from DATASET_PATH.
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python siammot/data/ingestion/ingest_mot.py --dataset_path $(DATASET_PATH) --anno_name $(ANNO_NAME) --mot17 $(MOT17) --det-options \"$(DET_OPTIONS)\""

baseline: ensure-storage-dirs ## Evaluate the default pre-trained checkpoint before HSPOT training.
	mkdir -p "$(HOST_BASELINE_ARTIFACT_DIR)"
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python tools/test_net.py --config-file $(CONFIG_FILE) --output-dir $(BASELINE_ARTIFACT_DIR) --model-file $(BASELINE_MODEL_FILE) --test-dataset $(DATASET_KEY) --set $(TEST_SET) --extra-mlflow-tags stage=baseline_eval workflow=baseline_hspot dataset_key=$(DATASET_KEY) eval_split=$(TEST_SET) --opts DATASETS.ROOT_DIR $(CONTAINER_DATASETS_DIR) INFERENCE.EVAL_METRIC $(EVAL_METRIC) $(COMMON_DEVICE_OPTS) $(TEST_EXTRA_OPTS)"

train: ensure-storage-dirs ## Train/fine-tune with tools/train_net.py.
	mkdir -p "$(HOST_TRAIN_ARTIFACT_DIR)"
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python tools/train_net.py --config-file $(CONFIG_FILE) --train-dir $(TRAIN_ARTIFACT_DIR) --extra-mlflow-tags stage=fine_tune workflow=fine_tune_hspot dataset_key=$(DATASET_KEY) train_split=$(TRAIN_SPLIT) --opts DATASETS.ROOT_DIR $(CONTAINER_DATASETS_DIR) DATASETS.TRAIN \"('$(DATASET_KEY)',)\" DATASETS.TRAIN_SET $(TRAIN_SPLIT) $(COMMON_DEVICE_OPTS) $(TRAIN_EXTRA_OPTS)"

test-finetune: ensure-storage-dirs ## Evaluate the standard fine-tuned HSPOT model with explicit MLflow tagging.
	mkdir -p "$(HOST_FINE_TUNE_EVAL_ARTIFACT_DIR)"
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python tools/test_net.py --config-file $(CONFIG_FILE) --output-dir $(FINE_TUNE_EVAL_ARTIFACT_DIR) --model-file $(FINE_TUNE_MODEL_FILE) --test-dataset $(DATASET_KEY) --set $(TEST_SET) --extra-mlflow-tags stage=fine_tune_eval workflow=fine_tune_eval_hspot dataset_key=$(DATASET_KEY) eval_split=$(TEST_SET) model_origin=fine_tune --opts DATASETS.ROOT_DIR $(CONTAINER_DATASETS_DIR) INFERENCE.EVAL_METRIC $(EVAL_METRIC) $(COMMON_DEVICE_OPTS) $(TEST_EXTRA_OPTS)"

test: ensure-storage-dirs ## Evaluate with tools/test_net.py (set MODEL_FILE and TEST_SET=val|test).
	@if [ -z "$(MODEL_FILE)" ]; then echo "MODEL_FILE is required, e.g. make test MODEL_FILE=/workspace/artifacts/train/.../model_final.pth"; exit 1; fi
	mkdir -p "$(HOST_INFER_ARTIFACT_DIR)"
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python tools/test_net.py --config-file $(CONFIG_FILE) --output-dir $(INFER_ARTIFACT_DIR) --model-file $(MODEL_FILE) --test-dataset $(DATASET_KEY) --set $(TEST_SET) --opts DATASETS.ROOT_DIR $(CONTAINER_DATASETS_DIR) INFERENCE.EVAL_METRIC $(EVAL_METRIC) $(COMMON_DEVICE_OPTS) $(TEST_EXTRA_OPTS)"

test-best-hpo: ensure-storage-dirs ## Evaluate the best trial checkpoint from HPO on the final split.
	mkdir -p "$(HOST_BEST_HPO_EVAL_ARTIFACT_DIR)"
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python tools/test_best_hpo.py --project-root . --config-file $(CONFIG_FILE) --best-trial-file $(BEST_TRIAL_FILE) --output-dir $(BEST_HPO_EVAL_ARTIFACT_DIR) --dataset-key $(DATASET_KEY) --datasets-root $(CONTAINER_DATASETS_DIR) --test-split $(BEST_HPO_TEST_SET) --eval-metric $(EVAL_METRIC) $(if $(BEST_HPO_MODEL_FILE),--model-file $(BEST_HPO_MODEL_FILE),) --base-opts $(COMMON_DEVICE_OPTS) $(TEST_EXTRA_OPTS)"

tune: ensure-storage-dirs ## Run Optuna HPO (set BASE_MODEL_FILE).
	@if [ -z "$(BASE_MODEL_FILE)" ]; then echo "BASE_MODEL_FILE is required, e.g. make tune BASE_MODEL_FILE=/workspace/weights/DLA-34-FPN_EMM_crowdhuman_mot17.pth"; exit 1; fi
	mkdir -p "$(HOST_HPO_ARTIFACT_DIR)"
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python tools/tune_optuna.py --project-root . --config-file $(CONFIG_FILE) --base-model-file $(BASE_MODEL_FILE) --output-dir $(HPO_ARTIFACT_DIR) --study-name $(STUDY_NAME) --dataset-key $(DATASET_KEY) --datasets-root $(CONTAINER_DATASETS_DIR) --train-split $(HPO_TRAIN_SPLIT) --val-split $(HPO_VAL_SPLIT) --eval-metric $(EVAL_METRIC) --n-trials $(N_TRIALS) --max-iter $(MAX_ITER) --prune-checkpoints $(PRUNE_CHECKPOINTS) --base-opts $(COMMON_DEVICE_OPTS) $(TUNE_MLFLOW_FLAG) $(TUNE_EXTRA_OPTS)"

trackeval-add: ## Vendor TrackEval into third_party/TrackEval.
	git subtree add --prefix third_party/TrackEval https://github.com/JonathonLuiten/TrackEval.git master --squash

trackeval-update: ## Update vendored TrackEval.
	git subtree pull --prefix third_party/TrackEval https://github.com/JonathonLuiten/TrackEval.git master --squash
