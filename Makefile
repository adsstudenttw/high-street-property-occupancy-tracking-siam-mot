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

ARTIFACT_ROOT ?= /workspace/artifacts
TRAIN_ARTIFACT_DIR ?= $(ARTIFACT_ROOT)/train
INFER_ARTIFACT_DIR ?= $(ARTIFACT_ROOT)/infer
HPO_ARTIFACT_DIR ?= $(ARTIFACT_ROOT)/hpo
BASELINE_ARTIFACT_DIR ?= $(ARTIFACT_ROOT)/baseline

DATASET_PATH ?= datasets/hspot
ANNO_NAME ?= anno.json
MOT17 ?= false
DET_OPTIONS ?=

TRAIN_SPLIT ?= train
TRAIN_EXTRA_OPTS ?=

TEST_SET ?= val
EVAL_METRIC ?= both
MODEL_FILE ?=
BASELINE_MODEL_FILE ?= $(WORKDIR)/weights/DLA-34-FPN_EMM_crowdhuman_mot17.pth
TEST_EXTRA_OPTS ?=

BASE_MODEL_FILE ?=
STUDY_NAME ?= hspot_hpo
HPO_TRAIN_SPLIT ?= val
HPO_VAL_SPLIT ?= val
HPO_TEST_SPLIT ?= test
N_TRIALS ?= 20
MAX_ITER ?= 6000
PRUNE_CHECKPOINTS ?= 1000,3000
TUNE_MLFLOW_FLAG ?= --mlflow-enabled
TUNE_EXTRA_OPTS ?=

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
	-w $(WORKDIR)

.PHONY: help vm-bootstrap-docker verify-docker-gpu docker-build docker-shell \
	verify-docker-cpu smoke-cpu ingest baseline train test tune trackeval-add trackeval-update

help: ## Show available targets.
	@grep -E '^[a-zA-Z0-9_.-]+:.*## ' Makefile | awk 'BEGIN {FS=":.*## "}; {printf "%-24s %s\n", $$1, $$2}'

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

verify-docker-gpu: ## Validate Docker + GPU runtime.
	docker --version
	docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

verify-docker-cpu: GPU=none
verify-docker-cpu: ## Validate container startup without GPU access.
	docker --version
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python -c 'import platform; print(platform.platform())'"

docker-build: ## Build the project image (dependencies installed with uv).
	docker build -t $(IMAGE) -f Dockerfile .

docker-shell: ## Open an interactive shell in the project container.
	$(DOCKER_RUN_BASE) -it $(IMAGE) bash

smoke-cpu: GPU=none
smoke-cpu: DEVICE=cpu
smoke-cpu: ## Run a CPU-only smoke test (no dataset required).
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python -c 'import torch; print(\"torch\", torch.__version__); print(\"cuda_available\", torch.cuda.is_available())'"
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python tools/train_net.py --help > /tmp/train_help.txt && python tools/test_net.py --help > /tmp/test_help.txt && python tools/tune_optuna.py --help > /tmp/tune_help.txt && echo smoke_ok"

ingest: ## Ingest custom MOT dataset from DATASET_PATH.
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python siammot/data/ingestion/ingest_mot.py --dataset_path $(DATASET_PATH) --anno_name $(ANNO_NAME) --mot17 $(MOT17) --det-options \"$(DET_OPTIONS)\""

baseline: ## Evaluate the default pre-trained checkpoint before HSPOT training.
	mkdir -p artifacts/baseline
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python tools/test_net.py --config-file $(CONFIG_FILE) --output-dir $(BASELINE_ARTIFACT_DIR) --model-file $(BASELINE_MODEL_FILE) --test-dataset $(DATASET_KEY) --set $(TEST_SET) --opts DATASETS.ROOT_DIR datasets INFERENCE.EVAL_METRIC $(EVAL_METRIC) $(COMMON_DEVICE_OPTS) $(TEST_EXTRA_OPTS)"

train: ## Train/fine-tune with tools/train_net.py.
	mkdir -p artifacts/train
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python tools/train_net.py --config-file $(CONFIG_FILE) --train-dir $(TRAIN_ARTIFACT_DIR) --opts DATASETS.ROOT_DIR datasets DATASETS.TRAIN \"('$(DATASET_KEY)',)\" DATASETS.TRAIN_SET $(TRAIN_SPLIT) $(COMMON_DEVICE_OPTS) $(TRAIN_EXTRA_OPTS)"

test: ## Evaluate with tools/test_net.py (set MODEL_FILE and TEST_SET=val|test).
	@if [ -z "$(MODEL_FILE)" ]; then echo "MODEL_FILE is required, e.g. make test MODEL_FILE=/workspace/artifacts/train/.../model_final.pth"; exit 1; fi
	mkdir -p artifacts/infer
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python tools/test_net.py --config-file $(CONFIG_FILE) --output-dir $(INFER_ARTIFACT_DIR) --model-file $(MODEL_FILE) --test-dataset $(DATASET_KEY) --set $(TEST_SET) --opts DATASETS.ROOT_DIR datasets INFERENCE.EVAL_METRIC $(EVAL_METRIC) $(COMMON_DEVICE_OPTS) $(TEST_EXTRA_OPTS)"

tune: ## Run Optuna HPO (set BASE_MODEL_FILE).
	@if [ -z "$(BASE_MODEL_FILE)" ]; then echo "BASE_MODEL_FILE is required, e.g. make tune BASE_MODEL_FILE=/workspace/artifacts/train/.../model_final.pth"; exit 1; fi
	mkdir -p artifacts/hpo
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python tools/tune_optuna.py --project-root . --config-file $(CONFIG_FILE) --base-model-file $(BASE_MODEL_FILE) --output-dir $(HPO_ARTIFACT_DIR) --study-name $(STUDY_NAME) --dataset-key $(DATASET_KEY) --train-split $(HPO_TRAIN_SPLIT) --val-split $(HPO_VAL_SPLIT) --test-split $(HPO_TEST_SPLIT) --eval-metric $(EVAL_METRIC) --n-trials $(N_TRIALS) --max-iter $(MAX_ITER) --prune-checkpoints $(PRUNE_CHECKPOINTS) --base-opts $(COMMON_DEVICE_OPTS) $(TUNE_MLFLOW_FLAG) $(TUNE_EXTRA_OPTS)"

trackeval-add: ## Vendor TrackEval into third_party/TrackEval.
	git subtree add --prefix third_party/TrackEval https://github.com/JonathonLuiten/TrackEval.git master --squash

trackeval-update: ## Update vendored TrackEval.
	git subtree pull --prefix third_party/TrackEval https://github.com/JonathonLuiten/TrackEval.git master --squash
