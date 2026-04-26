SHELL := /bin/bash
.DEFAULT_GOAL := help

IMAGE_NAME ?= siammot
IMAGE_TAG ?= ubuntu20-cu110-uv
IMAGE := $(IMAGE_NAME):$(IMAGE_TAG)

PROJECT_ROOT := $(abspath .)
WORKDIR ?= /workspace

GPU ?= all
SHM_SIZE ?= 16g
MLFLOW_TRACKING_URI ?= http://ubuntu2204sudo.property-occupa.src.surf-hosted.nl:80
DEVICE ?= cuda

CONFIG_FILE ?= configs/dla/DLA_34_FPN_EMM_HSPOT.yaml
DATASET_KEY ?= MOT_HSPOT

# Host-side storage locations.
HOST_STORAGE_ROOT ?= /data/siammot_storage/siammot
HOST_DATASETS_DIR ?= $(HOST_STORAGE_ROOT)/datasets
HOST_WEIGHTS_DIR ?= $(HOST_STORAGE_ROOT)/weights
HOST_ARTIFACT_ROOT ?= $(HOST_STORAGE_ROOT)/artifacts
HOST_TRAIN_ARTIFACT_DIR ?= $(HOST_ARTIFACT_ROOT)/train
HOST_INFER_ARTIFACT_DIR ?= $(HOST_ARTIFACT_ROOT)/infer
HOST_FINAL_EVAL_ARTIFACT_DIR ?= $(HOST_HPO_ARTIFACT_DIR)/best_hpo_eval
HOST_HPO_ARTIFACT_DIR ?= $(HOST_ARTIFACT_ROOT)/hpo
HOST_BASELINE_ARTIFACT_DIR ?= $(HOST_ARTIFACT_ROOT)/baseline

# Container-side mount points for the large data directories.
CONTAINER_DATASETS_DIR ?= $(WORKDIR)/datasets
CONTAINER_WEIGHTS_DIR ?= $(WORKDIR)/weights
CONTAINER_ARTIFACT_ROOT ?= $(WORKDIR)/artifacts

ARTIFACT_ROOT ?= $(CONTAINER_ARTIFACT_ROOT)
TRAIN_ARTIFACT_DIR ?= $(ARTIFACT_ROOT)/train
INFER_ARTIFACT_DIR ?= $(ARTIFACT_ROOT)/infer
FINAL_EVAL_ARTIFACT_DIR ?= $(HPO_ARTIFACT_DIR)/best_hpo_eval
HPO_ARTIFACT_DIR ?= $(ARTIFACT_ROOT)/hpo
BASELINE_ARTIFACT_DIR ?= $(ARTIFACT_ROOT)/baseline

DATASET_PATH ?= $(CONTAINER_DATASETS_DIR)/hspot
ANNO_NAME ?= anno.json
MOT17 ?= false
DET_OPTIONS ?=

TRAIN_SPLIT ?= train
TRAIN_EXTRA_OPTS ?=

TEST_SET ?= val
EVAL_METRIC ?= both
HPO_EVAL_METRIC ?= both
MODEL_FILE ?=
BASELINE_RUN_NAME ?= hspot_baseline_val
FINE_TUNE_RUN_NAME ?= hspot_finetune
BASELINE_MODEL_FILE ?= $(CONTAINER_WEIGHTS_DIR)/DLA-34-FPN_EMM_crowdhuman_mot17.pth
FINE_TUNE_MODEL_FILE ?= $(TRAIN_ARTIFACT_DIR)/DLA-34-FPN_box_EMM_MOT_HSPOT/model_final.pth
TEST_EXTRA_OPTS ?=
VIS_SEQUENCE_IDS ?=
VIS_PREDICTIONS_DIR ?= $(BASELINE_ARTIFACT_DIR)
VIS_OUTPUT_DIR ?= $(BASELINE_ARTIFACT_DIR)/visualizations
VIS_SPLIT ?=
VIS_FRAME_START ?= 0
VIS_FRAME_END ?= -1
VIS_MAX_FRAMES ?= 0
VIS_WITH_GT ?= true
VIS_ONLY_WITH_BOXES ?= false

BASE_MODEL_FILE ?= $(CONTAINER_WEIGHTS_DIR)/DLA-34-FPN_EMM_crowdhuman_mot17.pth
BEST_TRIAL_FILE ?= $(HPO_ARTIFACT_DIR)/best_trial.json
STUDY_NAME ?= hspot_hpo
HPO_RUN_NAME_PREFIX ?= hspot_hota
HPO_TRAIN_SPLIT ?= train
HPO_VAL_SPLIT ?= val
N_TRIALS ?= 40
MAX_ITER ?= 2000
PRUNE_CHECKPOINTS ?= auto
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
	verify-docker-cpu smoke-cpu smoke-mlflow ingest baseline train test tune trackeval-add trackeval-update \
	ensure-storage-dirs print-storage-config verify-docker-root verify-containerd-root verify-storage-root \
	visualize visualize-baseline visualize-finetuning visualize-final-eval visualize-hpo-best

help: ## Show available targets.
	@grep -E '^[a-zA-Z0-9_.-]+:.*## ' Makefile | awk 'BEGIN {FS=":.*## "}; {printf "%-24s %s\n", $$1, $$2}'

ensure-storage-dirs: ## Ensure host-side storage directories exist.
	mkdir -p "$(HOST_DATASETS_DIR)" "$(HOST_WEIGHTS_DIR)" "$(HOST_ARTIFACT_ROOT)"

print-storage-config: ## Print host/container storage mappings used by Docker runs.
	@echo "HOST_STORAGE_ROOT=$(HOST_STORAGE_ROOT)"
	@echo "HOST_DATASETS_DIR=$(HOST_DATASETS_DIR) -> $(CONTAINER_DATASETS_DIR)"
	@echo "HOST_WEIGHTS_DIR=$(HOST_WEIGHTS_DIR) -> $(CONTAINER_WEIGHTS_DIR)"
	@echo "HOST_ARTIFACT_ROOT=$(HOST_ARTIFACT_ROOT) -> $(CONTAINER_ARTIFACT_ROOT)"

verify-docker-root: ## Verify Docker Root Dir is not default /var/lib/docker.
	@docker_root="$$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"; \
	if [ -z "$$docker_root" ]; then \
		echo "Could not determine Docker Root Dir. Is Docker running?"; \
		exit 1; \
	fi; \
	echo "Docker Root Dir=$$docker_root"; \
	if [ "$$docker_root" = "/var/lib/docker" ]; then \
		echo "Docker is still using default /var/lib/docker on the root disk."; \
		echo "Set daemon.json data-root to a SURF-volume path and restart docker."; \
		exit 1; \
	fi; \
	echo "Docker root dir is non-default (good)."

verify-containerd-root: ## Verify containerd root is not default /var/lib/containerd.
	@config_file="/etc/containerd/config.toml"; \
	if [ ! -f "$$config_file" ]; then \
		echo "Missing $$config_file. Configure containerd root on a SURF-volume path."; \
		exit 1; \
	fi; \
	containerd_root="$$(awk -F= '/^[[:space:]]*root[[:space:]]*=/{gsub(/^[[:space:]]+|[[:space:]]+$$/,"",$$2); gsub(/"/,"",$$2); print $$2; exit}' "$$config_file")"; \
	containerd_state="$$(awk -F= '/^[[:space:]]*state[[:space:]]*=/{gsub(/^[[:space:]]+|[[:space:]]+$$/,"",$$2); gsub(/"/,"",$$2); print $$2; exit}' "$$config_file")"; \
	if [ -z "$$containerd_root" ]; then \
		echo "Could not determine containerd root from $$config_file."; \
		exit 1; \
	fi; \
	echo "containerd root=$$containerd_root"; \
	if [ -n "$$containerd_state" ]; then \
		echo "containerd state=$$containerd_state"; \
	fi; \
	if [ "$$containerd_root" = "/var/lib/containerd" ]; then \
		echo "containerd is still using default /var/lib/containerd on the root disk."; \
		echo "Set /etc/containerd/config.toml root to a SURF-volume path and restart containerd."; \
		exit 1; \
	fi; \
	echo "containerd root is non-default (good)."

verify-storage-root: verify-docker-root verify-containerd-root ## Verify Docker and containerd both use non-default storage roots.
	@echo "Docker and containerd storage roots are non-default (good)."

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
	docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi

verify-docker-cpu: GPU=none
verify-docker-cpu: ensure-storage-dirs ## Validate container startup without GPU access.
	docker --version
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python -c 'import platform; print(platform.platform())'"

docker-build: ## Build the project image (dependencies installed with uv in a Python 3.8 venv).
	docker build -t $(IMAGE) -f Dockerfile .

docker-shell: ensure-storage-dirs ## Open an interactive shell in the project container.
	$(DOCKER_RUN_BASE) -it $(IMAGE) bash

smoke-cpu: GPU=none
smoke-cpu: DEVICE=cpu
smoke-cpu: ensure-storage-dirs ## Run a CPU-only smoke test (no dataset required).
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python -c 'import torch; print(\"torch\", torch.__version__); print(\"cuda_available\", torch.cuda.is_available())'"
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python tools/train_net.py --help > /tmp/train_help.txt && python tools/test_net.py --help > /tmp/test_help.txt && python tools/tune_optuna.py --help > /tmp/tune_help.txt && echo smoke_ok"

smoke-mlflow: GPU=none
smoke-mlflow: ensure-storage-dirs ## Validate MLflow connectivity and artifact logging against MLFLOW_TRACKING_URI.
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python tools/mlflow_smoke_test.py"

ingest: ensure-storage-dirs ## Ingest custom MOT dataset from DATASET_PATH.
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python siammot/data/ingestion/ingest_mot.py --dataset_path $(DATASET_PATH) --anno_name $(ANNO_NAME) --mot17 $(MOT17) --det-options \"$(DET_OPTIONS)\""

baseline: ensure-storage-dirs ## Evaluate the default pre-trained checkpoint before HSPOT training.
	mkdir -p "$(HOST_BASELINE_ARTIFACT_DIR)"
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python tools/test_net.py --config-file $(CONFIG_FILE) --output-dir $(BASELINE_ARTIFACT_DIR) --model-file $(BASELINE_MODEL_FILE) --test-dataset $(DATASET_KEY) --set $(TEST_SET) --extra-mlflow-tags stage=baseline_eval workflow=baseline_hspot dataset_key=$(DATASET_KEY) eval_split=$(TEST_SET) --opts DATASETS.ROOT_DIR $(CONTAINER_DATASETS_DIR) INFERENCE.EVAL_METRIC $(EVAL_METRIC) MLFLOW.INFERENCE_RUN_NAME $(BASELINE_RUN_NAME) $(COMMON_DEVICE_OPTS) $(TEST_EXTRA_OPTS)"

train: ensure-storage-dirs ## Train/fine-tune with tools/train_net.py.
	mkdir -p "$(HOST_TRAIN_ARTIFACT_DIR)"
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python tools/train_net.py --config-file $(CONFIG_FILE) --train-dir $(TRAIN_ARTIFACT_DIR) --extra-mlflow-tags stage=fine_tune workflow=fine_tune_hspot dataset_key=$(DATASET_KEY) train_split=$(TRAIN_SPLIT) --opts DATASETS.ROOT_DIR $(CONTAINER_DATASETS_DIR) DATASETS.TRAIN \"('$(DATASET_KEY)',)\" DATASETS.TRAIN_SET $(TRAIN_SPLIT) MLFLOW.TRAIN_RUN_NAME $(FINE_TUNE_RUN_NAME) $(COMMON_DEVICE_OPTS) $(TRAIN_EXTRA_OPTS)"

test: ensure-storage-dirs ## Evaluate with tools/test_net.py (set MODEL_FILE and TEST_SET=val|test).
	@if [ -z "$(MODEL_FILE)" ]; then echo "MODEL_FILE is required, e.g. make test MODEL_FILE=/workspace/artifacts/train/.../model_final.pth"; exit 1; fi
	mkdir -p "$(HOST_INFER_ARTIFACT_DIR)"
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python tools/test_net.py --config-file $(CONFIG_FILE) --output-dir $(INFER_ARTIFACT_DIR) --model-file $(MODEL_FILE) --test-dataset $(DATASET_KEY) --set $(TEST_SET) --opts DATASETS.ROOT_DIR $(CONTAINER_DATASETS_DIR) INFERENCE.EVAL_METRIC $(EVAL_METRIC) $(COMMON_DEVICE_OPTS) $(TEST_EXTRA_OPTS)"

visualize: ensure-storage-dirs ## Render saved prediction boxes on top of sequence frames from any evaluation output directory.
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python tools/visualize_predictions.py $(if $(strip $(VIS_SEQUENCE_IDS)),--sequence-id $(VIS_SEQUENCE_IDS),) --predictions-dir $(VIS_PREDICTIONS_DIR) --dataset-root $(CONTAINER_DATASETS_DIR)/hspot --output-dir $(VIS_OUTPUT_DIR) --frame-start $(VIS_FRAME_START) --frame-end $(VIS_FRAME_END) --max-frames $(VIS_MAX_FRAMES) $(if $(VIS_SPLIT),--split $(VIS_SPLIT),) $(if $(filter true,$(VIS_WITH_GT)),--with-gt,) $(if $(filter true,$(VIS_ONLY_WITH_BOXES)),--only-with-boxes,)"

visualize-baseline: ## Render saved baseline prediction boxes on top of sequence frames.
	$(MAKE) visualize VIS_PREDICTIONS_DIR="$(BASELINE_ARTIFACT_DIR)" VIS_OUTPUT_DIR="$(BASELINE_ARTIFACT_DIR)/visualizations" VIS_SEQUENCE_IDS="$(VIS_SEQUENCE_IDS)" VIS_SPLIT="$(if $(strip $(VIS_SPLIT)),$(VIS_SPLIT),val)" VIS_FRAME_START="$(VIS_FRAME_START)" VIS_FRAME_END="$(VIS_FRAME_END)" VIS_MAX_FRAMES="$(VIS_MAX_FRAMES)" VIS_WITH_GT="$(VIS_WITH_GT)" VIS_ONLY_WITH_BOXES="$(VIS_ONLY_WITH_BOXES)"

visualize-finetuning: ## Render prediction boxes from the best fine-tuning validation pass.
	@pred_dir="$$(python3 -c 'exec("""import json\nimport os\n\ncontainer_artifact_root = os.path.abspath(\"$(CONTAINER_ARTIFACT_ROOT)\")\nhost_artifact_root = os.path.abspath(\"$(HOST_ARTIFACT_ROOT)\")\nmodel_file = os.path.abspath(\"$(FINE_TUNE_MODEL_FILE)\")\nif model_file.startswith(container_artifact_root + os.sep):\n    model_file = os.path.join(host_artifact_root, os.path.relpath(model_file, container_artifact_root))\ntrain_dir = os.path.dirname(model_file)\nrun_info_path = os.path.join(train_dir, \"run_info.json\")\nif not os.path.isfile(run_info_path):\n    raise FileNotFoundError(\"Missing fine-tuning run info: {}\".format(run_info_path))\nwith open(run_info_path, \"r\", encoding=\"utf-8\") as f:\n    run_info = json.load(f)\nvalidation = run_info.get(\"validation\", {}) or {}\nbest_epoch = validation.get(\"best_epoch\")\nbest_iteration = validation.get(\"best_iteration\")\nif best_epoch is None or best_iteration is None:\n    raise KeyError(\"Fine-tuning run info does not contain validation.best_epoch/best_iteration: {}\".format(run_info_path))\npred_dir = os.path.join(train_dir, \"validation\", \"epoch_{:04d}_iter_{:07d}\".format(int(best_epoch), int(best_iteration)))\nprint(os.path.join(container_artifact_root, os.path.relpath(pred_dir, host_artifact_root)))""")')" && \
	$(MAKE) visualize VIS_PREDICTIONS_DIR="$$pred_dir" VIS_OUTPUT_DIR="$$pred_dir/visualizations" VIS_SEQUENCE_IDS="$(VIS_SEQUENCE_IDS)" VIS_SPLIT="$(if $(strip $(VIS_SPLIT)),$(VIS_SPLIT),val)" VIS_FRAME_START="$(VIS_FRAME_START)" VIS_FRAME_END="$(VIS_FRAME_END)" VIS_MAX_FRAMES="$(VIS_MAX_FRAMES)" VIS_WITH_GT="$(VIS_WITH_GT)" VIS_ONLY_WITH_BOXES="$(VIS_ONLY_WITH_BOXES)"

visualize-final-eval: ## Render prediction boxes from the best HPO trial's final test evaluation.
	$(MAKE) visualize VIS_PREDICTIONS_DIR="$(FINAL_EVAL_ARTIFACT_DIR)" VIS_OUTPUT_DIR="$(FINAL_EVAL_ARTIFACT_DIR)/visualizations" VIS_SEQUENCE_IDS="$(VIS_SEQUENCE_IDS)" VIS_SPLIT="$(if $(strip $(VIS_SPLIT)),$(VIS_SPLIT),test)" VIS_FRAME_START="$(VIS_FRAME_START)" VIS_FRAME_END="$(VIS_FRAME_END)" VIS_MAX_FRAMES="$(VIS_MAX_FRAMES)" VIS_WITH_GT="$(VIS_WITH_GT)" VIS_ONLY_WITH_BOXES="$(VIS_ONLY_WITH_BOXES)"

visualize-hpo-best: ## Render prediction boxes from the best HPO trial's best validation pass.
	@pred_dir="$$(python3 -c 'exec("""import glob\nimport json\nimport os\n\ncontainer_artifact_root = os.path.abspath(\"$(CONTAINER_ARTIFACT_ROOT)\")\nhost_artifact_root = os.path.abspath(\"$(HOST_ARTIFACT_ROOT)\")\nbest_trial_file = os.path.abspath(\"$(BEST_TRIAL_FILE)\")\nif best_trial_file.startswith(container_artifact_root + os.sep):\n    best_trial_file = os.path.join(host_artifact_root, os.path.relpath(best_trial_file, container_artifact_root))\nif not os.path.isfile(best_trial_file):\n    raise FileNotFoundError(\"Missing best trial file: {}\".format(best_trial_file))\nwith open(best_trial_file, \"r\", encoding=\"utf-8\") as f:\n    best_trial = json.load(f)\ntrial_number = best_trial.get(\"number\")\nuser_attrs = best_trial.get(\"user_attrs\", {}) or {}\nbest_checkpoint = str(user_attrs.get(\"final_checkpoint\", \"\")).strip()\nif best_checkpoint.startswith(container_artifact_root + os.sep):\n    best_checkpoint = os.path.join(host_artifact_root, os.path.relpath(best_checkpoint, container_artifact_root))\nif trial_number is None or not best_checkpoint:\n    raise KeyError(\"best_trial.json is missing number or user_attrs.final_checkpoint: {}\".format(best_trial_file))\ntrial_dir = os.path.join(os.path.dirname(best_trial_file), \"trials\", \"trial_{:04d}\".format(int(trial_number)))\nmatches = sorted(glob.glob(os.path.join(trial_dir, \"stage_summary_*.json\")))\nif not matches:\n    raise FileNotFoundError(\"No HPO stage summaries found under {}\".format(trial_dir))\nresolved = \"\"\nfor stage_summary_path in matches:\n    with open(stage_summary_path, \"r\", encoding=\"utf-8\") as f:\n        stage_summary = json.load(f)\n    train_run_info = stage_summary.get(\"train_run_info\", {}) or {}\n    validation = train_run_info.get(\"validation\", {}) or {}\n    candidate_best_checkpoint = str(validation.get(\"best_checkpoint\", \"\")).strip()\n    if candidate_best_checkpoint.startswith(container_artifact_root + os.sep):\n        candidate_best_checkpoint = os.path.join(host_artifact_root, os.path.relpath(candidate_best_checkpoint, container_artifact_root))\n    if candidate_best_checkpoint != best_checkpoint:\n        continue\n    train_dir = str(train_run_info.get(\"train_dir\", \"\")).strip()\n    if train_dir.startswith(container_artifact_root + os.sep):\n        train_dir = os.path.join(host_artifact_root, os.path.relpath(train_dir, container_artifact_root))\n    best_epoch = validation.get(\"best_epoch\")\n    best_iteration = validation.get(\"best_iteration\")\n    if not train_dir or best_epoch is None or best_iteration is None:\n        raise KeyError(\"Incomplete HPO validation metadata in {}\".format(stage_summary_path))\n    resolved = os.path.join(train_dir, \"validation\", \"epoch_{:04d}_iter_{:07d}\".format(int(best_epoch), int(best_iteration)))\n    break\nif not resolved:\n    raise FileNotFoundError(\"Could not map best HPO checkpoint to a validation output directory under {}\".format(trial_dir))\nprint(os.path.join(container_artifact_root, os.path.relpath(resolved, host_artifact_root)))""")')" && \
	$(MAKE) visualize VIS_PREDICTIONS_DIR="$$pred_dir" VIS_OUTPUT_DIR="$$pred_dir/visualizations" VIS_SEQUENCE_IDS="$(VIS_SEQUENCE_IDS)" VIS_SPLIT="$(if $(strip $(VIS_SPLIT)),$(VIS_SPLIT),val)" VIS_FRAME_START="$(VIS_FRAME_START)" VIS_FRAME_END="$(VIS_FRAME_END)" VIS_MAX_FRAMES="$(VIS_MAX_FRAMES)" VIS_WITH_GT="$(VIS_WITH_GT)" VIS_ONLY_WITH_BOXES="$(VIS_ONLY_WITH_BOXES)"

tune: ensure-storage-dirs ## Run Optuna HPO (set BASE_MODEL_FILE).
	@if [ -z "$(BASE_MODEL_FILE)" ]; then echo "BASE_MODEL_FILE is required, e.g. make tune BASE_MODEL_FILE=/workspace/weights/DLA-34-FPN_EMM_crowdhuman_mot17.pth"; exit 1; fi
	mkdir -p "$(HOST_HPO_ARTIFACT_DIR)"
	$(DOCKER_RUN_BASE) $(IMAGE) bash -lc "python tools/tune_optuna.py --project-root . --config-file $(CONFIG_FILE) --base-model-file $(BASE_MODEL_FILE) --output-dir $(HPO_ARTIFACT_DIR) --study-name $(STUDY_NAME) --run-name-prefix $(HPO_RUN_NAME_PREFIX) --dataset-key $(DATASET_KEY) --datasets-root $(CONTAINER_DATASETS_DIR) --train-split $(HPO_TRAIN_SPLIT) --val-split $(HPO_VAL_SPLIT) --eval-metric $(HPO_EVAL_METRIC) --n-trials $(N_TRIALS) --max-iter $(MAX_ITER) --prune-checkpoints $(PRUNE_CHECKPOINTS) --base-opts $(COMMON_DEVICE_OPTS) $(TUNE_MLFLOW_FLAG) $(TUNE_EXTRA_OPTS)"

trackeval-add: ## Vendor TrackEval into third_party/TrackEval.
	git subtree add --prefix third_party/TrackEval https://github.com/JonathonLuiten/TrackEval.git master --squash

trackeval-update: ## Update vendored TrackEval.
	git subtree pull --prefix third_party/TrackEval https://github.com/JonathonLuiten/TrackEval.git master --squash
