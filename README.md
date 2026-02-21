## Installation
Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## End-to-End HSPOT Workflow
This is a simple step-by-step flow for custom HSPOT data:
1. Set up dataset and ingest annotations.
2. Start MLflow (optional but recommended).
3. Set up TrackEval (needed for HOTA).
4. Fine-tune with `train_net.py`.
5. Test with `test_net.py`.
6. Run hyperparameter tuning with `tune_optuna.py`.

### 1. Prepare Custom HSPOT Dataset
Expected dataset folder layout:
1. `datasets/hspot/train`
2. `datasets/hspot/val`
3. `datasets/hspot/test` (optional but needed for final test evaluation)

Ingest the MOT-format annotations:
~~~bash
python3 siammot/data/ingestion/ingest_mot.py \
  --dataset_path datasets/hspot \
  --anno_name anno.json \
  --mot17 true \
  --det-options ""
~~~

Notes:
1. `--det-options ""` ingests all sequence folders and does not require MOT17 suffixes (`DPM/FRCNN/SDP`).
2. Use `--mot17 true` when GT rows contain MOT17 class/visibility columns; otherwise use `--mot17 false`.
3. This repository uses dataset key `MOT_HSPOT`.

### 2. Configure MLflow (Optional)
MLflow is integrated in both `tools/train_net.py` and `tools/test_net.py`.

Set tracking URI:
~~~bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
~~~

Start a local MLflow server (example):
~~~bash
mlflow server --host 127.0.0.1 --port 5000
~~~

If needed, you can disable logging per command with:
~~~bash
--opts MLFLOW.ENABLED False
~~~

### 3. Set Up TrackEval (Required for HOTA)
HOTA evaluation requires vendored TrackEval under `third_party/TrackEval`.

Add TrackEval to this repository:
~~~bash
git subtree add --prefix third_party/TrackEval https://github.com/JonathonLuiten/TrackEval.git master --squash
~~~

Update later:
~~~bash
git subtree pull --prefix third_party/TrackEval https://github.com/JonathonLuiten/TrackEval.git master --squash
~~~

### 4. Fine-tune on HSPOT with `train_net.py`
Single-GPU fine-tuning on the `train` split:
~~~bash
python3 tools/train_net.py \
  --config-file configs/dla/DLA_34_FPN_EMM_HSPOT.yaml \
  --train-dir PATH_TO_TRAIN_DIR \
  --opts \
    DATASETS.ROOT_DIR datasets \
    DATASETS.TRAIN "('MOT_HSPOT',)" \
    DATASETS.TRAIN_SET train
~~~

`train_net.py` does training only. It does not run validation/test evaluation.

### 5. Evaluate with `test_net.py`
Run validation evaluation:
~~~bash
python3 tools/test_net.py \
  --config-file configs/dla/DLA_34_FPN_EMM_HSPOT.yaml \
  --output-dir PATH_TO_OUTPUT_DIR \
  --model-file PATH_TO_MODEL_FILE \
  --test-dataset MOT_HSPOT \
  --set val \
  --opts DATASETS.ROOT_DIR datasets INFERENCE.EVAL_METRIC both
~~~

Run final test evaluation:
~~~bash
python3 tools/test_net.py \
  --config-file configs/dla/DLA_34_FPN_EMM_HSPOT.yaml \
  --output-dir PATH_TO_OUTPUT_DIR \
  --model-file PATH_TO_MODEL_FILE \
  --test-dataset MOT_HSPOT \
  --set test \
  --opts DATASETS.ROOT_DIR datasets INFERENCE.EVAL_METRIC both
~~~

`INFERENCE.EVAL_METRIC` options:
1. `clear`
2. `hota`
3. `both`

### 6. Hyperparameter Tuning (1 GPU) with Optuna
`tools/tune_optuna.py` uses Bayesian optimization (`TPESampler`) with pruning (`MedianPruner`).
Per trial, it fine-tunes with `train_net.py` and evaluates on `val` with `test_net.py`.
After the study finishes, it runs one final `test` evaluation with the best trial checkpoint.

Objective metric is selected from `--eval-metric`:
1. `clear` optimizes `infer/mot/idf1`
2. `hota` optimizes `infer/mot/hota`
3. `both` optimizes `infer/mot/hota`

Example:
~~~bash
python3 tools/tune_optuna.py \
  --project-root . \
  --config-file configs/dla/DLA_34_FPN_EMM_HSPOT.yaml \
  --base-model-file PATH_TO_BASE_CHECKPOINT \
  --output-dir PATH_TO_HPO_OUTPUT \
  --study-name hspot_val_hpo \
  --dataset-key MOT_HSPOT \
  --train-split val \
  --val-split val \
  --test-split test \
  --eval-metric hota \
  --n-trials 20 \
  --max-iter 6000 \
  --prune-checkpoints 1000,3000
~~~

Important outputs:
1. `PATH_TO_HPO_OUTPUT/best_trial.json`
2. `PATH_TO_HPO_OUTPUT/study_trials.json`
3. `PATH_TO_HPO_OUTPUT/final_test_eval/final_test_metrics.json`

## Notes
Both `tools/train_net.py` and `tools/test_net.py` support config overrides:
~~~bash
--opts KEY1 VALUE1 KEY2 VALUE2 ...
~~~

If you get `ModuleNotFoundError: No module named 'siammot'`, set:
~~~bash
export PYTHONPATH=.:$PYTHONPATH
~~~
