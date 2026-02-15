


## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.


## Try SiamMOT demo
For demo  purposes,  we provide two tracking models -- tracking person (visible part) or jointly tracking person and vehicles (bus, car, truck, motorcycle, etc).
The person tracking model is trained on COCO-17 and CrowdHuman, while the latter model is trained on COCO-17 and VOC12.
Currently, both models used in demos use EMM as its motion model, which performs best among different alternatives.

In order to run the demo, use the following command:
~~~
python3 demos/demo.py --demo-video  PATH_TO_DEMO_VIDE --track-class person --dump-video True
~~~
You can choose `person` or  `person_vehicel` for `track-class` such that person tracking or person/vehicle tracking model is used accordingly.

The model would be automatically downloaded to `demos/models`,
and the visualization of tracking outputs is automatically saved to `demos/demo_vis`

![](readme/demo_volleyball.gif)

![](readme/demo_person_vehicle.gif) 

We also provide several pre-trained models in [model_zoos.md](readme/model_zoo.md) that can be used for demo. 

## Dataset Evaluation and Training
After [installation](readme/INSTALL.md), follow the instructions in [DATA.md](readme/DATA.md) to setup the datasets.
As a sanity check, the models presented in [model_zoos.md](readme/model_zoo.md) can be used to for benchmark testing. 

Use the following command to train a model on an 8-GPU machine:
Before running training / inference, setup the [configuration file](configs) properly
~~~
python3 -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file configs/dla/DLA_34_FPN.yaml --train-dir PATH_TO_TRAIN_DIR --model-suffix MODEL_SUFFIX 
~~~

Use the following command to test a model on a single-GPU machine:
~~~
python3 tools/test_net.py --config-file configs/dla/DLA_34_FPN.yaml --output-dir PATH_TO_OUTPUT_DIR --model-file PATH_TO_MODEL_FILE --test-dataset DATASET_KEY --set val
~~~

### MLflow Experiment Tracking
MLflow support is built into both `tools/train_net.py` and `tools/test_net.py`.

1. Set your tracking server URI if needed:
~~~
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
~~~
2. Enable MLflow in your config:
~~~yaml
MLFLOW:
  ENABLED: True
  TRACKING_URI: "http://127.0.0.1:5000"  # optional if env var is set
  EXPERIMENT_NAME: "siammot"
  TRAIN_RUN_NAME: ""
  INFERENCE_RUN_NAME: ""
  LOG_EVERY_N_STEPS: 20
  LOG_MODEL_CHECKPOINTS: True
  LOG_CONFIG_ARTIFACT: True
  LOG_INFERENCE_OUTPUTS: False
~~~
3. (Optional) Link inference runs to a training run:
~~~
python3 tools/test_net.py ... --parent-run-id TRAIN_RUN_ID
~~~

### Hyperparameter Tuning with Optuna
The project includes `tools/tune_optuna.py` to optimize post-processing / inference tracking hyperparameters and, optionally,
training hyperparameters in per-trial fine-tuning runs.

Inference-only tuning (recommended after you already trained a custom model):
~~~bash
python3 tools/tune_optuna.py \
  --project-root . \
  --config-file configs/dla/DLA_34_FPN_EMM_MOT17.yaml \
  --mode inference \
  --model-file PATH_TO_MODEL_FILE \
  --output-dir PATH_TO_TUNING_OUTPUT \
  --study-name my_study \
  --test-dataset MY_DATASET_KEY \
  --dataset-split val \
  --metric-name infer/mot/idf1 \
  --n-trials 30
~~~

Per-trial fine-tuning + tuning:
~~~bash
python3 tools/tune_optuna.py \
  --project-root . \
  --config-file configs/dla/DLA_34_FPN_EMM_MOT17.yaml \
  --mode finetune \
  --train-dir PATH_TO_TRAIN_ROOT \
  --output-dir PATH_TO_TUNING_OUTPUT \
  --study-name my_study \
  --test-dataset MY_DATASET_KEY \
  --dataset-split val \
  --metric-name infer/mot/idf1 \
  --n-trials 20 \
  --max-iter 5000
~~~

Both `tools/train_net.py` and `tools/test_net.py` now support config overrides through:
~~~bash
--opts KEY1 VALUE1 KEY2 VALUE2 ...
~~~
This is used by Optuna to evaluate each trial configuration.

**Note:** If you get an error `ModuleNotFoundError: No module named 'siammot'` when running in the git root then make
sure your PYTHONPATH includes the current directory, which you can add by running: `export PYTHONPATH=.:$PYTHONPATH`
or you can explicitly add the project to the path by replacing the '.' in the export command with the absolute path to
the git root.
