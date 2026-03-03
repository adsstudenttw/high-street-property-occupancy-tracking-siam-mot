import argparse
import json
import os
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a final SiamMOT model using the best hyperparameters from Optuna."
    )
    parser.add_argument("--project-root", default=".", type=str)
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument("--best-trial-file", required=True, type=str)
    parser.add_argument("--base-model-file", required=True, type=str)
    parser.add_argument("--train-dir", required=True, type=str)
    parser.add_argument("--dataset-key", default="MOT_HSPOT", type=str)
    parser.add_argument("--datasets-root", default="datasets", type=str)
    parser.add_argument("--train-split", default="train", type=str)
    parser.add_argument("--model-suffix", default="best_hpo", type=str)
    parser.add_argument(
        "--base-opts",
        nargs="*",
        default=[],
        help="Additional fixed cfg overrides passed through to train_net.py",
    )
    args = parser.parse_args()
    if len(args.base_opts) % 2 != 0:
        raise ValueError("--base-opts must contain an even number of KEY VALUE tokens")
    return args


def stringify_cfg_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return "{:.8g}".format(value)
    return str(value)


def cfg_dict_to_opts(cfg_values: Dict[str, Any]) -> List[str]:
    opts: List[str] = []
    for key in sorted(cfg_values.keys()):
        opts.extend([key, stringify_cfg_value(cfg_values[key])])
    return opts


def load_best_trial(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Invalid best trial payload: expected a JSON object")
    return payload


def main() -> None:
    args = parse_args()
    project_root = os.path.abspath(args.project_root)
    config_file = os.path.abspath(os.path.join(project_root, args.config_file))
    best_trial_file = os.path.abspath(os.path.join(project_root, args.best_trial_file))
    base_model_file = os.path.abspath(os.path.join(project_root, args.base_model_file))
    train_dir = os.path.abspath(os.path.join(project_root, args.train_dir))
    train_script = os.path.join(project_root, "tools", "train_net.py")

    if not os.path.isfile(config_file):
        raise FileNotFoundError("Missing config file: {}".format(config_file))
    if not os.path.isfile(best_trial_file):
        raise FileNotFoundError("Missing best trial file: {}".format(best_trial_file))
    if not os.path.isfile(base_model_file):
        raise FileNotFoundError("Missing base model file: {}".format(base_model_file))
    if not os.path.isfile(train_script):
        raise FileNotFoundError("Missing training script: {}".format(train_script))

    best_trial = load_best_trial(best_trial_file)
    best_trial_number = best_trial.get("number")
    user_attrs = best_trial.get("user_attrs", {})
    if not isinstance(user_attrs, dict):
        raise ValueError("Invalid best trial payload: user_attrs must be an object")
    sampled_cfg = user_attrs.get("sampled_cfg", {})
    if not isinstance(sampled_cfg, dict) or not sampled_cfg:
        raise ValueError(
            "Best trial file does not contain user_attrs.sampled_cfg: {}".format(best_trial_file)
        )

    fixed_cfg = {
        "DATASETS.ROOT_DIR": args.datasets_root,
        "DATASETS.TRAIN": "('{}',)".format(args.dataset_key),
        "DATASETS.TRAIN_SET": args.train_split,
        "MODEL.WEIGHT": base_model_file,
    }
    train_opts = list(args.base_opts)
    train_opts.extend(cfg_dict_to_opts(fixed_cfg))
    train_opts.extend(cfg_dict_to_opts(sampled_cfg))

    extra_mlflow_tags = [
        "stage=final_train_best_hpo",
        "hpo_best_trial_file={}".format(best_trial_file),
    ]
    if best_trial_number is not None:
        extra_mlflow_tags.append("hpo_best_trial_number={}".format(best_trial_number))

    cmd: Sequence[str] = [
        sys.executable,
        train_script,
        "--config-file",
        config_file,
        "--train-dir",
        train_dir,
        "--model-suffix",
        args.model_suffix,
        "--extra-mlflow-tags",
        *extra_mlflow_tags,
        "--opts",
        *train_opts,
    ]

    os.makedirs(train_dir, exist_ok=True)
    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "{}{}{}".format(project_root, os.pathsep, py_path) if py_path else project_root

    printable_cmd = " ".join(shlex.quote(part) for part in cmd)
    print("[run] {}".format(printable_cmd))
    subprocess.run(cmd, cwd=project_root, env=env, check=True)


if __name__ == "__main__":
    main()
