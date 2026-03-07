import argparse
import json
import os
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the best HPO trial checkpoint on a target split."
    )
    parser.add_argument("--project-root", default=".", type=str)
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument("--best-trial-file", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--dataset-key", default="MOT_HSPOT", type=str)
    parser.add_argument("--datasets-root", default="datasets", type=str)
    parser.add_argument("--test-split", default="test", type=str)
    parser.add_argument("--eval-metric", default="both", choices=["clear", "hota", "both"])
    parser.add_argument(
        "--model-file",
        default="",
        type=str,
        help="Optional explicit checkpoint. If omitted, resolves from best_trial.json user_attrs.final_checkpoint.",
    )
    parser.add_argument(
        "--base-opts",
        nargs="*",
        default=[],
        help="Additional fixed cfg overrides passed through to test_net.py",
    )
    args = parser.parse_args()
    if len(args.base_opts) % 2 != 0:
        raise ValueError("--base-opts must contain an even number of KEY VALUE tokens")
    return args


def load_best_trial(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Invalid best trial payload: expected a JSON object")
    return payload


def resolve_model_file(args: argparse.Namespace, best_trial: Dict[str, Any], project_root: str) -> str:
    explicit_model = str(args.model_file).strip()
    if explicit_model:
        resolved = os.path.abspath(os.path.join(project_root, explicit_model))
        if not os.path.isfile(resolved):
            raise FileNotFoundError("Missing --model-file: {}".format(resolved))
        return resolved

    user_attrs = best_trial.get("user_attrs", {})
    if not isinstance(user_attrs, dict):
        raise ValueError("Invalid best trial payload: user_attrs must be an object")

    checkpoint = str(user_attrs.get("final_checkpoint", "")).strip()
    if not checkpoint:
        raise ValueError(
            "Best trial file does not contain user_attrs.final_checkpoint: {}".format(
                args.best_trial_file
            )
        )
    resolved = os.path.abspath(os.path.join(project_root, checkpoint))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            "Best trial checkpoint from user_attrs.final_checkpoint does not exist: {}".format(
                resolved
            )
        )
    return resolved


def main() -> None:
    args = parse_args()
    project_root = os.path.abspath(args.project_root)
    config_file = os.path.abspath(os.path.join(project_root, args.config_file))
    best_trial_file = os.path.abspath(os.path.join(project_root, args.best_trial_file))
    output_dir = os.path.abspath(os.path.join(project_root, args.output_dir))
    test_script = os.path.join(project_root, "tools", "test_net.py")

    if not os.path.isfile(config_file):
        raise FileNotFoundError("Missing config file: {}".format(config_file))
    if not os.path.isfile(best_trial_file):
        raise FileNotFoundError("Missing best trial file: {}".format(best_trial_file))
    if not os.path.isfile(test_script):
        raise FileNotFoundError("Missing test script: {}".format(test_script))

    best_trial = load_best_trial(best_trial_file)
    model_file = resolve_model_file(args, best_trial, project_root)

    mlflow_tags: List[str] = [
        "stage=final_eval_best_hpo",
        "workflow=best_hpo_final_eval",
        "dataset_key={}".format(args.dataset_key),
        "eval_split={}".format(args.test_split),
        "model_origin=hpo_best_trial",
        "hpo_best_trial_file={}".format(best_trial_file),
    ]
    best_trial_number = best_trial.get("number")
    if best_trial_number is not None:
        mlflow_tags.append("hpo_best_trial_number={}".format(best_trial_number))

    test_opts: List[str] = list(args.base_opts)
    test_opts.extend(
        [
            "DATASETS.ROOT_DIR",
            args.datasets_root,
            "INFERENCE.EVAL_METRIC",
            args.eval_metric,
        ]
    )

    cmd: Sequence[str] = [
        sys.executable,
        test_script,
        "--config-file",
        config_file,
        "--output-dir",
        output_dir,
        "--model-file",
        model_file,
        "--test-dataset",
        args.dataset_key,
        "--set",
        args.test_split,
        "--extra-mlflow-tags",
        *mlflow_tags,
        "--opts",
        *test_opts,
    ]

    os.makedirs(output_dir, exist_ok=True)
    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "{}{}{}".format(project_root, os.pathsep, py_path) if py_path else project_root

    printable_cmd = " ".join(shlex.quote(part) for part in cmd)
    print("[run] {}".format(printable_cmd))
    subprocess.run(cmd, cwd=project_root, env=env, check=True)


if __name__ == "__main__":
    main()
