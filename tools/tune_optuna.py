import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List

import optuna


def _run_command(cmd: List[str], cwd: str):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def _build_cfg_opts(overrides: Dict[str, object]) -> List[str]:
    opts = []
    for key, value in overrides.items():
        opts.extend([key, str(value)])
    return opts


def _load_metrics(metrics_path: str) -> Dict[str, float]:
    if not os.path.exists(metrics_path):
        raise FileNotFoundError("Metrics file not found: {}".format(metrics_path))
    with open(metrics_path, "r") as f:
        payload = json.load(f)
    return payload["metrics"]


def _resolve_final_checkpoint(train_dir: str, model_suffix: str) -> str:
    if not os.path.isdir(train_dir):
        raise RuntimeError("Training directory does not exist: {}".format(train_dir))

    candidates = []
    suffix = "_{}".format(model_suffix)
    for d in os.listdir(train_dir):
        candidate_dir = os.path.join(train_dir, d)
        if os.path.isdir(candidate_dir) and d.endswith(suffix):
            checkpoint = os.path.join(candidate_dir, "model_final.pth")
            if os.path.exists(checkpoint):
                candidates.append((os.path.getmtime(checkpoint), checkpoint))

    if not candidates:
        raise RuntimeError("Unable to resolve model_final.pth for suffix '{}'".format(model_suffix))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _objective(trial: optuna.Trial, args):
    trial_id = "trial_{:04d}".format(trial.number)
    trial_root = os.path.join(args.study_dir, trial_id)
    os.makedirs(trial_root, exist_ok=True)

    inference_overrides = {
        "INFERENCE.TRACK_SCORE_THRESH": trial.suggest_float(
            "inference_track_score_thresh", args.track_score_low, args.track_score_high
        ),
        "INFERENCE.MIN_TRACK_LENGTH": trial.suggest_int(
            "inference_min_track_length", args.min_track_len_low, args.min_track_len_high
        ),
        "MODEL.TRACK_HEAD.TRACK_THRESH": trial.suggest_float(
            "model_track_thresh", args.track_thresh_low, args.track_thresh_high
        ),
        "MODEL.TRACK_HEAD.START_TRACK_THRESH": trial.suggest_float(
            "model_start_track_thresh", args.start_track_thresh_low, args.start_track_thresh_high
        ),
        "MODEL.TRACK_HEAD.RESUME_TRACK_THRESH": trial.suggest_float(
            "model_resume_track_thresh", args.resume_track_thresh_low, args.resume_track_thresh_high
        ),
    }

    if args.mode == "inference":
        model_file = args.model_file
    else:
        model_suffix = "{}_{}".format(args.model_suffix_prefix, trial.number)
        train_overrides = {
            "SOLVER.BASE_LR": trial.suggest_float("solver_base_lr", args.lr_low, args.lr_high, log=True),
            "SOLVER.WEIGHT_DECAY": trial.suggest_float(
                "solver_weight_decay", args.weight_decay_low, args.weight_decay_high, log=True
            ),
        }
        if args.max_iter > 0:
            train_overrides["SOLVER.MAX_ITER"] = args.max_iter

        run_info_file = os.path.join(trial_root, "train_run_info.json")
        train_cmd = [
            sys.executable,
            "tools/train_net.py",
            "--config-file",
            args.config_file,
            "--train-dir",
            args.train_dir,
            "--model-suffix",
            model_suffix,
            "--run-info-file",
            run_info_file,
        ]
        train_cmd += ["--opts"] + _build_cfg_opts(train_overrides)
        _run_command(train_cmd, cwd=args.project_root)

        if os.path.exists(run_info_file):
            with open(run_info_file, "r") as f:
                run_info = json.load(f)
            model_file = run_info.get("final_checkpoint")
        else:
            model_file = _resolve_final_checkpoint(args.train_dir, model_suffix)

    if not model_file or not os.path.exists(model_file):
        raise RuntimeError("Model checkpoint does not exist: {}".format(model_file))

    output_dir = os.path.join(trial_root, "inference")
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(trial_root, "metrics.json")

    test_cmd = [
        sys.executable,
        "tools/test_net.py",
        "--config-file",
        args.config_file,
        "--output-dir",
        output_dir,
        "--model-file",
        model_file,
        "--test-dataset",
        args.test_dataset,
        "--set",
        args.dataset_split,
        "--metrics-file",
        metrics_path,
    ]
    test_cmd += ["--opts"] + _build_cfg_opts(inference_overrides) 
    _run_command(test_cmd, cwd=args.project_root)

    metrics = _load_metrics(metrics_path)
    if args.metric_name not in metrics:
        raise KeyError("Metric '{}' not found in {}".format(args.metric_name, metrics_path))

    trial.set_user_attr("metrics_path", metrics_path)
    trial.set_user_attr("model_file", model_file)
    trial.set_user_attr("inference_overrides", inference_overrides)

    return float(metrics[args.metric_name])


def _parse_args():
    parser = argparse.ArgumentParser("Optuna tuner for SiamMOT")
    parser.add_argument("--project-root", default=".", type=str)
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument("--mode", choices=("inference", "finetune"), default="inference")
    parser.add_argument("--model-file", default="", type=str, help="required when mode=inference")
    parser.add_argument("--train-dir", default="", type=str, help="required when mode=finetune")
    parser.add_argument("--model-suffix-prefix", default="optuna", type=str)
    parser.add_argument("--max-iter", default=0, type=int, help="override solver max iter for mode=finetune")
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--study-name", default="siammot_optuna", type=str)
    parser.add_argument("--storage", default="", type=str, help="e.g., sqlite:///optuna.db")
    parser.add_argument("--direction", choices=("maximize", "minimize"), default="maximize")
    parser.add_argument("--metric-name", default="infer/mot/idf1", type=str)
    parser.add_argument("--test-dataset", required=True, type=str)
    parser.add_argument("--dataset-split", default="val", type=str)
    parser.add_argument("--n-trials", default=20, type=int)
    parser.add_argument("--timeout-sec", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--track-score-low", default=0.3, type=float)
    parser.add_argument("--track-score-high", default=0.95, type=float)
    parser.add_argument("--min-track-len-low", default=1, type=int)
    parser.add_argument("--min-track-len-high", default=30, type=int)
    parser.add_argument("--track-thresh-low", default=0.2, type=float)
    parser.add_argument("--track-thresh-high", default=0.8, type=float)
    parser.add_argument("--start-track-thresh-low", default=0.4, type=float)
    parser.add_argument("--start-track-thresh-high", default=0.98, type=float)
    parser.add_argument("--resume-track-thresh-low", default=0.2, type=float)
    parser.add_argument("--resume-track-thresh-high", default=0.8, type=float)
    parser.add_argument("--lr-low", default=1e-5, type=float)
    parser.add_argument("--lr-high", default=1e-2, type=float)
    parser.add_argument("--weight-decay-low", default=1e-6, type=float)
    parser.add_argument("--weight-decay-high", default=1e-3, type=float)
    return parser.parse_args()


def main():
    args = _parse_args()
    args.project_root = os.path.abspath(args.project_root)
    args.output_dir = os.path.abspath(args.output_dir)
    args.study_dir = os.path.join(args.output_dir, args.study_name)
    os.makedirs(args.study_dir, exist_ok=True)

    if args.mode == "inference" and not args.model_file:
        raise ValueError("--model-file is required when --mode inference")
    if args.mode == "finetune" and not args.train_dir:
        raise ValueError("--train-dir is required when --mode finetune")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=(args.storage or None),
        direction=args.direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    timeout = args.timeout_sec if args.timeout_sec > 0 else None
    study.optimize(lambda trial: _objective(trial, args), n_trials=args.n_trials, timeout=timeout)

    result = {
        "study_name": args.study_name,
        "direction": args.direction,
        "metric_name": args.metric_name,
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
        "num_trials": len(study.trials),
    }
    result_path = os.path.join(args.study_dir, "best_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print("Best value: {}".format(study.best_value))
    print("Best params: {}".format(study.best_params))
    print("Saved summary to {}".format(result_path))


if __name__ == "__main__":
    main()
