from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import optuna
from optuna.trial import TrialState

from siammot.engine.mlflow_logger import MLflowLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for SiamMOT using train_net.py and test_net.py"
    )
    parser.add_argument("--project-root", default=".", type=str)
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument("--base-model-file", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--study-name", default="hspot_hpo", type=str)
    parser.add_argument("--run-name-prefix", default="hspot_hota", type=str)
    parser.add_argument("--storage-file", default="", type=str)
    parser.add_argument("--datasets-root", default="datasets", type=str)
    parser.add_argument("--dataset-key", default="MOT_HSPOT", type=str)
    parser.add_argument("--train-split", default="train", type=str)
    parser.add_argument("--val-split", default="val", type=str)
    parser.add_argument(
        "--eval-metric", default="both", choices=["clear", "hota", "both"]
    )
    parser.add_argument(
        "--direction", default="maximize", choices=["maximize", "minimize"]
    )
    parser.add_argument("--n-trials", default=40, type=int)
    parser.add_argument("--timeout-sec", default=360000, type=int)
    parser.add_argument("--max-iter", default=2000, type=int)
    parser.add_argument(
        "--prune-checkpoints",
        default="auto",
        type=str,
        help="Comma-separated iteration checkpoints for intermediate pruning reports, or 'auto' to align stages to epoch boundaries.",
    )
    parser.add_argument("--pruner-startup-trials", default=3, type=int)
    parser.add_argument("--pruner-warmup-steps", default=1, type=int)
    parser.add_argument("--early-stop-patience", default=1, type=int)
    parser.add_argument("--early-stop-min-delta", default=0.0, type=float)
    parser.add_argument("--early-stop-warmup-stages", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--mlflow-enabled", action="store_true")
    parser.add_argument(
        "--base-opts",
        nargs="*",
        default=[],
        help="Additional fixed cfg overrides passed to both train_net.py and test_net.py",
    )
    args = parser.parse_args()
    if len(args.base_opts) % 2 != 0:
        raise ValueError("--base-opts must contain an even number of KEY VALUE tokens")
    if args.n_trials < 1:
        raise ValueError("--n-trials must be >= 1")
    if args.max_iter < 1:
        raise ValueError("--max-iter must be >= 1")
    return args


def to_optuna_storage_url(storage_file: str) -> str:
    abs_path = os.path.abspath(storage_file)
    return "sqlite:///{}".format(abs_path)


def objective_metric_name(eval_metric: str) -> str:
    mode = str(eval_metric).strip().lower()
    objective_map = {
        "clear": "infer/mot/idf1",
        "hota": "infer/mot/hota",
        "both": "infer/mot/hota",
    }
    if mode not in objective_map:
        raise ValueError(
            "Unsupported eval mode '{}'. Supported: clear, hota, both".format(
                eval_metric
            )
        )
    return objective_map[mode]


def parse_explicit_stage_iters(prune_checkpoints: str, max_iter: int) -> List[int]:
    stages: List[int] = []
    for raw in str(prune_checkpoints).split(","):
        token = raw.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            continue
        if value < max_iter:
            stages.append(value)
    stages = sorted(set(stages))
    stages.append(max_iter)
    return stages


def build_epoch_stage_iters(max_iter: int, steps_per_epoch: int) -> List[int]:
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if steps_per_epoch < 1:
        return [max_iter]

    stages: List[int] = []
    current = int(steps_per_epoch)
    while current < int(max_iter):
        stages.append(current)
        current += int(steps_per_epoch)
    stages.append(int(max_iter))
    return stages


def resolve_stage_iters(
    prune_checkpoints: str,
    max_iter: int,
    steps_per_epoch: int,
) -> List[int]:
    mode = str(prune_checkpoints).strip().lower()
    if mode in {"", "auto", "epoch", "epochs"}:
        return build_epoch_stage_iters(max_iter, steps_per_epoch)
    return parse_explicit_stage_iters(prune_checkpoints, max_iter)


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


def combine_opts(*option_lists: Sequence[str]) -> List[str]:
    merged: List[str] = []
    for option_list in option_lists:
        merged.extend(list(option_list))
    return merged


def hpo_mlflow_tags(
    args: argparse.Namespace,
    trial_number: int,
    run_stage: str,
    stage_iter: int,
) -> List[str]:
    return [
        "stage={}".format(run_stage),
        "hpo_study_name={}".format(args.study_name),
        "hpo_trial_number={}".format(trial_number),
        "hpo_stage_iter={}".format(stage_iter),
    ]


def hpo_trial_run_name(run_name_prefix: str, trial_number: int) -> str:
    prefix = str(run_name_prefix).strip()
    if not prefix:
        return "trial_{:04d}".format(trial_number)
    return "{}_trial_{:04d}".format(prefix, trial_number)


def run_command(
    cmd: Sequence[str], cwd: str, env: Dict[str, str], log_path: str
) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    printable_cmd = " ".join(shlex.quote(part) for part in cmd)
    print("\n[run] {}\n".format(printable_cmd))
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(cmd),
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            "Command failed with exit code {}. See log: {}\nCommand: {}".format(
                return_code, log_path, printable_cmd
            )
        )


def final_test_eval_metric(eval_metric: str) -> str:
    mode = str(eval_metric).strip().lower()
    if mode == "clear":
        return "hota"
    return mode or "hota"


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_improved(
    value: float,
    best_value: float,
    direction: str,
    min_delta: float,
) -> bool:
    if direction == "maximize":
        return value > (best_value + min_delta)
    return value < (best_value - min_delta)


def run_final_best_trial_eval(
    context: TuningContext,
    best_trial: optuna.trial.FrozenTrial,
    parent_mlflow_logger: Optional[MLflowLogger],
) -> Dict[str, Any]:
    args = context.args
    best_checkpoint = str(best_trial.user_attrs.get("final_checkpoint", "")).strip()
    if not best_checkpoint or not os.path.isfile(best_checkpoint):
        raise FileNotFoundError(
            "Best trial checkpoint does not exist: {}".format(
                best_checkpoint or "<missing>"
            )
        )

    output_dir = os.path.join(os.path.abspath(args.output_dir), "best_hpo_eval")
    os.makedirs(output_dir, exist_ok=True)
    final_eval_metric = final_test_eval_metric(args.eval_metric)
    log_path = os.path.join(output_dir, "final_eval.log")

    child_logger: Optional[MLflowLogger] = None
    child_run_status = "FINISHED"
    child_run_id = ""
    if args.mlflow_enabled:
        child_logger = make_mlflow_logger(
            enabled=True,
            experiment_name=context.mlflow_experiment_name,
        )
        child_logger.start_run(
            experiment_name=context.mlflow_experiment_name,
            run_name="{}_final_eval".format(
                str(args.run_name_prefix).strip()
                or str(args.study_name).strip()
                or "hpo"
            ),
            tags={
                "stage": "final_eval_best_hpo",
                "workflow": "tune_optuna_final_eval",
                "hpo_study_name": args.study_name,
                "hpo_best_trial_number": str(best_trial.number),
                "dataset_key": args.dataset_key,
                "eval_split": "test",
            },
            nested=True,
        )
        child_run_id = child_logger.run_id or ""

    test_opts = combine_opts(
        base_opts := list(args.base_opts),
        [
            "DATASETS.ROOT_DIR",
            args.datasets_root,
            "INFERENCE.EVAL_METRIC",
            final_eval_metric,
        ],
    )
    cmd: List[str] = [
        sys.executable,
        context.test_script,
        "--config-file",
        context.config_file,
        "--output-dir",
        output_dir,
        "--model-file",
        best_checkpoint,
        "--test-dataset",
        args.dataset_key,
        "--set",
        "test",
        "--parent-run-id",
        context.mlflow_parent_run_id,
        "--extra-mlflow-tags",
        "stage=final_eval_best_hpo",
        "workflow=tune_optuna_final_eval",
        "dataset_key={}".format(args.dataset_key),
        "eval_split=test",
        "model_origin=hpo_best_trial",
        "hpo_best_trial_number={}".format(best_trial.number),
    ]
    if child_run_id:
        cmd.extend(["--mlflow-run-id", child_run_id])
    cmd.extend(["--opts", *test_opts])

    try:
        run_command(
            cmd,
            cwd=context.project_root,
            env=context.env,
            log_path=log_path,
        )
        model_name = os.path.basename(os.path.dirname(best_checkpoint))
        metrics_path = os.path.join(output_dir, model_name, "inference_metrics.json")
        eval_summary = {
            "best_trial_number": best_trial.number,
            "best_checkpoint": best_checkpoint,
            "output_dir": output_dir,
            "metrics_file": metrics_path,
            "eval_metric_mode": final_eval_metric,
            "split": "test",
        }
        if os.path.isfile(metrics_path):
            eval_summary["metrics"] = load_json(metrics_path).get("metrics", {})
        if child_logger is not None:
            child_logger.log_params(
                {
                    "best_checkpoint": best_checkpoint,
                    "dataset_key": args.dataset_key,
                    "test_split": "test",
                    "eval_metric_mode": final_eval_metric,
                }
            )
            if os.path.isfile(metrics_path):
                metrics_payload = load_json(metrics_path).get("metrics", {})
                if isinstance(metrics_payload, dict):
                    child_logger.log_metrics(metrics_payload)
            child_logger.log_artifact(log_path, artifact_path="metadata")
            if os.path.isfile(metrics_path):
                child_logger.log_artifact(metrics_path, artifact_path="evaluation")
        return eval_summary
    except Exception:
        child_run_status = "FAILED"
        raise
    finally:
        if child_logger is not None:
            child_logger.end_run(status=child_run_status)


def sample_trial_cfg(trial: optuna.Trial) -> Dict[str, Any]:
    # Keep the search space tight for HSPOT to avoid wasting trials on
    # implausible settings for this small single-class dataset.
    track_thresh = trial.suggest_float("model_track_thresh", 0.35, 0.55)
    start_thresh = trial.suggest_float(
        "model_start_track_thresh",
        max(track_thresh, 0.55),
        0.75,
    )
    resume_thresh = trial.suggest_float(
        "model_resume_track_thresh",
        0.25,
        min(start_thresh, 0.45),
    )

    return {
        "SOLVER.BASE_LR": trial.suggest_float("solver_base_lr", 5e-4, 2e-3, log=True),
        "SOLVER.WEIGHT_DECAY": trial.suggest_float(
            "solver_weight_decay", 5e-5, 2e-4, log=True
        ),
        "MODEL.TRACK_HEAD.TRACK_THRESH": track_thresh,
        "MODEL.TRACK_HEAD.START_TRACK_THRESH": start_thresh,
        "MODEL.TRACK_HEAD.RESUME_TRACK_THRESH": resume_thresh,
        "MODEL.TRACK_HEAD.MAX_DORMANT_FRAMES": trial.suggest_int(
            "model_max_dormant_frames", 10, 25
        ),
        "INFERENCE.TRACK_SCORE_THRESH": trial.suggest_float(
            "infer_track_score_thresh", 0.5, 0.75
        ),
        "INFERENCE.MIN_TRACK_LENGTH": trial.suggest_int("infer_min_track_length", 2, 5),
    }


@dataclass
class TuningContext:
    args: argparse.Namespace
    project_root: str
    config_file: str
    train_script: str
    test_script: str
    env: Dict[str, str]
    common_data_cfg: Dict[str, Any]
    objective_metric: str
    stage_iters: List[int]
    steps_per_epoch: int
    mlflow_experiment_name: str
    mlflow_parent_run_id: str = ""


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def resolve_mlflow_experiment_name(
    config_file: str,
    base_opts: Sequence[str],
) -> str:
    from siammot.configs.defaults import cfg as default_cfg

    resolved_cfg = default_cfg.clone()
    resolved_cfg.merge_from_file(config_file)
    if base_opts:
        resolved_cfg.merge_from_list(list(base_opts))
    experiment_name = str(resolved_cfg.MLFLOW.EXPERIMENT_NAME).strip()
    return experiment_name or "SiamMOT"


def resolve_train_cfg(
    config_file: str,
    base_opts: Sequence[str],
    cfg_overrides: Dict[str, Any],
):
    from siammot.configs.defaults import cfg as default_cfg

    resolved_cfg = default_cfg.clone()
    resolved_cfg.merge_from_file(config_file)
    if base_opts:
        resolved_cfg.merge_from_list(list(base_opts))
    if cfg_overrides:
        resolved_cfg.merge_from_list(cfg_dict_to_opts(cfg_overrides))
    resolved_cfg.freeze()
    return resolved_cfg


def make_mlflow_logger(
    enabled: bool,
    experiment_name: str,
    artifact_path_prefix: str = "",
) -> MLflowLogger:
    mlflow_cfg = SimpleNamespace(
        MLFLOW=SimpleNamespace(
            ENABLED=bool(enabled),
            TRACKING_URI=os.environ.get("MLFLOW_TRACKING_URI", "").strip(),
            EXPERIMENT_NAME=str(experiment_name).strip() or "SiamMOT",
        )
    )
    return MLflowLogger(mlflow_cfg, artifact_path_prefix=artifact_path_prefix)


def hpo_parent_run_name(study_name: str, run_name_prefix: str) -> str:
    base_name = str(run_name_prefix).strip() or str(study_name).strip() or "hpo"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return "{}_tune_{}".format(base_name, timestamp)


def stage_artifact_subdir(stage_name: str, stage_kind: str) -> str:
    return "stages/{}/{}".format(stage_name, stage_kind)


def log_trial_stage_metrics(
    mlflow_logger: Optional[MLflowLogger],
    stage_name: str,
    stage_iter: int,
    objective_metric_name: str,
    objective_value: float,
    metrics: Dict[str, Any],
    best_metric: Optional[float],
) -> None:
    if mlflow_logger is None or not mlflow_logger.can_log:
        return

    mlflow_logger.log_metric("hpo/objective", objective_value, step=stage_iter)
    mlflow_logger.log_metric(
        "hpo/stage/{}/{}".format(stage_name, objective_metric_name),
        objective_value,
    )
    if best_metric is not None:
        mlflow_logger.log_metric(
            "hpo/best_objective_so_far", best_metric, step=stage_iter
        )

    for metric_name, metric_value in metrics.items():
        try:
            parsed_value = float(metric_value)
        except (TypeError, ValueError):
            continue
        mlflow_logger.log_metric(
            "hpo/stage/{}/{}".format(stage_name, metric_name),
            parsed_value,
        )


def build_context(args: argparse.Namespace) -> TuningContext:
    objective_metric = objective_metric_name(args.eval_metric)

    project_root = os.path.abspath(args.project_root)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    storage_file = args.storage_file
    if not storage_file:
        storage_file = os.path.join(output_dir, "optuna_study.db")
    args.storage_file = storage_file

    config_file = os.path.abspath(os.path.join(project_root, args.config_file))
    base_model_file = os.path.abspath(os.path.join(project_root, args.base_model_file))
    args.base_model_file = base_model_file
    train_script = os.path.join(project_root, "tools", "train_net.py")
    test_script = os.path.join(project_root, "tools", "test_net.py")
    if not os.path.isfile(train_script):
        raise FileNotFoundError("Missing training script: {}".format(train_script))
    if not os.path.isfile(test_script):
        raise FileNotFoundError("Missing test script: {}".format(test_script))
    if not os.path.isfile(config_file):
        raise FileNotFoundError("Missing config file: {}".format(config_file))
    if not os.path.isfile(args.base_model_file):
        raise FileNotFoundError(
            "Missing --base-model-file: {}".format(args.base_model_file)
        )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        "{}{}{}".format(project_root, os.pathsep, py_path) if py_path else project_root
    )

    common_data_cfg = {
        "DATASETS.ROOT_DIR": args.datasets_root,
        "DATASETS.TRAIN": "('{}',)".format(args.dataset_key),
        "DATASETS.TRAIN_SET": args.train_split,
        "DATASETS.VAL": "('{}',)".format(args.dataset_key),
        "DATASETS.VAL_SET": args.val_split,
        "SOLVER.VAL_EPOCH_PERIOD": 1,
        "SOLVER.VAL_EVAL_METRIC": args.eval_metric,
        "SOLVER.VAL_TARGET_METRIC": objective_metric,
        "MLFLOW.ENABLED": bool(args.mlflow_enabled),
    }
    mlflow_experiment_name = resolve_mlflow_experiment_name(config_file, args.base_opts)
    resolved_train_cfg = resolve_train_cfg(config_file, args.base_opts, common_data_cfg)
    from siammot.data.build_train_data_loader import summarize_train_dataset

    train_data_stats = summarize_train_dataset(resolved_train_cfg)
    steps_per_epoch = int(train_data_stats.get("steps_per_epoch", 0))
    stage_iters = resolve_stage_iters(
        args.prune_checkpoints, args.max_iter, steps_per_epoch
    )
    print("Pruning stages (iters): {}".format(stage_iters))
    print("Objective metric: {}".format(objective_metric))
    print("Steps per epoch: {}".format(steps_per_epoch))

    return TuningContext(
        args=args,
        project_root=project_root,
        config_file=config_file,
        train_script=train_script,
        test_script=test_script,
        env=env,
        common_data_cfg=common_data_cfg,
        objective_metric=objective_metric,
        stage_iters=stage_iters,
        steps_per_epoch=steps_per_epoch,
        mlflow_experiment_name=mlflow_experiment_name,
    )


def trial_objective(context: TuningContext, trial: optuna.Trial) -> float:
    args = context.args
    trial_dir = os.path.join(
        os.path.abspath(args.output_dir), "trials", "trial_{:04d}".format(trial.number)
    )
    train_root = os.path.join(trial_dir, "train")
    logs_root = os.path.join(trial_dir, "logs")
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(logs_root, exist_ok=True)

    sampled_cfg = sample_trial_cfg(trial)
    sampled_opts = cfg_dict_to_opts(sampled_cfg)
    base_opts = list(args.base_opts)

    trial_mlflow_logger: Optional[MLflowLogger] = None
    trial_run_status = "FINISHED"
    trial_state_label = "COMPLETE"
    current_model_file = os.path.abspath(args.base_model_file)
    current_metric: Optional[float] = None
    best_metric: Optional[float] = None
    best_checkpoint_path = current_model_file
    non_improve_stages = 0
    try:
        if args.mlflow_enabled:
            trial_mlflow_logger = make_mlflow_logger(
                enabled=True,
                experiment_name=context.mlflow_experiment_name,
            )
            trial_tags = {
                "stage": "hpo_trial",
                "workflow": "tune_optuna",
                "hpo_study_name": args.study_name,
                "hpo_trial_number": str(trial.number),
            }
            if context.mlflow_parent_run_id:
                trial_tags["hpo_parent_run_id"] = context.mlflow_parent_run_id
            trial_mlflow_logger.start_run(
                experiment_name=context.mlflow_experiment_name,
                run_name=hpo_trial_run_name(args.run_name_prefix, trial.number),
                tags=trial_tags,
                nested=True,
            )
            trial.set_user_attr("mlflow_run_id", trial_mlflow_logger.run_id)
            trial_mlflow_logger.log_params(
                {
                    "hpo_trial_number": trial.number,
                    "hpo_study_name": args.study_name,
                    "hpo_objective_metric": context.objective_metric,
                    "hpo_eval_metric_mode": args.eval_metric,
                    "hpo_train_split": args.train_split,
                    "hpo_val_split": args.val_split,
                    "hpo_direction": args.direction,
                    **{
                        "optuna.{}".format(param_name): param_value
                        for param_name, param_value in trial.params.items()
                    },
                }
            )
            trial_inputs_path = os.path.join(trial_dir, "trial_inputs.json")
            write_json(
                trial_inputs_path,
                {
                    "number": trial.number,
                    "params": trial.params,
                    "sampled_cfg": sampled_cfg,
                    "study_name": args.study_name,
                    "objective_metric": context.objective_metric,
                    "eval_metric_mode": args.eval_metric,
                    "train_split": args.train_split,
                    "val_split": args.val_split,
                },
            )
            trial_mlflow_logger.log_artifact(
                trial_inputs_path, artifact_path="metadata"
            )

        for stage_idx, stage_iter in enumerate(context.stage_iters, start=1):
            stage_name = "iter_{:07d}".format(stage_iter)

            stage_run_info_path = os.path.join(
                trial_dir, "run_info_{}.json".format(stage_name)
            )
            stage_train_cfg = {
                "SOLVER.MAX_ITER": stage_iter,
                "MODEL.WEIGHT": current_model_file,
            }
            train_opts = combine_opts(
                base_opts,
                cfg_dict_to_opts(context.common_data_cfg),
                cfg_dict_to_opts(stage_train_cfg),
                sampled_opts,
            )
            train_cmd = [
                sys.executable,
                context.train_script,
                "--config-file",
                context.config_file,
                "--train-dir",
                train_root,
                "--model-suffix",
                "trial_{:04d}".format(trial.number),
                "--run-info-file",
                stage_run_info_path,
                "--extra-mlflow-tags",
                *hpo_mlflow_tags(args, trial.number, "hpo_train", stage_iter),
            ]
            if (
                args.mlflow_enabled
                and trial_mlflow_logger is not None
                and trial_mlflow_logger.run_id
            ):
                train_cmd.extend(
                    [
                        "--mlflow-run-id",
                        trial_mlflow_logger.run_id,
                        "--mlflow-artifact-subdir",
                        stage_artifact_subdir(stage_name, "train"),
                        "--mlflow-stage-name",
                        stage_name,
                        "--mlflow-stage-iter",
                        str(stage_iter),
                    ]
                )
            train_cmd.extend(
                [
                    "--opts",
                    *train_opts,
                ]
            )
            run_command(
                train_cmd,
                cwd=context.project_root,
                env=context.env,
                log_path=os.path.join(logs_root, "train_{}.log".format(stage_name)),
            )

            run_info = load_json(stage_run_info_path)
            final_checkpoint = str(run_info.get("final_checkpoint", "")).strip()
            if not final_checkpoint or not os.path.isfile(final_checkpoint):
                raise FileNotFoundError(
                    "Could not find final checkpoint for {}: {}".format(
                        stage_name, final_checkpoint
                    )
                )
            current_model_file = final_checkpoint

            validation_payload = run_info.get("validation", {})
            if not isinstance(validation_payload, dict):
                raise ValueError(
                    "Training run info for {} does not contain a validation object: {}".format(
                        stage_name, stage_run_info_path
                    )
                )
            best_metric_value = validation_payload.get("best_metric_value")
            latest_validation = validation_payload.get("latest", {})
            latest_metrics = {}
            latest_metric_value = None
            if isinstance(latest_validation, dict):
                latest_metrics = latest_validation.get("metrics", {}) or {}
                latest_metric_value = latest_validation.get("target_value")
            best_checkpoint_candidate = str(
                validation_payload.get("best_checkpoint", "") or current_model_file
            ).strip()
            if best_checkpoint_candidate:
                best_checkpoint_path = best_checkpoint_candidate
            if best_metric_value is None:
                raise KeyError(
                    "Validation target metric '{}' was not produced by {}. Validation payload: {}".format(
                        context.objective_metric,
                        stage_run_info_path,
                        validation_payload,
                    )
                )
            metrics_payload = {
                "metrics": latest_metrics,
                "latest_validation": latest_validation,
                "validation": validation_payload,
            }
            current_metric = float(best_metric_value)
            trial.report(current_metric, step=stage_iter)
            trial.set_user_attr("metric_{}".format(stage_name), current_metric)
            if latest_metric_value is not None:
                trial.set_user_attr(
                    "latest_metric_{}".format(stage_name), float(latest_metric_value)
                )

            if best_metric is None or is_improved(
                current_metric,
                best_metric,
                direction=args.direction,
                min_delta=float(args.early_stop_min_delta),
            ):
                best_metric = current_metric
                non_improve_stages = 0
            else:
                non_improve_stages += 1

            log_trial_stage_metrics(
                trial_mlflow_logger,
                stage_name=stage_name,
                stage_iter=stage_iter,
                objective_metric_name=context.objective_metric,
                objective_value=current_metric,
                metrics=latest_metrics,
                best_metric=best_metric,
            )

            stage_summary_path = os.path.join(
                trial_dir, "stage_summary_{}.json".format(stage_name)
            )
            write_json(
                stage_summary_path,
                {
                    "stage_name": stage_name,
                    "stage_iter": stage_iter,
                    "objective_metric": context.objective_metric,
                    "objective_value": current_metric,
                    "best_metric_so_far": best_metric,
                    "train_run_info": run_info,
                    "validation": metrics_payload,
                },
            )
            if trial_mlflow_logger is not None:
                trial_mlflow_logger.log_artifact(
                    stage_summary_path,
                    artifact_path="stages/{}".format(stage_name),
                )

            if (
                int(args.early_stop_patience) > 0
                and stage_idx > int(args.early_stop_warmup_stages)
                and non_improve_stages >= int(args.early_stop_patience)
            ):
                trial.set_user_attr("stopped_early_at_iter", stage_iter)
                raise optuna.TrialPruned(
                    "Early stopping: no improvement for {} stage(s)".format(
                        non_improve_stages
                    )
                )
            if trial.should_prune():
                trial.set_user_attr("pruned_at_iter", stage_iter)
                raise optuna.TrialPruned("Optuna pruner stopped the trial")

        assert current_metric is not None
        trial.set_user_attr("final_checkpoint", best_checkpoint_path)
        trial.set_user_attr("sampled_cfg", sampled_cfg)
        return current_metric
    except optuna.TrialPruned:
        trial_run_status = "KILLED"
        trial_state_label = "PRUNED"
        raise
    except Exception:
        trial_run_status = "FAILED"
        trial_state_label = "FAILED"
        raise
    finally:
        if trial_mlflow_logger is not None:
            if current_metric is not None:
                trial_mlflow_logger.log_metric("hpo/final_objective", current_metric)
            if best_metric is not None:
                trial_mlflow_logger.log_metric("hpo/best_objective", best_metric)
            trial_mlflow_logger.set_tags(
                {
                    "hpo_trial_state": trial_state_label,
                    "hpo_final_checkpoint": best_checkpoint_path,
                }
            )
            trial_summary_path = os.path.join(trial_dir, "trial_summary.json")
            write_json(
                trial_summary_path,
                {
                    "number": trial.number,
                    "state": trial_state_label,
                    "value": current_metric,
                    "best_metric": best_metric,
                    "params": trial.params,
                    "sampled_cfg": sampled_cfg,
                    "user_attrs": dict(trial.user_attrs),
                    "final_checkpoint": best_checkpoint_path,
                    "mlflow_run_id": trial_mlflow_logger.run_id,
                },
            )
            trial_mlflow_logger.log_artifact(
                trial_summary_path, artifact_path="metadata"
            )
            trial_mlflow_logger.end_run(status=trial_run_status)


def dump_study_summary(study: optuna.study.Study, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    trials_payload: List[Dict[str, Any]] = []
    for trial in study.trials:
        trials_payload.append(
            {
                "number": trial.number,
                "state": str(trial.state),
                "value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            }
        )
    with open(
        os.path.join(output_dir, "study_trials.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trials_payload, f, indent=2, sort_keys=True)

    best_trial = study.best_trial
    best_payload = {
        "number": best_trial.number,
        "value": best_trial.value,
        "params": best_trial.params,
        "user_attrs": best_trial.user_attrs,
    }
    with open(os.path.join(output_dir, "best_trial.json"), "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    context = build_context(args)
    parent_mlflow_logger: Optional[MLflowLogger] = None
    parent_run_status = "FINISHED"

    if args.mlflow_enabled:
        parent_mlflow_logger = make_mlflow_logger(
            enabled=True,
            experiment_name=context.mlflow_experiment_name,
        )
        parent_mlflow_logger.start_run(
            experiment_name=context.mlflow_experiment_name,
            run_name=hpo_parent_run_name(args.study_name, args.run_name_prefix),
            tags={
                "stage": "hpo",
                "workflow": "tune_optuna",
                "hpo_study_name": args.study_name,
            },
        )
        context.mlflow_parent_run_id = parent_mlflow_logger.run_id or ""
        parent_mlflow_logger.log_params(
            {
                "study_name": args.study_name,
                "config_file": context.config_file,
                "base_model_file": args.base_model_file,
                "dataset_key": args.dataset_key,
                "datasets_root": args.datasets_root,
                "train_split": args.train_split,
                "val_split": args.val_split,
                "eval_metric_mode": args.eval_metric,
                "objective_metric": context.objective_metric,
                "direction": args.direction,
                "n_trials": args.n_trials,
                "max_iter": args.max_iter,
                "prune_checkpoints": args.prune_checkpoints,
                "run_name_prefix": args.run_name_prefix,
            }
        )

    try:
        sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=args.pruner_startup_trials,
            n_warmup_steps=args.pruner_warmup_steps,
            interval_steps=1,
        )
        study = optuna.create_study(
            study_name=args.study_name,
            direction=args.direction,
            sampler=sampler,
            pruner=pruner,
            storage=to_optuna_storage_url(args.storage_file),
            load_if_exists=True,
        )

        timeout = None if args.timeout_sec <= 0 else args.timeout_sec
        study.optimize(
            lambda trial: trial_objective(context, trial),
            n_trials=args.n_trials,
            timeout=timeout,
            gc_after_trial=True,
            show_progress_bar=False,
        )

        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not completed_trials:
            raise RuntimeError(
                "No completed trials found. Cannot select best checkpoint."
            )

        output_dir = os.path.abspath(args.output_dir)
        dump_study_summary(study, output_dir)

        summary = {
            "study_name": args.study_name,
            "storage_file": os.path.abspath(args.storage_file),
            "eval_metric_mode": args.eval_metric,
            "objective_metric": context.objective_metric,
            "best_trial_number": study.best_trial.number,
            "best_trial_value": study.best_value,
            "best_trial_params": study.best_trial.params,
            "best_checkpoint": study.best_trial.user_attrs.get("final_checkpoint"),
        }

        final_eval_summary = run_final_best_trial_eval(
            context,
            study.best_trial,
            parent_mlflow_logger,
        )
        summary["final_test_evaluation"] = final_eval_summary
        summary_file = os.path.join(output_dir, "hpo_summary.json")
        write_json(summary_file, summary)

        if parent_mlflow_logger is not None:
            parent_mlflow_logger.log_metrics(
                {
                    "hpo/best_trial_number": study.best_trial.number,
                    "hpo/best_trial_value": study.best_value,
                    "hpo/completed_trials": len(completed_trials),
                    "hpo/total_trials": len(study.trials),
                }
            )
            final_eval_metrics = final_eval_summary.get("metrics", {})
            if isinstance(final_eval_metrics, dict):
                metrics_to_log = {}
                for metric_name, metric_value in final_eval_metrics.items():
                    try:
                        metrics_to_log["hpo/final_test/{}".format(metric_name)] = float(
                            metric_value
                        )
                    except (TypeError, ValueError):
                        continue
                if metrics_to_log:
                    parent_mlflow_logger.log_metrics(metrics_to_log)
            for artifact_name in (
                "study_trials.json",
                "best_trial.json",
                "hpo_summary.json",
            ):
                artifact_path = os.path.join(output_dir, artifact_name)
                if os.path.isfile(artifact_path):
                    parent_mlflow_logger.log_artifact(
                        artifact_path, artifact_path="summary"
                    )

        print("\nHPO complete.")
        print("Summary: {}".format(summary_file))
        print(
            "Best trial: #{} value={}".format(study.best_trial.number, study.best_value)
        )
    except Exception:
        parent_run_status = "FAILED"
        raise
    finally:
        if parent_mlflow_logger is not None:
            parent_mlflow_logger.end_run(status=parent_run_status)


if __name__ == "__main__":
    main()
