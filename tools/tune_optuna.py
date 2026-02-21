import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import optuna
from optuna.trial import TrialState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for SiamMOT using train_net.py and test_net.py"
    )
    parser.add_argument("--project-root", default=".", type=str)
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument("--base-model-file", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--study-name", default="hspot_hpo", type=str)
    parser.add_argument("--storage-file", default="", type=str)
    parser.add_argument("--datasets-root", default="datasets", type=str)
    parser.add_argument("--dataset-key", default="MOT_HSPOT", type=str)
    parser.add_argument("--train-split", default="val", type=str)
    parser.add_argument("--val-split", default="val", type=str)
    parser.add_argument("--test-split", default="test", type=str)
    parser.add_argument("--eval-metric", default="clear", choices=["clear", "hota", "both"])
    parser.add_argument("--direction", default="maximize", choices=["maximize", "minimize"])
    parser.add_argument("--n-trials", default=20, type=int)
    parser.add_argument("--timeout-sec", default=0, type=int)
    parser.add_argument("--max-iter", default=6000, type=int)
    parser.add_argument(
        "--prune-checkpoints",
        default="1000,3000",
        type=str,
        help="Comma-separated iteration checkpoints used for intermediate pruning reports",
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


def parse_stage_iters(prune_checkpoints: str, max_iter: int) -> List[int]:
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


def run_command(cmd: Sequence[str], cwd: str, env: Dict[str, str], log_path: str) -> None:
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


def sample_trial_cfg(trial: optuna.Trial) -> Dict[str, Any]:
    track_thresh = trial.suggest_float("model_track_thresh", 0.2, 0.8)
    start_thresh = trial.suggest_float("model_start_track_thresh", track_thresh, 0.95)
    resume_thresh = trial.suggest_float("model_resume_track_thresh", 0.2, start_thresh)

    return {
        "SOLVER.BASE_LR": trial.suggest_float("solver_base_lr", 1e-4, 5e-3, log=True),
        "SOLVER.WEIGHT_DECAY": trial.suggest_float(
            "solver_weight_decay", 1e-5, 1e-3, log=True
        ),
        "MODEL.TRACK_HEAD.TRACK_THRESH": track_thresh,
        "MODEL.TRACK_HEAD.START_TRACK_THRESH": start_thresh,
        "MODEL.TRACK_HEAD.RESUME_TRACK_THRESH": resume_thresh,
        "MODEL.TRACK_HEAD.MAX_DORMANT_FRAMES": trial.suggest_int(
            "model_max_dormant_frames", 5, 60
        ),
        "INFERENCE.TRACK_SCORE_THRESH": trial.suggest_float(
            "infer_track_score_thresh", 0.3, 0.95
        ),
        "INFERENCE.MIN_TRACK_LENGTH": trial.suggest_int("infer_min_track_length", 1, 15),
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
    common_eval_cfg: Dict[str, Any]
    objective_metric: str
    stage_iters: List[int]


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
        raise FileNotFoundError("Missing --base-model-file: {}".format(args.base_model_file))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        "{}{}{}".format(project_root, os.pathsep, py_path)
        if py_path
        else project_root
    )

    common_data_cfg = {
        "DATASETS.ROOT_DIR": args.datasets_root,
        "DATASETS.TRAIN": "('{}',)".format(args.dataset_key),
        "DATASETS.TRAIN_SET": args.train_split,
        "MLFLOW.ENABLED": bool(args.mlflow_enabled),
    }
    common_eval_cfg = {
        "DATASETS.ROOT_DIR": args.datasets_root,
        "INFERENCE.EVAL_METRIC": args.eval_metric,
        "MLFLOW.ENABLED": bool(args.mlflow_enabled),
    }

    stage_iters = parse_stage_iters(args.prune_checkpoints, args.max_iter)
    print("Pruning stages (iters): {}".format(stage_iters))
    print("Objective metric: {}".format(objective_metric))

    return TuningContext(
        args=args,
        project_root=project_root,
        config_file=config_file,
        train_script=train_script,
        test_script=test_script,
        env=env,
        common_data_cfg=common_data_cfg,
        common_eval_cfg=common_eval_cfg,
        objective_metric=objective_metric,
        stage_iters=stage_iters,
    )


def trial_objective(context: TuningContext, trial: optuna.Trial) -> float:
    args = context.args
    trial_dir = os.path.join(os.path.abspath(args.output_dir), "trials", "trial_{:04d}".format(trial.number))
    train_root = os.path.join(trial_dir, "train")
    eval_root = os.path.join(trial_dir, "eval")
    logs_root = os.path.join(trial_dir, "logs")
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(eval_root, exist_ok=True)
    os.makedirs(logs_root, exist_ok=True)

    sampled_cfg = sample_trial_cfg(trial)
    sampled_opts = cfg_dict_to_opts(sampled_cfg)
    base_opts = list(args.base_opts)

    current_model_file = os.path.abspath(args.base_model_file)
    current_metric = None
    best_metric = None
    non_improve_stages = 0

    for stage_idx, stage_iter in enumerate(context.stage_iters, start=1):
        stage_name = "iter_{:07d}".format(stage_iter)

        stage_run_info_path = os.path.join(trial_dir, "run_info_{}.json".format(stage_name))
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
            "--opts",
            *train_opts,
        ]
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
                "Could not find final checkpoint for {}: {}".format(stage_name, final_checkpoint)
            )
        current_model_file = final_checkpoint

        stage_metrics_path = os.path.join(trial_dir, "metrics_{}.json".format(stage_name))
        test_opts = combine_opts(
            base_opts,
            cfg_dict_to_opts(context.common_eval_cfg),
            sampled_opts,
        )
        test_cmd = [
            sys.executable,
            context.test_script,
            "--config-file",
            context.config_file,
            "--output-dir",
            eval_root,
            "--model-file",
            current_model_file,
            "--test-dataset",
            args.dataset_key,
            "--set",
            args.val_split,
            "--metrics-file",
            stage_metrics_path,
            "--opts",
            *test_opts,
        ]
        run_command(
            test_cmd,
            cwd=context.project_root,
            env=context.env,
            log_path=os.path.join(logs_root, "eval_{}.log".format(stage_name)),
        )

        metrics_payload = load_json(stage_metrics_path)
        metrics = metrics_payload.get("metrics", {})
        if context.objective_metric not in metrics:
            raise KeyError(
                "Metric '{}' not found in {}. Available keys: {}".format(
                    context.objective_metric, stage_metrics_path, sorted(metrics.keys())
                )
            )
        current_metric = float(metrics[context.objective_metric])
        trial.report(current_metric, step=stage_iter)
        trial.set_user_attr("metric_{}".format(stage_name), current_metric)

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

        if (
            int(args.early_stop_patience) > 0
            and stage_idx > int(args.early_stop_warmup_stages)
            and non_improve_stages >= int(args.early_stop_patience)
        ):
            trial.set_user_attr("stopped_early_at_iter", stage_iter)
            raise optuna.TrialPruned(
                "Early stopping: no improvement for {} stage(s)".format(non_improve_stages)
            )
        if trial.should_prune():
            trial.set_user_attr("pruned_at_iter", stage_iter)
            raise optuna.TrialPruned("Optuna pruner stopped the trial")

    assert current_metric is not None
    trial.set_user_attr("final_checkpoint", current_model_file)
    trial.set_user_attr("sampled_cfg", sampled_cfg)
    return current_metric


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
    with open(os.path.join(output_dir, "study_trials.json"), "w", encoding="utf-8") as f:
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


def run_final_test_eval(context: TuningContext, study: optuna.study.Study) -> str:
    args = context.args
    best_trial = study.best_trial
    best_checkpoint = str(best_trial.user_attrs.get("final_checkpoint", "")).strip()
    if not best_checkpoint or not os.path.isfile(best_checkpoint):
        raise FileNotFoundError(
            "Best trial checkpoint not found: {} (trial #{})".format(
                best_checkpoint, best_trial.number
            )
        )

    sampled_cfg = best_trial.user_attrs.get("sampled_cfg", {})
    if not isinstance(sampled_cfg, dict):
        sampled_cfg = {}

    final_dir = os.path.join(os.path.abspath(args.output_dir), "final_test_eval")
    os.makedirs(final_dir, exist_ok=True)
    metrics_file = os.path.join(final_dir, "final_test_metrics.json")

    final_opts = combine_opts(
        list(args.base_opts),
        cfg_dict_to_opts(context.common_eval_cfg),
        cfg_dict_to_opts(sampled_cfg),
    )
    final_cmd = [
        sys.executable,
        context.test_script,
        "--config-file",
        context.config_file,
        "--output-dir",
        final_dir,
        "--model-file",
        best_checkpoint,
        "--test-dataset",
        args.dataset_key,
        "--set",
        args.test_split,
        "--metrics-file",
        metrics_file,
        "--opts",
        *final_opts,
    ]
    run_command(
        final_cmd,
        cwd=context.project_root,
        env=context.env,
        log_path=os.path.join(final_dir, "final_test_eval.log"),
    )
    return metrics_file


def main() -> None:
    args = parse_args()
    context = build_context(args)

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
        raise RuntimeError("No completed trials found. Cannot select best checkpoint.")

    output_dir = os.path.abspath(args.output_dir)
    dump_study_summary(study, output_dir)
    final_metrics_file = run_final_test_eval(context, study)

    summary = {
        "study_name": args.study_name,
        "storage_file": os.path.abspath(args.storage_file),
        "eval_metric_mode": args.eval_metric,
        "objective_metric": context.objective_metric,
        "best_trial_number": study.best_trial.number,
        "best_trial_value": study.best_value,
        "best_trial_params": study.best_trial.params,
        "best_checkpoint": study.best_trial.user_attrs.get("final_checkpoint"),
        "final_test_metrics_file": final_metrics_file,
    }
    summary_file = os.path.join(output_dir, "hpo_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("\nHPO complete.")
    print("Summary: {}".format(summary_file))
    print("Best trial: #{} value={}".format(study.best_trial.number, study.best_value))
    print("Final test metrics: {}".format(final_metrics_file))


if __name__ == "__main__":
    main()
