import argparse
import logging
import json
import os
import re
import traceback
from typing import Any, Dict, Optional, Tuple

import torch

from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config

from siammot.configs.defaults import cfg
from siammot.data.build_train_data_loader import build_train_data_loader, summarize_train_dataset
from siammot.modelling.rcnn import build_siammot
from siammot.engine.trainer import do_train
from siammot.utils.get_model_name import get_model_name
from siammot.engine.tensorboard_writer import TensorboardWriter
from siammot.engine.mlflow_logger import MLflowLogger
from siammot.engine.validation import ValidationRunner
from yacs.config import CfgNode


try:
    from apex import amp
except ImportError:
    amp = None


parser = argparse.ArgumentParser(description="PyTorch SiamMOT Training")
parser.add_argument(
    "--config-file", default="", metavar="FILE", help="path to config file", type=str
)
parser.add_argument(
    "--train-dir",
    default="",
    help="training folder where training artifacts are dumped",
    type=str,
)
parser.add_argument(
    "--model-suffix",
    default="",
    help="model suffix to differentiate different runs",
    type=str,
)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument(
    "--run-info-file",
    default="",
    help="optional path to dump run metadata as json",
    type=str,
)
parser.add_argument(
    "--extra-mlflow-tags",
    nargs="*",
    default=[],
    help="optional KEY=VALUE tags to attach to the MLflow run",
)
parser.add_argument(
    "--mlflow-run-id",
    default="",
    help="optional existing MLflow run id to attach to instead of creating a new run",
    type=str,
)
parser.add_argument(
    "--mlflow-artifact-subdir",
    default="",
    help="optional artifact subdirectory used when logging into an existing MLflow run",
    type=str,
)
parser.add_argument(
    "--mlflow-stage-name",
    default="",
    help="optional stage name used for stage-scoped MLflow logging when attaching to an existing run",
    type=str,
)
parser.add_argument(
    "--mlflow-stage-iter",
    default=0,
    help="optional stage iteration used as the MLflow metric step when attaching to an existing run",
    type=int,
)
parser.add_argument(
    "--opts",
    nargs=argparse.REMAINDER,
    default=[],
    help="modify config options using the command-line",
)


def parse_mlflow_tags(raw_tags: Any) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for raw in raw_tags or []:
        token = str(raw).strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                "--extra-mlflow-tags entries must be in KEY=VALUE form, got '{}'".format(token)
            )
        key, value = token.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("MLflow tag key cannot be empty")
        parsed[key] = value.strip()
    return parsed


def _resolve_mlflow_stage_scope(stage_name: str, stage_iter: int, fallback_iter: int) -> Tuple[str, int]:
    resolved_iter = int(stage_iter) if int(stage_iter) > 0 else int(fallback_iter)
    raw_stage_name = str(stage_name or "").strip()
    if not raw_stage_name:
        raw_stage_name = "iter_{:07d}".format(resolved_iter)
    stage_slug = re.sub(r"[^0-9A-Za-z_.-]+", "_", raw_stage_name).strip("._-")
    if not stage_slug:
        stage_slug = "iter_{:07d}".format(resolved_iter)
    return stage_slug, resolved_iter


def log_train_data_stats_to_mlflow(
    mlflow_logger: Optional[MLflowLogger],
    logger: logging.Logger,
    train_data_stats: Dict[str, Any],
    using_external_mlflow_run: bool = False,
    mlflow_stage_name: str = "",
    mlflow_stage_iter: int = 0,
) -> None:
    if mlflow_logger is None or not mlflow_logger.can_log:
        return

    if not using_external_mlflow_run:
        mlflow_logger.log_params(train_data_stats)
        return

    top_level_params = dict(train_data_stats)
    top_level_params.pop("max_iter", None)
    top_level_params.pop("effective_num_epochs", None)
    mlflow_logger.log_params(top_level_params)

    stage_key, stage_step = _resolve_mlflow_stage_scope(
        mlflow_stage_name,
        mlflow_stage_iter,
        int(train_data_stats.get("max_iter", 0) or 0),
    )
    logger.info(
        "MLflow attached run: stage-varying train stats will be logged under 'stage.%s.*' at step=%d",
        stage_key,
        stage_step,
    )
    mlflow_logger.log_params(
        {
            "stage.{}.max_iter".format(stage_key): train_data_stats.get("max_iter"),
            "stage.{}.effective_num_epochs".format(stage_key): train_data_stats.get(
                "effective_num_epochs"
            ),
            "stage.{}.steps_per_epoch".format(stage_key): train_data_stats.get(
                "steps_per_epoch"
            ),
        }
    )
    mlflow_logger.log_metrics(
        {
            "train/stage/max_iter": float(train_data_stats.get("max_iter", 0) or 0),
            "train/stage/effective_num_epochs": float(
                train_data_stats.get("effective_num_epochs", 0.0) or 0.0
            ),
            "train/stage/steps_per_epoch": float(
                train_data_stats.get("steps_per_epoch", 0) or 0
            ),
        },
        step=stage_step,
    )
    mlflow_logger.set_tags(
        {
            "latest_stage_name": stage_key,
            "latest_stage_iter": str(stage_step),
        }
    )


def train(
    cfg: CfgNode,
    train_dir: str,
    local_rank: int,
    distributed: bool,
    logger: logging.Logger,
    mlflow_logger: Optional[MLflowLogger] = None,
    using_external_mlflow_run: bool = False,
    mlflow_stage_name: str = "",
    mlflow_stage_iter: int = 0,
) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:

    # build model
    model = build_siammot(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = "O1" if use_mixed_precision else "O0"
    if use_mixed_precision and amp is None:
        raise ImportError("Mixed precision (DTYPE=float16) requires apex.amp to be installed.")
    if amp is not None:
        model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    arguments: Dict[str, Any] = {"iteration": 0}

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, train_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT) or {}
    arguments.update(extra_checkpoint_data)
    arguments.setdefault("validation_history", [])
    arguments.setdefault("best_validation_metric", None)
    arguments.setdefault("best_validation_iteration", 0)
    arguments.setdefault("best_validation_epoch", 0)
    arguments.setdefault("best_validation_checkpoint", "")
    arguments.setdefault("latest_validation", {})
    arguments.setdefault("last_validated_iteration", 0)

    data_loader = build_train_data_loader(
        cfg,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    train_data_stats = summarize_train_dataset(cfg, dataset=data_loader.dataset)
    logger.info(
        "Train dataset stats: split=%s clips=%d batch=%d steps_per_epoch=%d effective_epochs=%.4f",
        train_data_stats["train_split"],
        train_data_stats["train_dataset_num_clips"],
        train_data_stats["video_clips_per_batch"],
        train_data_stats["steps_per_epoch"],
        train_data_stats["effective_num_epochs"],
    )
    log_train_data_stats_to_mlflow(
        mlflow_logger,
        logger,
        train_data_stats,
        using_external_mlflow_run=using_external_mlflow_run,
        mlflow_stage_name=mlflow_stage_name,
        mlflow_stage_iter=mlflow_stage_iter,
    )

    validation_epoch_period = int(getattr(cfg.SOLVER, "VAL_EPOCH_PERIOD", 0))
    validation_target_metric = str(getattr(cfg.SOLVER, "VAL_TARGET_METRIC", "infer/mot/hota")).strip()
    validation_runner = None
    steps_per_epoch = int(train_data_stats["steps_per_epoch"])
    if validation_epoch_period > 0:
        if steps_per_epoch < 1:
            raise ValueError(
                "Epoch validation is enabled, but steps_per_epoch resolved to {}.".format(
                    steps_per_epoch
                )
            )
        validation_dir = os.path.join(train_dir, "validation")
        mkdir(validation_dir)
        runner = ValidationRunner(cfg, validation_dir, logger=logger)
        validation_model = model.module if hasattr(model, "module") else model
        validation_runner = lambda iteration, epoch: runner(validation_model, iteration, epoch)
        logger.info(
            "Epoch validation enabled: every %d epoch(s) on %s/%s using %s",
            validation_epoch_period,
            runner.dataset_key,
            runner.split,
            runner.eval_metric,
        )
        if mlflow_logger is not None and mlflow_logger.can_log:
            mlflow_logger.log_params(
                {
                    "val.dataset_key": runner.dataset_key,
                    "val.split": runner.split,
                    "val.epoch_period": validation_epoch_period,
                    "val.eval_metric": runner.eval_metric,
                    "val.target_metric": validation_target_metric,
                }
            )
    else:
        logger.info("Epoch validation disabled (SOLVER.VAL_EPOCH_PERIOD=%d)", validation_epoch_period)

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    tensorboard_writer = TensorboardWriter(cfg, train_dir)

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        logger,
        tensorboard_writer,
        mlflow_logger=mlflow_logger,
        mlflow_log_every_n_steps=cfg.MLFLOW.LOG_EVERY_N_STEPS,
        mlflow_log_checkpoints=cfg.MLFLOW.LOG_MODEL_CHECKPOINTS,
        validation_runner=validation_runner,
        steps_per_epoch=steps_per_epoch,
        validation_epoch_period=validation_epoch_period,
        validation_target_metric=validation_target_metric,
    )

    validation_state = {
        "history": arguments.get("validation_history", []),
        "best_metric_name": validation_target_metric,
        "best_metric_value": arguments.get("best_validation_metric"),
        "best_iteration": arguments.get("best_validation_iteration"),
        "best_epoch": arguments.get("best_validation_epoch"),
        "best_checkpoint": arguments.get("best_validation_checkpoint"),
        "latest": arguments.get("latest_validation", {}),
    }

    return model, train_data_stats, validation_state


def setup_env_and_logger(
    args: argparse.Namespace,
    cfg: CfgNode,
) -> Tuple[str, logging.Logger]:
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    model_name = get_model_name(cfg, args.model_suffix)
    train_dir = os.path.join(args.train_dir, model_name)
    if train_dir:
        mkdir(train_dir)

    logger = setup_logger("siammot", train_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(train_dir, "config.yml")
    logger.info("Saving config into: {}".format(output_config_path))
    save_config(cfg, output_config_path)

    return train_dir, logger


def main() -> None:
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    train_dir, logger = setup_env_and_logger(args, cfg)
    model_name = os.path.basename(train_dir)
    attached_mlflow_run_id = str(args.mlflow_run_id).strip()
    using_external_mlflow_run = bool(attached_mlflow_run_id)
    mlflow_logger = MLflowLogger(
        cfg,
        logger,
        artifact_path_prefix=str(args.mlflow_artifact_subdir).strip(),
    )
    run_status = "FINISHED"

    try:
        mlflow_run_name = (
            cfg.MLFLOW.TRAIN_RUN_NAME if cfg.MLFLOW.TRAIN_RUN_NAME else model_name
        )
        mlflow_tags: Dict[str, str] = {
            "stage": "train",
            "model_name": model_name,
        }
        mlflow_tags.update(parse_mlflow_tags(args.extra_mlflow_tags))
        if using_external_mlflow_run:
            mlflow_logger.start_run(
                experiment_name=cfg.MLFLOW.EXPERIMENT_NAME,
                run_id=attached_mlflow_run_id,
                manage_lifecycle=False,
            )
        else:
            mlflow_logger.start_run(
                experiment_name=cfg.MLFLOW.EXPERIMENT_NAME,
                run_name=mlflow_run_name,
                tags=mlflow_tags,
            )

            mlflow_logger.log_params(
                {
                    "config_file": args.config_file,
                    "train_dir": train_dir,
                    "model_suffix": args.model_suffix,
                    "distributed": args.distributed,
                    "dtype": cfg.DTYPE,
                    "num_train_datasets": len(cfg.DATASETS.TRAIN),
                    "train_split": str(getattr(cfg.DATASETS, "TRAIN_SET", "train")),
                }
            )
            mlflow_logger.log_cfg_params(cfg)

            if cfg.MLFLOW.LOG_CONFIG_ARTIFACT:
                mlflow_logger.log_artifact(
                    os.path.join(train_dir, "config.yml"), artifact_path="configs"
                )

        if mlflow_logger.run_id:
            run_id_file = os.path.join(train_dir, "mlflow_run_id.txt")
            with open(run_id_file, "w") as f:
                f.write(mlflow_logger.run_id + "\n")
            mlflow_logger.log_artifact(run_id_file, artifact_path="metadata")

        _model, train_data_stats, validation_state = train(
            cfg,
            train_dir,
            args.local_rank,
            args.distributed,
            logger,
            mlflow_logger=mlflow_logger,
            using_external_mlflow_run=using_external_mlflow_run,
            mlflow_stage_name=args.mlflow_stage_name,
            mlflow_stage_iter=args.mlflow_stage_iter,
        )

        run_info: Dict[str, Any] = {
            "model_name": model_name,
            "train_dir": train_dir,
            "final_checkpoint": os.path.join(train_dir, "model_final.pth"),
            "mlflow_run_id": mlflow_logger.run_id,
            "train_data_stats": train_data_stats,
            "validation": validation_state,
        }
        run_info_path = os.path.join(train_dir, "run_info.json")
        with open(run_info_path, "w") as f:
            json.dump(run_info, f, indent=2, sort_keys=True)
        if args.run_info_file:
            with open(args.run_info_file, "w") as f:
                json.dump(run_info, f, indent=2, sort_keys=True)
        mlflow_logger.log_artifact(run_info_path, artifact_path="metadata")
    except Exception:
        run_status = "FAILED"
        logger.error("Training failed:\n%s", traceback.format_exc())
        raise
    finally:
        mlflow_logger.end_run(status=run_status)


if __name__ == "__main__":
    main()
