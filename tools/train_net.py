import argparse
import logging
import json
import os
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
from siammot.data.build_train_data_loader import build_train_data_loader
from siammot.modelling.rcnn import build_siammot
from siammot.engine.trainer import do_train
from siammot.utils.get_model_name import get_model_name
from siammot.engine.tensorboard_writer import TensorboardWriter
from siammot.engine.mlflow_logger import MLflowLogger
from yacs.config import CfgNode


try:
    from apex import amp
except ImportError:
    raise ImportError("Use APEX for multi-precision via apex.amp")


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
    "--opts",
    nargs=argparse.REMAINDER,
    default=[],
    help="modify config options using the command-line",
)


def train(
    cfg: CfgNode,
    train_dir: str,
    local_rank: int,
    distributed: bool,
    logger: logging.Logger,
    mlflow_logger: Optional[MLflowLogger] = None,
) -> torch.nn.Module:

    # build model
    model = build_siammot(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = "O1" if use_mixed_precision else "O0"
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

    data_loader = build_train_data_loader(
        cfg,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

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
    )

    return model


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
    mlflow_logger = MLflowLogger(cfg, logger)
    run_status = "FINISHED"

    try:
        mlflow_run_name = (
            cfg.MLFLOW.TRAIN_RUN_NAME if cfg.MLFLOW.TRAIN_RUN_NAME else model_name
        )
        mlflow_tags: Dict[str, str] = {
            "stage": "train",
            "model_name": model_name,
        }
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

        train(
            cfg,
            train_dir,
            args.local_rank,
            args.distributed,
            logger,
            mlflow_logger=mlflow_logger,
        )

        run_info: Dict[str, Optional[str]] = {
            "model_name": model_name,
            "train_dir": train_dir,
            "final_checkpoint": os.path.join(train_dir, "model_final.pth"),
            "mlflow_run_id": mlflow_logger.run_id,
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
