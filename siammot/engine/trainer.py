import datetime
import logging
import math
import os
import re
import time
from typing import Any, Callable, Dict, Iterable, MutableMapping, Optional, Sequence, Tuple

try:
    from apex import amp
except ImportError:
    amp = None
import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.utils.comm import get_world_size

from .mlflow_logger import MLflowLogger
from .tensorboard_writer import TensorboardWriter


def _format_validation_metric_name(metric_name: str) -> str:
    clean_name = str(metric_name).strip().strip("/")
    if clean_name.startswith("infer/"):
        return "val/{}".format(clean_name[len("infer/"):])
    return "val/{}".format(clean_name)


def _checkpoint_metric_slug(metric_name: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", str(metric_name).strip().lower()).strip("_")
    return slug or "metric"


def _resolved_epoch(iteration: int, steps_per_epoch: int) -> int:
    if steps_per_epoch <= 0:
        return 0
    return int(math.ceil(float(iteration) / float(steps_per_epoch)))


def _run_validation(
    *,
    model: torch.nn.Module,
    checkpointer: Any,
    iteration: int,
    max_iter: int,
    steps_per_epoch: int,
    validation_epoch_period: int,
    validation_target_metric: str,
    validation_runner: Callable[[int, int], Dict[str, Any]],
    arguments: MutableMapping[str, Any],
    logger: logging.Logger,
    mlflow_logger: Optional[MLflowLogger] = None,
) -> None:
    if steps_per_epoch <= 0 or validation_epoch_period <= 0:
        return

    epoch = _resolved_epoch(iteration, steps_per_epoch)
    last_validated_iteration = int(arguments.get("last_validated_iteration", 0) or 0)
    is_epoch_boundary = iteration % steps_per_epoch == 0
    should_validate = False
    if iteration == max_iter and iteration != last_validated_iteration:
        should_validate = True
    elif is_epoch_boundary and epoch % validation_epoch_period == 0 and iteration != last_validated_iteration:
        should_validate = True

    if not should_validate:
        return

    if get_world_size() > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()

    payload: Optional[Dict[str, Any]] = None
    if get_world_size() < 2 or dist.get_rank() == 0:
        logger.info("Start validation at epoch=%d iter=%d", epoch, iteration)
        payload = validation_runner(iteration, epoch)
        metrics = payload.get("metrics", {})
        target_value = metrics.get(validation_target_metric)
        if target_value is None:
            logger.warning(
                "Validation metrics at epoch=%d iter=%d did not include target metric '%s'. "
                "Available metrics: %s",
                epoch,
                iteration,
                validation_target_metric,
                sorted(metrics.keys()),
            )

        history = list(arguments.get("validation_history", []))
        history_entry = {
            "epoch": int(epoch),
            "iteration": int(iteration),
            "target_metric": validation_target_metric,
            "target_value": float(target_value) if target_value is not None else None,
            "dataset_key": payload.get("dataset_key"),
            "split": payload.get("split"),
            "metrics": dict(metrics),
            "output_dir": payload.get("output_dir"),
            "metrics_file": payload.get("metrics_file"),
        }
        history.append(history_entry)
        arguments["validation_history"] = history
        arguments["latest_validation"] = history_entry
        arguments["last_validated_iteration"] = int(iteration)

        if mlflow_logger is not None and mlflow_logger.can_log:
            validation_metrics: Dict[str, float] = {
                "val/epoch": float(epoch),
            }
            for metric_name, metric_value in metrics.items():
                try:
                    validation_metrics[_format_validation_metric_name(metric_name)] = float(metric_value)
                except (TypeError, ValueError):
                    continue
            if target_value is not None:
                validation_metrics["val/objective"] = float(target_value)
            best_metric = arguments.get("best_validation_metric")
            if best_metric is not None:
                validation_metrics["val/best_objective_so_far"] = float(best_metric)
            mlflow_logger.log_metrics(validation_metrics, step=iteration)

        best_metric = arguments.get("best_validation_metric")
        is_improved = (
            target_value is not None
            and (best_metric is None or float(target_value) > float(best_metric))
        )
        if is_improved:
            metric_slug = _checkpoint_metric_slug(validation_target_metric)
            best_checkpoint_name = "model_best_{}".format(metric_slug)
            arguments["best_validation_metric"] = float(target_value)
            arguments["best_validation_iteration"] = int(iteration)
            arguments["best_validation_epoch"] = int(epoch)
            arguments["best_validation_checkpoint"] = os.path.join(
                checkpointer.save_dir, "{}.pth".format(best_checkpoint_name)
            )
            checkpointer.save(best_checkpoint_name, **arguments)
            if mlflow_logger is not None and mlflow_logger.can_log:
                best_checkpoint_path = arguments["best_validation_checkpoint"]
                mlflow_logger.log_artifact(best_checkpoint_path, artifact_path="checkpoints")
                mlflow_logger.log_metric("val/best_objective_so_far", float(target_value), step=iteration)
            logger.info(
                "Validation improved: %s=%.6f at epoch=%d iter=%d",
                validation_target_metric,
                float(target_value),
                epoch,
                iteration,
            )

    if get_world_size() > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()

    model.train()


def do_train(
    model: torch.nn.Module,
    data_loader: Iterable[Tuple[Any, Sequence[Any], Any]],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    checkpointer: Any,
    device: torch.device,
    checkpoint_period: int,
    arguments: MutableMapping[str, Any],
    logger: logging.Logger,
    tensorboard_writer: Optional[TensorboardWriter] = None,
    mlflow_logger: Optional[MLflowLogger] = None,
    mlflow_log_every_n_steps: int = 20,
    mlflow_log_checkpoints: bool = True,
    validation_runner: Optional[Callable[[int, int], Dict[str, Any]]] = None,
    steps_per_epoch: int = 0,
    validation_epoch_period: int = 0,
    validation_target_metric: str = "infer/mot/hota",
) -> None:
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):

        if any(len(target) < 1 for target in targets):
            logger.error(
                "Iteration={iteration + 1} || Image Ids used for training {_} || "
                "targets Length={[len(target) for target in targets]}")
            continue

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        _, loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        if amp is not None:
            # Note: If mixed precision is not used, this ends up doing nothing
            # Otherwise apply loss scaling for mixed-precision recipe
            with amp.scale_loss(losses, optimizer) as scaled_losses:
                scaled_losses.backward()
        else:
            losses.backward()
        optimizer.step()

        # write images / ground truth / evaluation metrics to tensorboard
        if tensorboard_writer is not None:
            tensorboard_writer(iteration, losses_reduced, loss_dict_reduced, images, targets)

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if mlflow_logger is not None and mlflow_logger.can_log:
            if iteration == 1 or iteration % max(1, mlflow_log_every_n_steps) == 0 or iteration == max_iter:
                train_metrics: Dict[str, float] = {
                    "train/loss_total": losses_reduced.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/batch_time_sec": batch_time,
                    "train/data_time_sec": data_time,
                }
                for _loss_key, _val in loss_dict_reduced.items():
                    train_metrics[f"train/{_loss_key}"] = _val.item()
                mlflow_logger.log_metrics(train_metrics, step=iteration)

        if get_world_size() < 2 or dist.get_rank() == 0:
            if iteration % 20 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        )
                )
        if validation_runner is not None:
            _run_validation(
                model=model,
                checkpointer=checkpointer,
                iteration=iteration,
                max_iter=max_iter,
                steps_per_epoch=steps_per_epoch,
                validation_epoch_period=validation_epoch_period,
                validation_target_metric=validation_target_metric,
                validation_runner=validation_runner,
                arguments=arguments,
                logger=logger,
                mlflow_logger=mlflow_logger,
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
            if mlflow_logger is not None and mlflow_logger.can_log and mlflow_log_checkpoints:
                checkpoint_path = os.path.join(checkpointer.save_dir, "model_{:07d}.pth".format(iteration))
                mlflow_logger.log_artifact(checkpoint_path, artifact_path="checkpoints")
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
            if mlflow_logger is not None and mlflow_logger.can_log and mlflow_log_checkpoints:
                checkpoint_path = os.path.join(checkpointer.save_dir, "model_final.pth")
                mlflow_logger.log_artifact(checkpoint_path, artifact_path="checkpoints")

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    if mlflow_logger is not None and mlflow_logger.can_log:
        mlflow_logger.log_metrics(
            {
                "train/total_time_sec": total_training_time,
                "train/sec_per_iter": total_training_time / max_iter,
            }
        )
