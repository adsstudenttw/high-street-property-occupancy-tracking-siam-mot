import datetime
import logging
import os
import time
from apex import amp
import torch.distributed as dist

from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.utils.comm import get_world_size

from .tensorboard_writer import TensorboardWriter


def do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        logger,
        tensorboard_writer: TensorboardWriter = None,
        mlflow_logger=None,
        mlflow_log_every_n_steps=20,
        mlflow_log_checkpoints=True,
):
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

        result, loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
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
                train_metrics = {
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
