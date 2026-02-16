"""
Basic testing script for PyTorch
Only support single-gpu now
"""
import argparse
import json
import os
import traceback
from typing import Any, Dict, Optional, cast

import torch

from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from siammot.configs.defaults import cfg
from siammot.modelling.rcnn import build_siammot
from siammot.engine.inferencer import DatasetInference
from siammot.utils.get_model_name import get_model_name
from siammot.data.adapters.utils.data_utils import load_dataset_anno, load_public_detection
from siammot.data.adapters.handler.data_filtering import build_data_filter_fn
from siammot.engine.mlflow_logger import MLflowLogger
from yacs.config import CfgNode

try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

parser = argparse.ArgumentParser(description="PyTorch Video Object Detection Inference")
parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file", type=str)
parser.add_argument("--output-dir", default="", help="path to output folder", type=str)
parser.add_argument("--model-file", default=None, metavar="FILE", help="path to model file", type=str)
parser.add_argument("--test-dataset", default="MOT17_DPM", type=str)
parser.add_argument("--set", default="test", type=str)
parser.add_argument("--gpu-id", default=0, type=int)
parser.add_argument("--num-gpus", default=1, type=int)
parser.add_argument("--parent-run-id", default="", type=str)
parser.add_argument("--metrics-file", default="", type=str, help="optional path to dump metrics json")
parser.add_argument(
    "--opts",
    nargs=argparse.REMAINDER,
    default=[],
    help="modify config options using the command-line",
)


def test(
    cfg: CfgNode,
    args: argparse.Namespace,
    output_dir: str,
) -> Dict[str, Any]:

    torch.cuda.empty_cache()

    # Construct model graph
    model = build_siammot(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    # Load model params
    model_file: Optional[str] = args.model_file
    if model_file is None:
        raise KeyError("No checkpoint is found")
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=model_file)
    if os.path.isfile(model_file):
        _ = checkpointer.load(model_file)
    elif os.path.isdir(model_file):
        _ = checkpointer.load(use_latest=True)
    else:
        raise KeyError("No checkpoint is found")

    # Load testing dataset
    dataset_key = args.test_dataset
    dataset, _dataset_info = load_dataset_anno(cfg, dataset_key, args.set)
    dataset = sorted(dataset)

    # do inference on dataset
    data_filter_fn = build_data_filter_fn(dataset_key)

    # load public detection
    public_detection: Optional[Any] = None
    if cfg.INFERENCE.USE_GIVEN_DETECTIONS:
        public_detection = load_public_detection(cfg, dataset_key)

    dataset_inference = DatasetInference(cfg, model, dataset, output_dir, data_filter_fn, public_detection)
    return dataset_inference()


def main() -> None:
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    model_name = get_model_name(cfg)
    output_dir = os.path.join(args.output_dir, model_name)
    if not os.path.exists(output_dir):
        mkdir(output_dir)

    mlflow_logger = MLflowLogger(cfg)
    run_status = "FINISHED"

    try:
        mlflow_run_name = cfg.MLFLOW.INFERENCE_RUN_NAME if cfg.MLFLOW.INFERENCE_RUN_NAME else model_name
        mlflow_tags: Dict[str, str] = {
            "stage": "inference",
            "model_name": model_name,
        }
        if args.parent_run_id:
            mlflow_tags["parent_train_run_id"] = args.parent_run_id

        mlflow_logger.start_run(
            experiment_name=cfg.MLFLOW.EXPERIMENT_NAME,
            run_name=mlflow_run_name,
            tags=mlflow_tags,
        )

        mlflow_logger.log_params(
            {
                "config_file": args.config_file,
                "output_dir": output_dir,
                "model_file": args.model_file,
                "test_dataset": args.test_dataset,
                "split": args.set,
                "use_given_detections": cfg.INFERENCE.USE_GIVEN_DETECTIONS,
            }
        )
        mlflow_logger.log_cfg_params(cfg)

        infer_results = test(cfg, args, output_dir)
        infer_metrics = cast(Dict[str, float], infer_results.get("metrics", {}))
        mlflow_logger.log_metrics(infer_metrics)

        metrics_payload = {
            "metrics": infer_metrics,
            "config_file": args.config_file,
            "model_file": args.model_file,
            "dataset": args.test_dataset,
            "split": args.set,
        }
        infer_metrics_path = os.path.join(output_dir, "inference_metrics.json")
        with open(infer_metrics_path, "w") as f:
            json.dump(metrics_payload, f, indent=2, sort_keys=True)
        if args.metrics_file:
            with open(args.metrics_file, "w") as f:
                json.dump(metrics_payload, f, indent=2, sort_keys=True)
        mlflow_logger.log_artifact(infer_metrics_path, artifact_path="evaluation")

        summary_text = cast(str, infer_results.get("mot_summary", ""))
        if summary_text:
            summary_path = os.path.join(output_dir, "mot_summary.txt")
            with open(summary_path, "w") as f:
                f.write(summary_text)
            mlflow_logger.log_artifact(summary_path, artifact_path="evaluation")

        if cfg.MLFLOW.LOG_INFERENCE_OUTPUTS:
            mlflow_logger.log_artifacts(output_dir, artifact_path="inference_outputs")
    except Exception:
        run_status = "FAILED"
        print("Inference failed:\n{}".format(traceback.format_exc()))
        raise
    finally:
        mlflow_logger.end_run(status=run_status)


if __name__ == "__main__":
    main()
