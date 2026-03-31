import json
import logging
import os
import shutil
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, cast

import torch
from yacs.config import CfgNode

from siammot.data.adapters.handler.data_filtering import build_data_filter_fn
from siammot.data.adapters.utils.data_utils import load_dataset_anno, load_public_detection

from .inferencer import DatasetInference

ValidationResult = Dict[str, Any]


def _resolve_validation_dataset_key(cfg: CfgNode) -> str:
    raw_val_datasets = getattr(cfg.DATASETS, "VAL", ())
    if raw_val_datasets:
        dataset_keys = tuple(raw_val_datasets)
    else:
        dataset_keys = tuple(cfg.DATASETS.TRAIN)

    if len(dataset_keys) != 1:
        raise ValueError(
            "Epoch validation currently requires exactly one validation dataset. "
            "Resolved DATASETS.VAL={} from config.".format(dataset_keys)
        )
    return str(dataset_keys[0])


def _write_text_if_present(path: str, content: str) -> Optional[str]:
    if not str(content or "").strip():
        return None
    with open(path, "w") as f:
        f.write(content)
    return path


class ValidationRunner(object):
    def __init__(
        self,
        cfg: CfgNode,
        output_dir: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        eval_cfg = cfg.clone()
        eval_cfg.defrost()
        eval_cfg.INFERENCE.EVAL_METRIC = str(cfg.SOLVER.VAL_EVAL_METRIC)
        eval_cfg.freeze()

        self._cfg = eval_cfg
        self._output_dir = output_dir
        self._logger = logger or logging.getLogger(__name__)
        self._dataset_key = _resolve_validation_dataset_key(eval_cfg)
        self._split = str(getattr(eval_cfg.DATASETS, "VAL_SET", "val")).strip().lower()
        self._data_filter_fn = build_data_filter_fn(self._dataset_key)

        dataset, _dataset_info = load_dataset_anno(eval_cfg, self._dataset_key, self._split)
        self._dataset = sorted(dataset)

        public_detection: Optional[Mapping[Any, Any]] = None
        if eval_cfg.INFERENCE.USE_GIVEN_DETECTIONS:
            public_detection = cast(Optional[Mapping[Any, Any]], load_public_detection(eval_cfg, self._dataset_key))
        self._public_detection = public_detection

    @property
    def dataset_key(self) -> str:
        return self._dataset_key

    @property
    def split(self) -> str:
        return self._split

    @property
    def eval_metric(self) -> str:
        return str(self._cfg.INFERENCE.EVAL_METRIC)

    def __call__(
        self,
        model: torch.nn.Module,
        iteration: int,
        epoch: int,
    ) -> ValidationResult:
        if str(self._cfg.MODEL.DEVICE).strip().lower().startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

        run_output_dir = os.path.join(
            self._output_dir,
            "epoch_{:04d}_iter_{:07d}".format(epoch, iteration),
        )
        if os.path.isdir(run_output_dir):
            shutil.rmtree(run_output_dir)
        os.makedirs(run_output_dir, exist_ok=True)

        dataset_inference = DatasetInference(
            self._cfg,
            model,
            self._dataset,
            run_output_dir,
            self._data_filter_fn,
            self._public_detection,
        )
        infer_results = dataset_inference()
        infer_metrics = cast(Dict[str, float], infer_results.get("metrics", {}))

        metrics_payload: ValidationResult = {
            "epoch": int(epoch),
            "iteration": int(iteration),
            "dataset_key": self._dataset_key,
            "split": self._split,
            "eval_metric": self.eval_metric,
            "metrics": infer_metrics,
            "output_dir": run_output_dir,
            "metrics_file": "",
            "artifacts": {},
        }
        metrics_path = os.path.join(run_output_dir, "validation_metrics.json")
        metrics_payload["metrics_file"] = metrics_path

        summary_paths: Sequence[Tuple[str, str]] = (
            ("eval_summary.txt", cast(str, infer_results.get("eval_summary", ""))),
            ("mot_summary.txt", cast(str, infer_results.get("mot_summary", ""))),
            ("hota_summary.txt", cast(str, infer_results.get("hota_summary", ""))),
        )
        artifact_paths: Dict[str, str] = {}
        for filename, content in summary_paths:
            written_path = _write_text_if_present(os.path.join(run_output_dir, filename), content)
            if written_path is not None:
                artifact_paths[filename] = written_path

        metrics_payload["artifacts"] = artifact_paths
        with open(metrics_path, "w") as f:
            json.dump(metrics_payload, f, indent=2, sort_keys=True)
        self._logger.info(
            "Validation finished: epoch=%d iter=%d dataset=%s split=%s metrics=%s",
            epoch,
            iteration,
            self._dataset_key,
            self._split,
            infer_metrics,
        )
        return metrics_payload
