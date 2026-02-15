import logging
import os
import tempfile
from typing import Any, Dict, Optional

import torch.distributed as dist


def _is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, bool, str))


def _flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) or hasattr(value, "items"):
            flat.update(_flatten_dict(value, full_key))
        else:
            flat[full_key] = value
    return flat


class MLflowLogger(object):
    def __init__(self, cfg, logger: Optional[logging.Logger] = None):
        self._cfg = cfg
        self._logger = logger or logging.getLogger(__name__)
        self._enabled = bool(getattr(cfg, "MLFLOW", None) and cfg.MLFLOW.ENABLED)
        self._can_log = self._enabled and _is_main_process()
        self._active_run = False
        self._mlflow = None
        self._run_id = None

        if self._enabled:
            try:
                import mlflow
            except ImportError as exc:
                raise ImportError(
                    "MLflow tracking is enabled, but mlflow is not installed. "
                    "Install dependencies from requirements.txt."
                ) from exc
            self._mlflow = mlflow

            tracking_uri = str(self._cfg.MLFLOW.TRACKING_URI).strip()
            if tracking_uri:
                self._mlflow.set_tracking_uri(tracking_uri)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def can_log(self) -> bool:
        return self._can_log

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id

    def start_run(self, experiment_name: str, run_name: str = "", tags: Optional[Dict[str, str]] = None):
        if not self._can_log:
            return

        self._mlflow.set_experiment(experiment_name)
        active_run = self._mlflow.start_run(run_name=(run_name or None), tags=tags)
        self._active_run = True
        self._run_id = active_run.info.run_id
        self._logger.info("MLflow run started: %s", self._run_id)

    def end_run(self, status: str = "FINISHED"):
        if not self._can_log or not self._active_run:
            return
        self._mlflow.end_run(status=status)
        self._active_run = False

    def set_tags(self, tags: Dict[str, str]):
        if not self._can_log or not self._active_run:
            return
        self._mlflow.set_tags(tags)

    def log_param(self, key: str, value: Any):
        if not self._can_log or not self._active_run:
            return
        self._mlflow.log_param(key, str(value))

    def log_params(self, params: Dict[str, Any]):
        if not self._can_log or not self._active_run:
            return
        parsed = {}
        for key, value in params.items():
            if value is None:
                continue
            parsed[key] = value if _is_scalar(value) else str(value)
        if parsed:
            self._mlflow.log_params(parsed)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        if not self._can_log or not self._active_run:
            return
        if value is None:
            return
        self._mlflow.log_metric(key, float(value), step=step)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if not self._can_log or not self._active_run:
            return
        parsed = {}
        for key, value in metrics.items():
            if value is None:
                continue
            try:
                parsed[key] = float(value)
            except (TypeError, ValueError):
                continue
        if parsed:
            self._mlflow.log_metrics(parsed, step=step)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None):
        if not self._can_log or not self._active_run:
            return
        if not os.path.exists(path):
            self._logger.warning("MLflow artifact path does not exist: %s", path)
            return
        self._mlflow.log_artifact(path, artifact_path=artifact_path)

    def log_artifacts(self, path: str, artifact_path: Optional[str] = None):
        if not self._can_log or not self._active_run:
            return
        if not os.path.isdir(path):
            self._logger.warning("MLflow artifacts directory does not exist: %s", path)
            return
        self._mlflow.log_artifacts(path, artifact_path=artifact_path)

    def log_config(self, cfg, artifact_file: str = "config.yml"):
        if not self._can_log or not self._active_run:
            return
        with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as tmp:
            tmp.write(cfg.dump())
            tmp_path = tmp.name
        try:
            self.log_artifact(tmp_path, artifact_path=os.path.dirname(artifact_file) or None)
        finally:
            os.remove(tmp_path)

    def log_cfg_params(self, cfg, prefix: str = "cfg"):
        if not self._can_log or not self._active_run:
            return
        flat_cfg = _flatten_dict(cfg)
        params = {}
        for key, value in flat_cfg.items():
            if _is_scalar(value):
                params[f"{prefix}.{key}"] = value
            elif isinstance(value, (list, tuple)):
                params[f"{prefix}.{key}"] = ",".join([str(v) for v in value])
            else:
                params[f"{prefix}.{key}"] = str(value)
        self.log_params(params)
