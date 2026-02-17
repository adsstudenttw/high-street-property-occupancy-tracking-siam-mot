import os
import logging
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from gluoncv.torch.data.gluoncv_motion_dataset.dataset import DataSample
from yacs.config import CfgNode

from ..data.build_inference_data_loader import build_video_loader
from ..data.adapters.augmentation.build_augmentation import build_siam_augmentation
from ..utils.boxlists_to_entities import boxlists_to_entities, convert_given_detections_to_boxlist
from ..eval.eval_clears_mot import eval_clears_mot
from ..eval.eval_hota import eval_hota

MetricsMap = Dict[str, float]
InferenceResult = Dict[str, Any]
DatasetEntries = Sequence[Tuple[Any, DataSample]]


def do_inference(
    cfg: CfgNode,
    model: torch.nn.Module,
    sample: DataSample,
    transforms: Optional[Callable[..., Any]] = None,
    given_detection: Optional[DataSample] = None,
) -> DataSample:
    """
    Do inference on a specific video (sample)
    :param cfg: configuration file of the model
    :param model: a pytorch model
    :param sample: a testing video
    :param transforms: image-wise transform that prepares
           video frames for processing
    :param given_detection: the cached detections from other model,
           it means that the detection branch is disabled in the
           model forward pass
    :return: the detection results in the format of DataSample
    """
    logger = logging.getLogger(__name__)
    model.eval()
    gpu_device = torch.device('cuda')

    video_loader = build_video_loader(cfg, sample, transforms)

    sample_result = DataSample(sample.id, raw_info=None, metadata=sample.metadata)
    network_time = 0.0
    for (video_clip, frame_id, timestamps) in tqdm(video_loader):
        frame_id = frame_id.item()
        timestamps = torch.squeeze(timestamps, dim=0).tolist()
        video_clip = torch.squeeze(video_clip, dim=0)

        frame_detection = None
        # used the public provided detection (e.g. MOT17, HiEve)
        # the public detection needs to be ingested to DataSample
        # the ingested detection has been provided, find the details in readme/DATA.md
        if given_detection:
            frame_detection = given_detection.get_entities_for_frame_num(frame_id)
            frame_detection = convert_given_detections_to_boxlist(frame_detection,
                                                                  sample.width,
                                                                  sample.height)
            frame_height, frame_width = video_clip.shape[-2:]
            frame_detection = frame_detection.resize((frame_width, frame_height))
            frame_detection = [frame_detection.to(gpu_device)]

        with torch.no_grad():
            video_clip = video_clip.to(gpu_device)
            torch.cuda.synchronize()
            network_start_time = time.time()
            output_boxlists= model(video_clip, given_detection=frame_detection)
            torch.cuda.synchronize()
            network_time += time.time() - network_start_time

        # Resize to original image size and to xywh mode
        output_boxlists = [o.resize([sample.width, sample.height]).convert('xywh')
                           for o in output_boxlists]
        output_boxlists = [o.to(torch.device("cpu")) for o in output_boxlists]
        output_entities = boxlists_to_entities(output_boxlists, frame_id, timestamps)
        for entity in output_entities:
            sample_result.add_entity(entity)

    logger.info('Sample_id {} / Speed {} fps'.format(sample.id, len(sample) / (network_time)))

    return sample_result


class DatasetInference(object):
    def __init__(
        self,
        cfg: CfgNode,
        model: torch.nn.Module,
        dataset: DatasetEntries,
        output_dir: str,
        data_filter_fn: Optional[Callable[..., Any]] = None,
        public_detection: Optional[Mapping[Any, DataSample]] = None,
        distributed: bool = False,
    ) -> None:

        self._cfg = cfg

        self._transform = build_siam_augmentation(cfg, is_train=False)
        self._model = model
        self._dataset = dataset
        self._output_dir = output_dir
        self._distributed = distributed
        self._data_filter_fn = data_filter_fn
        self._pub_detection = public_detection
        self._track_conf = cfg.INFERENCE.TRACK_SCORE_THRESH
        self._track_len = cfg.INFERENCE.MIN_TRACK_LENGTH
        self._logger = logging.getLogger(__name__)

        self.results: Dict[Any, DataSample] = {}

    def _eval_det_ap(self) -> Tuple[np.ndarray, str]:
        from ..eval.eval_det_ap import eval_det_ap
        iou_threshold = np.arange(0.5, 0.95, 0.05).tolist()
        ap_matrix = eval_det_ap(self._dataset, self.results,
                                data_filter_fn=self._data_filter_fn,
                                iou_threshold=iou_threshold)
        ap = np.mean(ap_matrix, axis=0)

        ap_str_summary = "\n"
        ap_str_summary += 'Detection AP @[ IoU=0.50:0.95 ] = {:.2f}\n'.format(np.mean(ap) * 100)
        ap_str_summary += 'Detection AP @[ IoU=0.50 ] = {:.2f}\n'.format(ap[0] * 100)
        ap_str_summary += 'Detection AP @[ IoU=0.75 ] = {:.2f}\n'.format(ap[5] * 100)

        return ap, ap_str_summary

    def _eval_clear_mot(self) -> Tuple[Any, str, MetricsMap]:

        motmetric, motstrsummary, overall_metrics = eval_clears_mot(
            self._dataset, self.results, data_filter_fn=self._data_filter_fn
        )
        return motmetric, motstrsummary, overall_metrics

    def _eval_hota(self) -> Tuple[str, MetricsMap]:
        hota_summary, overall_metrics = eval_hota(
            self._dataset,
            self.results,
            data_filter_fn=self._data_filter_fn,
            keep_debug_files=bool(self._cfg.INFERENCE.HOTA_KEEP_DEBUG_FILES),
            debug_dir=str(self._cfg.INFERENCE.HOTA_DEBUG_DIR).strip() or None,
        )
        return hota_summary, overall_metrics

    def _eval_mode(self) -> str:
        eval_metric = str(self._cfg.INFERENCE.EVAL_METRIC).strip().lower()
        valid_modes = {"clear", "hota", "both"}
        if eval_metric not in valid_modes:
            raise ValueError(
                "Invalid INFERENCE.EVAL_METRIC '{}'. Supported values: clear, hota, both".format(
                    self._cfg.INFERENCE.EVAL_METRIC
                )
            )
        return eval_metric

    def _inference_on_video(self, sample: DataSample) -> DataSample:
        cache_path = os.path.join(self._output_dir, '{}.json'.format(sample.id))
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        if os.path.exists(cache_path):
            sample_result = DataSample.load(cache_path)
        else:
            given_detection = None
            if self._pub_detection:
                given_detection = self._pub_detection[sample.id]
            sample_result = do_inference(self._cfg, self._model, sample,
                                         transforms=self._transform,
                                         given_detection=given_detection
                                         )
            sample_result.dump(cache_path)
        return sample_result

    def _postprocess_tracks(self, tracks: DataSample) -> DataSample:
        """
        post_process the tracks to filter out short and non-confident tracks
        :param tracks: un-filtered tracks
        :return: filtered tracks that would be used for evaluation
        """
        track_ids = set()
        for _entity in tracks.entities:
            if _entity.id not in track_ids and _entity.id >= 0:
                track_ids.add(_entity.id)

        filter_tracks = tracks.get_copy_without_entities()
        for _id in track_ids:
            _id_entities = tracks.get_entities_with_id(_id)
            _track_conf = np.mean([_e.confidence for _e in _id_entities])
            if len(_id_entities) >= self._track_len \
                    and _track_conf >= self._track_conf:
                for _entity in _id_entities:
                    filter_tracks.add_entity(_entity)
        return filter_tracks

    def __call__(self) -> InferenceResult:
        # todo: enable the inference in an efficient distributed framework
        start_time = time.time()
        total_frames = 0
        for (_, sample) in tqdm(self._dataset):
            # clean up the memory
            self._model.reset_siammot_status()
            total_frames += len(sample)

            sample_result = self._inference_on_video(sample)

            sample_result = self._postprocess_tracks(sample_result)
            self.results.update({sample.id: sample_result})

        self._logger.info("\n---------------- Start evaluating ----------------\n")
        eval_mode = self._eval_mode()
        clear_summary = ""
        clear_metrics: MetricsMap = {}
        hota_summary = ""
        hota_metrics: MetricsMap = {}

        if eval_mode in {"clear", "both"}:
            _, clear_summary, clear_metrics = self._eval_clear_mot()
            self._logger.info(clear_summary)
        if eval_mode in {"hota", "both"}:
            hota_summary, hota_metrics = self._eval_hota()
            self._logger.info(hota_summary)

        # ap, ap_str_summary = self._eval_det_ap()
        # self._logger.info(ap_str_summary)
        self._logger.info("\n---------------- Finish evaluating ----------------\n")

        inference_time = time.time() - start_time
        metrics: MetricsMap = {}
        metrics.update({f"infer/mot/{k}": v for k, v in clear_metrics.items()})
        metrics.update({f"infer/mot/hota/{k.lower()}": v for k, v in hota_metrics.items()})
        if "HOTA" in hota_metrics:
            metrics["infer/mot/hota"] = hota_metrics["HOTA"]
        metrics["infer/total_frames"] = float(total_frames)
        metrics["infer/total_time_sec"] = float(inference_time)
        if inference_time > 0:
            metrics["infer/fps"] = float(total_frames) / float(inference_time)
        metrics["infer/postprocess/track_score_thresh"] = float(self._track_conf)
        metrics["infer/postprocess/min_track_length"] = float(self._track_len)

        summary_parts = []
        if clear_summary.strip():
            summary_parts.append("CLEAR MOT\n{}".format(clear_summary.strip()))
        if hota_summary.strip():
            summary_parts.append("HOTA\n{}".format(hota_summary.strip()))
        eval_summary = "\n\n".join(summary_parts)

        return {
            "metrics": metrics,
            "eval_summary": eval_summary,
            "mot_summary": clear_summary,
            "hota_summary": hota_summary,
        }
