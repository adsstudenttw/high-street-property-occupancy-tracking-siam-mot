import logging
import math
import os
import subprocess
import sys
import tempfile
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from gluoncv.torch.data.gluoncv_motion_dataset.dataset import DataSample

DatasetEntries = Sequence[Tuple[Any, DataSample]]
PredictedSamples = Mapping[Any, DataSample]
MetricsMap = Dict[str, float]

HOTA_BENCHMARK = "SIAMMOT"
HOTA_SPLIT = "eval"
HOTA_TRACKER_NAME = "siammot_tracker"
HOTA_CLASS_NAME = "pedestrian"

logger = logging.getLogger(__name__)
_VALID_DUPLICATE_POLICIES = {"error", "keep_first", "keep_highest_conf"}


def _as_positive_track_id(track_id: Any, id_map: Dict[str, int]) -> int:
    try:
        parsed = int(track_id)
    except (TypeError, ValueError):
        parsed = 0
    if parsed > 0:
        return parsed
    key = str(track_id)
    if key not in id_map:
        id_map[key] = len(id_map) + 1
    return id_map[key]


def _next_generated_track_id(sample: DataSample) -> int:
    max_positive_id = 0
    for entity in sample.entities:
        try:
            parsed = int(getattr(entity, "id", 0))
        except (TypeError, ValueError):
            continue
        if parsed > max_positive_id:
            max_positive_id = parsed
    return max_positive_id + 1


def _prediction_track_id(track_id: Any, next_track_id_state: Dict[str, int]) -> int:
    try:
        parsed = int(track_id)
    except (TypeError, ValueError):
        parsed = 0
    if parsed > 0:
        return parsed
    generated_id = next_track_id_state["next_id"]
    next_track_id_state["next_id"] += 1
    return generated_id


def _entity_bbox_xywh(entity: Any) -> Optional[Tuple[float, float, float, float]]:
    bbox = getattr(entity, "bbox", None)
    if bbox is None or len(bbox) < 4:
        return None
    try:
        x = float(bbox[0])
        y = float(bbox[1])
        w = float(bbox[2])
        h = float(bbox[3])
    except (TypeError, ValueError):
        return None
    return x, y, w, h


def _entity_confidence(entity: Any) -> float:
    conf = getattr(entity, "confidence", 1.0)
    try:
        return float(conf)
    except (TypeError, ValueError):
        return 1.0


def _normalize_policy(policy: str, entity_kind: str) -> str:
    normalized = str(policy).strip().lower()
    if normalized not in _VALID_DUPLICATE_POLICIES:
        raise ValueError(
            "Invalid HOTA duplicate {} policy '{}'. Supported values: {}.".format(
                entity_kind, policy, ", ".join(sorted(_VALID_DUPLICATE_POLICIES))
            )
        )
    return normalized


def _format_duplicate_error(
    entity_kind: str,
    sample_id: str,
    frame_number: int,
    duplicate_ids: Sequence[Any],
) -> str:
    formatted_ids = ", ".join(str(duplicate_id) for duplicate_id in duplicate_ids)
    return (
        "Found duplicate {} track ids in HOTA export for sample '{}' frame {}: {}. "
        "Set the corresponding INFERENCE.HOTA_DUPLICATE_{}_POLICY option to "
        "'keep_first' or 'keep_highest_conf' to normalize at export time."
    ).format(entity_kind, sample_id, frame_number, formatted_ids, entity_kind.upper())


def _resolve_duplicate_entities(
    entities: Sequence[Any],
    policy: str,
    entity_kind: str,
    sample_id: str,
    frame_number: int,
) -> Tuple[Sequence[Any], int]:
    grouped_entities: Dict[Any, Sequence[Tuple[int, Any]]] = {}
    for order, entity in enumerate(entities):
        track_id = getattr(entity, "id", 0)
        mapped_track_id = int(track_id) if str(track_id).lstrip("-").isdigit() else track_id
        grouped_entities.setdefault(mapped_track_id, []).append((order, entity))

    duplicate_ids = [track_id for track_id, grouped in grouped_entities.items() if len(grouped) > 1]
    if not duplicate_ids:
        return entities, 0

    if policy == "error":
        raise RuntimeError(_format_duplicate_error(entity_kind, sample_id, frame_number, duplicate_ids))

    resolved_entities = []
    for grouped in grouped_entities.values():
        if policy == "keep_first":
            winner = min(grouped, key=lambda item: item[0])
        else:
            winner = max(grouped, key=lambda item: (_entity_confidence(item[1]), -item[0]))
        resolved_entities.append(winner)

    resolved_entities.sort(key=lambda item: item[0])
    return [item[1] for item in resolved_entities], len(entities) - len(resolved_entities)


def _is_valid_bbox_xywh(bbox: Optional[Tuple[float, float, float, float]]) -> bool:
    if bbox is None:
        return False
    x, y, w, h = bbox
    if any(not math.isfinite(v) for v in (x, y, w, h)):
        return False
    if w <= 0 or h <= 0:
        return False
    return True


def _write_seqinfo(path: str, sample_id: str, sample: DataSample) -> None:
    fps = 30
    try:
        fps = int(round(float(sample.metadata.get("fps", fps))))
    except Exception:
        pass
    width = int(getattr(sample, "width", 0) or 0)
    height = int(getattr(sample, "height", 0) or 0)
    seq_len = int(len(sample))
    with open(path, "w") as f:
        f.write("[Sequence]\n")
        f.write(f"name={sample_id}\n")
        f.write("imDir=img1\n")
        f.write(f"frameRate={max(fps, 1)}\n")
        f.write(f"seqLength={max(seq_len, 0)}\n")
        f.write(f"imWidth={max(width, 0)}\n")
        f.write(f"imHeight={max(height, 0)}\n")
        f.write("imExt=.jpg\n")


def _export_motchallenge_layout(
    samples: DatasetEntries,
    predicted_samples: PredictedSamples,
    data_filter_fn: Optional[Callable[..., Any]],
    root_dir: str,
    duplicate_gt_policy: str,
    duplicate_pred_policy: str,
) -> Tuple[str, str, str, Sequence[str], Dict[str, int]]:
    split_name = f"{HOTA_BENCHMARK}-{HOTA_SPLIT}"

    gt_root = os.path.join(root_dir, "gt")
    trackers_root = os.path.join(root_dir, "trackers")
    output_root = os.path.join(root_dir, "output")

    gt_split_root = os.path.join(gt_root, split_name)
    tracker_data_root = os.path.join(trackers_root, split_name, HOTA_TRACKER_NAME, "data")
    os.makedirs(gt_split_root, exist_ok=True)
    os.makedirs(tracker_data_root, exist_ok=True)
    os.makedirs(output_root, exist_ok=True)

    sequence_names = []
    export_stats: Dict[str, int] = {
        "gt_invalid_bbox_count": 0,
        "pred_invalid_bbox_count": 0,
        "gt_duplicate_track_count": 0,
        "pred_duplicate_track_count": 0,
    }

    for sample_id_raw, sample in samples:
        sample_id = str(sample_id_raw)
        if sample_id_raw in predicted_samples:
            predicted_tracks = predicted_samples[sample_id_raw]
        elif sample_id in predicted_samples:
            predicted_tracks = predicted_samples[sample_id]
        else:
            raise KeyError(f"Missing prediction for sample id: {sample_id_raw}")

        sequence_names.append(sample_id)
        gt_dir = os.path.join(gt_split_root, sample_id, "gt")
        os.makedirs(gt_dir, exist_ok=True)
        gt_path = os.path.join(gt_dir, "gt.txt")
        seqinfo_path = os.path.join(gt_split_root, sample_id, "seqinfo.ini")
        pred_path = os.path.join(tracker_data_root, f"{sample_id}.txt")

        _write_seqinfo(seqinfo_path, sample_id, sample)

        gt_id_map: Dict[str, int] = {}
        pred_next_id_state = {"next_id": _next_generated_track_id(predicted_tracks)}
        with open(gt_path, "w") as gt_file, open(pred_path, "w") as pred_file:
            for frame_idx in range(len(sample)):
                gt_entities = sample.get_entities_for_frame_num(frame_idx)
                ignore_gt = []
                if data_filter_fn is not None:
                    valid_gt, ignore_gt = data_filter_fn(gt_entities, meta_data=sample.metadata)
                else:
                    valid_gt = gt_entities
                valid_gt, gt_duplicate_count = _resolve_duplicate_entities(
                    valid_gt,
                    duplicate_gt_policy,
                    "gt",
                    sample_id,
                    frame_idx + 1,
                )
                export_stats["gt_duplicate_track_count"] += gt_duplicate_count

                pred_entities = predicted_tracks.get_entities_for_frame_num(frame_idx)
                if data_filter_fn is not None:
                    valid_pred, _ = data_filter_fn(pred_entities, ignore_gt)
                else:
                    valid_pred = pred_entities
                valid_pred, duplicate_count = _resolve_duplicate_entities(
                    valid_pred,
                    duplicate_pred_policy,
                    "pred",
                    sample_id,
                    frame_idx + 1,
                )
                export_stats["pred_duplicate_track_count"] += duplicate_count

                frame_number = frame_idx + 1
                for entity in valid_gt:
                    bbox = _entity_bbox_xywh(entity)
                    if not _is_valid_bbox_xywh(bbox):
                        export_stats["gt_invalid_bbox_count"] += 1
                        continue
                    x, y, w, h = bbox
                    track_id = _as_positive_track_id(getattr(entity, "id", 0), gt_id_map)
                    gt_file.write(
                        f"{frame_number},{track_id},{x:.3f},{y:.3f},{w:.3f},{h:.3f},1,1,1\n"
                    )
                for entity in valid_pred:
                    bbox = _entity_bbox_xywh(entity)
                    if not _is_valid_bbox_xywh(bbox):
                        export_stats["pred_invalid_bbox_count"] += 1
                        continue
                    x, y, w, h = bbox
                    conf = _entity_confidence(entity)
                    track_id = _prediction_track_id(getattr(entity, "id", 0), pred_next_id_state)
                    pred_file.write(
                        f"{frame_number},{track_id},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{conf:.6f},-1,-1,-1\n"
                    )

    return gt_root, trackers_root, output_root, sequence_names, export_stats


def _parse_summary_file(path: str) -> MetricsMap:
    metrics: MetricsMap = {}
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) < 2:
        return metrics

    headers = lines[0].split()
    values = lines[1].split()
    for key, value in zip(headers, values):
        value = value.strip().rstrip("%")
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def _expected_hota_summary_file(output_root: str) -> str:
    return os.path.join(output_root, HOTA_TRACKER_NAME, f"{HOTA_CLASS_NAME}_summary.txt")


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_vendored_trackeval() -> Tuple[str, str]:
    vendored_root = os.path.join(_project_root(), "third_party", "TrackEval")
    vendored_script = os.path.join(vendored_root, "scripts", "run_mot_challenge.py")
    if os.path.isdir(vendored_root) and os.path.isfile(vendored_script):
        return vendored_root, vendored_script
    raise ImportError(
        "HOTA evaluation requires vendored TrackEval under `third_party/TrackEval`. "
        "Add it with:\n"
        "git subtree add --prefix third_party/TrackEval "
        "https://github.com/JonathonLuiten/TrackEval.git master --squash"
    )


def eval_hota(
    samples: DatasetEntries,
    predicted_samples: PredictedSamples,
    data_filter_fn: Optional[Callable[..., Any]] = None,
    keep_debug_files: bool = False,
    debug_dir: Optional[str] = None,
    duplicate_gt_policy: str = "error",
    duplicate_pred_policy: str = "error",
) -> Tuple[str, MetricsMap]:
    """
    Evaluate tracking using HOTA via the TrackEval package.
    If keep_debug_files is True, intermediate MOT-format files are retained for inspection.
    """
    temp_dir_manager = None
    if keep_debug_files:
        base_dir = os.path.abspath(debug_dir) if debug_dir else None
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix="siammot_hota_", dir=base_dir)
    else:
        temp_dir_manager = tempfile.TemporaryDirectory(prefix="siammot_hota_")
        tmp_dir = temp_dir_manager.name
    try:
        duplicate_gt_policy = _normalize_policy(duplicate_gt_policy, "gt")
        duplicate_pred_policy = _normalize_policy(duplicate_pred_policy, "pred")
        (
            gt_root,
            trackers_root,
            output_root,
            sequence_names,
            export_stats,
        ) = _export_motchallenge_layout(
            samples,
            predicted_samples,
            data_filter_fn,
            tmp_dir,
            duplicate_gt_policy,
            duplicate_pred_policy,
        )

        if export_stats["gt_duplicate_track_count"] > 0:
            logger.warning(
                "HOTA export collapsed %d duplicate GT boxes with repeated track ids in the same frame.",
                export_stats["gt_duplicate_track_count"],
            )
        if export_stats["gt_invalid_bbox_count"] > 0:
            logger.warning(
                "HOTA export dropped %d GT boxes with invalid geometry/values.",
                export_stats["gt_invalid_bbox_count"],
            )
        if export_stats["pred_invalid_bbox_count"] > 0:
            logger.warning(
                "HOTA export dropped %d predicted boxes with invalid geometry/values.",
                export_stats["pred_invalid_bbox_count"],
            )
        if export_stats["pred_duplicate_track_count"] > 0:
            logger.warning(
                "HOTA export collapsed %d duplicate predicted boxes with repeated track ids in the same frame.",
                export_stats["pred_duplicate_track_count"],
            )

        base_args = [
            "--GT_FOLDER",
            gt_root,
            "--TRACKERS_FOLDER",
            trackers_root,
            "--OUTPUT_FOLDER",
            output_root,
            "--BENCHMARK",
            HOTA_BENCHMARK,
            "--SPLIT_TO_EVAL",
            HOTA_SPLIT,
            "--TRACKERS_TO_EVAL",
            HOTA_TRACKER_NAME,
            "--CLASSES_TO_EVAL",
            HOTA_CLASS_NAME,
            "--METRICS",
            "HOTA",
            "--DO_PREPROC",
            "False",
            "--USE_PARALLEL",
            "False",
            "--PRINT_CONFIG",
            "False",
            "--PLOT_CURVES",
            "False",
        ]
        # Vendored TrackEval's CLI parses None-default args with nargs='+', so
        # passing --SEQMAP_FILE produces a list and later crashes in os.path.isfile.
        # --SEQ_INFO is the stable path here: TrackEval turns it into a dict and
        # reads per-sequence lengths from the seqinfo.ini files we already export.
        base_args.extend(["--SEQ_INFO", *sequence_names])

        vendored_root, vendored_script = _resolve_vendored_trackeval()
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            vendored_root if not existing_pythonpath else vendored_root + os.pathsep + existing_pythonpath
        )
        cmd = [sys.executable, vendored_script] + base_args
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                cwd=vendored_root,
                env=env,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Failed to run vendored TrackEval script: {}".format(vendored_script)) from exc

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        if proc.returncode != 0:
            raise RuntimeError(
                "Vendored TrackEval HOTA evaluation failed with return code {}.\nSTDOUT:\n{}\nSTDERR:\n{}".format(
                    proc.returncode, stdout, stderr
                )
            )

        summary_file = _expected_hota_summary_file(output_root)
        if not os.path.isfile(summary_file):
            raise RuntimeError(
                "TrackEval completed but expected summary file was not found: {}\n"
                "STDOUT:\n{}\nSTDERR:\n{}".format(summary_file, stdout, stderr)
            )

        with open(summary_file, "r") as f:
            summary_text = "\n\n" + f.read().strip() + "\n\n"
        overall_metrics = _parse_summary_file(summary_file)
        if not overall_metrics:
            raise RuntimeError(
                "TrackEval summary file was found but no HOTA metrics could be parsed: {}".format(summary_file)
            )

        return summary_text, overall_metrics
    finally:
        if temp_dir_manager is not None:
            temp_dir_manager.cleanup()
        elif keep_debug_files:
            logger.info("HOTA debug artifacts kept at %s", tmp_dir)
