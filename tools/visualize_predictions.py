import argparse
import glob
import os
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from gluoncv.torch.data.gluoncv_motion_dataset.dataset import DataSample

from siammot.data.adapters.utils.data_utils import load_motion_anno


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
PRED_COLOR = (255, 140, 0)
GT_COLOR = (0, 200, 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render SiamMOT prediction boxes from a saved inference JSON onto sequence frames."
    )
    parser.add_argument(
        "--sequence-id",
        required=True,
        nargs="+",
        help="One or more sequence/sample ids, e.g. achter_clarenburg_ls choorstraat_2_ls",
    )
    parser.add_argument(
        "--predictions-json",
        default="",
        help="Optional explicit path to the saved prediction JSON. Defaults to <predictions-dir>/<sequence-id>.json.",
    )
    parser.add_argument(
        "--predictions-dir",
        default="/data/siammot_storage/siammot/artifacts/baseline",
        help="Directory that contains saved per-sequence prediction JSON files.",
    )
    parser.add_argument(
        "--dataset-root",
        default="/data/siammot_storage/siammot/datasets/hspot",
        help="HSPOT dataset root that contains raw_data/<split>/<sequence-id>/img1.",
    )
    parser.add_argument(
        "--split",
        default="",
        help="Optional explicit split: train, val, or test. If omitted, the script searches all three.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for rendered frames. Defaults to <predictions-dir>/visualizations/<sequence-id>.",
    )
    parser.add_argument("--frame-start", type=int, default=0, help="Inclusive zero-based frame index to start from.")
    parser.add_argument(
        "--frame-end",
        type=int,
        default=-1,
        help="Inclusive zero-based frame index to stop at. Use -1 for the sequence end.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional cap on the number of rendered frames after applying frame-start/frame-end.",
    )
    parser.add_argument(
        "--only-with-boxes",
        action="store_true",
        help="Render only frames that contain at least one prediction or GT entity.",
    )
    parser.add_argument(
        "--with-gt",
        action="store_true",
        help="Overlay GT boxes from annotation/anno.json and annotation/splits.json.",
    )
    return parser.parse_args()


def resolve_predictions_json(sequence_id: str, predictions_json: str, predictions_dir: str) -> str:
    if predictions_json:
        return predictions_json

    direct_path = os.path.join(predictions_dir, "{}.json".format(sequence_id))
    if os.path.isfile(direct_path):
        return direct_path

    recursive_pattern = os.path.join(predictions_dir, "**", "{}.json".format(sequence_id))
    matches = sorted(glob.glob(recursive_pattern, recursive=True))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileExistsError(
            "Found multiple prediction JSON files for '{}': {}".format(sequence_id, matches)
        )
    return direct_path


def resolve_split_and_img_dir(dataset_root: str, sequence_id: str, split: str) -> Tuple[str, str]:
    split_candidates: Sequence[str]
    if split:
        split_candidates = [split]
    else:
        split_candidates = ["train", "val", "test"]

    for candidate in split_candidates:
        img_dir = os.path.join(dataset_root, "raw_data", candidate, sequence_id, "img1")
        if os.path.isdir(img_dir):
            return candidate, img_dir

    raise FileNotFoundError(
        "Could not find img1 for sequence '{}' under {}.".format(sequence_id, dataset_root)
    )


def list_frame_paths(img_dir: str) -> List[str]:
    frame_paths = [
        os.path.join(img_dir, file_name)
        for file_name in sorted(os.listdir(img_dir))
        if file_name.lower().endswith(IMAGE_EXTENSIONS)
    ]
    if not frame_paths:
        raise FileNotFoundError("No frame images found in '{}'.".format(img_dir))
    return frame_paths


def load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        return ImageFont.load_default()


def clip_box_xywh(
    bbox: Sequence[float], image_size: Tuple[int, int]
) -> Optional[Tuple[float, float, float, float]]:
    if len(bbox) < 4:
        return None
    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    if w <= 0 or h <= 0:
        return None
    image_w, image_h = image_size
    x1 = max(0.0, min(float(image_w - 1), x))
    y1 = max(0.0, min(float(image_h - 1), y))
    x2 = max(0.0, min(float(image_w - 1), x + w))
    y2 = max(0.0, min(float(image_h - 1), y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def format_label(entity: object, entity_kind: str) -> str:
    track_id = getattr(entity, "id", -1)
    confidence = float(getattr(entity, "confidence", 0.0))
    if entity_kind == "gt":
        return "GT id={}".format(track_id)
    return "P id={} conf={:.2f}".format(track_id, confidence)


def draw_entities(
    image: Image.Image,
    entities: Iterable[object],
    font: ImageFont.ImageFont,
    entity_kind: str,
) -> int:
    draw = ImageDraw.Draw(image)
    count = 0
    for entity in entities:
        bbox = clip_box_xywh(getattr(entity, "bbox", []), image.size)
        if bbox is None:
            continue

        color = GT_COLOR if entity_kind == "gt" else PRED_COLOR
        line_width = 2 if entity_kind == "gt" else 3
        x1, y1, x2, y2 = bbox
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        label = format_label(entity, entity_kind)
        text_left = x1
        text_top = max(0.0, y1 - (18.0 if entity_kind == "pred" else 34.0))
        if hasattr(draw, "textbbox"):
            text_bbox = draw.textbbox((text_left, text_top), label, font=font)
            draw.rectangle(text_bbox, fill=color)
        draw.text((text_left, text_top), label, fill=(0, 0, 0), font=font)
        count += 1
    return count


def load_ground_truth_sample(dataset_root: str, sequence_id: str, split: str) -> DataSample:
    dataset = load_motion_anno(dataset_root, "anno.json", "splits.json", set=split)
    for sample in dataset:
        if sample.id == sequence_id:
            return sample
    raise KeyError(
        "Sequence '{}' was not found in annotation/anno.json for split '{}'.".format(
            sequence_id, split
        )
    )


def render_frames(
    sample_result: DataSample,
    ground_truth: Optional[DataSample],
    frame_paths: Sequence[str],
    output_dir: str,
    frame_start: int,
    frame_end: int,
    max_frames: int,
    only_with_boxes: bool,
) -> int:
    os.makedirs(output_dir, exist_ok=True)
    font = load_font()
    last_frame = len(frame_paths) - 1 if frame_end < 0 else min(frame_end, len(frame_paths) - 1)
    rendered = 0

    for frame_idx in range(max(0, frame_start), last_frame + 1):
        if max_frames > 0 and rendered >= max_frames:
            break

        pred_entities = list(sample_result.get_entities_for_frame_num(frame_idx))
        gt_entities = []
        if ground_truth is not None:
            gt_entities = list(ground_truth.get_entities_for_frame_num(frame_idx))
        if only_with_boxes and not pred_entities and not gt_entities:
            continue

        with Image.open(frame_paths[frame_idx]) as frame_image:
            image = frame_image.convert("RGB")
        gt_count = draw_entities(image, gt_entities, font, entity_kind="gt")
        pred_count = draw_entities(image, pred_entities, font, entity_kind="pred")
        if only_with_boxes and gt_count == 0 and pred_count == 0:
            continue

        output_path = os.path.join(output_dir, os.path.basename(frame_paths[frame_idx]))
        image.save(output_path)
        rendered += 1

    return rendered


def main() -> None:
    args = parse_args()
    if args.predictions_json and len(args.sequence_id) > 1:
        raise ValueError("--predictions-json can only be used with a single --sequence-id.")

    output_root = args.output_dir or os.path.join(args.predictions_dir, "visualizations")
    total_rendered = 0
    for sequence_id in args.sequence_id:
        predictions_json = resolve_predictions_json(
            sequence_id, args.predictions_json, args.predictions_dir
        )
        if not os.path.isfile(predictions_json):
            raise FileNotFoundError("Prediction JSON not found: '{}'".format(predictions_json))

        split, img_dir = resolve_split_and_img_dir(args.dataset_root, sequence_id, args.split)
        frame_paths = list_frame_paths(img_dir)
        sample_result = DataSample.load(predictions_json)
        ground_truth = load_ground_truth_sample(args.dataset_root, sequence_id, split) if args.with_gt else None
        output_dir = os.path.join(output_root, sequence_id)

        rendered = render_frames(
            sample_result=sample_result,
            ground_truth=ground_truth,
            frame_paths=frame_paths,
            output_dir=output_dir,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            max_frames=args.max_frames,
            only_with_boxes=args.only_with_boxes,
        )
        total_rendered += rendered
        print("Rendered {} frame(s) for {} to {}".format(rendered, sequence_id, output_dir))

    print("Rendered {} frame(s) total.".format(total_rendered))


if __name__ == "__main__":
    main()
