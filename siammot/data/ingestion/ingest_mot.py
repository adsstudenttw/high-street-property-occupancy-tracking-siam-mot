import argparse
import csv
import configparser
import datetime
import glob
import os

from PIL import Image
from pathlib import Path

from gluoncv.torch.data.gluoncv_motion_dataset.dataset import GluonCVMotionDataset, DataSample, AnnoEntity, FieldNames, SplitNames
from gluoncv.torch.data.gluoncv_motion_dataset.utils.ingestion_utils import process_dataset_splits

# From paper, see table 5 and 6: https://arxiv.org/pdf/1603.00831.pdf
MOT_LABEL_MAP = {
    1: "Pedestrian",
    2: "Person on vehicle",
    3: "Car",
    4: "Bicycle",
    5: "Motorbike",
    6: "Non motorized vehicle",
    7: "Static person",
    8: "Distractor",
    9: "Occluder",
    10: "Occluder on the ground",
    11: "Occluder full",
    12: "Reflection",
}

DET_OPTIONS = {"SDP", "FRCNN", "DPM"}


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    value_str = str(value).strip().lower()
    if value_str in {"true", "1", "yes", "y", "on"}:
        return True
    if value_str in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")


def _parse_det_options(value):
    if value is None:
        return [""]
    value_str = str(value).strip()
    if not value_str:
        return [""]
    options = [item.strip() for item in value_str.split(",") if item.strip()]
    return options if options else [""]


def sample_from_mot_csv(csv_path, fps, sample=None, mot17=True, has_gt=False):
    if sample is None:
        id_ = Path(csv_path).stem
        sample = DataSample(id_)
    else:
        sample = sample.get_copy_without_entities()
    with open(csv_path, newline='') as f:
        reader = csv.reader(f, delimiter=',')

        def coord(x):
            return round(float(x))

        for row in reader:
            frame_num = int(row[0])
            obj_id = row[1]
            x = coord(row[2])
            y = coord(row[3])
            w = coord(row[4])
            h = coord(row[5])
            conf = float(row[6])
            # If not mot17 the last 3 are 3D coords which are usually -1
            # (see pg. 9 https://arxiv.org/pdf/1504.01942.pdf)
            if has_gt and mot17:
                label = int(row[7])
                visibility = float(row[8])
            else:
                label = 1
                visibility = 1

            label_text = MOT_LABEL_MAP[label]

            # NOTE: Actually all classes that aren't Pedestrian have confidence 0 and so should be ingested
            # but are ignored at evaluation time
            # i.e. (label != 1 and conf) is never true
            assert not (label != 1 and conf)
            has_person_label = label_text in ("Pedestrian")

            time_ms = int((frame_num - 1) / fps * 1000)
            entity = AnnoEntity(time=time_ms, id=obj_id)
            entity.bbox = [x, y, w, h]
            blob = {
                "frame_csv": frame_num,
                "frame_idx": frame_num - 1,
                "visibility": visibility
            }
            entity.labels = {}
            # entity.labels["person"] = 1
            if has_person_label:
                entity.labels["person"] = 1
            else:
                entity.labels[str(label)] = 1
            entity.labels["vis"] = visibility

            entity.confidence = conf
            entity.blob = blob

            sample.add_entity(entity)
    return sample


def main(args, description="Initial ingestion", det_options=None, mot17=True):
    if det_options is None:
        det_options = [""]
    if len(det_options) == 0:
        det_options = [""]

    dataset_path = args.dataset_path
    out_filename = args.anno_name

    out_dataset = GluonCVMotionDataset(out_filename, dataset_path, load_anno=False)
    metadata = {
        FieldNames.DESCRIPTION: description,
        FieldNames.DATE_MODIFIED: str(datetime.datetime.now()),
    }
    out_dataset.metadata = metadata

    splits = {
        "train": os.path.join(out_dataset.data_root_path, "train"),
        "val": os.path.join(out_dataset.data_root_path, "val"),
        "test": os.path.join(out_dataset.data_root_path, "test"), # No gt for MOT test
    }

    for det_option in det_options:
        for split_name, split_path in splits.items():
            if not os.path.isdir(split_path):
                continue

            subdir_pattern = "*" if det_option == "" else "*" + det_option
            subdirs = [d for d in glob.glob(os.path.join(split_path, subdir_pattern)) if os.path.isdir(d)]
            for i, subdir in enumerate(subdirs):
                vid_id = os.path.basename(subdir)
                vid_path = os.path.join(split_path, subdir)

                sample = DataSample(vid_id)

                if mot17:
                    info_path = os.path.join(vid_path, "seqinfo.ini")
                    config = configparser.ConfigParser()
                    config.read(info_path)
                    seq_conf = config["Sequence"]
                    fps = float(seq_conf['frameRate'])
                    num_frames = int(seq_conf['seqLength'])
                    width = int(seq_conf['imWidth'])
                    height = int(seq_conf['imHeight'])
                else:
                    # Assume 30 fps
                    fps = 30
                    im_paths = glob.glob(os.path.join(vid_path, "img1", "*.jpg"))
                    num_frames = len(im_paths)
                    im_example = Image.open(im_paths[0])
                    width = im_example.width
                    height = im_example.height

                rel_base_dir = vid_path.replace(out_dataset.data_root_path, "").lstrip(os.path.sep)
                rel_base_dir = os.path.join(rel_base_dir, "img1")
                metadata = {
                    FieldNames.DATA_PATH: rel_base_dir,
                    FieldNames.FPS: fps,
                    FieldNames.NUM_FRAMES: num_frames,
                    FieldNames.RESOLUTION: {"width": width, "height": height},
                }
                sample.metadata = metadata

                gt_path = os.path.join(vid_path, "gt/gt.txt")
                det_path = os.path.join(vid_path, "det/det.txt")
                has_gt = os.path.exists(gt_path)
                anno_path = gt_path if has_gt else det_path

                sample = sample_from_mot_csv(anno_path, fps, sample, mot17, has_gt)

                out_dataset.add_sample(sample)

                print("Done {} sample {}/{}, {}".format(split_name, i+1, len(subdirs), vid_id))

    out_dataset.dump()

    return out_dataset


def write_data_split(args, dataset):
    if dataset is None:
        dataset = GluonCVMotionDataset(args.anno_name, args.dataset_path)

    def split_func(sample):
        data_path = sample.data_relative_path
        if data_path.startswith("train"):
            return SplitNames.TRAIN
        elif data_path.startswith("val"):
            return SplitNames.VAL
        elif data_path.startswith("test"):
            return SplitNames.TEST

        raise Exception("Shouldn't happen")

    process_dataset_splits(dataset, split_func, save=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ingest mot dataset')
    parser.add_argument('--dataset_path', default="",
                        help="The path of dataset folder")
    parser.add_argument('--anno_name', default="anno.json",
                        help="The file name (with json) of ingested annotation file")
    parser.add_argument(
        '--mot17',
        default=None,
        type=_parse_bool,
        help="Whether annotation rows follow MOT17 GT format (columns 8-9 are class and visibility). "
             "If omitted, this is inferred from whether dataset_path contains 'MOT17'.",
    )
    parser.add_argument(
        '--det-options',
        default="",
        type=str,
        help="Optional comma-separated sequence suffix filters (e.g. DPM,FRCNN,SDP). "
             "Leave empty to ingest all sequence folders (recommended for custom datasets).",
    )
    args = parser.parse_args()

    if args.mot17 is None:
        mot17 = "MOT17" in args.dataset_path
    else:
        mot17 = args.mot17
    det_options = _parse_det_options(args.det_options)
    if det_options != [""] and any(opt not in DET_OPTIONS for opt in det_options):
        print("Using custom detector suffix filters: {}".format(det_options))
    dataset = main(args, det_options=det_options, mot17=mot17)
    write_data_split(args, dataset)
