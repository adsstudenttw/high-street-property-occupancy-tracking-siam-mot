import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList
from gluoncv.torch.data.gluoncv_motion_dataset.dataset import AnnoEntity


def _normalize_class_index(label, class_table):
    if isinstance(label, torch.Tensor):
        label = label.item()

    label_index = int(label)
    if float(label_index) != float(label):
        raise ValueError("Expected an integer class label, got {!r}".format(label))

    if label_index < 1 or label_index > len(class_table):
        raise ValueError(
            "Class label {} is out of range for class table of size {}".format(
                label_index, len(class_table)
            )
        )
    return label_index - 1


def boxlists_to_entities(boxlists, firstframe_idx, timestamps, class_table=None):
    """
    Convert a list of boxlist to entities
    :return:
    """

    if isinstance(boxlists, BoxList):
        boxlists = [boxlists]

    # default class is person only
    if class_table is None:
        class_table = ["person"]

    assert isinstance(boxlists, list), "The input has to be a list"

    entities = []
    for i, boxlist in enumerate(boxlists):
        for j in range(len(boxlist)):
            entity = AnnoEntity()
            entity.bbox = boxlist.bbox[j].tolist()
            entity.confidence = boxlist.get_field('scores')[j].item()
            _label = boxlist.get_field('labels')[j].item()
            entity.labels = {
                class_table[_normalize_class_index(_label, class_table)]: entity.confidence
            }
            # the default id is -1
            entity.id = -1
            if boxlist.has_field('ids'):
                entity.id = boxlist.get_field('ids')[j].item()
            entity.frame_num = firstframe_idx + i
            entity.time = timestamps[i]
            entities.append(entity)

    return entities


def convert_given_detections_to_boxlist(entities: [AnnoEntity], video_width, video_height,
                                         class_table=None):
    # default class is person only
    if class_table is None:
        class_table = ["person"]

    boxes = [_entity.bbox for _entity in entities]
    boxes = torch.as_tensor(boxes).reshape(-1, 4)
    _labels = [class_table.index(list(_entity.labels.keys())[0]) + 1 for _entity in entities]
    _labels = torch.tensor(_labels, dtype=torch.int64)
    _scores = torch.tensor([_entity.confidence for _entity in entities])
    _ids = torch.tensor([-1 for _entity in entities], dtype=torch.int64)
    boxlist = BoxList(boxes,
                      [video_width, video_height],
                      mode='xywh').convert('xyxy')
    boxlist.add_field('labels', _labels)
    boxlist.add_field('scores', _scores)
    boxlist.add_field('ids', _ids)

    return boxlist
