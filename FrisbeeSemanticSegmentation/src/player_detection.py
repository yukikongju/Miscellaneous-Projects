import numpy as np
import supervision as sv

from constants import COCO_PERSON_CLASS_ID


def _get_player_bounding_boxes(
    detections: sv.Detections, field_polygon: np.ndarray, bottom_buffer: float = 25.0
) -> sv.Detections:
    """
    Get player bounding box inside the playing field

    Notes:
    - player who touch the boundaries will be removed
    """
    if len(detections.class_id) == 0:
        return detections

    person_mask = detections.class_id == COCO_PERSON_CLASS_ID
    person_bboxes = detections.xyxy[person_mask]

    y_min = field_polygon[:, 1].min()
    y_max = field_polygon[:, 1].max()  # 1006

    inside = np.array(
        [y1 > y_min and y2 < y_max - bottom_buffer for _, y1, _, y2 in person_bboxes],
        dtype=bool,
    )

    person_indices = np.where(person_mask)[0]
    keep_indices = person_indices[inside]

    # supervision's __getitem__ can't index tuple values in data (e.g. source_shape)
    stashed = {
        k: detections.data.pop(k)
        for k in list(detections.data)
        if isinstance(detections.data[k], tuple)
    }
    result = detections[keep_indices]
    #  detections.data.update(stashed)
    result.data.update(stashed)

    return result
