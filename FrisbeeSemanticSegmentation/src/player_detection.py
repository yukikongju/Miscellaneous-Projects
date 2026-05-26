import cv2
import numpy as np
import math
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
from typing import List

from constants import COCO_PERSON_CLASS_ID


def _get_player_bounding_boxes(
    detections: sv.Detections, field_polygon: np.ndarray, bottom_buffer: float = 70.0
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


def _get_players_images_from_detections(
    frame: np.ndarray,
    detections: sv.Detections,
) -> List[Image]:
    """
    Get list of frames with player in them
    """
    players = []
    for coords, label in zip(detections.xyxy, detections.class_id):
        if label != COCO_PERSON_CLASS_ID:
            continue
        x1, y1, x2, y2 = coords.astype(int)
        crop = frame[y1:y2, x1:x2]
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        players.append(pil_img)
    return players


def get_cropped_players_from_segmentation(
    seg_model: RFDETRBase,
    img: Image,
    upper: float = 0.25,
    lower: float = 0.25,
    detection_threshold: float = 0.4,
) -> List[np.ndarray]:
    """
    Cropping image by removing upper and lower based on provided percentages

    Returns
    -------
        List[(h, w, 3)]
    """
    detections = seg_model.predict(img, threshold=detection_threshold)

    # filter out class id that are not PERSON
    mask = detections.mask[detections.class_id == COCO_PERSON_CLASS_ID]

    s, h, w = mask.shape  # note: if s > 1, then more than 2 person in the picture
    y_min = math.floor(h * lower)
    y_max = math.floor(h * (1 - upper))

    cropped_image = detections.data["source_image"][y_min:y_max, :, :]
    cropped_mask = mask[:, y_min:y_max, :]
    result = cropped_image * cropped_mask[..., None]

    ## normal case: only one player found in image crop
    if result.shape[0] == 1 and len(result.shape) == 4:
        return list(result)
    ## FIXME: what to do when 2 players (occlusion?)
    ## only return the first for now (assumption: is the non-occluded player)
    #  if result.shape[0] > 1 and len(result.shape) == 4: # this line doesn't work
    #  return list(result)[0]

    ## fallback: if no person found or there is an error, return original image as numpy array
    cropped_image = np.asarray(img)[y_min:y_max, :, :]
    cropped_image = cropped_image[np.newaxis, :]
    return list(cropped_image)
