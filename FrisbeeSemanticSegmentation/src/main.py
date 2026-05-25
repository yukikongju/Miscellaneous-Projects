"""
Notes:
- RF-DETR outputs 1-indexed class IDs, where as COCO is 0-indexed
- Type of fields: (1) soccer field (2) turf with yard marks (3)
"""

import argparse
import cv2
import os
import numpy as np
import supervision as sv

from PIL import Image
from rfdetr import RFDETRSmall
from rfdetr.assets.coco_classes import COCO_CLASSES
from field_homography import (
    _get_field_polygon,
    _get_field_white_lines,
    _get_frame_edges,
)

COCO_PERSON_CLASS_ID = 1
COCO_FRISBEE_CLASS_ID = 34

DEFAULT_VIDEO_PATH = os.path.expanduser(
    #  "~/Data/AUDLClips/ufa_championship_game_clip.mp4"
    "~/Data/AUDLClips/ufa_championship_game_clip_trimmed.mp4"
)
DEFAULT_SHOW_PLAYER_NUMBERS = True
DEFAULT_DETECTION_THRESHOLD = 0.4


def parse_args():
    parser = argparse.ArgumentParser(description="Frisbee player detection")
    parser.add_argument("--video-path", default=DEFAULT_VIDEO_PATH, help="Path to input video file")
    parser.add_argument(
        "--show-player-numbers",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SHOW_PLAYER_NUMBERS,
        help="Show player numbers on detections",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=DEFAULT_DETECTION_THRESHOLD,
        help="Detection confidence threshold",
    )
    return parser.parse_args()


def draw_left_pane(frame, detections):
    """
    Video Frame and Player Boxes
    """
    # TODO: filter out only accepted class
    #  if len(detections.class_id) > 0:
    #  person_mask = detections.class_id == COCO_PERSON_CLASS_ID
    #  disc_mask = detections.class_id == COCO_FRISBEE_CLASS_ID
    #  person_bboxes = detections.xyxy[person_mask]
    #  disc_bboxes = detections.xyxy[disc_mask]
    #  mask = np.isin(
    #  detections.class_id, [COCO_PERSON_CLASS_ID, COCO_FRISBEE_CLASS_ID]
    #  )
    #  detections = detections[mask]

    labels = [COCO_CLASSES.get(class_id, str(class_id)) for class_id in detections.class_id]
    annotated_frame = sv.MaskAnnotator().annotate(frame.copy(), detections)
    annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels=labels)
    annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
    return annotated_frame


def draw_right_pane(frame, detections):
    """
    Bird's eye view of the field via homography.
    Falls back to the last known good frame when detection fails.
    """
    h, w = frame.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for x1, y1, x2, y2 in _get_frame_edges(frame):
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    box = _get_field_polygon(frame)
    cv2.drawContours(canvas, [box], -1, (0, 0, 255), 3)

    for x1, y1, x2, y2 in _get_field_white_lines(frame, box):
        cv2.line(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return canvas


def main():
    args = parse_args()
    #  VIDEO_PATH = args.video_path
    #  SHOW_PLAYER_NUMBERS = args.show_player_numbers
    #  DETECTION_THRESHOLD = args.detection_threshold

    model = RFDETRSmall()
    model.optimize_for_inference()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ------------------------- DETECTION ---------------------------
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections = model.predict(img, threshold=args.detection_threshold)

        # ------------------------- VISUALIZATION ---------------------------
        left = draw_left_pane(frame, detections)
        right = draw_right_pane(frame, detections)
        combined = np.hstack([left, right])
        cv2.imshow("FR-DETR Real-Time |  Left: clip   Right: positions", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
