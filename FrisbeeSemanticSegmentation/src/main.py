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
from typing import List, Optional
from sklearn.cluster import KMeans
from rfdetr import RFDETRSegSmall, RFDETRSmall
from rfdetr.assets.coco_classes import COCO_CLASSES
from field_homography import (
    _get_field_polygon,
    _get_field_white_lines,
    _get_frame_edges,
)
from player_detection import (
    _get_player_bounding_boxes,
    _get_players_images_from_detections,
    get_cropped_players_from_segmentation,
)
from player_position_tracker import PlayerPositionTracker


DEFAULT_VIDEO_PATH = os.path.expanduser(
    #  "~/Data/AUDLClips/ufa_championship_game_clip.mp4"
    #  "~/Data/AUDLClips/ufa_championship_game_clip_trimmed.mp4" # field trouble because too angled
    "~/Data/AUDLClips/ATL_CAR_turf.mp4"
    #  "~/Data/AUDLClips/CAR_SD_turf_bright_angled.mp4"  # doesn't work!!
    #  "~/Data/AUDLClips/NY_DC_turf_red_endzone.mp4"  # top border of field not recognized properly
    #  "~/Data/AUDLClips/USA_BEL_green_field_no_line.mp4"
)
DEFAULT_SHOW_PLAYER_NUMBERS = True
DEFAULT_DETECTION_THRESHOLD = 0.4
DEFAULT_MIN_KMEANS_CLUSTERING_POINTS = 20


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
    parser.add_argument(
        "--min-kmeans-clustering-points",
        type=float,
        default=DEFAULT_MIN_KMEANS_CLUSTERING_POINTS,
        help="Minimum of points before computing kmeans clustering for team assignment",
    )
    return parser.parse_args()


def draw_left_pane(
    frame: np.ndarray,
    player_detections: sv.Detections,
    player_team_labels: Optional[List[int]],
):
    """
    Video Frame and Player Boxes
    """
    #  colors = sv.ColorPalette.from_hex(
    #  [
    #  "#FF0000",  # red
    #  #  "#00FF00",  # green
    #  #  "#0000FF",  # blue
    #  #  "#FFFF00",  # yellow
    #  ]
    #  )
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

    ## Note: tried renaming class_id to label so supervision COLOR_LOOKUP.CLASS
    ## will color labels, but this solution hasn't worked, so going the ugly route
    ## and splitting detections in two teams
    #  if player_team_labels is not None:
    #  player_detections.class_id = np.array(player_team_labels)

    # rename team labels
    #  if player_team_labels is not None:
    #  print(player_team_labels)
    #  team_mappings = {0: "Team 1", 1: "Team 2"}
    #  player_team_labels = [
    #  team_mappings.get(label, label) for label in player_team_labels
    #  ]

    labels = [COCO_CLASSES.get(class_id, str(class_id)) for class_id in player_detections.class_id]
    if player_team_labels is None:
        labels = [
            COCO_CLASSES.get(class_id, str(class_id)) for class_id in player_detections.class_id
        ]
        annotated_frame = sv.MaskAnnotator().annotate(frame.copy(), player_detections)
        annotated_frame = sv.LabelAnnotator().annotate(
            annotated_frame, player_detections, labels=labels
        )
        annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, player_detections)
    else:
        player_team_labels = np.array(player_team_labels)

        team1_mask = player_team_labels == 0
        team2_mask = player_team_labels == 1
        team1_detections = player_detections[team1_mask]
        team2_detections = player_detections[team2_mask]

        team1_color = sv.ColorPalette.from_hex(["#FF0000"])
        team2_color = sv.ColorPalette.from_hex(["#0000FF"])

        ## team1 annotations
        annotated_frame = sv.MaskAnnotator(
            color=team1_color, color_lookup=sv.ColorLookup.CLASS
        ).annotate(frame.copy(), team1_detections)
        annotated_frame = sv.LabelAnnotator(
            color=team1_color, color_lookup=sv.ColorLookup.CLASS
        ).annotate(
            annotated_frame,
            team1_detections,
            #  labels=labels if player_team_labels is None else player_team_labels,
            labels=["Team 1"] * len(team1_detections),
        )
        annotated_frame = sv.BoxAnnotator(
            color=team1_color, color_lookup=sv.ColorLookup.CLASS
        ).annotate(annotated_frame, team1_detections)

        ## team2 annotations
        annotated_frame = sv.MaskAnnotator(
            color=team2_color, color_lookup=sv.ColorLookup.CLASS
        ).annotate(annotated_frame, team2_detections)
        annotated_frame = sv.LabelAnnotator(
            color=team2_color, color_lookup=sv.ColorLookup.CLASS
        ).annotate(
            annotated_frame,
            team2_detections,
            #  labels=labels if player_team_labels is None else player_team_labels,
            labels=["Team 2"] * len(team2_detections),
        )
        annotated_frame = sv.BoxAnnotator(
            color=team2_color, color_lookup=sv.ColorLookup.CLASS
        ).annotate(annotated_frame, team2_detections)

    return annotated_frame


def draw_right_pane(h: int, w: int, frame_edges, field_polygon, field_white_lines):
    """
    Bird's eye view of the field via homography.
    Falls back to the last known good frame when detection fails.
    """
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for x1, y1, x2, y2 in frame_edges:
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.drawContours(canvas, [field_polygon], -1, (0, 0, 255), 3)

    for x1, y1, x2, y2 in field_white_lines:
        cv2.line(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return canvas


def main():
    args = parse_args()
    #  VIDEO_PATH = args.video_path
    #  SHOW_PLAYER_NUMBERS = args.show_player_numbers
    #  DETECTION_THRESHOLD = args.detection_threshold

    ## VARIABLE INITIALIZATION
    bbox_model = RFDETRSmall()
    bbox_model.optimize_for_inference()
    seg_model = RFDETRSegSmall()
    seg_model.optimize_for_inference()

    position_tracker = PlayerPositionTracker()
    #  team_clustering = KMeans(init="k-means++", n_clusters=2, n_init=4)
    team_clustering = None
    player_torso_training_set = []

    ## Read video from .mp4 file path
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video_path}")

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ------------------------- DETECTION ---------------------------
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections = bbox_model.predict(img, threshold=args.detection_threshold)

        # ------------------------- FIELD DETECTION ---------------------------
        frame_edges = _get_frame_edges(frame)
        field_polygon = _get_field_polygon(frame)
        field_white_lines = _get_field_white_lines(frame, field_polygon)

        # ------------------------ PLAYER IN-FIELD DETECTION ---------------------------
        player_detections = _get_player_bounding_boxes(detections, field_polygon)

        # ------------------------ PLAYER TRAJECTORY ---------------------------
        tracked_players = position_tracker.update(
            detections=player_detections, frame_index=frame_index
        )
        positions = position_tracker.get_history()

        # ------------------------ FIXME: TEAM ASSIGNMENT ---------------------------
        player_images = _get_players_images_from_detections(frame, player_detections)
        players_torso = []
        for pic in player_images:
            cropped_torsos = get_cropped_players_from_segmentation(
                seg_model,
                pic,
                upper=0.25,
                lower=0.25,
                detection_threshold=args.detection_threshold,
            )
            if len(cropped_torsos) == 0:  # note: segmentation model was able to find someone
                raise ValueError(f"No player in player image, please check!")
            players_torso.extend(cropped_torsos)

        # FIXME: why less player image than player torso
        #  print(len(player_images), len(players_torso))
        #  print(players_torso)

        ## Finding average color vector
        player_median_color = []
        for array in players_torso:
            med = np.median(array, axis=0)
            player_median_color.append(med[med.shape[0] // 2])
        player_median_color = np.array(player_median_color)

        player_team_labels = None
        if team_clustering is None:
            player_torso_training_set.extend(player_median_color)
            if len(player_torso_training_set) > args.min_kmeans_clustering_points:
                print(f"enough points to train kmeans!")
                team_clustering = KMeans(init="k-means++", n_clusters=2, n_init=4).fit(
                    player_torso_training_set
                )
                #  player_labels = team_clustering.predict(player_median_color)
        else:
            player_team_labels = team_clustering.predict(player_median_color)

        # ----------------- TODO: BIRD-EYE FIELD HOMOGRAPHY ---------------

        # ------------------------- VISUALIZATION ---------------------------

        ## drawing panes
        h, w = frame.shape[:2]
        left = draw_left_pane(frame, player_detections, player_team_labels)
        right = draw_right_pane(h, w, frame_edges, field_polygon, field_white_lines)
        combined = np.hstack([left, right])
        cv2.imshow("FR-DETR Real-Time |  Left: clip   Right: positions", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
