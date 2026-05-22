"""
Script which does the following:

1. Open mp4 clip
2. Use RFDETRBase to visualize where the players and frisbee are using dots
3. Additional information to retrieve on players
     - Get player's number
     - Get Player team using jersey color
4. Create side-by-side window to visualize:
    - left window: clip
    - right window: player and disc movement (on black background). player from different team should be in different colors. Boolean flag to show player number as label.


- `~/Data/AUDLClips/ufa_championship_game_clip.mp4`

"""

import os
import cv2
import numpy as np
from PIL import Image
from rfdetr import RFDETRBase
import supervision as sv

# ── Configuration ──────────────────────────────────────────────────────────────
VIDEO_PATH = os.path.expanduser("~/Data/AUDLClips/ufa_championship_game_clip.mp4")
SHOW_PLAYER_NUMBERS = True  # set False to hide jersey-number labels
DETECTION_THRESHOLD = 0.4
DOT_RADIUS = 7

# COCO class indices used by RFDETRBase's default weights.
# If you load a frisbee-specific checkpoint, adjust FRISBEE_CLASS_ID accordingly.
PERSON_CLASS_ID = 0
FRISBEE_CLASS_ID = 32  # "sports ball" is the closest COCO class

# BGR colours
DISC_COLOR = (0, 255, 255)  # cyan/yellow for the disc
TEAM_COLORS = [
    (50, 120, 255),  # orange-ish – team A
    (255, 60, 60),  # blue-ish   – team B
]


# ── Helpers ────────────────────────────────────────────────────────────────────


def bbox_center(xyxy) -> tuple[int, int]:
    x1, y1, x2, y2 = xyxy
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def get_jersey_color(frame_bgr: np.ndarray, xyxy) -> np.ndarray:
    """
    Return the mean BGR colour of the jersey region of a player crop.
    Uses the middle vertical third of the bounding box to avoid head/shorts noise.
    """
    x1, y1, x2, y2 = map(int, xyxy)
    h = y2 - y1
    top = y1 + h // 4
    bot = y1 + h // 2
    crop = frame_bgr[top:bot, x1:x2]
    if crop.size == 0:
        return np.zeros(3, dtype=np.float32)
    return crop.reshape(-1, 3).mean(axis=0).astype(np.float32)


def assign_teams(colors: list) -> list[int]:
    """
    Cluster player jersey colours into 2 teams via K-Means.
    Falls back gracefully when fewer than 2 players are detected.
    """
    if len(colors) == 0:
        return []
    if len(colors) == 1:
        return [0]

    data = np.stack(colors)  # (N, 3)
    k = min(2, len(colors))

    # Simple K-Means implemented with numpy (avoids sklearn dependency)
    rng = np.random.default_rng(42)
    centers = data[rng.choice(len(data), k, replace=False)]

    for _ in range(20):  # max 20 iterations is plenty
        dists = np.linalg.norm(data[:, None] - centers[None], axis=2)  # (N, k)
        labels = dists.argmin(axis=1)
        new_centers = np.array(
            [
                data[labels == i].mean(axis=0) if (labels == i).any() else centers[i]
                for i in range(k)
            ]
        )
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    return labels.tolist()


def extract_player_number(frame_bgr: np.ndarray, xyxy) -> str:
    """
    Best-effort jersey-number extraction via pytesseract (optional dependency).
    Returns an empty string when tesseract is not installed or no digit is found.
    """
    try:
        import pytesseract  # type: ignore

        x1, y1, x2, y2 = map(int, xyxy)
        h = y2 - y1
        top = y1 + h // 4
        bot = y1 + (3 * h) // 4
        crop = frame_bgr[top:bot, x1:x2]
        if crop.size == 0:
            return ""
        crop = cv2.resize(crop, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cfg = "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789"
        text = pytesseract.image_to_string(thresh, config=cfg).strip()
        return text if text.isdigit() and 1 <= int(text) <= 99 else ""
    except Exception:
        return ""


def draw_dot(
    img: np.ndarray,
    center: tuple[int, int],
    color: tuple,
    radius: int = DOT_RADIUS,
    label: str | None = None,
) -> None:
    cv2.circle(img, center, radius, color, -1)
    cv2.circle(img, center, radius, (255, 255, 255), 1)  # white border
    if label:
        cv2.putText(
            img,
            label,
            (center[0] + radius + 3, center[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def build_movement_panel(
    shape: tuple,
    player_positions: list,
    player_teams: list[int],
    player_numbers: list[str],
    disc_positions: list,
    show_numbers: bool,
) -> np.ndarray:
    """Render the right-side panel: coloured dots on a black background."""
    h, w = shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for (cx, cy), team, number in zip(player_positions, player_teams, player_numbers):
        color = TEAM_COLORS[team % len(TEAM_COLORS)]
        label = number if show_numbers and number else None
        draw_dot(canvas, (cx, cy), color, label=label)

    for cx, cy in disc_positions:
        draw_dot(canvas, (cx, cy), DISC_COLOR, radius=DOT_RADIUS - 1)

    return canvas


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    # Load model – use local checkpoint if present, else fall back to default weights
    local_weights = os.path.join(os.path.dirname(__file__), "rf-detr-base.pth")
    if os.path.isfile(local_weights):
        model = RFDETRBase(pretrain_weights=local_weights)
    else:
        model = RFDETRBase()
    model.optimize_for_inference()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    print(f"Processing  : {VIDEO_PATH}")
    print(f"Show numbers: {SHOW_PLAYER_NUMBERS}")
    print("Press  q  to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Detection ──────────────────────────────────────────────
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections = model.predict(pil_img, threshold=DETECTION_THRESHOLD)
        # detections.xyxy
        annotated_frame = sv.BoxAnnotator().annotate(frame, detections)

        cv2.imshow("RF-DETR Real-Time", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Index numpy arrays directly to avoid supervision's __getitem__ mismatched
        # data-dict shapes (e.g. stride tensors whose size != detection count).
        #  if detections.class_id is not None and len(detections.class_id) > 0:
        #  person_mask = detections.class_id == PERSON_CLASS_ID
        #  disc_mask   = detections.class_id == FRISBEE_CLASS_ID
        #  person_bboxes = detections.xyxy[person_mask]
        #  disc_bboxes   = detections.xyxy[disc_mask]
        #  else:
        #  person_bboxes = np.empty((0, 4), dtype=np.float32)
        #  disc_bboxes   = np.empty((0, 4), dtype=np.float32)

        #  # ── Team assignment via jersey colour ──────────────────────
        #  jersey_colors = [get_jersey_color(frame, bb) for bb in person_bboxes]
        #  team_labels   = assign_teams(jersey_colors)

        #  # ── Player number extraction ────────────────────────────────
        #  player_numbers = [extract_player_number(frame, bb) for bb in person_bboxes]

        #  # ── Dot positions ───────────────────────────────────────────
        #  player_positions = [bbox_center(bb) for bb in person_bboxes]
        #  disc_positions   = [bbox_center(bb) for bb in disc_bboxes]

        #  # ── Left panel: original frame + dot overlay ────────────────
        #  left = frame.copy()
        #  for (cx, cy), team, number in zip(player_positions, team_labels, player_numbers):
        #  color = TEAM_COLORS[team % len(TEAM_COLORS)]
        #  label = number if SHOW_PLAYER_NUMBERS and number else None
        #  draw_dot(left, (cx, cy), color, label=label)
        #  for (cx, cy) in disc_positions:
        #  draw_dot(left, (cx, cy), DISC_COLOR, radius=DOT_RADIUS - 1)

        #  # ── Right panel: movement map on black background ───────────
        #  right = build_movement_panel(
        #  frame.shape,
        #  player_positions,
        #  team_labels,
        #  player_numbers,
        #  disc_positions,
        #  SHOW_PLAYER_NUMBERS,
        #  )

        #  # ── Side-by-side display ─────────────────────────────────────
        #  combined = np.hstack([left, right])
        #  cv2.imshow("Player Movements  |  Left: clip   Right: positions", combined)

        #  if cv2.waitKey(1) & 0xFF == ord("q"):
        #  break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
