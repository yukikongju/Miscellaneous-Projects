import cv2
import numpy as np
from typing import Optional

from constants import UFA_BOTTOM_STREAM_OVERLAY_HEIGHT


def _get_field_polygon(frame: np.ndarray) -> np.ndarray:
    ## FIXME: rectangle shouldn't be changing a lot between frames
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    field_mask = cv2.inRange(hsv, lower_green, upper_green)
    #  print(field_mask.shape)

    # remove ufa bottom overlay from frame
    field_mask[-UFA_BOTTOM_STREAM_OVERLAY_HEIGHT:, :] = 0

    # clean mask
    kernel = np.ones((15, 15), np.uint8)
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel)
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, kernel)

    # find biggest green region
    contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    field_contour = max(contours, key=cv2.contourArea)

    # Approximate boundary polygon
    #  epsilon = 0.02 * cv2.arcLength(field_contour, True)
    #  field_poly = cv2.approxPolyDP(field_contour, epsilon, True)
    #  cv2.drawContours(canvas, [field_poly], -1, (0, 0, 255), 3)

    # draw rectangle
    rect = cv2.minAreaRect(field_contour)
    box = cv2.boxPoints(rect)
    return box.astype(int)


def _angle_diff(a1: float, a2: float) -> float:
    """Smallest angular difference between two undirected line angles, result in [0, pi/2]."""
    diff = abs(a1 - a2) % np.pi
    return min(diff, np.pi - diff)


def _get_field_white_lines(
    frame: np.ndarray,
    field_poly: np.ndarray,
    angle_tol_deg: float = 85.0,
    min_length: float = 80.0,
) -> list[tuple]:
    h, w = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # rasterize field polygon into a binary mask
    field_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(field_mask, [field_poly.astype(np.int32)], 255)

    # find white markings on the field, clipped to field boundary
    lower_white = np.array([0, 0, 160])
    upper_white = np.array([180, 80, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    white_mask = cv2.bitwise_and(white_mask, white_mask, mask=field_mask)

    # remove noise
    small_kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, small_kernel)

    # interpolate/connect broken white lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3))
    connected_h = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, horizontal_kernel)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 35))
    connected_v = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, vertical_kernel)
    white_mask = cv2.bitwise_and(cv2.bitwise_or(connected_h, connected_v), field_mask)

    edges = cv2.Canny(white_mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=80,
        maxLineGap=30,
    )
    if lines is None:
        return []

    # derive field long axis from the longest box edge, then compute perpendicular angle
    edge_vecs = [field_poly[(i + 1) % 4].astype(float) - field_poly[i] for i in range(4)]
    long_axis = max(edge_vecs, key=np.linalg.norm)
    perp_angle = np.arctan2(long_axis[1], long_axis[0]) + np.pi / 2

    tol = np.deg2rad(angle_tol_deg)
    result = []
    for x1, y1, x2, y2 in (pts[0] for pts in lines):
        line_angle = np.arctan2(y2 - y1, x2 - x1)
        if _angle_diff(line_angle, perp_angle) > tol:
            continue
        if np.hypot(x2 - x1, y2 - y1) < min_length:
            continue
        extended = extend_line_to_box(x1, y1, x2, y2, field_poly)
        if extended is not None:
            pt1, pt2 = extended
            result.append((pt1[0], pt1[1], pt2[0], pt2[1]))
    return result


def _get_frame_edges(frame: np.ndarray) -> list[tuple]:
    """
    https://www.geeksforgeeks.org/python/line-detection-python-opencv-houghline-method/
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=5,
        maxLineGap=10,
    )
    if lines is None:
        return []
    return [tuple(pts[0]) for pts in lines]


def extend_line_to_box(x1, y1, x2, y2, box_pts):
    """
    Extend a line segment to the edges of a rotated bounding box.
    box_pts: (4,2) int array from cv2.boxPoints
    Returns two clipped endpoints, or None if no intersection found.
    """
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None

    # Build the 4 edges of the box
    box_edges = [(box_pts[i], box_pts[(i + 1) % 4]) for i in range(4)]

    def line_intersect(p1, d, p3, p4):
        """Ray p1+t*d intersected with segment p3->p4. Returns t or None."""
        p3, p4 = np.array(p3, float), np.array(p4, float)
        p1 = np.array(p1, float)
        d = np.array(d, float)
        v = p4 - p3
        denom = d[0] * v[1] - d[1] * v[0]
        if abs(denom) < 1e-10:
            return None
        t = ((p3[0] - p1[0]) * v[1] - (p3[1] - p1[1]) * v[0]) / denom
        u = ((p3[0] - p1[0]) * d[1] - (p3[1] - p1[1]) * d[0]) / denom
        if 0.0 <= u <= 1.0:
            return t
        return None

    origin = np.array([x1, y1], float)
    direction = np.array([dx, dy], float)

    ts = []
    for a, b in box_edges:
        t = line_intersect(origin, direction, a, b)
        if t is not None:
            ts.append(t)

    if len(ts) < 2:
        return None

    ts.sort()
    t_min, t_max = ts[0], ts[-1]

    pt1 = (int(origin[0] + t_min * direction[0]), int(origin[1] + t_min * direction[1]))
    pt2 = (int(origin[0] + t_max * direction[0]), int(origin[1] + t_max * direction[1]))
    return pt1, pt2
