import numpy as np
import supervision as sv


class PlayerPositionTracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()
        self.history = {}  # tracker_id -> list of positions

    def update(self, detections: sv.Detections, frame_index: int):
        """
        Tracks players and stores xy feet position for each frame.

        detections.xyxy must be in the same coordinate system as the frame.
        """
        # note: need to remove `source_shape` because `supervision` does not accept tuple values during tracking/indexing
        detections.data.pop("source_shape", None)
        #  stashed = {
        #  k: detections.data.pop(k)
        #  for k in list(detections.data)
        #  if isinstance(detections.data[k], tuple)
        #  }

        # assign tracker_id to each detection
        tracked_detections = self.tracker.update_with_detections(detections)

        # Restore metadata
        #  detections.data.update(stashed)
        #  tracked_detections.data.update(stashed)

        if tracked_detections.tracker_id is None:
            return tracked_detections

        for xyxy, tracker_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id):
            x1, _, x2, y2 = xyxy

            # player position = bottom-center of bbox, aka feet
            x = float((x1 + x2) / 2)
            y = float(y2)

            tracker_id = int(tracker_id)

            if tracker_id not in self.history:
                self.history[tracker_id] = []

            self.history[tracker_id].append(
                {
                    "frame": frame_index,
                    "x": x,
                    "y": y,
                }
            )

        return tracked_detections

    def get_history(self):
        return self.history

    def get_latest_positions(self):
        latest = {}

        for tracker_id, positions in self.history.items():
            if len(positions) > 0:
                latest[tracker_id] = positions[-1]

        return latest
