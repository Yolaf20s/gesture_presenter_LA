"""
Temporal smoothing for per-frame gesture predictions.
"""

from collections import deque

import config


class GestureStabilizer:
    """
    Require a gesture to remain consistent for several frames before it becomes
    trigger-eligible. This reduces single-frame flicker and accidental actions.
    """

    def __init__(self, stable_frames=None):
        self.stable_frames = stable_frames or config.TRIGGER_STABLE_FRAMES
        self.history = deque(maxlen=self.stable_frames)

    def reset(self):
        self.history.clear()

    def update(self, raw_gesture, raw_confidence, tracking_state):
        """
        Return `(stable_gesture, stable_confidence)`.

        Non-ready tracking states collapse to `idle` and clear the history so the
        trigger layer only sees actionable poses when the signal is stable.
        """
        if tracking_state != "ready":
            self.history.clear()
            return "idle", 0.0

        self.history.append((raw_gesture, float(raw_confidence)))
        if len(self.history) < self.stable_frames:
            return "idle", 0.0

        gestures = [gesture for gesture, _ in self.history]
        if len(set(gestures)) != 1:
            return "idle", 0.0

        stable_gesture = gestures[-1]
        if stable_gesture == "idle":
            return "idle", 0.0

        avg_confidence = sum(conf for _, conf in self.history) / len(self.history)
        return stable_gesture, avg_confidence
