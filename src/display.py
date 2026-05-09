"""
OpenCV overlay rendering for the CLI preview.
"""

import time

import cv2
import numpy as np

import config


class Display:
    """Render gesture state, performance stats, and trigger feedback."""

    COLOR_BG = (0, 0, 0)
    COLOR_TEXT = (255, 255, 255)
    COLOR_GESTURE = (0, 255, 0)
    COLOR_IDLE = (128, 128, 128)
    COLOR_WARNING = (0, 165, 255)
    COLOR_TRIGGER = (0, 255, 255)
    COLOR_TIMING = (255, 200, 100)
    COLOR_COOLDOWN_BAR = (0, 165, 255)

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.last_trigger_message = ""
        self.last_trigger_time = 0.0
        print("[Display] Ready.")

    def render(
        self,
        frame,
        gesture,
        confidence,
        timings,
        tracking_state="ready",
        fps=None,
        cooldown_remaining=0.0,
        just_triggered=False,
        trigger_message="",
    ):
        """Render all overlay elements on a BGR frame."""
        h, w = frame.shape[:2]
        display_name = config.GESTURE_DISPLAY_NAME.get(gesture, gesture)
        tracking_text = config.TRACKING_DISPLAY_NAME.get(tracking_state, tracking_state)
        conf_text = f"Confidence: {confidence:.0%}"

        if tracking_state != "ready":
            color = self.COLOR_WARNING
        else:
            color = self.COLOR_IDLE if gesture == "idle" else self.COLOR_GESTURE

        cv2.rectangle(frame, (5, 5), (380, 95), self.COLOR_BG, -1)
        cv2.rectangle(frame, (5, 5), (380, 95), color, 2)
        cv2.putText(frame, display_name, (15, 35), self.font, 0.8, color, 2)
        cv2.putText(frame, conf_text, (15, 60), self.font, 0.5, self.COLOR_TEXT, 1)
        cv2.putText(frame, tracking_text, (15, 82), self.font, 0.5, self.COLOR_TEXT, 1)

        if fps is None and "total" in timings and timings["total"] > 0:
            fps = 1000.0 / timings["total"]
        if fps is not None:
            fps_text = f"FPS: {fps:.1f}"
            (tw, th), _ = cv2.getTextSize(fps_text, self.font, 0.6, 2)
            cv2.rectangle(frame, (w - tw - 20, 5), (w - 5, th + 20), self.COLOR_BG, -1)
            cv2.putText(
                frame,
                fps_text,
                (w - tw - 12, th + 12),
                self.font,
                0.6,
                self.COLOR_TEXT,
                2,
            )

        if config.SHOW_TIMING and timings:
            box_height = 25 * (len(timings) + 1)
            cv2.rectangle(frame, (5, h - box_height - 5), (290, h - 5), self.COLOR_BG, -1)
            cv2.putText(
                frame,
                "Timing (ms):",
                (10, h - box_height + 15),
                self.font,
                0.45,
                self.COLOR_TEXT,
                1,
            )
            for i, (step, ms) in enumerate(timings.items()):
                line = f"  {step:12s}: {ms:6.2f}"
                cv2.putText(
                    frame,
                    line,
                    (10, h - box_height + 35 + i * 22),
                    self.font,
                    0.4,
                    self.COLOR_TIMING,
                    1,
                )

        if cooldown_remaining > 0:
            bar_w = 200
            bar_h = 12
            bar_x = w - bar_w - 10
            bar_y = h - 25
            ratio = min(1.0, cooldown_remaining / config.TRIGGER_COOLDOWN)
            filled_w = int(bar_w * ratio)

            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + filled_w, bar_y + bar_h),
                self.COLOR_COOLDOWN_BAR,
                -1,
            )

            label = f"Cooldown: {cooldown_remaining:.1f}s"
            cv2.putText(frame, label, (bar_x, bar_y - 5), self.font, 0.4, self.COLOR_COOLDOWN_BAR, 1)

        if just_triggered:
            self.last_trigger_message = trigger_message
            self.last_trigger_time = time.time()

        if time.time() - self.last_trigger_time < 1.5 and self.last_trigger_message:
            banner = f">> {self.last_trigger_message} <<"
            (tw, th), _ = cv2.getTextSize(banner, self.font, 0.7, 2)
            x = (w - tw) // 2
            y = 120
            cv2.rectangle(frame, (x - 10, y - th - 5), (x + tw + 10, y + 8), self.COLOR_BG, -1)
            cv2.rectangle(
                frame,
                (x - 10, y - th - 5),
                (x + tw + 10, y + 8),
                self.COLOR_TRIGGER,
                2,
            )
            cv2.putText(frame, banner, (x, y), self.font, 0.7, self.COLOR_TRIGGER, 2)

        return frame


if __name__ == "__main__":
    display = Display()
    frame = np.full((480, 640, 3), 50, dtype=np.uint8)
    timings = {
        "capture": 2.34,
        "preprocess": 3.12,
        "detect": 18.50,
        "features": 0.80,
        "classify": 0.45,
        "trigger": 0.05,
        "total": 25.26,
    }

    print("Showing display preview. Press ESC to quit.")
    states = [
        ("idle", 0.20, "idle", False, ""),
        ("right_arm", 0.92, "ready", True, "Triggered: Next Slide"),
        ("right_arm", 0.92, "ready", False, ""),
        ("idle", 0.00, "no_pose", False, ""),
    ]

    state_idx = 0
    last_change = time.time()
    cooldown_remaining = 0.0

    while True:
        if time.time() - last_change > 2:
            state_idx = (state_idx + 1) % len(states)
            last_change = time.time()
            _, _, _, triggered, _ = states[state_idx]
            if triggered:
                cooldown_remaining = config.TRIGGER_COOLDOWN

        cooldown_remaining = max(0.0, cooldown_remaining - 0.03)

        gesture, conf, tracking_state, _, msg = states[state_idx]
        just_triggered = (time.time() - last_change < 0.1) and bool(msg)

        canvas = frame.copy()
        canvas = display.render(
            canvas,
            gesture,
            conf,
            timings,
            tracking_state=tracking_state,
            fps=39.6,
            cooldown_remaining=cooldown_remaining,
            just_triggered=just_triggered,
            trigger_message=msg,
        )

        cv2.imshow("Display Test", canvas)
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
