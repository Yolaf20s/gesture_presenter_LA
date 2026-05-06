"""
Class Display: ve overlay len frame.
Step 7 cua pipeline.

Hien thi:
1. Skeleton (tu MediaPipe)
2. Gesture detected + confidence
3. Performance metrics: FPS, timing ms tung step
4. Cooldown bar (con bao nhieu giay nua moi trigger duoc)
"""

import cv2
import numpy as np
import config


class Display:
    """Ve overlay info len frame."""

    # Mau theo BGR (OpenCV dung BGR)
    COLOR_BG = (0, 0, 0)            # den
    COLOR_TEXT = (255, 255, 255)    # trang
    COLOR_GESTURE = (0, 255, 0)     # xanh la (gesture active)
    COLOR_IDLE = (128, 128, 128)    # xam (idle)
    COLOR_TRIGGER = (0, 255, 255)   # vang (vua trigger)
    COLOR_TIMING = (255, 200, 100)  # cam nhat (timing)
    COLOR_COOLDOWN_BAR = (0, 165, 255)  # cam (cooldown)

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.last_trigger_message = ""
        self.last_trigger_time = 0.0
        print("[Display] San sang.")

    def render(self, frame, gesture, confidence, timings, cooldown_remaining=0,
               just_triggered=False, trigger_message=""):
        """
        Render tat ca overlay len frame.

        frame: BGR frame (da co skeleton vẽ truoc do).
        gesture: string label hien tai.
        confidence: float [0, 1].
        timings: dict {step_name: ms}.
        cooldown_remaining: float, giay con lai.
        just_triggered: bool, vua trigger trong frame nay khong.
        trigger_message: string mo ta trigger.
        Return: frame voi overlay.
        """
        h, w = frame.shape[:2]

        # ====================================================
        # Top-left: Gesture label + confidence
        # ====================================================
        display_name = config.GESTURE_DISPLAY_NAME.get(gesture, gesture)
        gesture_text = f"{display_name}"
        conf_text = f"Confidence: {confidence:.0%}"

        # Mau theo gesture
        color = self.COLOR_IDLE if gesture == 'idle' else self.COLOR_GESTURE

        # Background box cho de doc
        cv2.rectangle(frame, (5, 5), (350, 75), self.COLOR_BG, -1)
        cv2.rectangle(frame, (5, 5), (350, 75), color, 2)

        cv2.putText(frame, gesture_text, (15, 35),
                    self.font, 0.8, color, 2)
        cv2.putText(frame, conf_text, (15, 60),
                    self.font, 0.5, self.COLOR_TEXT, 1)

        # ====================================================
        # Top-right: FPS
        # ====================================================
        if 'total' in timings and timings['total'] > 0:
            fps = 1000.0 / timings['total']
            fps_text = f"FPS: {fps:.0f}"
            (tw, th), _ = cv2.getTextSize(fps_text, self.font, 0.6, 2)
            cv2.rectangle(frame, (w - tw - 20, 5), (w - 5, th + 20),
                          self.COLOR_BG, -1)
            cv2.putText(frame, fps_text, (w - tw - 12, th + 12),
                        self.font, 0.6, self.COLOR_TEXT, 2)

        # ====================================================
        # Bottom-left: Timing per step (yeu cau cua mon CV)
        # ====================================================
        if config.SHOW_TIMING and timings:
            box_height = 25 * (len(timings) + 1)
            cv2.rectangle(frame, (5, h - box_height - 5),
                          (260, h - 5), self.COLOR_BG, -1)
            cv2.putText(frame, "Timing (ms):", (10, h - box_height + 15),
                        self.font, 0.45, self.COLOR_TEXT, 1)
            for i, (step, ms) in enumerate(timings.items()):
                line = f"  {step:12s}: {ms:6.2f}"
                cv2.putText(frame, line,
                            (10, h - box_height + 35 + i * 22),
                            self.font, 0.4, self.COLOR_TIMING, 1)

        # ====================================================
        # Bottom-right: Cooldown bar
        # ====================================================
        if cooldown_remaining > 0:
            bar_w = 200
            bar_h = 12
            bar_x = w - bar_w - 10
            bar_y = h - 25
            # Tinh ti le con lai
            ratio = min(1.0, cooldown_remaining / config.TRIGGER_COOLDOWN)
            filled_w = int(bar_w * ratio)

            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h),
                          (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + filled_w, bar_y + bar_h),
                          self.COLOR_COOLDOWN_BAR, -1)

            label = f"Cooldown: {cooldown_remaining:.1f}s"
            cv2.putText(frame, label, (bar_x, bar_y - 5),
                        self.font, 0.4, self.COLOR_COOLDOWN_BAR, 1)

        # ====================================================
        # Trigger banner (giua tren) - hien 1.5s sau khi trigger
        # ====================================================
        import time
        if just_triggered:
            self.last_trigger_message = trigger_message
            self.last_trigger_time = time.time()

        # Hien banner trong 1.5s sau trigger
        if time.time() - self.last_trigger_time < 1.5 and self.last_trigger_message:
            banner = f">> {self.last_trigger_message} <<"
            (tw, th), _ = cv2.getTextSize(banner, self.font, 0.7, 2)
            x = (w - tw) // 2
            y = 110
            cv2.rectangle(frame, (x - 10, y - th - 5),
                          (x + tw + 10, y + 8),
                          self.COLOR_BG, -1)
            cv2.rectangle(frame, (x - 10, y - th - 5),
                          (x + tw + 10, y + 8),
                          self.COLOR_TRIGGER, 2)
            cv2.putText(frame, banner, (x, y),
                        self.font, 0.7, self.COLOR_TRIGGER, 2)

        return frame


# ============================================================
# Test khi chay file truc tiep
# ============================================================
if __name__ == '__main__':
    """Test Display: ve overlay len frame trang voi gia tri gia."""
    import time

    display = Display()

    # Tao frame trang
    frame = np.full((480, 640, 3), 50, dtype=np.uint8)  # xam toi

    # Gia lap timings
    timings = {
        'capture':    2.34,
        'preprocess': 3.12,
        'detect':    18.50,
        'features':   0.80,
        'classify':   0.45,
        'trigger':    0.05,
        'total':     25.26,
    }

    print("Hien preview cua Display overlay. Nhan ESC de thoat.")
    print("Cycle qua cac state khac nhau...")

    states = [
        ('idle', 0.20, False, ""),
        ('right_arm', 0.92, True, "Triggered: Next Slide ->"),
        ('right_arm', 0.92, False, ""),
        ('thumb_up', 0.78, True, "Triggered: Confetti!"),
        ('idle', 0.30, False, ""),
    ]

    state_idx = 0
    last_change = time.time()
    cooldown_remaining = 0

    while True:
        # Doi state moi 2s
        if time.time() - last_change > 2:
            state_idx = (state_idx + 1) % len(states)
            last_change = time.time()
            gesture, conf, triggered, msg = states[state_idx]
            if triggered:
                cooldown_remaining = config.TRIGGER_COOLDOWN

        # Update cooldown
        cooldown_remaining = max(0, cooldown_remaining - 0.03)

        gesture, conf, _, msg = states[state_idx]
        just_triggered = (time.time() - last_change < 0.1) and msg != ""

        canvas = frame.copy()
        canvas = display.render(
            canvas, gesture, conf, timings,
            cooldown_remaining=cooldown_remaining,
            just_triggered=just_triggered,
            trigger_message=msg,
        )

        cv2.imshow('Display Test', canvas)
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    print("Test hoan tat.")