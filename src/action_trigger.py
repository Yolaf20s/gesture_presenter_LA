"""
Class ActionTrigger: gui phim tat den ung dung dang focused (Canva).
Step 6 cua pipeline.

Logic:
- Nhan gesture + confidence
- Check confidence > threshold
- Check cooldown da het
- pyautogui.press(key) -> phim tat duoc gui den ung dung dang focus
"""

import time
import pyautogui
import config


# Tat fail-safe cua pyautogui.
pyautogui.FAILSAFE = False


class ActionTrigger:
    """Trigger phim tat dua tren gesture detected."""

    def __init__(self, cooldown=None):
        self.cooldown = cooldown if cooldown is not None else config.TRIGGER_COOLDOWN
        self.last_trigger_time = 0.0
        self.last_gesture = None
        self.trigger_count = 0

        print(f"[ActionTrigger] Cooldown = {self.cooldown}s")

    def trigger(self, gesture, confidence):
        """
        Trigger action neu du dieu kien.

        Return: tuple (triggered: bool, message: str)
        """
        # Skip neu confidence thap
        if confidence < config.CONFIDENCE_THRESHOLD:
            return False, f"Skipped (low confidence {confidence:.2f})"

        # Skip neu la idle
        if gesture == 'idle':
            return False, "Idle"

        # Skip neu chua het cooldown
        elapsed = time.time() - self.last_trigger_time
        if elapsed < self.cooldown:
            remaining = self.cooldown - elapsed
            return False, f"Cooldown ({remaining:.1f}s)"

        # Lay phim tat tu config
        key = config.GESTURE_TO_KEY.get(gesture)
        if key is None:
            return False, f"No key mapping for {gesture}"

        # TRIGGER!
        try:
            pyautogui.press(key)
        except Exception as e:
            return False, f"Error: {e}"

        # Update state
        self.last_trigger_time = time.time()
        self.last_gesture = gesture
        self.trigger_count += 1

        display_name = config.GESTURE_DISPLAY_NAME.get(gesture, gesture)
        return True, f"Triggered: {display_name} (key='{key}')"

    def get_cooldown_remaining(self):
        """Tra ve thoi gian cooldown con lai (giay), 0 neu da het."""
        elapsed = time.time() - self.last_trigger_time
        return max(0.0, self.cooldown - elapsed)


# ============================================================
# Test khi chay file truc tiep
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("TEST ActionTrigger")
    print("=" * 60)
    print("\nLuu y: Test nay se gui phim tat that den ung dung dang focused!")
    print("Hay mo Notepad len lam target de xem ket qua.")
    print("Doi 3 giay de switch sang Notepad...\n")
    time.sleep(3)

    trigger = ActionTrigger(cooldown=2.0)

    test_sequence = [
        ('right_arm', 0.95),    # Trigger -> phim 'right'
        ('right_arm', 0.95),    # Skip (cooldown)
        ('idle', 0.99),         # Skip (idle)
        ('left_arm', 0.85),     # Skip (cooldown chua het)
        ('thumb_up', 0.30),     # Skip (low confidence)
    ]

    print("Bat dau test sequence (cooldown=1s):")
    for gesture, conf in test_sequence:
        triggered, msg = trigger.trigger(gesture, conf)
        status = "OK" if triggered else "--"
        print(f"  [{status}] gesture={gesture:14s} conf={conf:.2f}  -> {msg}")
        time.sleep(0.3)

    print(f"\nDoi 1.5s cho cooldown het...")
    time.sleep(1.5)

    print("Test sau cooldown:")
    triggered, msg = trigger.trigger('left_arm', 0.90)
    status = "OK" if triggered else "--"
    print(f"  [{status}] gesture=left_arm  conf=0.90  -> {msg}")

    print(f"\nTotal triggers: {trigger.trigger_count}")
    print("\nTest hoan tat. ActionTrigger hoat dong dung.")