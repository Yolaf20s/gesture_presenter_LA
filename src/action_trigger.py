"""
Convert stable gesture predictions into keyboard actions.

This layer enforces:
- a minimum confidence threshold
- cooldown between actions
- release-to-rearm behavior
- optional frontmost-application checks
"""

import platform
import subprocess
import time

import config

try:
    import pyautogui
    pyautogui.FAILSAFE = config.PYAUTOGUI_FAILSAFE
    PYAUTOGUI_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local GUI environment
    pyautogui = None
    PYAUTOGUI_IMPORT_ERROR = exc


class ActionTrigger:
    """Dispatch keyboard shortcuts for stable gesture predictions."""

    def __init__(self, cooldown=None):
        self.cooldown = cooldown if cooldown is not None else config.TRIGGER_COOLDOWN
        self.last_trigger_time = 0.0
        self.last_gesture = None
        self.trigger_count = 0
        self.armed = True
        self._focus_provider = None

        print(f"[ActionTrigger] Cooldown = {self.cooldown:.1f}s")

    def set_focus_provider(self, provider):
        """Inject a focus lookup function for tests."""
        self._focus_provider = provider

    def trigger(self, gesture, confidence, tracking_state="ready"):
        """
        Trigger an action if all safety checks pass.

        Return `(triggered: bool, message: str)`.
        """
        if tracking_state in config.REARM_STATES:
            self.armed = True
            if tracking_state != "ready":
                return False, config.TRACKING_DISPLAY_NAME.get(tracking_state, tracking_state)

        if confidence < config.CONFIDENCE_THRESHOLD:
            return False, f"Skipped (low confidence {confidence:.2f})"

        if gesture == "idle":
            self.armed = True
            return False, "Idle"

        if not self.armed:
            return False, "Waiting for release"

        elapsed = time.time() - self.last_trigger_time
        if elapsed < self.cooldown:
            remaining = self.cooldown - elapsed
            return False, f"Cooldown ({remaining:.1f}s)"

        focus_ok, focus_message = self._focus_allows_trigger()
        if not focus_ok:
            return False, focus_message

        key = config.GESTURE_TO_KEY.get(gesture)
        if key is None:
            return False, f"No key mapping for {gesture}"

        if pyautogui is None:
            return False, f"Trigger backend unavailable: {PYAUTOGUI_IMPORT_ERROR}"

        try:
            pyautogui.press(key)
        except Exception as exc:
            return False, f"Trigger failed: {exc}"

        self.last_trigger_time = time.time()
        self.last_gesture = gesture
        self.trigger_count += 1
        self.armed = False

        display_name = config.GESTURE_DISPLAY_NAME.get(gesture, gesture)
        return True, f"Triggered: {display_name} (key='{key}')"

    def get_cooldown_remaining(self):
        elapsed = time.time() - self.last_trigger_time
        return max(0.0, self.cooldown - elapsed)

    def _focus_allows_trigger(self):
        if not config.ENFORCE_TARGET_APP_FOCUS:
            return True, "Focus guard disabled"

        app_name = self._get_frontmost_app_name()
        if app_name is None:
            if config.ALLOW_TRIGGER_WHEN_FOCUS_UNKNOWN:
                return True, "Focus unknown"
            return False, "Blocked: could not determine the focused app"

        if config.TARGET_APP_NAME.lower() in app_name.lower():
            return True, f"Focused app: {app_name}"
        return False, f"Blocked: focused app is '{app_name}', expected '{config.TARGET_APP_NAME}'"

    def _get_frontmost_app_name(self):
        if self._focus_provider is not None:
            return self._focus_provider()

        system = platform.system().lower()
        if system == "darwin":
            script = (
                'tell application "System Events" '
                'to get name of first application process whose frontmost is true'
            )
            try:
                result = subprocess.run(
                    ["osascript", "-e", script],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=1.5,
                )
            except (subprocess.SubprocessError, OSError):
                return None
            return result.stdout.strip() or None

        return None


if __name__ == "__main__":
    print("=" * 60)
    print("ActionTrigger smoke test")
    print("=" * 60)
    print("Warning: this sends real key presses to the focused application.")
    print("Switch to a harmless target window if you want to test it.\n")
    time.sleep(3)

    trigger = ActionTrigger(cooldown=2.0)
    test_sequence = [
        ("right_arm", 0.95, "ready"),
        ("right_arm", 0.95, "ready"),
        ("idle", 0.99, "idle"),
        ("left_arm", 0.85, "ready"),
        ("thumb_up", 0.30, "ready"),
    ]

    for gesture, conf, state in test_sequence:
        triggered, msg = trigger.trigger(gesture, conf, tracking_state=state)
        status = "OK" if triggered else "--"
        print(f"[{status}] gesture={gesture:14s} conf={conf:.2f} -> {msg}")
        time.sleep(0.3)
