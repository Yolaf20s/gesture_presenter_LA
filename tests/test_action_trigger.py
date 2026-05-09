import unittest

import src.action_trigger as action_trigger_module
from src.action_trigger import ActionTrigger


class _FakePyAutoGUI:
    def __init__(self):
        self.pressed = []

    def press(self, key):
        self.pressed.append(key)


class ActionTriggerTests(unittest.TestCase):
    def setUp(self):
        self.original_pyautogui = action_trigger_module.pyautogui
        self.original_import_error = action_trigger_module.PYAUTOGUI_IMPORT_ERROR
        self.fake_pyautogui = _FakePyAutoGUI()
        action_trigger_module.pyautogui = self.fake_pyautogui
        action_trigger_module.PYAUTOGUI_IMPORT_ERROR = None

    def tearDown(self):
        action_trigger_module.pyautogui = self.original_pyautogui
        action_trigger_module.PYAUTOGUI_IMPORT_ERROR = self.original_import_error

    def test_requires_release_before_retriggering(self):
        trigger = ActionTrigger(cooldown=0.0)
        trigger.set_focus_provider(lambda: "Canva")

        fired, message = trigger.trigger("right_arm", 0.95, tracking_state="ready")
        self.assertTrue(fired, message)
        self.assertEqual(self.fake_pyautogui.pressed, ["right"])

        fired, message = trigger.trigger("right_arm", 0.95, tracking_state="ready")
        self.assertFalse(fired)
        self.assertEqual(message, "Waiting for release")
        self.assertEqual(self.fake_pyautogui.pressed, ["right"])

        fired, _ = trigger.trigger("idle", 0.95, tracking_state="idle")
        self.assertFalse(fired)

        fired, message = trigger.trigger("right_arm", 0.95, tracking_state="ready")
        self.assertTrue(fired, message)
        self.assertEqual(self.fake_pyautogui.pressed, ["right", "right"])

    def test_blocks_when_wrong_app_is_focused(self):
        trigger = ActionTrigger(cooldown=0.0)
        trigger.set_focus_provider(lambda: "TextEdit")

        fired, message = trigger.trigger("left_arm", 0.95, tracking_state="ready")
        self.assertFalse(fired)
        self.assertIn("focused app", message)
        self.assertEqual(self.fake_pyautogui.pressed, [])


if __name__ == "__main__":
    unittest.main()
