import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
from src import mediapipe_models


class MediaPipeModelTests(unittest.TestCase):
    def test_pose_model_spec_matches_complexity(self):
        self.assertEqual(
            mediapipe_models.get_pose_model_spec(0).filename,
            "pose_landmarker_lite.task",
        )
        self.assertEqual(
            mediapipe_models.get_pose_model_spec(1).filename,
            "pose_landmarker_full.task",
        )
        self.assertEqual(
            mediapipe_models.get_pose_model_spec(2).filename,
            "pose_landmarker_heavy.task",
        )

    def test_optional_model_returns_none_when_missing_and_no_download(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hand_landmarker.task"
            spec = mediapipe_models.get_hand_model_spec()
            with mock.patch.object(config, "MEDIAPIPE_AUTO_DOWNLOAD_MODELS", False):
                result = mediapipe_models.ensure_model(
                    path=path,
                    spec=spec,
                    auto_download=False,
                    required=False,
                    model_name="hand",
                )
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
