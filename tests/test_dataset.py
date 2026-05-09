import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.dataset import load_feature_sessions, split_feature_dataset


def _write_session(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(rows, dtype=np.float32))


class DatasetSplitTests(unittest.TestCase):
    def test_prefers_person_holdout_when_multiple_people_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_session(root / "alice" / "right_arm" / "session_1.npy", [[1, 2, 3], [1, 2, 4]])
            _write_session(root / "alice" / "idle" / "session_1.npy", [[0, 0, 0], [0, 0, 1]])
            _write_session(root / "bob" / "right_arm" / "session_1.npy", [[9, 9, 9], [9, 9, 8]])
            _write_session(root / "bob" / "idle" / "session_1.npy", [[5, 5, 5], [5, 5, 4]])

            sessions = load_feature_sessions(dataset_dir=root.as_posix())
            X_train, y_train, X_test, y_test, summary = split_feature_dataset(
                sessions,
                test_ratio=0.5,
                seed=7,
            )

            self.assertEqual(summary["strategy"], "person_holdout")
            self.assertEqual(len(summary["test_groups"]), 1)
            self.assertGreater(len(X_train), 0)
            self.assertGreater(len(X_test), 0)
            self.assertSetEqual(set(np.unique(y_train)) | set(np.unique(y_test)), {"right_arm", "idle"})

    def test_falls_back_to_contiguous_split_for_single_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_session(root / "alice" / "right_arm.npy", np.arange(30).reshape(10, 3))
            _write_session(root / "alice" / "idle.npy", np.arange(30, 60).reshape(10, 3))

            sessions = load_feature_sessions(dataset_dir=root.as_posix())
            X_train, y_train, X_test, y_test, summary = split_feature_dataset(
                sessions,
                test_ratio=0.2,
                seed=7,
            )

            self.assertEqual(summary["strategy"], "contiguous_within_session")
            self.assertIn("contiguous holdout chunk", summary["warning"])
            self.assertEqual(len(X_train) + len(X_test), 20)
            self.assertEqual(len(y_train) + len(y_test), 20)


if __name__ == "__main__":
    unittest.main()
