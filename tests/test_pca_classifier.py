import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.pca_classifier import PCAClassifier


class PCAClassifierTests(unittest.TestCase):
    def test_fit_predict_and_save_roundtrip(self):
        rng = np.random.default_rng(123)
        class_a = np.column_stack(
            [
                rng.normal(-2.0, 0.3, size=30),
                rng.normal(100.0, 10.0, size=30),
                rng.normal(0.1, 0.02, size=30),
            ]
        )
        class_b = np.column_stack(
            [
                rng.normal(2.0, 0.3, size=30),
                rng.normal(300.0, 10.0, size=30),
                rng.normal(0.8, 0.02, size=30),
            ]
        )
        X = np.vstack([class_a, class_b])
        y = np.array(["left"] * len(class_a) + ["right"] * len(class_b))

        clf = PCAClassifier(n_components=2, knn_k=3).fit(X, y)
        pred, conf = clf.predict(class_a[0])
        self.assertEqual(pred, "left")
        self.assertGreater(conf, 0.7)
        self.assertEqual(clf.project(class_a[0]).shape, (2,))
        self.assertEqual(clf.project(X[:5]).shape, (5, 2))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.npz"
            clf.save(path.as_posix())
            loaded = PCAClassifier.load(path.as_posix())
            loaded_pred, loaded_conf = loaded.predict(class_b[0])
            self.assertEqual(loaded_pred, "right")
            self.assertGreater(loaded_conf, 0.7)
            np.testing.assert_allclose(loaded.scale, clf.scale)


if __name__ == "__main__":
    unittest.main()
