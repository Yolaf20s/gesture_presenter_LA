"""
PCA + weighted kNN gesture classifier.

The project intentionally implements PCA directly instead of delegating to
`sklearn.PCA`, but now standardizes mixed-scale features and uses a more honest
distance-aware confidence score.
"""

import os

import numpy as np

import config


class PCAClassifier:
    """Linear PCA projection plus weighted kNN classification."""

    def __init__(self, n_components=None, knn_k=None):
        self.n_components = n_components or config.PCA_N_COMPONENTS
        self.knn_k = knn_k or config.KNN_K

        self.mean = None
        self.scale = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.explained_variance_ratio = None
        self.train_projections = None
        self.train_labels = None
        self.classes = None
        self.distance_scale = 1.0

        self.is_fitted = False

    def fit(self, X, y):
        """Fit PCA and cache the projected training set for weighted kNN."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        if self.n_components > n_features:
            raise ValueError(
                f"n_components={self.n_components} exceeds n_features={n_features}."
            )

        self.mean = X.mean(axis=0)
        if config.PCA_STANDARDIZE_FEATURES:
            self.scale = X.std(axis=0)
            self.scale[self.scale < 1e-6] = 1.0
        else:
            self.scale = np.ones(n_features, dtype=np.float64)

        X_standardized = self._prepare_matrix(X)
        cov_matrix = (X_standardized.T @ X_standardized) / max(1, n_samples - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        self.eigenvalues = eigenvalues[: self.n_components]
        self.eigenvectors = eigenvectors[:, : self.n_components]
        total_variance = float(np.sum(eigenvalues)) or 1.0
        self.explained_variance_ratio = self.eigenvalues / total_variance

        self.train_projections = X_standardized @ self.eigenvectors
        self.train_labels = y
        self.classes = np.unique(y)
        self.distance_scale = self._estimate_distance_scale(self.train_projections)
        self.is_fitted = True

        print(
            "[PCAClassifier] Fit complete:\n"
            f"  - samples: {n_samples}\n"
            f"  - features: {n_features}\n"
            f"  - components: {self.n_components}\n"
            f"  - classes: {list(self.classes)}\n"
            f"  - retained variance: {self.explained_variance_ratio.sum() * 100:.1f}%\n"
            f"  - neighbor distance scale: {self.distance_scale:.4f}"
        )
        return self

    def project(self, x):
        """Project a single sample or batch into PCA space."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() or load() before projecting.")

        x = np.asarray(x, dtype=np.float64)
        is_single = x.ndim == 1
        if is_single:
            x = x.reshape(1, -1)

        standardized = self._prepare_matrix(x)
        projected = standardized @ self.eigenvectors
        return projected[0] if is_single else projected

    def predict(self, x):
        """
        Predict a label and a distance-aware confidence in [0, 1].

        The confidence combines:
        - weighted vote share among the nearest neighbors
        - a penalty when the sample sits farther away than a typical training
          neighborhood
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() or load() before predict().")

        x_proj = self.project(x).reshape(1, -1)
        diffs = self.train_projections - x_proj
        distances = np.linalg.norm(diffs, axis=1)

        k = min(self.knn_k, len(distances))
        nearest_idx = np.argsort(distances)[:k]
        nearest_distances = distances[nearest_idx]
        nearest_labels = self.train_labels[nearest_idx]

        weights = 1.0 / (nearest_distances + 1e-6)
        label_weights = {}
        for label, weight in zip(nearest_labels, weights):
            label_weights[label] = label_weights.get(label, 0.0) + float(weight)

        predicted_label = max(label_weights, key=label_weights.get)
        winner_weight = label_weights[predicted_label]
        total_weight = sum(label_weights.values()) or 1.0
        vote_confidence = winner_weight / total_weight

        mean_neighbor_distance = float(np.mean(nearest_distances)) if len(nearest_distances) else 0.0
        overrun = max(0.0, mean_neighbor_distance - self.distance_scale)
        proximity_confidence = 1.0 / (1.0 + overrun / (self.distance_scale + 1e-6))
        confidence = float(np.clip(vote_confidence * proximity_confidence, 0.0, 1.0))

        return str(predicted_label), confidence

    def predict_batch(self, X):
        return [self.predict(x) for x in X]

    def save(self, path=None):
        """Save the fitted model to an `.npz` file."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        path = path or config.PCA_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)

        np.savez(
            path,
            mean=self.mean,
            scale=self.scale,
            eigenvectors=self.eigenvectors,
            eigenvalues=self.eigenvalues,
            explained_variance_ratio=self.explained_variance_ratio,
            train_projections=self.train_projections,
            train_labels=self.train_labels,
            classes=self.classes,
            n_components=self.n_components,
            knn_k=self.knn_k,
            distance_scale=self.distance_scale,
        )
        print(f"[PCAClassifier] Saved model -> {path}")

    @classmethod
    def load(cls, path=None):
        """Load a model from an `.npz` file. Supports legacy saves."""
        path = path or config.PCA_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        data = np.load(path, allow_pickle=True)
        clf = cls(
            n_components=int(data["n_components"]),
            knn_k=int(data["knn_k"]),
        )
        clf.mean = data["mean"]
        clf.scale = data["scale"] if "scale" in data else np.ones_like(clf.mean)
        clf.eigenvectors = data["eigenvectors"]
        clf.eigenvalues = data["eigenvalues"]
        if "explained_variance_ratio" in data:
            clf.explained_variance_ratio = data["explained_variance_ratio"]
        else:
            total = float(np.sum(clf.eigenvalues)) or 1.0
            clf.explained_variance_ratio = clf.eigenvalues / total
        clf.train_projections = data["train_projections"]
        clf.train_labels = data["train_labels"]
        clf.classes = data["classes"]
        clf.distance_scale = (
            float(data["distance_scale"])
            if "distance_scale" in data
            else clf._estimate_distance_scale(clf.train_projections)
        )
        clf.is_fitted = True

        print(f"[PCAClassifier] Loaded model from {path}")
        return clf

    def _prepare_matrix(self, X):
        return (X - self.mean) / self.scale

    def _estimate_distance_scale(self, projections):
        if len(projections) < 2:
            return 1.0

        k = min(self.knn_k, len(projections) - 1)
        diffs = projections[:, None, :] - projections[None, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(distances, np.inf)
        nearest = np.sort(distances, axis=1)[:, :k]
        neighborhood_mean = np.mean(nearest, axis=1)
        scale = float(np.median(neighborhood_mean))
        return scale if scale > 1e-6 else 1.0


if __name__ == "__main__":
    print("=" * 60)
    print("Synthetic PCAClassifier smoke test")
    print("=" * 60)

    np.random.seed(42)
    n_per_class = 50
    centroids = {
        "right_arm": np.array([0.5, 1.5, 0.0, 1.5, -0.5, 0.0, 0.5, 0.0, 1.0, -1, -1, -1, -1]),
        "left_arm": np.array([1.5, 0.5, 1.5, 0.0, 0.0, -0.5, 0.0, -0.5, 1.0, -1, -1, -1, -1]),
        "five_fingers": np.array([1.5, 1.5, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.5, 5.0, 1.2, -1, -1]),
        "thumb_up": np.array([1.5, 1.5, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.2, -1, -1]),
        "idle": np.array([1.5, 1.5, 0.3, 0.3, 1.0, 1.0, -0.3, 0.3, 1.5, -1, -1, -1, -1]),
    }

    X_list, y_list = [], []
    for label, center in centroids.items():
        samples = center + np.random.randn(n_per_class, 13) * 0.2
        X_list.append(samples)
        y_list.extend([label] * n_per_class)

    X = np.vstack(X_list)
    y = np.array(y_list)
    clf = PCAClassifier(n_components=5, knn_k=5).fit(X, y)

    print("\nPredictions for a few training samples:")
    for i in [0, 50, 100, 150, 200]:
        pred, conf = clf.predict(X[i])
        verdict = "OK" if pred == y[i] else "WRONG"
        print(
            f"  sample {i:3d}: true={y[i]:14s} pred={pred:14s} "
            f"conf={conf:.2f} [{verdict}]"
        )
