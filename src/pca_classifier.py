"""
Class PCAClassifier: Linear Algebra core cua project.
Step 5 cua pipeline.

Tu implement PCA + kNN tu dau (KHONG dung sklearn.PCA).
Muc dich: chung minh hieu ly thuyet PCA cho mon Linear Algebra.

Linear Algebra dung trong file nay:
- Mean centering (vector subtraction)
- Covariance matrix: C = (1/(n-1)) * X^T * X
- Eigendecomposition: C = V * Lambda * V^T
- Projection xuong khong gian k chieu: X_proj = X_centered * V_k
- Euclidean distance trong khong gian PCA (cho kNN)
- Spectral theorem cho symmetric matrices
"""

import numpy as np
import os
import config


class PCAClassifier:
    """
    PCA + kNN classifier.

    Workflow:
    1. fit(X, y): train tu dataset
    2. predict(x): du doan label cho 1 sample moi
    3. save(path) / load(path): luu/tai model
    """

    def __init__(self, n_components=None, knn_k=None):
        """
        n_components: so principal components giu lai. None -> lay tu config.
        knn_k: so neighbors trong kNN. None -> lay tu config.
        """
        self.n_components = n_components or config.PCA_N_COMPONENTS
        self.knn_k = knn_k or config.KNN_K

        # Cac tham so se duoc fit
        self.mean = None              # vector trung binh shape (n_features,)
        self.eigenvectors = None      # ma tran (n_features, n_components)
        self.eigenvalues = None       # vector eigenvalues (n_components,)
        self.train_projections = None # toa do training samples trong PCA space
        self.train_labels = None      # labels (string)
        self.classes = None           # list cac class duy nhat

        self.is_fitted = False

    # ====================================================
    # TRAINING
    # ====================================================
    def fit(self, X, y):
        """
        Train PCA + memorize training data cho kNN.

        X: numpy array shape (n_samples, n_features). VD: (1500, 13)
        y: list/array string labels shape (n_samples,)

        Linear Algebra:
        1. Tinh mean -> centering
        2. Tinh covariance matrix
        3. Eigendecomposition -> giu top-k eigenvectors
        4. Project training data xuong PCA space -> luu de kNN dung
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        if self.n_components > n_features:
            raise ValueError(
                f"n_components={self.n_components} > n_features={n_features}. "
                f"Giam n_components di."
            )

        # Buoc 1: Mean centering
        # X_centered = X - mean(X)
        self.mean = X.mean(axis=0)  # shape (n_features,)
        X_centered = X - self.mean  # shape (n_samples, n_features)

        # Buoc 2: Covariance matrix
        # C = (1 / (n-1)) * X_centered.T @ X_centered
        # Shape: (n_features, n_features), e.g. (13, 13)
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

        # Buoc 3: Eigendecomposition
        # Vi cov_matrix la symmetric -> dung np.linalg.eigh (nhanh hon, sap xep tang dan)
        # eigh tra ve: eigenvalues, eigenvectors (cot la eigenvector)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sap xep giam dan theo eigenvalue (eigh tra ve tang dan)
        sorted_idx = np.argsort(eigenvalues)[::-1]  # reverse de duoc giam dan
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # Giu top-k components
        self.eigenvalues = eigenvalues[:self.n_components]
        self.eigenvectors = eigenvectors[:, :self.n_components]

        # Buoc 4: Project training data xuong PCA space
        # X_proj = X_centered @ V_k
        # Shape: (n_samples, n_components), e.g. (1500, 5)
        self.train_projections = X_centered @ self.eigenvectors

        # Luu labels cho kNN
        self.train_labels = y
        self.classes = np.unique(y)

        self.is_fitted = True

        # In info
        total_variance = np.sum(eigenvalues)
        kept_variance = np.sum(self.eigenvalues)
        print(
            f"[PCAClassifier] Da fit:\n"
            f"  - n_samples: {n_samples}\n"
            f"  - n_features: {n_features}\n"
            f"  - n_components: {self.n_components}\n"
            f"  - classes: {list(self.classes)}\n"
            f"  - variance giai thich: {kept_variance/total_variance*100:.1f}%\n"
            f"  - eigenvalues top-{self.n_components}: "
            f"{[f'{e:.3f}' for e in self.eigenvalues]}"
        )

        return self

    # ====================================================
    # PREDICTION
    # ====================================================
    def predict(self, x):
        """
        Du doan label cho 1 sample.

        x: numpy array shape (n_features,)
        Return: tuple (label_str, confidence_float [0-1])
        """
        if not self.is_fitted:
            raise RuntimeError("Phai goi fit() truoc!")

        x = np.asarray(x, dtype=np.float64).reshape(1, -1)

        # Project x xuong PCA space
        x_centered = x - self.mean
        x_proj = x_centered @ self.eigenvectors  # shape (1, n_components)

        # Tinh khoang cach Euclidean toi tat ca training samples
        # train_projections shape: (n_samples, n_components)
        # Hieu: shape (n_samples, n_components)
        diffs = self.train_projections - x_proj  # broadcast
        distances = np.linalg.norm(diffs, axis=1)  # shape (n_samples,)

        # Lay k neighbors gan nhat
        k = min(self.knn_k, len(distances))
        nearest_idx = np.argsort(distances)[:k]
        nearest_labels = self.train_labels[nearest_idx]

        # Majority vote
        unique, counts = np.unique(nearest_labels, return_counts=True)
        winner_idx = np.argmax(counts)
        predicted_label = unique[winner_idx]
        confidence = counts[winner_idx] / k

        return str(predicted_label), float(confidence)

    def predict_batch(self, X):
        """Du doan cho nhieu sample. Tra ve list (label, confidence)."""
        return [self.predict(x) for x in X]

    # ====================================================
    # SAVE / LOAD
    # ====================================================
    def save(self, path=None):
        """Luu model ra file .npz."""
        if not self.is_fitted:
            raise RuntimeError("Khong the save model chua fit!")
        path = path or config.PCA_MODEL_PATH

        # Tao folder neu chua co
        os.makedirs(os.path.dirname(path), exist_ok=True)

        np.savez(
            path,
            mean=self.mean,
            eigenvectors=self.eigenvectors,
            eigenvalues=self.eigenvalues,
            train_projections=self.train_projections,
            train_labels=self.train_labels,
            classes=self.classes,
            n_components=self.n_components,
            knn_k=self.knn_k,
        )
        print(f"[PCAClassifier] Da save model -> {path}")

    @classmethod
    def load(cls, path=None):
        """Load model tu file .npz."""
        path = path or config.PCA_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Khong tim thay model file: {path}")

        data = np.load(path, allow_pickle=True)
        clf = cls(
            n_components=int(data['n_components']),
            knn_k=int(data['knn_k']),
        )
        clf.mean = data['mean']
        clf.eigenvectors = data['eigenvectors']
        clf.eigenvalues = data['eigenvalues']
        clf.train_projections = data['train_projections']
        clf.train_labels = data['train_labels']
        clf.classes = data['classes']
        clf.is_fitted = True

        print(f"[PCAClassifier] Da load model tu {path}")
        return clf


# ============================================================
# Test khi chay file truc tiep
# ============================================================
if __name__ == '__main__':
    """Test PCAClassifier voi du lieu gia (synthetic) de verify dung."""
    print("=" * 60)
    print("TEST PCAClassifier voi du lieu synthetic")
    print("=" * 60)

    # Tao 4 cluster trong khong gian 13D
    np.random.seed(42)
    n_per_class = 50

    # Moi class la 1 cluster Gaussian quanh 1 centroid khac nhau
    centroids = {
        'right_arm':    np.array([0.5, 1.5, 0.0, 1.5, -0.5, 0.0, 0.5, 0.0, 1.0, -1, -1, -1, -1]),
        'left_arm':     np.array([1.5, 0.5, 1.5, 0.0, 0.0, -0.5, 0.0, -0.5, 1.0, -1, -1, -1, -1]),
        'five_fingers': np.array([1.5, 1.5, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.5, 5.0, 1.2, -1, -1]),
        'thumb_up':     np.array([1.5, 1.5, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.2, -1, -1]),
        'idle':         np.array([1.5, 1.5, 0.3, 0.3, 1.0, 1.0, -0.3, 0.3, 1.5, -1, -1, -1, -1]),
    }

    X_list, y_list = [], []
    for label, center in centroids.items():
        samples = center + np.random.randn(n_per_class, 13) * 0.2  # noise nho
        X_list.append(samples)
        y_list.extend([label] * n_per_class)

    X = np.vstack(X_list)
    y = np.array(y_list)
    print(f"Synthetic dataset: X shape {X.shape}, y len {len(y)}")

    # Fit
    clf = PCAClassifier(n_components=5, knn_k=5)
    clf.fit(X, y)

    # Test predict tren training data (de verify dung)
    print("\nTest predict tren 5 sample bat ky:")
    for i in [0, 50, 100, 150, 200]:
        label_pred, conf = clf.predict(X[i])
        match = "OK" if label_pred == y[i] else "WRONG"
        print(f"  Sample {i}: true={y[i]:14s}  pred={label_pred:14s}  conf={conf:.2f}  [{match}]")

    # Test save/load
    test_path = 'models/test_pca_model.npz'
    clf.save(test_path)
    clf2 = PCAClassifier.load(test_path)
    label_pred, conf = clf2.predict(X[0])
    print(f"\nSau khi load: predict sample 0 = {label_pred} (conf {conf:.2f})")

    # Cleanup
    os.remove(test_path)
    print("\nTest hoan tat. PCAClassifier hoat dong dung.")