"""
Train the PCA + weighted kNN gesture classifier.

Usage:
    python3 -m models.train_pca
"""

from collections import defaultdict

import numpy as np

import config
from src.dataset import load_feature_sessions, split_feature_dataset, summarize_sessions
from src.pca_classifier import PCAClassifier


def stratified_validation_split(X, y, val_ratio=0.2, seed=42):
    """Simple per-class validation split inside the development set."""
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []

    for label in np.unique(y):
        label_idx = np.flatnonzero(y == label)
        shuffled = rng.permutation(label_idx)
        n_val = min(max(1, int(round(len(shuffled) * val_ratio))), max(1, len(shuffled) - 1))
        val_indices.extend(shuffled[:n_val])
        train_indices.extend(shuffled[n_val:])

    train_indices = np.array(sorted(train_indices))
    val_indices = np.array(sorted(val_indices))
    return X[train_indices], X[val_indices], y[train_indices], y[val_indices]


def evaluate(clf, X_eval, y_eval):
    """Return accuracy and confusion matrix for a fitted classifier."""
    correct = 0
    confusion = {
        cls: {pred: 0 for pred in config.GESTURE_CLASSES}
        for cls in config.GESTURE_CLASSES
    }

    for x, true_label in zip(X_eval, y_eval):
        pred, _ = clf.predict(x)
        if pred == true_label:
            correct += 1
        confusion[true_label][pred] += 1

    accuracy = correct / len(X_eval)
    return accuracy, confusion


def print_confusion_matrix(confusion):
    classes = config.GESTURE_CLASSES
    print("\nConfusion Matrix:")
    header = "True / Pred"
    print(f"{header:<14}", end="")
    for label in classes:
        print(f"{label[:10]:>11}", end="")
    print()

    for true_label in classes:
        print(f"{true_label:<14}", end="")
        for pred_label in classes:
            print(f"{confusion[true_label][pred_label]:>11}", end="")
        print()


def print_label_distribution(name, labels):
    counts = defaultdict(int)
    for label in labels:
        counts[label] += 1
    parts = [f"{label}={counts[label]}" for label in config.GESTURE_CLASSES if counts[label]]
    print(f"{name}: {' '.join(parts)}")


def flatten_all_sessions(sessions):
    X_list = [session.samples for session in sessions]
    y_list = []
    for session in sessions:
        y_list.extend([session.gesture] * len(session.samples))
    return np.vstack(X_list), np.array(y_list)


def main():
    print("=" * 60)
    print("TRAIN PCA CLASSIFIER")
    print("=" * 60 + "\n")

    sessions = load_feature_sessions()
    total_samples, session_lines = summarize_sessions(sessions)
    print("Dataset sessions:")
    for line in session_lines:
        print(line)
    print(f"\nTotal samples: {total_samples}")

    X_dev, y_dev, X_test, y_test, split_summary = split_feature_dataset(sessions)
    print(f"\nEvaluation split strategy: {split_summary['strategy']}")
    if split_summary["warning"]:
        print(f"Warning: {split_summary['warning']}")
    print(f"Train/dev groups: {split_summary['train_groups']}")
    print(f"Test groups: {split_summary['test_groups']}")
    print(f"Development set: {len(X_dev)} samples")
    print(f"Test set: {len(X_test)} samples")
    print_label_distribution("Development labels", y_dev)
    print_label_distribution("Test labels", y_test)

    X_train, X_val, y_train, y_val = stratified_validation_split(
        X_dev,
        y_dev,
        val_ratio=0.2,
        seed=config.EVALUATION_RANDOM_SEED,
    )
    print(f"\nModel-selection split inside development set:")
    print(f"  train={len(X_train)} samples")
    print(f"  val={len(X_val)} samples")

    candidate_components = [3, 5, 7, 10, 13]
    print("\nSelecting PCA dimensionality on validation data:")
    print(f"{'n_comp':>8} {'train_acc':>12} {'val_acc':>12}")

    best_n = config.PCA_N_COMPONENTS
    best_val_acc = -1.0

    for n_comp in candidate_components:
        if n_comp > X_train.shape[1]:
            continue
        clf = PCAClassifier(n_components=n_comp, knn_k=config.KNN_K).fit(X_train, y_train)
        train_acc, _ = evaluate(clf, X_train, y_train)
        val_acc, _ = evaluate(clf, X_val, y_val)
        print(f"{n_comp:>8} {train_acc * 100:>11.1f}% {val_acc * 100:>11.1f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_n = n_comp

    print(f"\nSelected n_components={best_n} based on validation accuracy.")

    eval_model = PCAClassifier(n_components=best_n, knn_k=config.KNN_K).fit(X_dev, y_dev)
    test_acc, confusion = evaluate(eval_model, X_test, y_test)
    print(f"\nHonest test accuracy: {test_acc * 100:.1f}%")
    print_confusion_matrix(confusion)

    X_all, y_all = flatten_all_sessions(sessions)
    deployment_model = PCAClassifier(n_components=best_n, knn_k=config.KNN_K).fit(
        X_all,
        y_all,
    )
    deployment_model.save(config.PCA_MODEL_PATH)
    print(
        f"\nSaved deployment model trained on all available data -> "
        f"{config.PCA_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
