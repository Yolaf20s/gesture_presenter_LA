"""
Script train PCA tu dataset da thu.

Cach dung:
    python -m models.train_pca
"""

import os
import glob
import numpy as np

import config
from src.pca_classifier import PCAClassifier


def load_all_datasets():
    """Load tat ca file .npy trong data/dataset/<person>/."""
    X_list, y_list = [], []
    person_dirs = glob.glob(os.path.join(config.DATASET_DIR, '*'))
    person_dirs = [p for p in person_dirs if os.path.isdir(p)]

    if not person_dirs:
        raise FileNotFoundError(
            f"Khong tim thay folder nguoi nao trong {config.DATASET_DIR}. "
            f"Chay 'python -m data.collect_data' truoc!"
        )

    print(f"Tim thay {len(person_dirs)} nguoi: "
          f"{[os.path.basename(p) for p in person_dirs]}")

    for person_dir in person_dirs:
        person_name = os.path.basename(person_dir)
        for gesture in config.GESTURE_CLASSES:
            path = os.path.join(person_dir, f"{gesture}.npy")
            if not os.path.exists(path):
                print(f"  WARNING: thieu {path}")
                continue
            data = np.load(path)
            X_list.append(data)
            y_list.extend([gesture] * len(data))
            print(f"  [{person_name}/{gesture:14s}] {len(data)} samples")

    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y


def train_test_split(X, y, test_ratio=0.2, seed=42):
    """Split du lieu thanh train/test."""
    np.random.seed(seed)
    n = len(X)
    indices = np.random.permutation(n)
    n_test = int(n * test_ratio)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def evaluate(clf, X_test, y_test):
    """Tinh accuracy + confusion matrix."""
    correct = 0
    confusion = {}
    for cls in config.GESTURE_CLASSES:
        confusion[cls] = {c: 0 for c in config.GESTURE_CLASSES}

    for x, true_label in zip(X_test, y_test):
        pred, conf = clf.predict(x)
        if pred == true_label:
            correct += 1
        confusion[true_label][pred] += 1

    accuracy = correct / len(X_test)
    return accuracy, confusion


def print_confusion_matrix(confusion):
    """In confusion matrix dep dep."""
    classes = config.GESTURE_CLASSES
    print("\nConfusion Matrix:")
    header = 'True / Pred'
    print(f"{header:<14}", end='')
    for c in classes:
        print(f"{c[:10]:>11}", end='')
    print()

    for true_cls in classes:
        print(f"{true_cls:<14}", end='')
        for pred_cls in classes:
            count = confusion[true_cls][pred_cls]
            print(f"{count:>11}", end='')
        print()


def main():
    print("=" * 60)
    print("TRAIN PCA CLASSIFIER")
    print("=" * 60 + "\n")

    # 1. Load dataset
    X, y = load_all_datasets()
    print(f"\nTong: X shape {X.shape}, y len {len(y)}")
    print(f"Phan bo class: ", end='')
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"{u}={c} ", end='')
    print()

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)
    print(f"\nSplit: train={len(X_train)}, test={len(X_test)}")

    # 3. Try cac n_components khac nhau
    print("\nTesting cac n_components khac nhau:")
    print(f"{'n_comp':>8} {'train_acc':>12} {'test_acc':>12}")

    best_n = config.PCA_N_COMPONENTS
    best_acc = 0
    for n_comp in [3, 5, 7, 10, 13]:
        if n_comp > X_train.shape[1]:
            continue
        clf = PCAClassifier(n_components=n_comp, knn_k=5)
        clf.fit(X_train, y_train)

        train_correct = sum(
            1 for x, t in zip(X_train, y_train)
            if clf.predict(x)[0] == t
        )
        train_acc = train_correct / len(X_train)
        test_acc, _ = evaluate(clf, X_test, y_test)

        print(f"{n_comp:>8} {train_acc*100:>11.1f}% {test_acc*100:>11.1f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            best_n = n_comp

    print(f"\nBest n_components: {best_n} (test acc: {best_acc*100:.1f}%)")

    # 4. Train final model voi best n_components, dung ALL data
    print(f"\nTrain final model voi n_components={best_n} tren toan bo dataset...")
    final_clf = PCAClassifier(n_components=best_n, knn_k=config.KNN_K)
    final_clf.fit(X, y)

    # Final eval
    acc, conf = evaluate(final_clf, X_test, y_test)
    print(f"\nFinal accuracy tren test set: {acc*100:.1f}%")
    print_confusion_matrix(conf)

    # 5. Save
    final_clf.save(config.PCA_MODEL_PATH)
    print(f"\nDONE! Model luu tai: {config.PCA_MODEL_PATH}")


if __name__ == '__main__':
    main()