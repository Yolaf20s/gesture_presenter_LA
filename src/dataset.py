"""
Dataset loading and evaluation splitting utilities.

Supports both:
- legacy layout: data/dataset/<person>/<gesture>.npy
- session layout: data/dataset/<person>/<gesture>/session_*.npy
"""

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np

import config


@dataclass(frozen=True)
class SessionRecord:
    person: str
    gesture: str
    path: Path
    samples: np.ndarray


def load_feature_sessions(dataset_dir=None, gesture_classes=None):
    dataset_dir = Path(dataset_dir or config.DATASET_DIR)
    gesture_classes = gesture_classes or config.GESTURE_CLASSES
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    sessions = []
    for person_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        for gesture in gesture_classes:
            legacy_path = person_dir / f"{gesture}.npy"
            if legacy_path.exists():
                sessions.append(
                    SessionRecord(
                        person=person_dir.name,
                        gesture=gesture,
                        path=legacy_path,
                        samples=_load_samples(legacy_path),
                    )
                )

            gesture_dir = person_dir / gesture
            if gesture_dir.is_dir():
                for session_path in sorted(gesture_dir.glob("*.npy")):
                    sessions.append(
                        SessionRecord(
                            person=person_dir.name,
                            gesture=gesture,
                            path=session_path,
                            samples=_load_samples(session_path),
                        )
                    )

    if not sessions:
        raise FileNotFoundError(
            f"No dataset sessions found under {dataset_dir}. "
            "Collect data first with `python3 -m data.collect_data`."
        )
    return sessions


def split_feature_dataset(sessions, test_ratio=None, seed=None):
    """
    Split feature sessions with the strongest grouping available.

    Priority:
    1. hold out full people if multiple people exist
    2. hold out full sessions if multiple sessions per gesture exist
    3. fall back to contiguous in-session splits with a small gap
    """
    test_ratio = test_ratio if test_ratio is not None else config.EVALUATION_TEST_RATIO
    seed = seed if seed is not None else config.EVALUATION_RANDOM_SEED
    rng = random.Random(seed)

    people = sorted({session.person for session in sessions})
    if len(people) >= 2:
        n_test_people = min(len(people) - 1, max(1, round(len(people) * test_ratio)))
        shuffled = people[:]
        rng.shuffle(shuffled)
        test_people = set(shuffled[:n_test_people])
        train_sessions = [session for session in sessions if session.person not in test_people]
        test_sessions = [session for session in sessions if session.person in test_people]
        summary = {
            "strategy": "person_holdout",
            "train_groups": sorted({session.person for session in train_sessions}),
            "test_groups": sorted(test_people),
            "warning": "",
        }
        return *_sessions_to_arrays(train_sessions), *_sessions_to_arrays(test_sessions), summary

    gesture_session_counts = {}
    for session in sessions:
        gesture_session_counts.setdefault(session.gesture, 0)
        gesture_session_counts[session.gesture] += 1

    if any(count > 1 for count in gesture_session_counts.values()):
        grouped = {}
        for session in sessions:
            grouped.setdefault(session.gesture, []).append(session)

        train_sessions = []
        test_sessions = []
        for gesture, gesture_sessions in grouped.items():
            if len(gesture_sessions) == 1:
                train_sessions.extend(gesture_sessions)
                continue
            shuffled = gesture_sessions[:]
            rng.shuffle(shuffled)
            n_test_sessions = min(len(shuffled) - 1, max(1, round(len(shuffled) * test_ratio)))
            test_sessions.extend(shuffled[:n_test_sessions])
            train_sessions.extend(shuffled[n_test_sessions:])

        summary = {
            "strategy": "session_holdout",
            "train_groups": [session.path.name for session in train_sessions],
            "test_groups": [session.path.name for session in test_sessions],
            "warning": "",
        }
        return *_sessions_to_arrays(train_sessions), *_sessions_to_arrays(test_sessions), summary

    train_X, train_y, test_X, test_y = [], [], [], []
    for session in sessions:
        session_train_X, session_test_X = _split_contiguous_session(session.samples, test_ratio)
        train_X.append(session_train_X)
        test_X.append(session_test_X)
        train_y.extend([session.gesture] * len(session_train_X))
        test_y.extend([session.gesture] * len(session_test_X))

    summary = {
        "strategy": "contiguous_within_session",
        "train_groups": [session.path.name for session in sessions],
        "test_groups": [session.path.name for session in sessions],
        "warning": (
            "Only one session per gesture was available, so evaluation uses a "
            "contiguous holdout chunk within each recording. This is better than "
            "random frame shuffling, but less trustworthy than person/session holdout."
        ),
    }
    return (
        np.vstack(train_X),
        np.array(train_y),
        np.vstack(test_X),
        np.array(test_y),
        summary,
    )


def summarize_sessions(sessions):
    lines = []
    total_samples = 0
    for session in sessions:
        total_samples += len(session.samples)
        lines.append(
            f"  [{session.person}/{session.gesture}] "
            f"{session.path.name}: {len(session.samples)} samples"
        )
    return total_samples, lines


def _sessions_to_arrays(sessions):
    X_list = [session.samples for session in sessions]
    y_list = []
    for session in sessions:
        y_list.extend([session.gesture] * len(session.samples))
    return np.vstack(X_list), np.array(y_list)


def _split_contiguous_session(samples, test_ratio):
    n = len(samples)
    n_test = min(max(1, int(round(n * test_ratio))), max(1, n - 1))
    max_reasonable_gap = max(0, int(round(n * 0.05)))
    gap = min(
        config.EVALUATION_GAP_SAMPLES,
        max_reasonable_gap,
        max(0, n - n_test - 1),
    )
    test_start = max(1, n - n_test)
    train_end = max(1, test_start - gap)
    train = samples[:train_end]
    test = samples[test_start:]
    return train, test


def _load_samples(path):
    data = np.load(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return np.asarray(data, dtype=np.float32)
