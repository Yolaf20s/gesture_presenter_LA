"""Quick dataset sanity check."""

from collections import defaultdict

import numpy as np

from src.dataset import load_feature_sessions


def verify(person_name):
    sessions = [session for session in load_feature_sessions() if session.person == person_name]
    if not sessions:
        print(f"No dataset sessions found for: {person_name}")
        return

    print(f"\n{'=' * 60}")
    print(f"VERIFY DATASET FOR: {person_name}")
    print(f"{'=' * 60}\n")

    grouped = defaultdict(list)
    for session in sessions:
        grouped[session.gesture].append(session)

    total = 0
    per_gesture_counts = []
    for gesture, gesture_sessions in sorted(grouped.items()):
        gesture_samples = np.vstack([session.samples for session in gesture_sessions])
        n_samples, n_features = gesture_samples.shape
        total += n_samples
        per_gesture_counts.append(n_samples)

        mean = gesture_samples.mean(axis=0)
        std = gesture_samples.std(axis=0)

        print(
            f"[{gesture:14s}] {len(gesture_sessions)} session(s), "
            f"{n_samples} samples x {n_features} features"
        )
        print(f"                 mean (first 3): {mean[:3].round(2)}")
        print(f"                 std  (first 3): {std[:3].round(2)}")

    print(f"\nTotal samples: {total}")
    if len(set(per_gesture_counts)) == 1:
        print(f"Balanced: every gesture has {per_gesture_counts[0]} samples.")
    else:
        print(f"Warning: gesture sample counts differ: {per_gesture_counts}")
        print("Training can still run, but class balance is no longer uniform.")


if __name__ == "__main__":
    name = input("Enter a person name (for example: alex): ").strip()
    verify(name)
