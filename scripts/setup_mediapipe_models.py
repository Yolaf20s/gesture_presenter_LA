"""
Download the MediaPipe Tasks model bundles used by this repo.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mediapipe_models import ensure_hand_model, ensure_pose_model


def main():
    pose_path = ensure_pose_model(auto_download=True)
    hand_path = ensure_hand_model(auto_download=True, required=False)

    print("\nModel setup complete:")
    print(f"- pose: {pose_path}")
    if hand_path is not None:
        print(f"- hand: {hand_path}")
    else:
        print("- hand: unavailable")


if __name__ == "__main__":
    main()
