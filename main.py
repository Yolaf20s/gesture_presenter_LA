"""
CLI entry point for the Gesture-Controlled Presentation Assistant.

Usage:
    python3 main.py

Requirements:
    - A connected camera (webcam or DroidCam)
    - A trained PCA model (`python3 -m models.train_pca`)
    - Optionally Canva in presentation mode if you want to test the shortcuts
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the gesture-controlled presentation assistant."
    )
    parser.add_argument(
        "--camera",
        type=_camera_source,
        default=None,
        help=(
            "Camera source override. Use an integer index like 0/1, or a stream "
            "URL such as http://192.168.1.10:4747/video."
        ),
    )
    return parser.parse_args()


def _camera_source(value):
    try:
        return int(value)
    except ValueError:
        return value


def main():
    args = parse_args()

    print("=" * 60)
    print("  GESTURE-CONTROLLED PRESENTATION ASSISTANT")
    print("=" * 60)
    print("\nSupported gestures:")
    print("  1. Right arm extended   -> Next slide")
    print("  2. Left arm extended    -> Previous slide")
    print("  3. Open hand (5 fingers)-> Countdown 5s")
    print("  4. Thumb up             -> Confetti")
    print("\nFor Canva demos:")
    print("  - Open Canva in Present mode")
    print("  - Keep Canva focused")
    print("  - Perform a gesture and the app will send a shortcut")
    print("\nPress ESC in the preview window to quit.\n")

    try:
        from src.pipeline import Pipeline

        pipeline = Pipeline(camera_source=args.camera)
    except Exception as exc:
        print(f"[Startup] {type(exc).__name__}: {exc}")
        print(
            "[Startup] If MediaPipe Tasks is installed, make sure the `.task` "
            "bundles exist under `models/mediapipe/`, or run "
            "`python3 scripts/setup_mediapipe_models.py`."
        )
        return 1

    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
