"""List available camera indices and optionally preview one of them."""

import sys

import cv2


def list_available_cameras(max_test=5):
    available = []
    print(f"Testing camera indices 0..{max_test - 1}...\n")

    for index in range(max_test):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"  Index {index}: unavailable")
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"  Index {index}: opened, but could not read a frame")
            cap.release()
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        backend = cap.getBackendName()

        available.append({"index": index, "width": width, "height": height})
        print(f"  Index {index}: OK -- {width}x{height} @ {fps:.0f}fps ({backend})")
        cap.release()

    return available


def preview_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Could not open camera index {index}")
        return

    print(f"\nPreviewing camera index {index}. Press ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(f"Camera {index} preview", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    cameras = list_available_cameras(max_test=5)
    if not cameras:
        print("\nNo usable cameras found.")
        sys.exit(1)

    print(f"\nFound {len(cameras)} usable camera(s).\n")
    choice = input("Enter a camera index to preview (Enter to skip): ").strip()
    if choice.isdigit():
        preview_camera(int(choice))


if __name__ == "__main__":
    main()
