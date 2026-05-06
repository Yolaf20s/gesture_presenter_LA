"""Script do tat ca camera kha dung tren may."""

import cv2
import sys


def list_available_cameras(max_test=5):
    available = []
    print(f"Dang test camera index 0..{max_test - 1}...\n")

    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            print(f"  Index {i}: Khong kha dung")
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"  Index {i}: Mo duoc nhung khong doc duoc frame")
            cap.release()
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        backend = cap.getBackendName()

        available.append({'index': i, 'width': width, 'height': height})
        print(f"  Index {i}: OK -- {width}x{height} @ {fps:.0f}fps ({backend})")
        cap.release()

    return available


def preview_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Khong mo duoc camera index {index}")
        return

    print(f"\nDang preview camera index {index}. Nhan ESC de thoat.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(f'Camera {index} preview', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    cams = list_available_cameras(max_test=5)
    if not cams:
        print("\nKhong tim thay camera nao!")
        sys.exit(1)

    print(f"\nTim thay {len(cams)} camera kha dung.\n")
    choice = input("Nhap index camera muon preview (Enter de bo qua): ").strip()
    if choice.isdigit():
        preview_camera(int(choice))


if __name__ == '__main__':
    main()