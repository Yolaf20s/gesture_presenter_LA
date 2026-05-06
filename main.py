"""
Entry point cho Gesture-Controlled Presentation Assistant.

Cach dung:
    python main.py

Yeu cau:
    - Camera (webcam hoac DroidCam) da connect
    - PCA model da train (chay `python -m models.train_pca` truoc)
    - (Optional) Canva o che do Present de test phim tat
"""

from src.pipeline import Pipeline


def main():
    print("=" * 60)
    print("  GESTURE-CONTROLLED PRESENTATION ASSISTANT")
    print("=" * 60)
    print("\n4 GESTURE:")
    print("  1. Dang tay phai sang ngang  -> Next slide")
    print("  2. Dang tay trai sang ngang  -> Back slide")
    print("  3. Gio ban tay 5 ngon         -> Countdown 5s")
    print("  4. Gio ngon cai len           -> Confetti")
    print("\nDe demo voi Canva:")
    print("  - Mo Canva, vao mode Present")
    print("  - Lam gesture, app se gui phim tat den Canva")
    print("\nNhan ESC tren cua so camera de thoat.\n")

    pipeline = Pipeline()
    pipeline.run()


if __name__ == '__main__':
    main()