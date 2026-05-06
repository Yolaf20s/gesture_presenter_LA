"""Script kiem tra dataset da thu co dung khong."""

import os
import numpy as np
import config


def verify(person_name):
    save_dir = os.path.join(config.DATASET_DIR, person_name)

    if not os.path.isdir(save_dir):
        print(f"Khong tim thay folder: {save_dir}")
        return

    print(f"\n{'='*60}")
    print(f"VERIFY DATASET cho: {person_name}")
    print(f"{'='*60}\n")

    total = 0
    for gesture in config.GESTURE_CLASSES:
        path = os.path.join(save_dir, f"{gesture}.npy")

        if not os.path.exists(path):
            print(f"  [{gesture:14s}] KHONG CO FILE")
            continue

        data = np.load(path)
        n_samples, n_features = data.shape
        total += n_samples

        # Kiem tra basic stats
        mean = data.mean(axis=0)
        std = data.std(axis=0)

        print(f"  [{gesture:14s}] {n_samples} samples x {n_features} features")
        print(f"                  mean (3 first): {mean[:3].round(2)}")
        print(f"                  std  (3 first): {std[:3].round(2)}")

    print(f"\nTotal: {total} samples")

    # Kiem tra kich thuoc co dong nhat khong
    sizes = []
    for gesture in config.GESTURE_CLASSES:
        path = os.path.join(save_dir, f"{gesture}.npy")
        if os.path.exists(path):
            sizes.append(np.load(path).shape[0])

    if len(set(sizes)) == 1:
        print(f"OK: tat ca cac gesture co {sizes[0]} samples (can bang).")
    elif sizes:
        print(f"WARNING: cac gesture co so samples khac nhau: {sizes}")
        print("(Khong sao, training van chay duoc.)")


if __name__ == '__main__':
    name = input("Nhap ten nguoi (vd: 'an'): ").strip()
    verify(name)