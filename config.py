"""
Cấu hình toàn cục cho Gesture-Controlled Presentation Assistant.
Sửa file này để tune behavior, không cần đụng vào logic code.
"""

# ============================================================
# CAMERA CONFIG
# ============================================================
# DroidCam thường ở index 1 sau khi cài Client (laptop webcam = 0).
# Chạy `python scripts/list_cameras.py` để biết index của DroidCam trên máy bạn.
CAMERA_SOURCE = 1
# Nếu dùng WiFi mode: CAMERA_SOURCE = 'http://192.168.1.x:4747/video'
# Nếu test bằng webcam laptop: CAMERA_SOURCE = 0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# ============================================================
# PREPROCESSING CONFIG
# ============================================================
RESIZE_WIDTH = 640      # giảm kích thước để tăng FPS
RESIZE_HEIGHT = 360
GAUSSIAN_KERNEL = 5     # kernel size cho Gaussian blur (phải là số lẻ)

# ============================================================
# POSE & HAND DETECTION CONFIG
# ============================================================
# MediaPipe Pose model complexity:
#   0 = Lite (nhanh, kém chính xác)
#   1 = Full (cân bằng — RECOMMENDED)
#   2 = Heavy (chậm, chính xác nhất)
POSE_MODEL_COMPLEXITY = 0
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5

HAND_MAX_NUM_HANDS = 2
HAND_MIN_DETECTION_CONFIDENCE = 0.5
HAND_MIN_TRACKING_CONFIDENCE = 0.5

# ============================================================
# PCA CLASSIFIER CONFIG
# ============================================================
PCA_N_COMPONENTS = 5    # số principal components giữ lại (test 3, 5, 7, 10)
KNN_K = 5               # số neighbors trong kNN
CONFIDENCE_THRESHOLD = 0.6  # ngưỡng để trigger action (tăng nếu false positive nhiều)

# ============================================================
# ACTION TRIGGER CONFIG
# ============================================================
TRIGGER_COOLDOWN = 2.0  # giây — thời gian chờ giữa 2 trigger liên tiếp

# Mapping gesture → phím tắt Canva
GESTURE_TO_KEY = {
    'right_arm':    'right',  # Next slide
    'left_arm':     'left',   # Back slide
    'five_fingers': '5',      # Countdown 5s (Canva Magic Shortcut)
    'thumb_up':     'c',      # Confetti (Canva Magic Shortcut)
    'idle':         None,     # Không làm gì
}

# Tên hiển thị (cho UI)
GESTURE_DISPLAY_NAME = {
    'right_arm':    'Next Slide ->',
    'left_arm':     '<- Back Slide',
    'five_fingers': 'Countdown 5s',
    'thumb_up':     'Confetti!',
    'idle':         '...',
}

# ============================================================
# DATASET CONFIG
# ============================================================
GESTURE_CLASSES = ['right_arm', 'left_arm', 'five_fingers', 'thumb_up', 'idle']
SAMPLES_PER_GESTURE = 100  # số sample thu cho mỗi gesture mỗi người
DATASET_DIR = 'data/dataset'

# ============================================================
# MODEL PATH
# ============================================================
PCA_MODEL_PATH = 'models/pca_model.npz'

# ============================================================
# DISPLAY CONFIG
# ============================================================
SHOW_SKELETON = True
SHOW_TIMING = True
WINDOW_NAME = 'Gesture Presenter'