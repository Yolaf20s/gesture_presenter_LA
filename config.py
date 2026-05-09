"""
Global configuration for the Gesture-Controlled Presentation Assistant.

Tune behavior here instead of modifying the application logic.
"""

# ============================================================
# CAMERA
# ============================================================
# Typical defaults:
#   - Laptop webcam: 0
#   - DroidCam client: often 1
# Use `python3 scripts/list_cameras.py` to inspect available indices.
CAMERA_SOURCE = 0

# Camera capture properties. The camera backend may choose the nearest match.
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Most selfie-style camera previews are mirrored. Set this to False for
# non-mirrored sources so left/right hand mapping stays correct.
CAMERA_IS_MIRRORED = True

# ============================================================
# PREPROCESSING
# ============================================================
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 360
GAUSSIAN_KERNEL = 5

# ============================================================
# POSE / HAND DETECTION
# ============================================================
# Backend selection:
#   - "auto": prefer the legacy `mp.solutions` API when present, otherwise use
#     the newer MediaPipe Tasks API.
#   - "solutions": force the legacy API.
#   - "tasks": force the newer Tasks API.
MEDIAPIPE_BACKEND = "auto"

# MediaPipe Pose model complexity:
#   0 = Lite (fastest)
#   1 = Full (balanced)
#   2 = Heavy (slowest, most accurate)
POSE_MODEL_COMPLEXITY = 0
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5

# Minimum visibility for the pose landmarks that feed the feature extractor.
POSE_MIN_LANDMARK_VISIBILITY = 0.35

HAND_MAX_NUM_HANDS = 2
HAND_MIN_DETECTION_CONFIDENCE = 0.5
HAND_MIN_TRACKING_CONFIDENCE = 0.5

# Newer MediaPipe Tasks wheels require external `.task` model bundles.
# The repo can download them on first run when they are missing.
MEDIAPIPE_MODELS_DIR = "models/mediapipe"
MEDIAPIPE_POSE_MODEL_PATH = None
MEDIAPIPE_HAND_MODEL_PATH = None
MEDIAPIPE_AUTO_DOWNLOAD_MODELS = True
MEDIAPIPE_MODEL_DOWNLOAD_TIMEOUT = 60

# ============================================================
# FEATURE EXTRACTION / GESTURES
# ============================================================
GESTURE_CLASSES = [
    "right_arm",
    "left_arm",
    "five_fingers",
    "thumb_up",
    "idle",
]
HAND_REQUIRED_GESTURES = {"five_fingers", "thumb_up"}

# ============================================================
# PCA / CLASSIFIER
# ============================================================
PCA_N_COMPONENTS = 5
KNN_K = 5

# Standardize mixed-scale feature dimensions before fitting PCA.
PCA_STANDARDIZE_FEATURES = True

# Confidence threshold used by the trigger layer after smoothing/calibration.
CONFIDENCE_THRESHOLD = 0.7

# ============================================================
# TRIGGERING
# ============================================================
TRIGGER_COOLDOWN = 2.0
PYAUTOGUI_FAILSAFE = True

# Require the same non-idle prediction to stay stable for N consecutive frames
# before it becomes trigger-eligible.
TRIGGER_STABLE_FRAMES = 3

# After a trigger fires, require the user to release back to a neutral / invalid
# state before another action is allowed.
REARM_STATES = {"idle", "no_pose", "low_visibility"}

# Optional focus guard. When enabled, actions only fire if the configured app is
# focused. The macOS implementation works through `osascript`; Windows uses a
# lightweight ctypes call. Unsupported systems can fall back to allowing the
# trigger when focus cannot be determined.
TARGET_APP_NAME = "Canva"
ENFORCE_TARGET_APP_FOCUS = True
ALLOW_TRIGGER_WHEN_FOCUS_UNKNOWN = True

# Gesture -> key mapping for Canva presentation controls / shortcuts.
GESTURE_TO_KEY = {
    "right_arm": "right",
    "left_arm": "left",
    "five_fingers": "5",
    "thumb_up": "c",
    "idle": None,
}

GESTURE_DISPLAY_NAME = {
    "right_arm": "Next Slide",
    "left_arm": "Previous Slide",
    "five_fingers": "Countdown 5s",
    "thumb_up": "Confetti",
    "idle": "Idle",
}

TRACKING_DISPLAY_NAME = {
    "ready": "Ready",
    "idle": "Idle",
    "no_pose": "No pose detected",
    "low_visibility": "Pose visibility too low",
}

# ============================================================
# DATASET / EVALUATION
# ============================================================
SAMPLES_PER_GESTURE = 100
DATASET_DIR = "data/dataset"

# New recordings are stored as session files so evaluation can split by
# recording session instead of shuffling near-duplicate adjacent frames.
SESSION_FILENAME_TEMPLATE = "session_{timestamp}.npy"

EVALUATION_TEST_RATIO = 0.2
EVALUATION_RANDOM_SEED = 42

# When a single recording must be split in-place, leave a small gap between the
# train and test chunks to reduce temporal leakage.
EVALUATION_GAP_SAMPLES = 5

# ============================================================
# MODEL PATHS
# ============================================================
PCA_MODEL_PATH = "models/pca_model.npz"

# ============================================================
# DISPLAY
# ============================================================
SHOW_SKELETON = True
SHOW_TIMING = True
WINDOW_NAME = "Gesture Presenter"
