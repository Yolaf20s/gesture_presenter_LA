"""
Pose and hand landmark detection with MediaPipe.

Supports both:
- the legacy `mp.solutions` API
- the newer MediaPipe Tasks API, which requires `.task` model bundles
"""

from dataclasses import dataclass
import time

import mediapipe as mp

import config
from src.mediapipe_models import ensure_hand_model, ensure_pose_model


@dataclass
class _LegacyLandmarkListAdapter:
    """Adapter that mimics the old `NormalizedLandmarkList` shape."""

    landmark: list


@dataclass
class _LegacyClassificationAdapter:
    """Adapter that mimics the old handedness classification object."""

    label: str


@dataclass
class _LegacyHandednessAdapter:
    """Adapter that mimics the old handedness list object."""

    classification: list


class PoseHandDetector:
    """MediaPipe wrapper for body pose + hand landmarks."""

    def __init__(self):
        self.backend = self._select_backend()
        self.pose = None
        self.hands = None

        self.mp_drawing = None
        self.mp_drawing_styles = None
        self.pose_connections = None
        self.hand_connections = None
        self._last_timestamp_ms = 0

        if self.backend == "solutions":
            self._init_solutions_backend()
        elif self.backend == "tasks":
            self._init_tasks_backend()
        else:
            raise RuntimeError(f"Unsupported MediaPipe backend: {self.backend}")

        print(f"[Detector] MediaPipe initialized with backend: {self.backend}")

    def _select_backend(self):
        requested = str(config.MEDIAPIPE_BACKEND).strip().lower()
        has_solutions = self._has_legacy_solutions()
        has_tasks = self._has_tasks_api()

        if requested == "auto":
            if has_solutions:
                return "solutions"
            if has_tasks:
                return "tasks"
        elif requested == "solutions":
            if has_solutions:
                return "solutions"
            raise RuntimeError(
                "config.MEDIAPIPE_BACKEND is set to 'solutions', but this "
                "MediaPipe install does not expose `mp.solutions`."
            )
        elif requested == "tasks":
            if has_tasks:
                return "tasks"
            raise RuntimeError(
                "config.MEDIAPIPE_BACKEND is set to 'tasks', but this "
                "MediaPipe install does not expose `mp.tasks`."
            )
        else:
            raise RuntimeError(
                "config.MEDIAPIPE_BACKEND must be one of: auto, solutions, tasks."
            )

        raise RuntimeError(
            "This MediaPipe install exposes neither the legacy Solutions API nor "
            "the newer Tasks API."
        )

    def _has_legacy_solutions(self):
        solutions = getattr(mp, "solutions", None)
        return (
            solutions is not None
            and hasattr(solutions, "pose")
            and hasattr(solutions, "hands")
        )

    def _has_tasks_api(self):
        tasks = getattr(mp, "tasks", None)
        return tasks is not None and hasattr(tasks, "vision")

    def _init_solutions_backend(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=config.POSE_MODEL_COMPLEXITY,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
        )

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.HAND_MAX_NUM_HANDS,
            model_complexity=1,
            min_detection_confidence=config.HAND_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.HAND_MIN_TRACKING_CONFIDENCE,
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose_connections = self.mp_pose.POSE_CONNECTIONS
        self.hand_connections = self.mp_hands.HAND_CONNECTIONS

    def _init_tasks_backend(self):
        vision = mp.tasks.vision
        base_options = mp.tasks.BaseOptions
        cpu_delegate = base_options.Delegate.CPU
        running_mode = vision.RunningMode.VIDEO

        pose_model_path = ensure_pose_model()
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=base_options(
                    model_asset_path=str(pose_model_path),
                    delegate=cpu_delegate,
                ),
                running_mode=running_mode,
                num_poses=1,
                min_pose_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
                min_pose_presence_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
                output_segmentation_masks=False,
            )
        )

        hand_model_path = ensure_hand_model(required=False)
        if hand_model_path is not None and hand_model_path.exists():
            self.hands = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=base_options(
                        model_asset_path=str(hand_model_path),
                        delegate=cpu_delegate,
                    ),
                    running_mode=running_mode,
                    num_hands=config.HAND_MAX_NUM_HANDS,
                    min_hand_detection_confidence=(
                        config.HAND_MIN_DETECTION_CONFIDENCE
                    ),
                    min_hand_presence_confidence=(
                        config.HAND_MIN_DETECTION_CONFIDENCE
                    ),
                    min_tracking_confidence=config.HAND_MIN_TRACKING_CONFIDENCE,
                )
            )
        else:
            print(
                "[Detector] Hand model is unavailable. Pose detection will still "
                "run, but hand-based gestures will not work until the hand model "
                "bundle is present."
            )

        self.mp_drawing = vision.drawing_utils
        self.mp_drawing_styles = vision.drawing_styles
        self.pose_connections = vision.PoseLandmarksConnections.POSE_LANDMARKS
        self.hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    def detect(self, rgb_frame):
        """Run pose and hand detection on an RGB frame."""
        if self.backend == "solutions":
            rgb_frame.flags.writeable = False
            pose_result = self.pose.process(rgb_frame)
            hands_result = self.hands.process(rgb_frame)
            rgb_frame.flags.writeable = True
            return {
                "pose_landmarks": pose_result.pose_landmarks,
                "hand_landmarks": hands_result.multi_hand_landmarks,
                "handedness": hands_result.multi_handedness,
                "_raw_pose_result": pose_result,
                "_raw_hands_result": hands_result,
            }

        timestamp_ms = self._next_timestamp_ms()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        pose_result = self.pose.detect_for_video(mp_image, timestamp_ms)
        hands_result = (
            self.hands.detect_for_video(mp_image, timestamp_ms)
            if self.hands is not None
            else None
        )

        return {
            "pose_landmarks": self._adapt_pose_landmarks(pose_result),
            "hand_landmarks": self._adapt_hand_landmarks(hands_result),
            "handedness": self._adapt_handedness(hands_result),
            "_raw_pose_result": pose_result,
            "_raw_hands_result": hands_result,
        }

    def _next_timestamp_ms(self):
        timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def _adapt_pose_landmarks(self, pose_result):
        if pose_result is None or not pose_result.pose_landmarks:
            return None
        return _LegacyLandmarkListAdapter(landmark=pose_result.pose_landmarks[0])

    def _adapt_hand_landmarks(self, hands_result):
        if hands_result is None or not hands_result.hand_landmarks:
            return None
        return [
            _LegacyLandmarkListAdapter(landmark=landmarks)
            for landmarks in hands_result.hand_landmarks
        ]

    def _adapt_handedness(self, hands_result):
        if hands_result is None or not hands_result.handedness:
            return None

        adapted = []
        for categories in hands_result.handedness:
            label = "Unknown"
            if categories:
                category = categories[0]
                label = getattr(category, "category_name", None) or getattr(
                    category, "label", "Unknown"
                )
            adapted.append(
                _LegacyHandednessAdapter(
                    classification=[_LegacyClassificationAdapter(label=label)]
                )
            )
        return adapted

    def draw_skeleton(self, bgr_frame, detection_result):
        """Draw pose + hand landmarks on a BGR frame."""
        if self.backend == "solutions":
            if detection_result["pose_landmarks"] is not None:
                self.mp_drawing.draw_landmarks(
                    bgr_frame,
                    detection_result["pose_landmarks"],
                    self.pose_connections,
                    landmark_drawing_spec=(
                        self.mp_drawing_styles.get_default_pose_landmarks_style()
                    ),
                )

            if detection_result["hand_landmarks"] is not None:
                for hand_landmarks in detection_result["hand_landmarks"]:
                    self.mp_drawing.draw_landmarks(
                        bgr_frame,
                        hand_landmarks,
                        self.hand_connections,
                        landmark_drawing_spec=(
                            self.mp_drawing_styles.get_default_hand_landmarks_style()
                        ),
                        connection_drawing_spec=(
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        ),
                    )
            return bgr_frame

        pose_result = detection_result.get("_raw_pose_result")
        if pose_result is not None and pose_result.pose_landmarks:
            for pose_landmarks in pose_result.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    bgr_frame,
                    pose_landmarks,
                    self.pose_connections,
                    landmark_drawing_spec=(
                        self.mp_drawing_styles.get_default_pose_landmarks_style()
                    ),
                )

        hands_result = detection_result.get("_raw_hands_result")
        if hands_result is not None and hands_result.hand_landmarks:
            for hand_landmarks in hands_result.hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    bgr_frame,
                    hand_landmarks,
                    self.hand_connections,
                    landmark_drawing_spec=(
                        self.mp_drawing_styles.get_default_hand_landmarks_style()
                    ),
                    connection_drawing_spec=(
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    ),
                )

        return bgr_frame

    def close(self):
        if self.pose is not None:
            self.pose.close()
        if self.hands is not None:
            self.hands.close()


if __name__ == "__main__":
    import cv2

    from src.camera import Camera
    from src.preprocessor import Preprocessor

    cam = Camera()
    pre = Preprocessor()
    detector = PoseHandDetector()

    print("Press ESC to quit.")
    while True:
        frame = cam.read_frame()
        if frame is None:
            break

        rgb, bgr = pre.process(frame)
        result = detector.detect(rgb)
        annotated = detector.draw_skeleton(bgr.copy(), result)
        cv2.imshow("Pose & Hand Detection", annotated)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    detector.close()
    cv2.destroyAllWindows()
