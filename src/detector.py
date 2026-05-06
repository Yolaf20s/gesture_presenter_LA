"""
Class PoseHandDetector: phat hien skeleton co the va ban tay.
Step 3 cua pipeline.

Su dung MediaPipe:
- Pose: 33 landmark co the
- Hands: 21 landmark moi ban tay (toi da 2 tay)
"""

import mediapipe as mp
import config


class PoseHandDetector:
    """Wrapper quanh MediaPipe Pose & Hands."""

    def __init__(self):
        # Khoi tao MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # Track giua cac frame -> nhanh hon
            model_complexity=config.POSE_MODEL_COMPLEXITY,
            smooth_landmarks=True,    # Lam muot landmark giua frame
            enable_segmentation=False,
            min_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
        )

        # Khoi tao MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.HAND_MAX_NUM_HANDS,
            model_complexity=1,
            min_detection_confidence=config.HAND_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.HAND_MIN_TRACKING_CONFIDENCE,
        )

        # De ve skeleton
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        print("[Detector] MediaPipe Pose & Hands san sang.")

    def detect(self, rgb_frame):
        """
        rgb_frame: numpy array shape (H, W, 3), RGB color space.
        Return: dict chua ket qua detection.
        """
        # MediaPipe co cu phap Optimize: set writeable=False de tang toc
        rgb_frame.flags.writeable = False

        pose_result = self.pose.process(rgb_frame)
        hands_result = self.hands.process(rgb_frame)

        rgb_frame.flags.writeable = True

        return {
            'pose_landmarks': pose_result.pose_landmarks,
            'hand_landmarks': hands_result.multi_hand_landmarks,  # list cac tay
            'handedness': hands_result.multi_handedness,  # left/right info
        }

    def draw_skeleton(self, bgr_frame, detection_result):
        """
        Ve skeleton len frame de hien thi.
        bgr_frame: frame BGR de hien thi (modify in-place).
        detection_result: output cua detect().
        Return: frame da ve skeleton.
        """
        # Ve pose skeleton (33 landmark + connections)
        if detection_result['pose_landmarks'] is not None:
            self.mp_drawing.draw_landmarks(
                bgr_frame,
                detection_result['pose_landmarks'],
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )

        # Ve hand skeleton (21 landmark x mỗi tay)
        if detection_result['hand_landmarks'] is not None:
            for hand_landmarks in detection_result['hand_landmarks']:
                self.mp_drawing.draw_landmarks(
                    bgr_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style(),
                )

        return bgr_frame

    def close(self):
        """Giai phong tai nguyen MediaPipe."""
        self.pose.close()
        self.hands.close()


# ============================================================
# Test khi chay file truc tiep
# ============================================================
if __name__ == '__main__':
    """Chay file nay de test detector. Hien camera + skeleton overlay."""
    import cv2
    from src.camera import Camera
    from src.preprocessor import Preprocessor

    cam = Camera()
    pre = Preprocessor()
    detector = PoseHandDetector()

    print("Nhan ESC de thoat.")
    print("Dung truoc camera, gio tay, lam vai gesture de test!")

    while True:
        frame = cam.read_frame()
        if frame is None:
            break

        rgb, bgr = pre.process(frame)
        result = detector.detect(rgb)
        annotated = detector.draw_skeleton(bgr.copy(), result)

        cv2.imshow('Pose & Hand Detection', annotated)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cam.release()
    detector.close()
    cv2.destroyAllWindows()