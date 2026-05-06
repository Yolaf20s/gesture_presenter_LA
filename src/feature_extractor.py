"""
Class FeatureExtractor: bien landmark tho thanh vector dac trung.
Step 4 cua pipeline.

Tu thiet ke 13 features bat bien voi:
- Khoang cach camera
- Vi tri trong frame
- Chieu cao nguoi dung

Linear Algebra dung trong file nay:
- Vector subtraction (vi tri tuong doi)
- Vector norm (chuan hoa)
- Dot product (tinh goc giua 2 vector)
"""

import numpy as np
import mediapipe as mp


# Landmark indices cua MediaPipe Pose (33 diem)
# Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
class PoseLM:
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24


# Landmark indices cua MediaPipe Hands (21 diem moi tay)
class HandLM:
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    INDEX_MCP = 5    # Khop ngon tro voi long ban tay
    MIDDLE_MCP = 9
    RING_MCP = 13
    PINKY_MCP = 17


class FeatureExtractor:
    """
    Tinh feature vector tu landmark.
    Output: numpy array shape (13,)
    """

    FEATURE_DIM = 13  # so chieu cua feature vector

    def extract(self, detection_result):
        """
        detection_result: dict tu PoseHandDetector.detect()
        Return: numpy array (13,) hoac None neu khong co pose
        """
        pose_lms = detection_result['pose_landmarks']
        hand_lms_list = detection_result['hand_landmarks']
        handedness_list = detection_result['handedness']

        # Phai co pose moi extract duoc
        if pose_lms is None:
            return None

        # Convert pose landmarks sang numpy array (33, 3)
        pose = np.array([[lm.x, lm.y, lm.z] for lm in pose_lms.landmark])

        features = []

        # ====================================================
        # Pose features (9 features) - dung 2D (x, y)
        # ====================================================
        # Lay ra cac diem chinh
        l_sh = pose[PoseLM.LEFT_SHOULDER, :2]
        r_sh = pose[PoseLM.RIGHT_SHOULDER, :2]
        l_el = pose[PoseLM.LEFT_ELBOW, :2]
        r_el = pose[PoseLM.RIGHT_ELBOW, :2]
        l_wr = pose[PoseLM.LEFT_WRIST, :2]
        r_wr = pose[PoseLM.RIGHT_WRIST, :2]
        l_hp = pose[PoseLM.LEFT_HIP, :2]
        r_hp = pose[PoseLM.RIGHT_HIP, :2]

        # Tinh "shoulder width" lam don vi chuan hoa (bat bien voi khoang cach)
        shoulder_width = np.linalg.norm(l_sh - r_sh) + 1e-6  # tranh chia 0

        # Feature 1: Goc vai-khuyu-co tay PHAI
        features.append(self._angle(r_sh, r_el, r_wr))

        # Feature 2: Goc vai-khuyu-co tay TRAI
        features.append(self._angle(l_sh, l_el, l_wr))

        # Feature 3: Goc than-vai-khuyu PHAI (tay phai dang ngang hay xuoi?)
        features.append(self._angle(r_hp, r_sh, r_el))

        # Feature 4: Goc than-vai-khuyu TRAI
        features.append(self._angle(l_hp, l_sh, l_el))

        # Feature 5: Vi tri Y co tay phai so voi vai (chuan hoa theo shoulder_width)
        # Am = co tay tren vai, Duong = co tay duoi vai
        features.append((r_wr[1] - r_sh[1]) / shoulder_width)

        # Feature 6: Vi tri Y co tay trai so voi vai
        features.append((l_wr[1] - l_sh[1]) / shoulder_width)

        # Feature 7: Vi tri X co tay phai so voi vai (chuan hoa)
        # Am = co tay nam ben trai vai (vao trong), Duong = ben phai (dang ra)
        features.append((r_wr[0] - r_sh[0]) / shoulder_width)

        # Feature 8: Vi tri X co tay trai so voi vai
        features.append((l_wr[0] - l_sh[0]) / shoulder_width)

        # Feature 9: Khoang cach 2 co tay (chuan hoa)
        features.append(np.linalg.norm(l_wr - r_wr) / shoulder_width)

        # ====================================================
        # Hand features (4 features) - 2 cho moi tay
        # ====================================================
        right_hand_lms, left_hand_lms = self._split_hands(
            hand_lms_list, handedness_list
        )

        # Feature 10, 11: tay PHAI (so ngon duoi, ti le ngon cai)
        if right_hand_lms is not None:
            n_extended_r, thumb_ratio_r = self._hand_features(right_hand_lms)
        else:
            n_extended_r, thumb_ratio_r = -1.0, -1.0  # khong detect duoc tay
        features.extend([n_extended_r, thumb_ratio_r])

        # Feature 12, 13: tay TRAI
        if left_hand_lms is not None:
            n_extended_l, thumb_ratio_l = self._hand_features(left_hand_lms)
        else:
            n_extended_l, thumb_ratio_l = -1.0, -1.0
        features.extend([n_extended_l, thumb_ratio_l])

        return np.array(features, dtype=np.float32)

    # ====================================================
    # Helper methods
    # ====================================================

    def _angle(self, p1, p2, p3):
        """
        Tinh goc tai diem p2 giua vector p2->p1 va p2->p3.
        Output: goc tinh bang radian, range [0, pi].

        Cong thuc: cos(theta) = (v1 . v2) / (||v1|| * ||v2||)
        """
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # tranh nan do floating point
        return float(np.arccos(cos_angle))

    def _split_hands(self, hand_lms_list, handedness_list):
        """
        Tach hand landmark thanh tay phai va tay trai.
        Return: (right_hand_lms, left_hand_lms), moi cai la None hoac numpy (21, 3)
        """
        if hand_lms_list is None or handedness_list is None:
            return None, None

        right_hand = None
        left_hand = None

        for hand_lms, handedness in zip(hand_lms_list, handedness_list):
            label = handedness.classification[0].label  # 'Right' or 'Left'
            arr = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])

            # MediaPipe tra ve 'Right' la tu goc nhin nguoi (mirror).
            # Trong frame webcam (mirror), 'Right' = tay phai cua nguoi.
            if label == 'Right':
                right_hand = arr
            else:
                left_hand = arr

        return right_hand, left_hand

    def _hand_features(self, hand_lms):
        """
        Tinh 2 feature tu 1 ban tay (21 landmark):
        - n_extended: so ngon tay duoi (0 -> 5)
        - thumb_ratio: ti le do duoi cua ngon cai
        """
        wrist = hand_lms[HandLM.WRIST, :2]

        # Diem dau ngon
        tips = {
            'thumb': hand_lms[HandLM.THUMB_TIP, :2],
            'index': hand_lms[HandLM.INDEX_TIP, :2],
            'middle': hand_lms[HandLM.MIDDLE_TIP, :2],
            'ring': hand_lms[HandLM.RING_TIP, :2],
            'pinky': hand_lms[HandLM.PINKY_TIP, :2],
        }

        # Diem khop noi voi long ban tay (MCP)
        mcps = {
            'index': hand_lms[HandLM.INDEX_MCP, :2],
            'middle': hand_lms[HandLM.MIDDLE_MCP, :2],
            'ring': hand_lms[HandLM.RING_MCP, :2],
            'pinky': hand_lms[HandLM.PINKY_MCP, :2],
        }

        # Don vi chuan hoa: khoang cach co tay -> khop ngon giua
        hand_size = np.linalg.norm(wrist - mcps['middle']) + 1e-6

        # Dem ngon duoi: dau ngon xa co tay hon khop MCP cua no
        # (logic don gian, du dung cho 4 gesture)
        n_extended = 0

        # 4 ngon (tro, giua, ap, ut): kiem tra dau ngon xa co tay hon MCP
        for finger in ['index', 'middle', 'ring', 'pinky']:
            tip_dist = np.linalg.norm(tips[finger] - wrist)
            mcp_dist = np.linalg.norm(mcps[finger] - wrist)
            if tip_dist > mcp_dist:  # ngon duoi
                n_extended += 1

        # Ngon cai: dung khoang cach tu thumb_tip den index_MCP (goc ngon tro)
        # Khi cup -> thumb_tip gan index_MCP
        # Khi duoi -> thumb_tip xa index_MCP
        thumb_to_index_mcp = np.linalg.norm(tips['thumb'] - mcps['index']) / hand_size

        if thumb_to_index_mcp > 0.6:  # nguong nay tunable
            n_extended += 1

        thumb_ratio = thumb_to_index_mcp  # 0 = cup, ~1.5 = duoi

        return float(n_extended), float(thumb_ratio)


# ============================================================
# Test khi chay file truc tiep
# ============================================================
if __name__ == '__main__':
    """Test FeatureExtractor: hien feature vector real-time tren cua so."""
    import cv2
    from src.camera import Camera
    from src.preprocessor import Preprocessor
    from src.detector import PoseHandDetector

    cam = Camera()
    pre = Preprocessor()
    detector = PoseHandDetector()
    extractor = FeatureExtractor()

    print("Nhan ESC de thoat.")
    print("Lam vai gesture (dang tay, gio ngon cai...) de xem feature vector thay doi!")

    while True:
        frame = cam.read_frame()
        if frame is None:
            break

        rgb, bgr = pre.process(frame)
        result = detector.detect(rgb)
        annotated = detector.draw_skeleton(bgr.copy(), result)
        features = extractor.extract(result)

        # Hien feature vector len cua so
        if features is not None:
            # In 13 feature thanh 1 dong
            text_lines = [
                f"Angles R/L: {features[0]:.2f} / {features[1]:.2f}",
                f"Body-Sh-Elb R/L: {features[2]:.2f} / {features[3]:.2f}",
                f"Wrist Y R/L: {features[4]:.2f} / {features[5]:.2f}",
                f"Wrist X R/L: {features[6]:.2f} / {features[7]:.2f}",
                f"Wrists dist: {features[8]:.2f}",
                f"R hand: ext={features[9]:.0f} thumb={features[10]:.2f}",
                f"L hand: ext={features[11]:.0f} thumb={features[12]:.2f}",
            ]
            for i, line in enumerate(text_lines):
                cv2.putText(
                    annotated, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )
        else:
            cv2.putText(
                annotated, "No pose detected", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

        cv2.imshow('Feature Extractor Test', annotated)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    detector.close()
    cv2.destroyAllWindows()