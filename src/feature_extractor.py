"""
Convert raw pose / hand landmarks into a compact gesture feature vector.

The current runtime keeps the existing 13-dimensional feature schema so the
saved dataset format remains compatible. The code now also returns metadata so
the rest of the app can distinguish "idle" from "tracking lost".
"""

import numpy as np

import config


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


class HandLM:
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    INDEX_MCP = 5
    MIDDLE_MCP = 9
    RING_MCP = 13
    PINKY_MCP = 17


class FeatureExtractor:
    """Compute the 13-D feature vector used by the PCA classifier."""

    FEATURE_DIM = 13
    REQUIRED_POSE_LANDMARKS = (
        PoseLM.LEFT_SHOULDER,
        PoseLM.RIGHT_SHOULDER,
        PoseLM.LEFT_ELBOW,
        PoseLM.RIGHT_ELBOW,
        PoseLM.LEFT_WRIST,
        PoseLM.RIGHT_WRIST,
    )
    def extract(self, detection_result, return_meta=False):
        """
        Return a feature vector or `None`.

        When `return_meta=True`, return `(features, meta)` where `meta` includes:
        - `tracking_state`: ready / no_pose / low_visibility
        - `min_pose_visibility`
        - `hands_detected`
        - `right_hand_detected`
        - `left_hand_detected`
        """
        pose_lms = detection_result["pose_landmarks"]
        hand_lms_list = detection_result["hand_landmarks"]
        handedness_list = detection_result["handedness"]

        if pose_lms is None:
            meta = self._build_meta(
                tracking_state="no_pose",
                min_pose_visibility=0.0,
                hands_detected=0,
                right_hand_detected=False,
                left_hand_detected=False,
            )
            return (None, meta) if return_meta else None

        min_visibility = self._min_required_visibility(pose_lms)
        if min_visibility < config.POSE_MIN_LANDMARK_VISIBILITY:
            right_hand_lms, left_hand_lms = self._split_hands(
                hand_lms_list,
                handedness_list,
            )
            meta = self._build_meta(
                tracking_state="low_visibility",
                min_pose_visibility=min_visibility,
                hands_detected=int(right_hand_lms is not None) + int(left_hand_lms is not None),
                right_hand_detected=right_hand_lms is not None,
                left_hand_detected=left_hand_lms is not None,
            )
            return (None, meta) if return_meta else None

        pose = np.array([[lm.x, lm.y, lm.z] for lm in pose_lms.landmark], dtype=np.float32)

        features = []
        l_sh = pose[PoseLM.LEFT_SHOULDER, :2]
        r_sh = pose[PoseLM.RIGHT_SHOULDER, :2]
        l_el = pose[PoseLM.LEFT_ELBOW, :2]
        r_el = pose[PoseLM.RIGHT_ELBOW, :2]
        l_wr = pose[PoseLM.LEFT_WRIST, :2]
        r_wr = pose[PoseLM.RIGHT_WRIST, :2]
        l_hp = pose[PoseLM.LEFT_HIP, :2]
        r_hp = pose[PoseLM.RIGHT_HIP, :2]

        shoulder_width = np.linalg.norm(l_sh - r_sh) + 1e-6

        features.append(self._angle(r_sh, r_el, r_wr))
        features.append(self._angle(l_sh, l_el, l_wr))
        features.append(self._angle(r_hp, r_sh, r_el))
        features.append(self._angle(l_hp, l_sh, l_el))
        features.append((r_wr[1] - r_sh[1]) / shoulder_width)
        features.append((l_wr[1] - l_sh[1]) / shoulder_width)
        features.append((r_wr[0] - r_sh[0]) / shoulder_width)
        features.append((l_wr[0] - l_sh[0]) / shoulder_width)
        features.append(np.linalg.norm(l_wr - r_wr) / shoulder_width)

        right_hand_lms, left_hand_lms = self._split_hands(
            hand_lms_list,
            handedness_list,
        )

        if right_hand_lms is not None:
            n_extended_r, thumb_ratio_r = self._hand_features(right_hand_lms)
        else:
            n_extended_r, thumb_ratio_r = -1.0, -1.0
        features.extend([n_extended_r, thumb_ratio_r])

        if left_hand_lms is not None:
            n_extended_l, thumb_ratio_l = self._hand_features(left_hand_lms)
        else:
            n_extended_l, thumb_ratio_l = -1.0, -1.0
        features.extend([n_extended_l, thumb_ratio_l])

        feature_array = np.array(features, dtype=np.float32)
        meta = self._build_meta(
            tracking_state="ready",
            min_pose_visibility=min_visibility,
            hands_detected=int(right_hand_lms is not None) + int(left_hand_lms is not None),
            right_hand_detected=right_hand_lms is not None,
            left_hand_detected=left_hand_lms is not None,
        )
        return (feature_array, meta) if return_meta else feature_array

    def _build_meta(
        self,
        tracking_state,
        min_pose_visibility,
        hands_detected,
        right_hand_detected,
        left_hand_detected,
    ):
        return {
            "tracking_state": tracking_state,
            "min_pose_visibility": float(min_pose_visibility),
            "hands_detected": int(hands_detected),
            "right_hand_detected": bool(right_hand_detected),
            "left_hand_detected": bool(left_hand_detected),
        }

    def _min_required_visibility(self, pose_lms):
        qualities = [
            self._landmark_quality(pose_lms.landmark[idx])
            for idx in self.REQUIRED_POSE_LANDMARKS
        ]
        return min(qualities) if qualities else 0.0

    def _landmark_quality(self, landmark):
        values = []
        for attr in ("visibility", "presence"):
            value = getattr(landmark, attr, None)
            if value is not None:
                values.append(float(value))
        if not values:
            return 1.0
        return max(values)

    def _angle(self, p1, p2, p3):
        """Angle at `p2` between vectors `p2->p1` and `p2->p3`."""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.arccos(cos_angle))

    def _split_hands(self, hand_lms_list, handedness_list):
        """Return `(right_hand, left_hand)` in the user's coordinate system."""
        if hand_lms_list is None or handedness_list is None:
            return None, None

        right_hand = None
        left_hand = None

        for hand_lms, handedness in zip(hand_lms_list, handedness_list):
            label = handedness.classification[0].label
            arr = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
            user_label = self._resolve_user_hand_label(label)
            if user_label == "Right":
                right_hand = arr
            elif user_label == "Left":
                left_hand = arr

        return right_hand, left_hand

    def _resolve_user_hand_label(self, raw_label):
        """Map MediaPipe handedness into the user's left/right hand."""
        if raw_label not in {"Left", "Right"}:
            return raw_label
        if config.CAMERA_IS_MIRRORED:
            return raw_label
        return "Left" if raw_label == "Right" else "Right"

    def _hand_features(self, hand_lms):
        """
        Return two compact hand descriptors:
        - number of extended fingers (0-5)
        - thumb openness ratio
        """
        wrist = hand_lms[HandLM.WRIST, :2]

        tips = {
            "thumb": hand_lms[HandLM.THUMB_TIP, :2],
            "index": hand_lms[HandLM.INDEX_TIP, :2],
            "middle": hand_lms[HandLM.MIDDLE_TIP, :2],
            "ring": hand_lms[HandLM.RING_TIP, :2],
            "pinky": hand_lms[HandLM.PINKY_TIP, :2],
        }
        mcps = {
            "index": hand_lms[HandLM.INDEX_MCP, :2],
            "middle": hand_lms[HandLM.MIDDLE_MCP, :2],
            "ring": hand_lms[HandLM.RING_MCP, :2],
            "pinky": hand_lms[HandLM.PINKY_MCP, :2],
        }

        hand_size = np.linalg.norm(wrist - mcps["middle"]) + 1e-6
        n_extended = 0

        for finger in ("index", "middle", "ring", "pinky"):
            tip_dist = np.linalg.norm(tips[finger] - wrist)
            mcp_dist = np.linalg.norm(mcps[finger] - wrist)
            if tip_dist > mcp_dist:
                n_extended += 1

        thumb_to_index_mcp = np.linalg.norm(tips["thumb"] - mcps["index"]) / hand_size
        if thumb_to_index_mcp > 0.6:
            n_extended += 1

        thumb_ratio = thumb_to_index_mcp
        return float(n_extended), float(thumb_ratio)


if __name__ == "__main__":
    import cv2

    from src.camera import Camera
    from src.preprocessor import Preprocessor
    from src.detector import PoseHandDetector

    cam = Camera()
    pre = Preprocessor()
    detector = PoseHandDetector()
    extractor = FeatureExtractor()

    print("Press ESC to quit.")
    while True:
        frame = cam.read_frame()
        if frame is None:
            break

        rgb, bgr = pre.process(frame)
        detection = detector.detect(rgb)
        features, meta = extractor.extract(detection, return_meta=True)
        annotated = detector.draw_skeleton(bgr.copy(), detection)

        if features is not None:
            text = f"{meta['tracking_state']} | dim={len(features)}"
        else:
            text = meta["tracking_state"]
        cv2.putText(
            annotated,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Feature Extractor", annotated)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    detector.close()
    cv2.destroyAllWindows()
