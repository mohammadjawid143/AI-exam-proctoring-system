import os
import time
from pathlib import Path
from collections import deque

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp


BASE_DIR = Path(__file__).resolve().parent

POSSIBLE_MODEL_PATHS = [
    BASE_DIR / "models" / "model.h5",
    BASE_DIR / "models" / "liveness_model.h5",
    BASE_DIR.parent / "models" / "model.h5",
    Path("models") / "model.h5",
]


class CNNLivenessDetector:
    """
    CNN-based face liveness detector.

    This class does ONLY liveness detection:
    - It does not verify student identity.
    - It uses MediaPipe FaceMesh for face landmarks and blink detection.
    - It uses the H5 CNN model only for Live/Spoof scoring.
    """

    def __init__(
        self,
        model_path=None,
        threshold=0.70,
        sigmoid_live_high=True,
        live_class_index=0,
        padding=0.38,
        enable_blink_challenge=True,
        blink_valid_seconds=12.0,
        blink_challenge_interval_seconds=18.0,
        min_face_area_ratio=0.015,
        max_face_area_ratio=0.55,
        max_face_area_change=0.25,
        min_blur_score=12.0,
        min_brightness=25.0,
        max_brightness=240.0,
        ear_threshold=0.20,
        required_closed_frames=2,
        stability_window=3,
        spoof_threshold=0.35,
    ):
        if model_path is None:
            model_path = self.find_model_path()

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Liveness model not found: {self.model_path}")

        # Small H5 liveness models can be over-sensitive.
        # We cap the effective threshold to reduce false Spoof detection.
        self.threshold = min(float(threshold), 0.60)
        self.original_threshold = float(threshold)
        self.spoof_threshold = float(spoof_threshold)

        self.sigmoid_live_high = sigmoid_live_high
        self.live_class_index = live_class_index
        self.padding = padding

        # Blink challenge settings.
        self.enable_blink_challenge = enable_blink_challenge
        self.blink_valid_seconds = blink_valid_seconds
        self.blink_challenge_interval_seconds = blink_challenge_interval_seconds

        # Quality thresholds.
        self.min_face_area_ratio = min_face_area_ratio
        self.max_face_area_ratio = max_face_area_ratio
        self.max_face_area_change = max_face_area_change
        self.min_blur_score = min_blur_score
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

        self.ear_threshold = ear_threshold
        self.required_closed_frames = required_closed_frames

        self.closed_frames = 0
        self.last_blink_time = 0.0
        self.next_blink_challenge_time = 0.0

        self.last_face_area_ratio = None
        self.score_history = deque(maxlen=stability_window)

        self.model = tf.keras.models.load_model(
            str(self.model_path),
            compile=False
        )

        self.input_size = self.get_model_input_size()

        # MediaPipe is used for face landmarks and blink detection.
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55
        )

        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        print("CNN Liveness model loaded:", self.model_path)
        print("Model input shape:", self.model.input_shape)
        print("Model output shape:", self.model.output_shape)
        print("Input size:", self.input_size)
        print("Original threshold:", self.original_threshold)
        print("Effective threshold:", self.threshold)
        print("Spoof threshold:", self.spoof_threshold)
        print("Sigmoid live high:", self.sigmoid_live_high)
        print("Blink challenge enabled:", self.enable_blink_challenge)
        print("Blink valid seconds:", self.blink_valid_seconds)
        print("Blink challenge interval:", self.blink_challenge_interval_seconds)

    def find_model_path(self):
        """
        Find the liveness H5 model automatically.
        """
        for path in POSSIBLE_MODEL_PATHS:
            if path.exists():
                return path

        return POSSIBLE_MODEL_PATHS[0]

    def get_model_input_size(self):
        """
        Read input size from the loaded Keras model.
        """
        input_shape = self.model.input_shape

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if len(input_shape) != 4:
            return (150, 150)

        # NHWC format: (None, H, W, C)
        if input_shape[1] not in (1, 3):
            height = input_shape[1]
            width = input_shape[2]

        # NCHW format: (None, C, H, W)
        else:
            height = input_shape[2]
            width = input_shape[3]

        if height is None or width is None:
            return (150, 150)

        return (int(width), int(height))

    def reset_blink(self):
        """
        Reset blink state only.
        """
        self.closed_frames = 0
        self.last_blink_time = 0.0
        self.next_blink_challenge_time = 0.0

    def reset_tracking_state(self):
        """
        Reset all temporal liveness state.
        """
        self.reset_blink()
        self.last_face_area_ratio = None
        self.score_history.clear()

    def _landmark_to_point(self, landmark, frame_w, frame_h):
        """
        Convert MediaPipe normalized landmark to pixel point.
        """
        x = int(landmark.x * frame_w)
        y = int(landmark.y * frame_h)

        x = max(0, min(frame_w - 1, x))
        y = max(0, min(frame_h - 1, y))

        return np.array([x, y], dtype=np.int32)

    def _landmarks_to_square_bbox(self, landmarks, frame_w, frame_h):
        """
        Build a square padded face box from landmarks.
        """
        points = [
            self._landmark_to_point(lm, frame_w, frame_h)
            for lm in landmarks
        ]

        points = np.array(points, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(points)

        if w <= 0 or h <= 0:
            return None, 0.0

        face_area_ratio = (w * h) / float(frame_w * frame_h)

        cx = x + w / 2.0
        cy = y + h / 2.0
        side = max(w, h)
        side = int(side * (1.0 + self.padding * 2.0))

        x1 = int(cx - side / 2.0)
        y1 = int(cy - side / 2.0)
        x2 = x1 + side
        y2 = y1 + side

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_w, x2)
        y2 = min(frame_h, y2)

        if x2 <= x1 or y2 <= y1:
            return None, 0.0

        return (x1, y1, x2, y2), face_area_ratio

    def _face_error(
        self,
        status,
        warning=True,
        face_count=0,
        face_box=None,
        face_crop=None,
        landmarks=None,
        face_area_ratio=0.0
    ):
        """
        Build a standard face detection error dictionary.
        """
        return {
            "ok": False,
            "status": status,
            "warning": warning,
            "face_count": face_count,
            "face_box": face_box,
            "face_crop": face_crop,
            "landmarks": landmarks,
            "face_area_ratio": face_area_ratio
        }

    def _detect_face_data(self, frame):
        """
        Detect face and return crop, landmarks, and metadata.
        """
        if frame is None:
            self.reset_tracking_state()
            return self._face_error("No frame", warning=True)

        frame_h, frame_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self.reset_tracking_state()
            return self._face_error("No face detected", warning=True)

        face_items = []

        for face_landmarks in results.multi_face_landmarks:
            bbox, area_ratio = self._landmarks_to_square_bbox(
                face_landmarks.landmark,
                frame_w,
                frame_h
            )

            if bbox is None:
                continue

            face_items.append({
                "bbox": bbox,
                "area_ratio": area_ratio,
                "landmarks": face_landmarks.landmark
            })

        if not face_items:
            self.reset_tracking_state()
            return self._face_error("Face crop error", warning=True)

        face_items = sorted(
            face_items,
            key=lambda item: item["area_ratio"],
            reverse=True
        )

        largest_face = face_items[0]
        x1, y1, x2, y2 = largest_face["bbox"]
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            self.reset_tracking_state()
            return self._face_error(
                "Face crop error",
                warning=True,
                face_count=len(face_items),
                face_box=largest_face["bbox"],
                landmarks=largest_face["landmarks"],
                face_area_ratio=largest_face["area_ratio"]
            )

        if len(face_items) > 1:
            self.reset_tracking_state()
            return self._face_error(
                "Multiple faces detected",
                warning=True,
                face_count=len(face_items),
                face_box=largest_face["bbox"],
                face_crop=face_crop,
                landmarks=largest_face["landmarks"],
                face_area_ratio=largest_face["area_ratio"]
            )

        return {
            "ok": True,
            "status": "Face detected",
            "warning": False,
            "face_count": 1,
            "face_box": largest_face["bbox"],
            "face_crop": face_crop,
            "landmarks": largest_face["landmarks"],
            "face_area_ratio": largest_face["area_ratio"]
        }

    def _check_face_quality(self, face_crop, face_area_ratio):
        """
        Check face size, blur, and lighting.
        """
        if face_area_ratio < self.min_face_area_ratio:
            return False, "Face too small", False, None

        if face_area_ratio > self.max_face_area_ratio:
            return False, "Face/photo too close", True, None

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))

        quality = {
            "blur_score": blur_score,
            "brightness": brightness
        }

        if blur_score < self.min_blur_score:
            return False, "Face too blurry", False, quality

        if brightness < self.min_brightness:
            return False, "Poor lighting: too dark", False, quality

        if brightness > self.max_brightness:
            return False, "Poor lighting: too bright", False, quality

        return True, "Quality OK", False, quality

    def _preprocess_face(self, face_crop):
        """
        Prepare face crop for CNN model.
        """
        image = cv2.resize(face_crop, self.input_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def _predict_model(self, face_crop):
        """
        Run CNN model and return stable scores.
        """
        face_input = self._preprocess_face(face_crop)
        prediction = self.model.predict(face_input, verbose=0)

        prediction = np.array(prediction).squeeze().astype("float32").flatten()
        raw_prediction = prediction.tolist()

        if prediction.size == 1:
            value = float(prediction[0])

            if self.sigmoid_live_high:
                live_score = value
                spoof_score = 1.0 - value
            else:
                spoof_score = value
                live_score = 1.0 - value

        else:
            values = prediction.astype("float32")

            if np.max(values) > 1.0 or np.sum(values) <= 0:
                exp_values = np.exp(values - np.max(values))
                values = exp_values / np.sum(exp_values)
            else:
                values = values / max(float(np.sum(values)), 1e-6)

            live_index = int(self.live_class_index)
            live_index = max(0, min(len(values) - 1, live_index))

            live_score = float(values[live_index])
            spoof_score = float(1.0 - live_score)

        self.score_history.append(live_score)

        stable_live_score = float(np.mean(self.score_history))
        stable_spoof_score = 1.0 - stable_live_score

        model_live = stable_live_score >= self.threshold
        definite_spoof = stable_live_score <= self.spoof_threshold

        return (
            model_live,
            definite_spoof,
            stable_live_score,
            stable_spoof_score,
            raw_prediction
        )

    def _calculate_ear(self, landmarks, eye_points, frame_w, frame_h):
        """
        Calculate Eye Aspect Ratio.
        """
        points = []

        for index in eye_points:
            lm = landmarks[index]
            points.append(
                np.array([
                    lm.x * frame_w,
                    lm.y * frame_h
                ], dtype=np.float32)
            )

        horizontal = np.linalg.norm(points[0] - points[3])
        vertical_1 = np.linalg.norm(points[1] - points[5])
        vertical_2 = np.linalg.norm(points[2] - points[4])

        if horizontal <= 1e-6:
            return 0.0

        return float((vertical_1 + vertical_2) / (2.0 * horizontal))

    def _detect_blink(self, frame, landmarks):
        """
        Detect blink using EAR.
        """
        frame_h, frame_w = frame.shape[:2]

        left_ear = self._calculate_ear(
            landmarks,
            self.LEFT_EYE,
            frame_w,
            frame_h
        )

        right_ear = self._calculate_ear(
            landmarks,
            self.RIGHT_EYE,
            frame_w,
            frame_h
        )

        ear = (left_ear + right_ear) / 2.0
        blink_now = False

        if ear < self.ear_threshold:
            self.closed_frames += 1

        else:
            if self.closed_frames >= self.required_closed_frames:
                blink_now = True
                self.last_blink_time = time.time()
                self.next_blink_challenge_time = (
                    time.time() + self.blink_challenge_interval_seconds
                )

            self.closed_frames = 0

        blink_valid = (
            time.time() - self.last_blink_time
        ) <= self.blink_valid_seconds

        return blink_now, ear, blink_valid

    def _check_face_change(self, face_area_ratio):
        """
        Reset blink when face size changes too much.
        """
        if self.last_face_area_ratio is None:
            self.last_face_area_ratio = face_area_ratio
            return False

        change = abs(face_area_ratio - self.last_face_area_ratio)
        self.last_face_area_ratio = face_area_ratio

        if change > self.max_face_area_change:
            self.reset_blink()
            return True

        return False

    def _needs_blink_challenge(self, blink_valid):
        """
        Decide whether blink challenge is required.
        """
        if not self.enable_blink_challenge:
            return False

        now = time.time()

        if self.last_blink_time <= 0:
            return not blink_valid

        if now >= self.next_blink_challenge_time and not blink_valid:
            return True

        return False

    def _base_result(
        self,
        is_live=False,
        model_live=False,
        warning=True,
        status="Unknown",
        live_score=None,
        spoof_score=None,
        face_box=None,
        raw_prediction=None,
        blink_valid=False,
        blink_now=False,
        ear=0.0,
        face_area_ratio=0.0,
        face_changed=False,
        requires_action=False,
        face_count=0,
        quality=None,
        decision_reason=""
    ):
        """
        Build a standard liveness result dictionary.
        """
        return {
            "is_live": is_live,
            "model_live": model_live,
            "warning": warning,
            "status": status,
            "live_score": live_score,
            "spoof_score": spoof_score,
            "face_box": face_box,
            "raw_prediction": raw_prediction,
            "raw": raw_prediction,
            "blink_valid": blink_valid,
            "blink_now": blink_now,
            "ear": ear,
            "face_area_ratio": face_area_ratio,
            "face_changed": face_changed,
            "requires_action": requires_action,
            "face_count": face_count,
            "quality": quality or {},
            "decision_reason": decision_reason,
        }

    def predict(self, frame):
        """
        Main liveness detection function.
        """
        face_data = self._detect_face_data(frame)

        if not face_data["ok"]:
            return self._base_result(
                is_live=False,
                model_live=False,
                warning=face_data["warning"],
                status=face_data["status"],
                face_box=face_data["face_box"],
                face_area_ratio=face_data["face_area_ratio"],
                face_count=face_data["face_count"],
                decision_reason=face_data["status"]
            )

        face_crop = face_data["face_crop"]
        face_box = face_data["face_box"]
        landmarks = face_data["landmarks"]
        face_area_ratio = face_data["face_area_ratio"]
        face_count = face_data["face_count"]

        quality_ok, quality_status, quality_warning, quality = self._check_face_quality(
            face_crop,
            face_area_ratio
        )

        blink_now, ear, blink_valid = self._detect_blink(frame, landmarks)
        face_changed = self._check_face_change(face_area_ratio)

        if face_changed:
            blink_valid = False

        if not quality_ok:
            if quality_warning:
                self.reset_tracking_state()

            return self._base_result(
                is_live=False,
                model_live=False,
                warning=quality_warning,
                status=quality_status,
                face_box=face_box,
                blink_valid=False,
                blink_now=blink_now,
                ear=ear,
                face_area_ratio=face_area_ratio,
                face_changed=face_changed,
                requires_action=False,
                face_count=face_count,
                quality=quality,
                decision_reason=quality_status
            )

        try:
            (
                model_live,
                definite_spoof,
                live_score,
                spoof_score,
                raw_prediction,
            ) = self._predict_model(face_crop)

        except Exception as error:
            return self._base_result(
                is_live=False,
                model_live=False,
                warning=True,
                status=f"Liveness model error: {error}",
                face_box=face_box,
                blink_valid=blink_valid,
                blink_now=blink_now,
                ear=ear,
                face_area_ratio=face_area_ratio,
                face_changed=face_changed,
                requires_action=False,
                face_count=face_count,
                quality=quality,
                decision_reason="model_exception"
            )

        if definite_spoof:
            self.reset_blink()

            return self._base_result(
                is_live=False,
                model_live=False,
                warning=True,
                status="Spoof face detected",
                live_score=live_score,
                spoof_score=spoof_score,
                face_box=face_box,
                raw_prediction=raw_prediction,
                blink_valid=False,
                blink_now=blink_now,
                ear=ear,
                face_area_ratio=face_area_ratio,
                face_changed=face_changed,
                requires_action=False,
                face_count=face_count,
                quality=quality,
                decision_reason="definite_spoof_score"
            )

        if not model_live:
            return self._base_result(
                is_live=False,
                model_live=False,
                warning=False,
                status="Checking liveness",
                live_score=live_score,
                spoof_score=spoof_score,
                face_box=face_box,
                raw_prediction=raw_prediction,
                blink_valid=blink_valid,
                blink_now=blink_now,
                ear=ear,
                face_area_ratio=face_area_ratio,
                face_changed=face_changed,
                requires_action=False,
                face_count=face_count,
                quality=quality,
                decision_reason="borderline_live_score"
            )

        needs_blink = self._needs_blink_challenge(blink_valid)

        if needs_blink:
            return self._base_result(
                is_live=False,
                model_live=True,
                warning=False,
                status="Please blink once",
                live_score=live_score,
                spoof_score=spoof_score,
                face_box=face_box,
                raw_prediction=raw_prediction,
                blink_valid=False,
                blink_now=blink_now,
                ear=ear,
                face_area_ratio=face_area_ratio,
                face_changed=face_changed,
                requires_action=True,
                face_count=face_count,
                quality=quality,
                decision_reason="blink_required"
            )

        return self._base_result(
            is_live=True,
            model_live=True,
            warning=False,
            status="Live",
            live_score=live_score,
            spoof_score=spoof_score,
            face_box=face_box,
            raw_prediction=raw_prediction,
            blink_valid=blink_valid,
            blink_now=blink_now,
            ear=ear,
            face_area_ratio=face_area_ratio,
            face_changed=face_changed,
            requires_action=False,
            face_count=face_count,
            quality=quality,
            decision_reason="live_score_and_blink_ok"
        )

    def draw_result(self, frame, result):
        """
        Draw liveness result on a frame.
        """
        face_box = result.get("face_box")

        if result.get("is_live"):
            color = (0, 255, 0)
        elif result.get("requires_action"):
            color = (0, 255, 255)
        elif result.get("warning"):
            color = (0, 0, 255)
        else:
            color = (255, 255, 0)

        if face_box is not None:
            x1, y1, x2, y2 = face_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        live_score = result.get("live_score")
        spoof_score = result.get("spoof_score")
        status = result.get("status", "Unknown")

        if live_score is not None and spoof_score is not None:
            label = (
                f"{status} | "
                f"Live: {live_score:.2f} | "
                f"Spoof: {spoof_score:.2f}"
            )
        else:
            label = status

        cv2.putText(
            frame,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
            cv2.LINE_AA
        )

        blink_text = (
            f"Blink: {result.get('blink_valid')} | "
            f"EAR: {result.get('ear', 0.0):.2f} | "
            f"Faces: {result.get('face_count', 0)}"
        )

        cv2.putText(
            frame,
            blink_text,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            color,
            2,
            cv2.LINE_AA
        )

        return frame