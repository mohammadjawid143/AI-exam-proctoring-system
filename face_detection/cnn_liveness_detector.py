import os
import time
from pathlib import Path

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
    def __init__(
        self,
        model_path=None,
        threshold=0.70,
        sigmoid_live_high=True,
        padding=0.30,
        enable_blink_challenge=True,
        blink_valid_seconds=2.0
    ):
        if model_path is None:
            model_path = self.find_model_path()

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Liveness model not found: {self.model_path}")

        self.threshold = threshold

        # For your H5 model:
        # raw close to 1 = Live
        # raw close to 0 = Spoof
        self.sigmoid_live_high = sigmoid_live_high

        self.padding = padding
        self.enable_blink_challenge = enable_blink_challenge
        self.blink_valid_seconds = blink_valid_seconds

        self.model = tf.keras.models.load_model(
            str(self.model_path),
            compile=False
        )

        self.input_size = self.get_model_input_size()

        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        if self.face_detector.empty():
            raise RuntimeError("OpenCV Haar Cascade face detector could not be loaded.")

        # Blink challenge values
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.ear_threshold = 0.20
        self.closed_frames = 0
        self.required_closed_frames = 2
        self.last_blink_time = 0

        # Reset blink if face changes too much
        self.last_face_area_ratio = None
        self.max_face_area_change = 0.18

        # If face/photo is too close, consider suspicious
        self.max_face_area_ratio = 0.42

        print("CNN Liveness model loaded:", self.model_path)
        print("Model input shape:", self.model.input_shape)
        print("Model output shape:", self.model.output_shape)
        print("Input size:", self.input_size)
        print("Threshold:", self.threshold)
        print("Sigmoid live high:", self.sigmoid_live_high)
        print("Blink challenge enabled:", self.enable_blink_challenge)

    def find_model_path(self):
        for path in POSSIBLE_MODEL_PATHS:
            if path.exists():
                return path

        return POSSIBLE_MODEL_PATHS[0]

    def get_model_input_size(self):
        input_shape = self.model.input_shape

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        height = input_shape[1]
        width = input_shape[2]

        if height is None or width is None:
            return (150, 150)

        return (int(width), int(height))

    def reset_blink(self):
        self.last_blink_time = 0
        self.closed_frames = 0

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) == 0:
            return None, None, 0.0

        faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
        x, y, w, h = faces[0]

        frame_h, frame_w = frame.shape[:2]
        face_area_ratio = (w * h) / float(frame_w * frame_h)

        pad_x = int(w * self.padding)
        pad_y = int(h * self.padding)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame_w, x + w + pad_x)
        y2 = min(frame_h, y + h + pad_y)

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None, None, 0.0

        return face_crop, (x1, y1, x2, y2), face_area_ratio

    def preprocess_face(self, face_crop):
        image = cv2.resize(face_crop, self.input_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)

        return image

    def predict_model(self, face_crop):
        face_input = self.preprocess_face(face_crop)
        prediction = self.model.predict(face_input, verbose=0)

        prediction = np.array(prediction).squeeze().astype("float32").flatten()
        raw_prediction = prediction.tolist()

        value = float(prediction[0])

        if self.sigmoid_live_high:
            live_score = value
            spoof_score = 1.0 - value
        else:
            spoof_score = value
            live_score = 1.0 - value

        model_live = live_score >= self.threshold

        return model_live, live_score, spoof_score, raw_prediction

    def calculate_ear(self, landmarks, eye_points, frame_w, frame_h):
        points = []

        for index in eye_points:
            lm = landmarks[index]
            points.append(
                np.array([
                    lm.x * frame_w,
                    lm.y * frame_h
                ])
            )

        horizontal = np.linalg.norm(points[0] - points[3])
        vertical_1 = np.linalg.norm(points[1] - points[5])
        vertical_2 = np.linalg.norm(points[2] - points[4])

        if horizontal == 0:
            return 0.0

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    def detect_blink(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self.closed_frames = 0
            return False, 0.0, False

        landmarks = results.multi_face_landmarks[0].landmark
        frame_h, frame_w = frame.shape[:2]

        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]

        left_ear = self.calculate_ear(landmarks, left_eye, frame_w, frame_h)
        right_ear = self.calculate_ear(landmarks, right_eye, frame_w, frame_h)

        ear = (left_ear + right_ear) / 2.0

        blink_now = False

        if ear < self.ear_threshold:
            self.closed_frames += 1
        else:
            if self.closed_frames >= self.required_closed_frames:
                blink_now = True
                self.last_blink_time = time.time()

            self.closed_frames = 0

        blink_valid = (time.time() - self.last_blink_time) <= self.blink_valid_seconds

        return blink_now, ear, blink_valid

    def check_face_change(self, face_area_ratio):
        if self.last_face_area_ratio is None:
            self.last_face_area_ratio = face_area_ratio
            return False

        change = abs(face_area_ratio - self.last_face_area_ratio)
        self.last_face_area_ratio = face_area_ratio

        if change > self.max_face_area_change:
            self.reset_blink()
            return True

        return False

    def predict(self, frame):
        face_crop, face_box, face_area_ratio = self.detect_face(frame)

        if face_crop is None:
            self.reset_blink()
            self.last_face_area_ratio = None

            return {
                "is_live": False,
                "model_live": False,
                "warning": True,
                "status": "No face detected",
                "live_score": None,
                "spoof_score": None,
                "face_box": None,
                "raw_prediction": None,
                "blink_valid": False,
                "blink_now": False,
                "ear": 0.0,
                "face_area_ratio": 0.0,
                "face_changed": False,
                "requires_action": False
            }

        model_live, live_score, spoof_score, raw_prediction = self.predict_model(face_crop)

        blink_now, ear, blink_valid = self.detect_blink(frame)

        face_changed = self.check_face_change(face_area_ratio)

        if face_changed:
            blink_valid = False

        if face_area_ratio > self.max_face_area_ratio:
            self.reset_blink()

            return {
                "is_live": False,
                "model_live": model_live,
                "warning": True,
                "status": "Spoof detected: face/photo too close",
                "live_score": live_score,
                "spoof_score": spoof_score,
                "face_box": face_box,
                "raw_prediction": raw_prediction,
                "blink_valid": False,
                "blink_now": blink_now,
                "ear": ear,
                "face_area_ratio": face_area_ratio,
                "face_changed": face_changed,
                "requires_action": False
            }

        if not model_live:
            self.reset_blink()

            return {
                "is_live": False,
                "model_live": False,
                "warning": True,
                "status": "Spoof face detected",
                "live_score": live_score,
                "spoof_score": spoof_score,
                "face_box": face_box,
                "raw_prediction": raw_prediction,
                "blink_valid": False,
                "blink_now": blink_now,
                "ear": ear,
                "face_area_ratio": face_area_ratio,
                "face_changed": face_changed,
                "requires_action": False
            }

        if self.enable_blink_challenge and not blink_valid:
            return {
                "is_live": False,
                "model_live": True,
                "warning": False,
                "status": "Waiting for blink challenge",
                "live_score": live_score,
                "spoof_score": spoof_score,
                "face_box": face_box,
                "raw_prediction": raw_prediction,
                "blink_valid": False,
                "blink_now": blink_now,
                "ear": ear,
                "face_area_ratio": face_area_ratio,
                "face_changed": face_changed,
                "requires_action": True
            }

        return {
            "is_live": True,
            "model_live": True,
            "warning": False,
            "status": "Live face detected",
            "live_score": live_score,
            "spoof_score": spoof_score,
            "face_box": face_box,
            "raw_prediction": raw_prediction,
            "blink_valid": blink_valid,
            "blink_now": blink_now,
            "ear": ear,
            "face_area_ratio": face_area_ratio,
            "face_changed": face_changed,
            "requires_action": False
        }

    def draw_result(self, frame, result):
        face_box = result.get("face_box")

        if result["is_live"]:
            color = (0, 255, 0)
        elif result.get("requires_action"):
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        if face_box is not None:
            x1, y1, x2, y2 = face_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        live_score = result.get("live_score")
        spoof_score = result.get("spoof_score")

        if live_score is not None:
            label = (
                f"{result['status']} | "
                f"Live: {live_score:.2f} | "
                f"Spoof: {spoof_score:.2f}"
            )
        else:
            label = result["status"]

        cv2.putText(
            frame,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2
        )

        blink_text = (
            f"Blink valid: {result.get('blink_valid')} | "
            f"EAR: {result.get('ear', 0.0):.2f}"
        )

        cv2.putText(
            frame,
            blink_text,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        return frame