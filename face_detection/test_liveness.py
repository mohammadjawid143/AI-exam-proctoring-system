import os
import time
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp


MODEL_PATHS = [
    Path("face_detection/models/model.h5"),
    Path("face_detection/models/liveness_model.h5"),
    Path("models/model.h5"),
]


class H5LivenessWithBlinkTest:
    def __init__(self):
        self.model_path = self.find_model()

        self.model = tf.keras.models.load_model(
            str(self.model_path),
            compile=False
        )

        self.input_size = self.get_input_size()

        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.threshold = 0.70

        # Your model output:
        # raw close to 1 = Live
        # raw close to 0 = Spoof
        self.sigmoid_live_high = True

        # Blink challenge settings
        self.ear_threshold = 0.20
        self.closed_frames = 0
        self.required_closed_frames = 2
        self.last_blink_time = 0
        self.blink_valid_seconds = 1.2

        # Reset challenge if face size changes too much
        self.last_face_area_ratio = None
        self.max_face_area_change = 0.18

        # If face is extremely close, mark suspicious
        self.max_face_area_ratio = 0.42

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        print("Model loaded:", self.model_path)
        print("Model input shape:", self.model.input_shape)
        print("Input size:", self.input_size)
        print("Threshold:", self.threshold)
        print("sigmoid_live_high:", self.sigmoid_live_high)

    def find_model(self):
        for path in MODEL_PATHS:
            if path.exists():
                return path

        raise FileNotFoundError("model.h5 not found in face_detection/models/")

    def get_input_size(self):
        shape = self.model.input_shape

        if isinstance(shape, list):
            shape = shape[0]

        h = shape[1]
        w = shape[2]

        if h is None or w is None:
            return (150, 150)

        return (int(w), int(h))

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

        padding = 0.30
        pad_x = int(w * padding)
        pad_y = int(h * padding)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame_w, x + w + pad_x)
        y2 = min(frame_h, y + h + pad_y)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None, None, 0.0

        return crop, (x1, y1, x2, y2), face_area_ratio

    def preprocess(self, face_crop):
        image = cv2.resize(face_crop, self.input_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def predict_liveness(self, frame):
        face_crop, face_box, face_area_ratio = self.detect_face(frame)

        if face_crop is None:
            return {
                "model_live": False,
                "live_score": None,
                "spoof_score": None,
                "raw": None,
                "face_box": None,
                "face_area_ratio": 0.0
            }

        input_tensor = self.preprocess(face_crop)
        prediction = self.model.predict(input_tensor, verbose=0)
        raw = np.array(prediction).squeeze().astype("float32").flatten()

        value = float(raw[0])

        if self.sigmoid_live_high:
            live_score = value
            spoof_score = 1.0 - value
        else:
            spoof_score = value
            live_score = 1.0 - value

        model_live = live_score >= self.threshold

        return {
            "model_live": model_live,
            "live_score": live_score,
            "spoof_score": spoof_score,
            "raw": raw.tolist(),
            "face_box": face_box,
            "face_area_ratio": face_area_ratio
        }

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

    def detect_blink_now(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self.closed_frames = 0
            return False, 0.0

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

        return blink_now, ear

    def reset_blink(self):
        self.last_blink_time = 0
        self.closed_frames = 0

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

    def final_decision(self, frame):
        live_result = self.predict_liveness(frame)
        blink_now, ear = self.detect_blink_now(frame)

        face_area_ratio = live_result["face_area_ratio"]
        face_changed = self.check_face_change(face_area_ratio)

        blink_valid = (time.time() - self.last_blink_time) <= self.blink_valid_seconds

        model_live = live_result["model_live"]

        if live_result["face_box"] is None:
            status = "No face detected"
            final_live = False

        elif face_area_ratio > self.max_face_area_ratio:
            status = "SPOOF: face/photo too close"
            final_live = False
            self.reset_blink()

        elif face_changed:
            status = "WAITING: face changed, blink again"
            final_live = False

        elif not model_live:
            status = "SPOOF: model says fake"
            final_live = False
            self.reset_blink()

        elif not blink_valid:
            status = "WAITING: blink now"
            final_live = False

        else:
            status = "LIVE: real person"
            final_live = True

        return {
            "status": status,
            "final_live": final_live,
            "model_live": model_live,
            "blink_now": blink_now,
            "blink_valid": blink_valid,
            "ear": ear,
            "face_changed": face_changed,
            **live_result
        }

    def draw(self, frame, result):
        if result["final_live"]:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.putText(
            frame,
            result["status"],
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        if result["face_box"] is not None:
            x1, y1, x2, y2 = result["face_box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        live_score = result["live_score"]
        spoof_score = result["spoof_score"]

        if live_score is not None:
            score_text = f"Model Live: {live_score:.2f} | Spoof: {spoof_score:.2f}"
        else:
            score_text = "Model: no face"

        blink_text = (
            f"Blink valid: {result['blink_valid']} | "
            f"EAR: {result['ear']:.2f}"
        )

        face_text = (
            f"Face area: {result['face_area_ratio']:.2f} | "
            f"Changed: {result['face_changed']}"
        )

        setting_text = (
            f"Threshold: {self.threshold:.2f} | "
            f"sigmoid_live_high: {self.sigmoid_live_high}"
        )

        cv2.putText(
            frame,
            score_text,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            blink_text,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            face_text,
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            setting_text,
            (20, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


def run_test():
    detector = H5LivenessWithBlinkTest()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera could not be opened.")
        return

    print("H5 Liveness + Blink Challenge started.")
    print("Press q = quit")
    print("Press b = reset blink")
    print("Press p = reverse prediction meaning")
    print("Press + = increase threshold")
    print("Press - = decrease threshold")
    print("")
    print("Real test: look at camera and blink once.")
    print("Phone photo test: press b first, then show phone photo.")

    last_print = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        result = detector.final_decision(frame)
        detector.draw(frame, result)

        now = time.time()

        if now - last_print >= 1:
            print("--------------------------------")
            print("Status:", result["status"])
            print("Final Live:", result["final_live"])
            print("Model Live:", result["model_live"])
            print("Live Score:", result["live_score"])
            print("Spoof Score:", result["spoof_score"])
            print("Raw:", result["raw"])
            print("Blink valid:", result["blink_valid"])
            print("EAR:", result["ear"])
            print("Face area:", result["face_area_ratio"])
            print("Face changed:", result["face_changed"])
            last_print = now

        cv2.imshow("H5 Liveness + Blink Challenge", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("b"):
            detector.reset_blink()
            detector.last_face_area_ratio = None
            print("Blink challenge reset.")

        elif key == ord("p"):
            detector.sigmoid_live_high = not detector.sigmoid_live_high
            print("sigmoid_live_high:", detector.sigmoid_live_high)

        elif key == ord("+") or key == ord("="):
            detector.threshold = min(0.95, detector.threshold + 0.05)
            print("Threshold:", detector.threshold)

        elif key == ord("-"):
            detector.threshold = max(0.05, detector.threshold - 0.05)
            print("Threshold:", detector.threshold)

    cap.release()
    cv2.destroyAllWindows()


def run_deepface_authentication():
    run_test()


if __name__ == "__main__":
    run_test()