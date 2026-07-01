import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


BASE_DIR = Path(__file__).resolve().parent

POSSIBLE_MODEL_PATHS = [
    BASE_DIR / "models" / "minifasnet_v2.onnx",
    BASE_DIR.parent / "models" / "minifasnet_v2.onnx",
    Path("models") / "minifasnet_v2.onnx",
]


class MiniFASNetONNXLivenessDetector:
    def __init__(
        self,
        model_path=None,
        threshold=0.70,
        live_class_index=1,
        padding=0.35
    ):
        if model_path is None:
            model_path = self._find_model_path()

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MiniFASNet ONNX model not found: {self.model_path}"
            )

        self.threshold = threshold
        self.live_class_index = live_class_index
        self.padding = padding

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name

        self.input_layout, self.input_size = self._detect_input_format()

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        if self.face_cascade.empty():
            raise RuntimeError("OpenCV Haar Cascade face detector could not be loaded.")

        print("MiniFASNet ONNX liveness model loaded:", self.model_path)
        print("Input name:", self.input_name)
        print("Input shape:", self.input_shape)
        print("Input layout:", self.input_layout)
        print("Input size:", self.input_size)
        print("Liveness threshold:", self.threshold)
        print("Live class index:", self.live_class_index)

    def _find_model_path(self):
        for path in POSSIBLE_MODEL_PATHS:
            if path.exists():
                return path
        return POSSIBLE_MODEL_PATHS[0]

    def _detect_input_format(self):
        shape = self.input_shape

        clean_shape = []
        for value in shape:
            if isinstance(value, int):
                clean_shape.append(value)
            else:
                clean_shape.append(-1)

        if len(clean_shape) == 4 and clean_shape[1] == 3:
            height = clean_shape[2]
            width = clean_shape[3]

            if height <= 0:
                height = 80
            if width <= 0:
                width = 80

            return "NCHW", (width, height)

        if len(clean_shape) == 4 and clean_shape[3] == 3:
            height = clean_shape[1]
            width = clean_shape[2]

            if height <= 0:
                height = 80
            if width <= 0:
                width = 80

            return "NHWC", (width, height)

        return "NCHW", (80, 80)

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) == 0:
            return None, None

        faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
        x, y, w, h = faces[0]

        pad_x = int(w * self.padding)
        pad_y = int(h * self.padding)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + w + pad_x)
        y2 = min(frame.shape[0], y + h + pad_y)

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None, None

        return face_crop, (x1, y1, x2, y2)

    def preprocess_face(self, face_crop):
        resized = cv2.resize(face_crop, self.input_size)

        # اصلاح نرمال‌سازی: نگاشت به بازه‌ی [-1, 1]
        image = (resized.astype(np.float32) - 127.5) / 127.5

        if self.input_layout == "NCHW":
            image = np.transpose(image, (2, 0, 1))  # HWC -> CHW

        image = np.expand_dims(image, axis=0).astype(np.float32)
        return image

    def _softmax(self, values):
        values = np.array(values, dtype=np.float32)
        values = values - np.max(values)
        exp_values = np.exp(values)
        return exp_values / np.sum(exp_values)

    def _parse_prediction(self, output):
        prediction = np.array(output).squeeze()
        prediction = prediction.astype(np.float32).flatten()

        raw_prediction = prediction.tolist()

        if len(prediction) == 1:
            value = float(prediction[0])
            live_score = value
            spoof_score = 1.0 - value
            return live_score, spoof_score, raw_prediction

        prediction_sum = float(np.sum(prediction))
        if prediction_sum < 0.95 or prediction_sum > 1.05 or np.any(prediction < 0):
            probabilities = self._softmax(prediction)
        else:
            probabilities = prediction

        if self.live_class_index >= len(probabilities):
            self.live_class_index = 1 if len(probabilities) > 1 else 0

        live_score = float(probabilities[self.live_class_index])
        spoof_score = float(1.0 - live_score)

        return live_score, spoof_score, raw_prediction

    def predict(self, frame):
        face_crop, face_box = self.detect_face(frame)

        if face_crop is None:
            return {
                "is_live": False,
                "warning": True,
                "status": "No face detected",
                "live_score": None,
                "spoof_score": None,
                "face_box": None,
                "raw_prediction": None
            }

        face_input = self.preprocess_face(face_crop)

        output = self.session.run(
            [self.output_name],
            {self.input_name: face_input}
        )[0]

        live_score, spoof_score, raw_prediction = self._parse_prediction(output)

        is_live = live_score >= self.threshold

        if is_live:
            status = "Live face detected"
            warning = False
        else:
            status = "Spoof face detected"
            warning = True

        return {
            "is_live": is_live,
            "warning": warning,
            "status": status,
            "live_score": live_score,
            "spoof_score": spoof_score,
            "face_box": face_box,
            "raw_prediction": raw_prediction
        }

    def draw_result(self, frame, result):
        face_box = result.get("face_box")

        if face_box is None:
            return frame

        x1, y1, x2, y2 = face_box

        if result["is_live"]:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        live_score = result.get("live_score")
        spoof_score = result.get("spoof_score")

        if live_score is not None and spoof_score is not None:
            label = (
                f"{result['status']} | "
                f"Live: {live_score:.2f} | "
                f"Spoof: {spoof_score:.2f}"
            )
        else:
            label = result["status"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )
        return frame