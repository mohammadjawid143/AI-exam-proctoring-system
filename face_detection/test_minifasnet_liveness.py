from pathlib import Path
import time
import cv2
import numpy as np
import onnxruntime as ort


try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


MODEL_PATHS = [
    Path("face_detection/models/minifasnet_v2.onnx"),
    Path("models/minifasnet_v2.onnx"),
    Path("minifasnet_v2.onnx"),
]


YOLO_PATHS = [
    Path("yolo11n.pt"),
    Path("yolov8n.pt"),
    Path("models/yolo11n.pt"),
    Path("models/yolov8n.pt"),
]


class MiniFASNetTest:
    def __init__(self):
        self.model_path = self.find_model()

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name

        self.input_layout, self.input_size = self.get_input_info()

        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        if self.face_detector.empty():
            raise RuntimeError("Could not load OpenCV face detector.")

        # Your model showed Live is class index 2
        self.live_class_index = 2
        self.threshold = 0.50

        self.preprocess_modes = [
            "bgr_minus1_1",
            "bgr_0_1",
            "rgb_minus1_1",
            "rgb_0_1",
        ]

        self.crop_scales = [2.0, 2.7, 3.0, 3.5]
        self.preprocess_index = 0
        self.crop_scale_index = 1

        # Blink challenge
        self.blink_detected = False
        self.eye_closed_frames = 0
        self.ear_threshold = 0.20
        self.required_closed_frames = 2

        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.face_mesh = None

        # Phone detection
        self.yolo_model = None

        if YOLO_AVAILABLE:
            yolo_path = self.find_yolo_model()

            if yolo_path is not None:
                try:
                    self.yolo_model = YOLO(str(yolo_path))
                    print("YOLO loaded:", yolo_path)
                except Exception as error:
                    print("YOLO load error:", error)

        print("Model loaded:", self.model_path)
        print("Input name:", self.input_name)
        print("Input shape:", self.input_shape)
        print("Output name:", self.output_name)
        print("Input layout:", self.input_layout)
        print("Input size:", self.input_size)
        print("Live class index:", self.live_class_index)
        print("MediaPipe available:", MEDIAPIPE_AVAILABLE)
        print("YOLO available:", self.yolo_model is not None)

    def find_model(self):
        for path in MODEL_PATHS:
            if path.exists():
                return path

        raise FileNotFoundError(
            "minifasnet_v2.onnx not found. Put it in face_detection/models/"
        )

    def find_yolo_model(self):
        for path in YOLO_PATHS:
            if path.exists():
                return path

        return None

    def get_input_info(self):
        shape = self.input_shape

        clean_shape = []

        for value in shape:
            if isinstance(value, int):
                clean_shape.append(value)
            else:
                clean_shape.append(-1)

        if len(clean_shape) == 4 and clean_shape[1] == 3:
            h = clean_shape[2] if clean_shape[2] > 0 else 80
            w = clean_shape[3] if clean_shape[3] > 0 else 80
            return "NCHW", (w, h)

        if len(clean_shape) == 4 and clean_shape[3] == 3:
            h = clean_shape[1] if clean_shape[1] > 0 else 80
            w = clean_shape[2] if clean_shape[2] > 0 else 80
            return "NHWC", (w, h)

        return "NCHW", (80, 80)

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) == 0:
            return None

        faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
        return faces[0]

    def crop_face(self, frame, face_box):
        x, y, w, h = face_box
        img_h, img_w = frame.shape[:2]

        scale = self.crop_scales[self.crop_scale_index]

        center_x = x + w / 2
        center_y = y + h / 2

        box_size = max(w, h) * scale

        x1 = int(center_x - box_size / 2)
        y1 = int(center_y - box_size / 2)
        x2 = int(center_x + box_size / 2)
        y2 = int(center_y + box_size / 2)

        left_pad = max(0, -x1)
        top_pad = max(0, -y1)
        right_pad = max(0, x2 - img_w)
        bottom_pad = max(0, y2 - img_h)

        padded = frame

        if left_pad > 0 or top_pad > 0 or right_pad > 0 or bottom_pad > 0:
            padded = cv2.copyMakeBorder(
                frame,
                top_pad,
                bottom_pad,
                left_pad,
                right_pad,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )

            x1 += left_pad
            x2 += left_pad
            y1 += top_pad
            y2 += top_pad

        crop = padded[y1:y2, x1:x2]

        draw_box = (
            max(0, x1 - left_pad),
            max(0, y1 - top_pad),
            min(img_w, x2 - left_pad),
            min(img_h, y2 - top_pad)
        )

        return crop, draw_box

    def preprocess(self, crop):
        mode = self.preprocess_modes[self.preprocess_index]

        image = cv2.resize(crop, self.input_size)

        if mode == "bgr_minus1_1":
            image = image.astype(np.float32)
            image = (image - 127.5) / 128.0

        elif mode == "bgr_0_1":
            image = image.astype(np.float32) / 255.0

        elif mode == "rgb_minus1_1":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)
            image = (image - 127.5) / 128.0

        elif mode == "rgb_0_1":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0

        else:
            image = image.astype(np.float32)
            image = (image - 127.5) / 128.0

        if self.input_layout == "NCHW":
            image = np.transpose(image, (2, 0, 1))

        image = np.expand_dims(image, axis=0).astype(np.float32)

        return image

    def softmax(self, values):
        values = np.array(values, dtype=np.float32)
        values = values - np.max(values)
        exp_values = np.exp(values)
        return exp_values / np.sum(exp_values)

    def parse_output(self, output):
        raw = np.array(output).squeeze().astype(np.float32).flatten()

        if len(raw) >= 2:
            probabilities = self.softmax(raw)

            live_score = float(probabilities[self.live_class_index])
            spoof_score = float(1.0 - live_score)

            return raw.tolist(), probabilities.tolist(), live_score, spoof_score

        value = float(raw[0])

        if value < 0.0 or value > 1.0:
            value = float(1.0 / (1.0 + np.exp(-value)))

        spoof_score = value
        live_score = 1.0 - value

        return raw.tolist(), [live_score, spoof_score], live_score, spoof_score

    def calculate_ear(self, landmarks, eye_points, frame_w, frame_h):
        points = []

        for index in eye_points:
            landmark = landmarks[index]
            points.append(
                np.array([
                    landmark.x * frame_w,
                    landmark.y * frame_h
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
        if self.face_mesh is None:
            return False, 0.0

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return False, 0.0

        face_landmarks = results.multi_face_landmarks[0].landmark
        frame_h, frame_w = frame.shape[:2]

        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]

        left_ear = self.calculate_ear(face_landmarks, left_eye, frame_w, frame_h)
        right_ear = self.calculate_ear(face_landmarks, right_eye, frame_w, frame_h)

        ear = (left_ear + right_ear) / 2.0

        if ear < self.ear_threshold:
            self.eye_closed_frames += 1
        else:
            if self.eye_closed_frames >= self.required_closed_frames:
                self.blink_detected = True

            self.eye_closed_frames = 0

        return self.blink_detected, ear

    def detect_phone(self, frame):
        if self.yolo_model is None:
            return False, []

        detected_phones = []

        try:
            results = self.yolo_model(frame, verbose=False)[0]

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = self.yolo_model.names.get(cls_id, str(cls_id))

                if conf < 0.35:
                    continue

                name_lower = name.lower()

                if "phone" in name_lower or "cell phone" in name_lower or "mobile" in name_lower:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    detected_phones.append({
                        "name": name,
                        "confidence": conf,
                        "box": (x1, y1, x2, y2)
                    })

        except Exception as error:
            print("Phone detection error:", error)

        return len(detected_phones) > 0, detected_phones

    def predict(self, frame):
        face_box = self.detect_face(frame)

        phone_detected, phones = self.detect_phone(frame)
        blink_detected, ear = self.detect_blink(frame)

        if face_box is None:
            return {
                "status": "No face detected",
                "is_live": False,
                "final_live": False,
                "live_score": None,
                "spoof_score": None,
                "raw": None,
                "probabilities": None,
                "draw_box": None,
                "phone_detected": phone_detected,
                "phones": phones,
                "blink_detected": blink_detected,
                "ear": ear,
            }

        crop, draw_box = self.crop_face(frame, face_box)

        input_tensor = self.preprocess(crop)

        output = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )[0]

        raw, probabilities, live_score, spoof_score = self.parse_output(output)

        model_live = live_score >= self.threshold

        # Final decision:
        # 1. MiniFASNet must say live
        # 2. User must blink
        # 3. Phone must not be detected
        if phone_detected:
            status = "SPOOF: Phone replay attack"
            final_live = False

        elif not blink_detected:
            status = "WAITING: Blink once"
            final_live = False

        elif model_live:
            status = "LIVE: Real person"
            final_live = True

        else:
            status = "SPOOF: MiniFASNet spoof"
            final_live = False

        return {
            "status": status,
            "is_live": model_live,
            "final_live": final_live,
            "live_score": live_score,
            "spoof_score": spoof_score,
            "raw": raw,
            "probabilities": probabilities,
            "draw_box": draw_box,
            "phone_detected": phone_detected,
            "phones": phones,
            "blink_detected": blink_detected,
            "ear": ear,
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

        if result["draw_box"] is not None:
            x1, y1, x2, y2 = result["draw_box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        for phone in result["phones"]:
            x1, y1, x2, y2 = phone["box"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv2.putText(
                frame,
                f"PHONE {phone['confidence']:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        live_score = result["live_score"]
        spoof_score = result["spoof_score"]

        if live_score is not None:
            score_text = (
                f"MiniFASNet Live: {live_score:.2f} | "
                f"Spoof: {spoof_score:.2f}"
            )
        else:
            score_text = "MiniFASNet: No face"

        blink_text = f"Blink: {result['blink_detected']} | EAR: {result['ear']:.2f}"
        phone_text = f"Phone detected: {result['phone_detected']}"

        cv2.putText(
            frame,
            score_text,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            blink_text,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            phone_text,
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        mode = self.preprocess_modes[self.preprocess_index]
        scale = self.crop_scales[self.crop_scale_index]

        cv2.putText(
            frame,
            f"Mode: {mode} | Scale: {scale} | Threshold: {self.threshold:.2f}",
            (20, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )


def run_minifasnet_liveness_test():
    detector = MiniFASNetTest()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera could not be opened.")
        return

    print("\nMiniFASNet + Blink + Phone test started.")
    print("Press q = quit")
    print("Press b = reset blink challenge")
    print("Press m = change preprocess mode")
    print("Press c = change crop scale")
    print("Press + = increase threshold")
    print("Press - = decrease threshold")
    print("\nFor real face: blink once.")
    print("For phone photo: it should stay spoof because no blink or phone detected.")

    last_print_time = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame.")
            break

        result = detector.predict(frame)
        detector.draw(frame, result)

        now = time.time()

        if now - last_print_time >= 1:
            print("--------------------------------")
            print("Status:", result["status"])
            print("Final Live:", result["final_live"])
            print("MiniFASNet Live:", result["is_live"])
            print("Live Score:", result["live_score"])
            print("Spoof Score:", result["spoof_score"])
            print("Raw:", result["raw"])
            print("Probabilities:", result["probabilities"])
            print("Blink Detected:", result["blink_detected"])
            print("EAR:", result["ear"])
            print("Phone Detected:", result["phone_detected"])
            last_print_time = now

        cv2.imshow("MiniFASNet Combined Liveness Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("b"):
            detector.blink_detected = False
            detector.eye_closed_frames = 0
            print("Blink challenge reset.")

        elif key == ord("m"):
            detector.preprocess_index += 1
            detector.preprocess_index %= len(detector.preprocess_modes)
            print("Changed preprocess mode:", detector.preprocess_modes[detector.preprocess_index])

        elif key == ord("c"):
            detector.crop_scale_index += 1
            detector.crop_scale_index %= len(detector.crop_scales)
            print("Changed crop scale:", detector.crop_scales[detector.crop_scale_index])

        elif key == ord("+") or key == ord("="):
            detector.threshold = min(0.95, detector.threshold + 0.05)
            print("Threshold:", detector.threshold)

        elif key == ord("-"):
            detector.threshold = max(0.05, detector.threshold - 0.05)
            print("Threshold:", detector.threshold)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_minifasnet_liveness_test()