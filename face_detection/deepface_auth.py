import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import cv2
from datetime import datetime
from collections import deque
from deepface import DeepFace

from face_detection.student_manager import (
    find_student_by_id,
    create_exam_report_folder
)

from face_detection.cnn_liveness_detector import CNNLivenessDetector


MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "opencv"


class DeepFaceAuthenticator:
    def __init__(self, student_id, face_threshold=None):
        self.student_id = student_id
        self.student = find_student_by_id(student_id)

        if self.student is None:
            raise ValueError(f"Student not found with ID: {student_id}")

        self.profile_image_path = self.student["profile_image_path"]

        if not os.path.exists(self.profile_image_path):
            raise FileNotFoundError(
                f"Profile image not found: {self.profile_image_path}"
            )

        self.face_threshold = face_threshold

        self.identity_verified = False
        self.identity_status = "Face not checked"
        self.identity_distance = None
        self.identity_threshold = None

        self.liveness_history = deque(maxlen=7)

        try:
            self.liveness_detector = CNNLivenessDetector(
                threshold=0.70,
                sigmoid_live_high=True,
                enable_blink_challenge=True,
                blink_valid_seconds=4.0
            )
        except TypeError:
            self.liveness_detector = CNNLivenessDetector(
                threshold=0.70,
                sigmoid_live_high=True
            )

        print(f"Loaded student: {self.student['full_name']}")
        print(f"Using model: {MODEL_NAME}")
        print(f"Using detector: {DETECTOR_BACKEND}")
        print("CNN H5 Liveness enabled.")

    def verify_frame(self, frame):
        try:
            result = DeepFace.verify(
                img1_path=self.profile_image_path,
                img2_path=frame,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                align=True,
                silent=True
            )

            distance = result["distance"]
            deepface_threshold = result["threshold"]

            if self.face_threshold is not None:
                threshold = self.face_threshold
                verified = distance <= threshold
            else:
                threshold = deepface_threshold
                verified = result["verified"]

            if verified:
                status = "Student Verified"
            else:
                status = "Unknown Person"

            self.identity_verified = verified
            self.identity_status = status
            self.identity_distance = distance
            self.identity_threshold = threshold

            return {
                "status": status,
                "verified": verified,
                "distance": distance,
                "threshold": threshold
            }

        except Exception as error:
            self.identity_verified = False
            self.identity_status = f"Face Verification Error: {str(error)}"
            self.identity_distance = None
            self.identity_threshold = self.face_threshold

            return {
                "status": self.identity_status,
                "verified": False,
                "distance": None,
                "threshold": self.face_threshold
            }

    def check_liveness(self, frame):
        return self.liveness_detector.predict(frame)

    def get_stable_liveness(self, liveness_result):
        is_live_now = bool(liveness_result.get("is_live", False))

        self.liveness_history.append(1 if is_live_now else 0)

        live_votes = sum(self.liveness_history)
        total_votes = len(self.liveness_history)

        stable_live = live_votes >= max(1, total_votes // 2 + 1)

        if stable_live:
            stable_status = "Live"
        else:
            stable_status = "Spoof"

        return stable_live, stable_status

    def verify_frame_with_liveness(self, frame, run_identity_check=True):
        liveness_result = self.check_liveness(frame)

        liveness_status_raw = liveness_result.get("status", "")
        no_face = liveness_status_raw == "No face detected"
        requires_action = bool(liveness_result.get("requires_action", False))

        if requires_action:
            stable_live = False
            stable_liveness_status = liveness_status_raw or "Waiting for blink challenge"
        else:
            stable_live, stable_liveness_status = self.get_stable_liveness(
                liveness_result
            )

        if run_identity_check:
            identity_result = self.verify_frame(frame)
        else:
            identity_result = {
                "status": self.identity_status,
                "verified": self.identity_verified,
                "distance": self.identity_distance,
                "threshold": self.identity_threshold
            }

        if no_face:
            final_verified = False
            final_status = "No face detected"

        elif identity_result["status"] == "Face not checked":
            final_verified = False
            final_status = "Waiting for face verification"

        elif not identity_result["verified"]:
            final_verified = False
            final_status = "Unknown Person"

        elif requires_action:
            final_verified = False
            final_status = "Waiting for blink challenge"

        elif not stable_live:
            final_verified = False
            final_status = "Spoof"

        else:
            final_verified = True
            final_status = "Live"

        warning = False

        if no_face:
            warning = True

        if identity_result["status"] != "Face not checked" and not identity_result["verified"]:
            warning = True

        if not stable_live and not requires_action:
            warning = True

        return {
            "final_verified": final_verified,
            "final_status": final_status,
            "warning": warning,

            "identity_verified": identity_result["verified"],
            "identity_status": identity_result["status"],
            "identity_distance": identity_result["distance"],
            "identity_threshold": identity_result["threshold"],

            "is_live": stable_live,
            "model_live": liveness_result.get("model_live", stable_live),

            "liveness_status": stable_liveness_status,

            "live_score": liveness_result.get("live_score"),
            "spoof_score": liveness_result.get("spoof_score"),
            "face_box": liveness_result.get("face_box"),
            "raw_prediction": liveness_result.get("raw_prediction"),

            "blink_valid": liveness_result.get("blink_valid", False),
            "blink_now": liveness_result.get("blink_now", False),
            "ear": liveness_result.get("ear", 0.0),
            "face_area_ratio": liveness_result.get("face_area_ratio", 0.0),
            "face_changed": liveness_result.get("face_changed", False),
            "requires_action": requires_action
        }

    def draw_liveness_result(self, frame, result):
        return self.liveness_detector.draw_result(frame, result)


def save_exam_log(exam_id, student_id, message):
    report_folder = create_exam_report_folder(exam_id, student_id)
    log_path = os.path.join(report_folder, "log.txt")

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, "a", encoding="utf-8") as file:
        file.write(f"[{current_time}] {message}\n")


def save_violation_screenshot(exam_id, student_id, frame, violation_type):
    report_folder = create_exam_report_folder(exam_id, student_id)
    screenshots_folder = os.path.join(report_folder, "screenshots")

    os.makedirs(screenshots_folder, exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = f"{violation_type}_{current_time}.jpg"
    image_path = os.path.join(screenshots_folder, image_name)

    cv2.imwrite(
        image_path,
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 85]
    )

    return image_path