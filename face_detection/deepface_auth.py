import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import cv2
from deepface import DeepFace

from face_detection.student_manager import find_student_by_id


MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "opencv"


class DeepFaceAuthenticator:
    def __init__(
        self,
        student_id,
        face_threshold=0.55,
        detector_backend=DETECTOR_BACKEND,
        model_name=MODEL_NAME,
        enforce_detection=True,
        align=True,
        max_frame_side=640
    ):
        self.student_id = student_id
        self.face_threshold = face_threshold
        self.detector_backend = detector_backend
        self.model_name = model_name
        self.enforce_detection = enforce_detection
        self.align = align
        self.max_frame_side = max_frame_side

        self.student = find_student_by_id(student_id)

        if self.student is None:
            raise ValueError(f"Student not found with ID: {student_id}")

        self.profile_image_path = self.student.get("profile_image_path")

        if not self.profile_image_path:
            raise ValueError("Student profile_image_path is missing.")

        if not os.path.exists(self.profile_image_path):
            raise FileNotFoundError(
                f"Profile image not found: {self.profile_image_path}"
            )

        self.identity_verified = False
        self.identity_status = "Face not checked"
        self.identity_distance = None
        self.identity_threshold = self.face_threshold

        print(f"Loaded student: {self.student.get('full_name', self.student_id)}")
        print(f"Using face model: {self.model_name}")
        print(f"Using detector: {self.detector_backend}")
        print(f"Face threshold: {self.face_threshold}")

    def _resize_max(self, frame):
        if frame is None:
            return None

        h, w = frame.shape[:2]

        if max(h, w) <= self.max_frame_side:
            return frame

        scale = self.max_frame_side / max(h, w)

        return cv2.resize(
            frame,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_LINEAR
        )

    def _make_error_result(self, message):
        self.identity_verified = False
        self.identity_status = message
        self.identity_distance = None
        self.identity_threshold = self.face_threshold

        return {
            "status": self.identity_status,
            "verified": False,
            "distance": None,
            "threshold": self.identity_threshold
        }

    def verify_frame(self, frame):
        if frame is None:
            return self._make_error_result("Face Verification Error: empty frame")

        try:
            frame_for_verify = self._resize_max(frame)

            try:
                result = DeepFace.verify(
                    img1_path=self.profile_image_path,
                    img2_path=frame_for_verify,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    enforce_detection=self.enforce_detection,
                    align=self.align,
                    silent=True
                )

            except TypeError:
                result = DeepFace.verify(
                    img1_path=self.profile_image_path,
                    img2_path=frame_for_verify,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    enforce_detection=self.enforce_detection,
                    align=self.align
                )

            distance = float(result.get("distance"))
            deepface_threshold = float(result.get("threshold"))

            if self.face_threshold is not None:
                threshold = float(self.face_threshold)
                verified = distance <= threshold
            else:
                threshold = deepface_threshold
                verified = bool(result.get("verified", False))

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
            return self._make_error_result(
                f"Face Verification Error: {str(error)}"
            )

    def get_cached_result(self):
        return {
            "status": self.identity_status,
            "verified": self.identity_verified,
            "distance": self.identity_distance,
            "threshold": self.identity_threshold
        }

    def reset(self):
        self.identity_verified = False
        self.identity_status = "Face not checked"
        self.identity_distance = None
        self.identity_threshold = self.face_threshold