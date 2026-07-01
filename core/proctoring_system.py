import os
import cv2
import time
from datetime import datetime

from eye_tracking.eye_tracker import EyeTracker
from object_detection.cheating_object_detector import CheatingObjectDetector
from core.async_workers import LatestFrameWorker, AsyncReportWorker


_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LINE = cv2.LINE_AA

_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (0, 0, 255)
_CYAN = (0, 255, 255)
_YELLOW = (0, 255, 255)

# Worker input resolutions.
# Higher object resolution improves phone/book detection.
_EYE_RES = 640
_OBJECT_RES = 640
_IDENTITY_RES = 640
_LIVENESS_RES = 480
_DISPLAY_W = 960


def _resize_max(frame, max_side: int):
    """
    Resize a frame so the longest side becomes max_side.
    Keeps aspect ratio.
    """
    if frame is None:
        return None

    if max_side <= 0:
        return frame

    h, w = frame.shape[:2]

    if max(h, w) <= max_side:
        return frame

    scale = max_side / max(h, w)

    return cv2.resize(
        frame,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_LINEAR
    )


def _put(frame, text: str, y: int, color, scale: float = 0.65, thick: int = 2):
    """
    Draw text on the dashboard.
    """
    cv2.putText(
        frame,
        text,
        (20, y),
        _FONT,
        scale,
        color,
        thick,
        _LINE
    )


def _short(text, max_len=70):
    """
    Shorten long dashboard messages.
    """
    text = str(text)

    if len(text) <= max_len:
        return text

    return text[:max_len] + "..."


def _color_status(
    text: str,
    good_kw=("Verified", "Live", "OK"),
    bad_kw=("error", "Unknown", "Spoof", "WARNING", "No face", "Multiple"),
):
    """
    Return dashboard color based on status keywords.
    """
    text = str(text).lower()

    if any(keyword.lower() in text for keyword in good_kw):
        return _GREEN

    if any(keyword.lower() in text for keyword in bad_kw):
        return _RED

    return _CYAN


class ProctoringSystem:
    """
    Main online exam proctoring controller.

    This class coordinates:
    - Eye tracking
    - Object detection
    - Face identity verification
    - CNN liveness detection
    - Audio/noise detection
    - Report logging and screenshot saving

    Heavy modules run in worker threads to keep the camera UI responsive.
    """

    # Worker scheduling.
    EYE_EVERY = 2
    OBJECT_EVERY = 5
    LIVENESS_EVERY = 3
    AUDIO_EVERY = 5

    def __init__(
        self,
        student_id,
        exam_id,
        enable_face=True,
        enable_eye=True,
        enable_object=True,
        enable_audio=True,
        enable_liveness=None,
    ):
        self.student_id = student_id
        self.exam_id = exam_id

        self.enable_face = enable_face
        self.enable_eye = enable_eye
        self.enable_object = enable_object
        self.enable_audio = enable_audio

        # Backward compatibility:
        # If enable_liveness is not passed, it follows enable_face.
        if enable_liveness is None:
            self.enable_liveness = enable_face
        else:
            self.enable_liveness = enable_liveness

        # Report folders.
        self.base_report_dir = os.path.join(
            "reports",
            "exams",
            self.exam_id,
            self.student_id
        )

        self.screenshot_dir = os.path.join(
            self.base_report_dir,
            "screenshots"
        )

        os.makedirs(self.screenshot_dir, exist_ok=True)

        self.log_path = os.path.join(
            self.base_report_dir,
            "log.txt"
        )

        # Frame counter and cooldowns.
        self.frame_count = 0
        self.last_logged_times = {}

        # Identity verification interval.
        self.face_check_interval_seconds = 8
        self.last_identity_check_time = 0.0

        # Eye cached values.
        self.last_gaze_direction = "Eye disabled"
        self.last_eye_ratio = 0.0
        self.last_eye_warning = False
        self.eye_warning_start = None
        self.eye_warning_seconds = 5

        # Object cached values.
        self.last_detected_objects = []
        self.last_object_warning = False
        self.last_person_count = 0
        self.last_applied_object_time = 0.0

        # Face identity cached values.
        self.last_face_status = "Face disabled"
        self.last_identity_distance = None
        self.last_identity_threshold = None
        self.last_applied_face_time = 0.0

        # Liveness cached values.
        self.last_liveness_status = "Liveness disabled"
        self.last_liveness_warning = False
        self.last_live_score = None
        self.last_spoof_score = None
        self.last_liveness_face_count = 0
        self.last_applied_liveness_time = 0.0

        # Final face security verdict.
        self.last_final_face_status = "Face security disabled"

        # Audio cached values.
        self.last_audio_status = "Audio disabled"
        self.last_audio_warning = False
        self.last_audio_volume = 0.0
        self.last_audio_db = -100.0
        self.last_audio_threshold = 0.0
        self.last_audio_calibrated = False

        # FPS values.
        self._fps = 0.0
        self._fps_frames = 0
        self._fps_t0 = time.perf_counter()

        # Module objects.
        self.eye_tracker = None
        self.object_detector = None

        self.face_authenticator = None
        self.face_available = False

        self.liveness_detector = None
        self.liveness_available = False

        self.audio_detector = None

        # Workers.
        self.report_worker = AsyncReportWorker()
        self.eye_worker = None
        self.object_worker = None
        self.face_worker = None
        self.liveness_worker = None

        # Initialize all systems.
        self._init_eye_tracking()
        self._init_object_detection()
        self._init_face_system()
        self._init_liveness_system()
        self._init_audio_detection()
        self._init_workers()

    # --------------------------------------------------
    # Initializers
    # --------------------------------------------------

    def _init_eye_tracking(self):
        """
        Initialize MediaPipe-based eye and gaze tracking.
        """
        if not self.enable_eye:
            self.last_gaze_direction = "Eye disabled"
            return

        try:
            eye_config = {
                "detection": {
                    "eyes": {
                        "iris_horizontal_threshold": 0.15,
                        "iris_vertical_threshold": 0.14,

                        "head_horizontal_threshold": 18,
                        "head_vertical_threshold": 15,

                        "closed_eye_threshold": 0.18,
                        "calibration_frames": 50,

                        "draw": False,
                        "show_debug_text": False,
                        "iris_point_radius": 2,

                        "mirror_output": True,
                        "invert_head_vertical": False,
                        "invert_head_horizontal": False,

                        "smoothing_alpha": 0.20,
                        "stable_frames_required": 7,
                    }
                }
            }

            self.eye_tracker = EyeTracker(eye_config)
            self.last_gaze_direction = "Eye system ready"

            self.log_alert(
                "EYE_SYSTEM_READY",
                "Eye tracking initialized."
            )

        except Exception as error:
            self.enable_eye = False
            self.eye_tracker = None
            self.last_gaze_direction = f"Eye system error: {error}"

            self.log_alert(
                "EYE_SYSTEM_ERROR",
                f"Eye tracking disabled: {error}"
            )

    def _init_object_detection(self):
        """
        Initialize YOLO object detection.
        """
        if not self.enable_object:
            return

        try:
            # Prefer YOLO small model for better accuracy.
            # Fall back to nano model if yolo11s.pt is not available.
            try:
                self.object_detector = CheatingObjectDetector(
                    model_path="yolo11s.pt",
                    confidence_threshold=0.40,
                    img_size=_OBJECT_RES,
                    skip_frames=1,
                    show_only_relevant=True,
                )

            except TypeError:
                self.object_detector = CheatingObjectDetector(
                    model_path="yolo11s.pt",
                    confidence_threshold=0.40,
                    img_size=_OBJECT_RES,
                    skip_frames=1,
                )

            except Exception:
                try:
                    self.object_detector = CheatingObjectDetector(
                        model_path="yolo11n.pt",
                        confidence_threshold=0.40,
                        img_size=_OBJECT_RES,
                        skip_frames=1,
                        show_only_relevant=True,
                    )

                except TypeError:
                    self.object_detector = CheatingObjectDetector(
                        model_path="yolo11n.pt",
                        confidence_threshold=0.40,
                        img_size=_OBJECT_RES,
                        skip_frames=1,
                    )

            self.log_alert(
                "OBJECT_SYSTEM_READY",
                "Object detection initialized."
            )

        except Exception as error:
            self.enable_object = False
            self.object_detector = None

            self.log_alert(
                "OBJECT_SYSTEM_ERROR",
                f"Object detection disabled: {error}"
            )

    def _init_face_system(self):
        """
        Initialize student identity verification only.
        Liveness is initialized separately.
        """
        if not self.enable_face:
            self.last_face_status = "Face disabled"
            self.last_final_face_status = "Face disabled"
            return

        try:
            from face_detection.deepface_auth import DeepFaceAuthenticator

            self.face_authenticator = DeepFaceAuthenticator(
                self.student_id,
                face_threshold=0.55
            )

            self.face_available = True
            self.last_face_status = "Face system ready"
            self.last_final_face_status = "Waiting"

            self.log_alert(
                "FACE_SYSTEM_READY",
                "Face identity verification initialized."
            )

        except Exception as error:
            self.face_available = False
            self.face_authenticator = None
            self.last_face_status = f"Face system error: {error}"
            self.last_final_face_status = "Face security error"

            self.log_alert(
                "FACE_SYSTEM_ERROR",
                f"Face system disabled: {error}"
            )

    def _init_liveness_system(self):
        """
        Initialize CNN liveness detection separately from DeepFace identity verification.
        """
        if not self.enable_liveness:
            self.last_liveness_status = "Liveness disabled"
            return

        try:
            from face_detection.cnn_liveness_detector import CNNLivenessDetector

            self.liveness_detector = CNNLivenessDetector(
                threshold=0.70,
                sigmoid_live_high=True,
                enable_blink_challenge=True,

                # A valid blink remains accepted for 12 seconds.
                blink_valid_seconds=12.0,

                # The system asks for a new blink after a longer interval.
                blink_challenge_interval_seconds=18.0,
            )

            self.liveness_available = True
            self.last_liveness_status = "Liveness system ready"

            self.log_alert(
                "LIVENESS_SYSTEM_READY",
                "CNN liveness detection initialized."
            )

        except Exception as error:
            self.liveness_available = False
            self.liveness_detector = None
            self.last_liveness_status = f"Liveness system error: {error}"

            self.log_alert(
                "LIVENESS_SYSTEM_ERROR",
                f"Liveness system disabled: {error}"
            )

    def _init_audio_detection(self):
        """
        Initialize audio/noise detection.

        device=None means the OS default input device is used.
        This avoids invalid device errors when Windows changes device indexes.
        """
        if not self.enable_audio:
            self.last_audio_status = "Audio disabled"
            return

        try:
            from audio_detection.noise_detector import NoiseDetector

            self.audio_detector = NoiseDetector(
                sample_rate=48000,
                frame_duration_ms=30,
                vad_mode=2,
                calibration_seconds=3,
                warning_seconds=3,
                speech_ratio_threshold=0.60,
                min_volume_threshold=0.015,
                device=None
            )

            self.last_audio_status = "Audio system ready"
            self.last_audio_threshold = 0.015

            self.log_alert(
                "AUDIO_SYSTEM_READY",
                "Audio detection initialized."
            )

        except Exception as error:
            self.audio_detector = None
            self.last_audio_status = f"Audio init error: {error}"

            self.log_alert(
                "AUDIO_SYSTEM_ERROR",
                f"Audio init error: {error}"
            )

    def _init_workers(self):
        """
        Initialize background workers for heavy processing.
        """
        if self.enable_eye and self.eye_tracker is not None:
            self.eye_worker = LatestFrameWorker(
                name="eye_worker",
                process_func=self._process_eye_frame
            )

        if self.enable_object and self.object_detector is not None:
            self.object_worker = LatestFrameWorker(
                name="object_worker",
                process_func=self._process_object_frame
            )

        if self.face_available and self.face_authenticator is not None:
            self.face_worker = LatestFrameWorker(
                name="face_worker",
                process_func=self._process_face_frame
            )

        if self.liveness_available and self.liveness_detector is not None:
            self.liveness_worker = LatestFrameWorker(
                name="liveness_worker",
                process_func=self._process_liveness_frame
            )

    # --------------------------------------------------
    # Worker process functions
    # --------------------------------------------------

    def _process_eye_frame(self, small_frame):
        """
        Process eye tracking in a worker thread.
        """
        gaze, ratio = self.eye_tracker.track_eyes(small_frame)

        return {
            "gaze": gaze,
            "ratio": ratio,
            "time": time.time()
        }

    def _process_object_frame(self, frame):
        """
        Process YOLO object detection in a worker thread.
        """
        cheating_detected, detected_objects = self.object_detector.detect_objects(frame)

        person_count = sum(
            1 for obj in detected_objects
            if obj.get("name") == "person"
        )

        return {
            "cheating_detected": cheating_detected,
            "detected_objects": detected_objects,
            "person_count": person_count,
            "time": time.time()
        }

    def _process_face_frame(self, small_frame):
        """
        Process student identity verification in a worker thread.
        """
        result = self.face_authenticator.verify_frame(small_frame)

        return {
            "mode": "face_identity",
            "result": result,
            "time": time.time()
        }

    def _process_liveness_frame(self, small_frame):
        """
        Process CNN liveness detection in a worker thread.
        """
        result = self.liveness_detector.predict(small_frame)

        return {
            "mode": "liveness",
            "result": result,
            "time": time.time()
        }

    # --------------------------------------------------
    # Apply worker results
    # --------------------------------------------------

    def _apply_eye_result(self, frame):
        """
        Apply latest eye tracking result.
        """
        if self.eye_worker is None:
            return

        data = self.eye_worker.get_latest_result()

        if not data:
            return

        gaze = data.get("gaze", "center")
        ratio = data.get("ratio", 0.0)

        suspicious = {
            "left",
            "right",
            "up",
            "down",
            "no_face",
            "eyes_closed"
        }

        if gaze in suspicious:
            if self.eye_warning_start is None:
                self.eye_warning_start = time.time()

            elapsed = time.time() - self.eye_warning_start

            if elapsed >= self.eye_warning_seconds:
                self.last_eye_warning = True

                if self.should_log("EYE_WARNING", cooldown_seconds=8):
                    screenshot_path = self.save_screenshot(
                        frame,
                        "eye_warning"
                    )

                    self.log_alert(
                        "EYE_WARNING",
                        f"Gaze: {gaze} | Screenshot: {screenshot_path}"
                    )

            else:
                self.last_eye_warning = False

        else:
            self.eye_warning_start = None
            self.last_eye_warning = False

        self.last_gaze_direction = gaze
        self.last_eye_ratio = ratio

    def _apply_object_result(self, frame):
        """
        Apply latest object detection result.
        """
        if self.object_worker is None:
            return

        data = self.object_worker.get_latest_result()

        if not data:
            return

        result_time = data.get("time", 0.0)

        if result_time <= self.last_applied_object_time:
            return

        self.last_applied_object_time = result_time

        self.last_detected_objects = data["detected_objects"]
        self.last_object_warning = data["cheating_detected"]
        self.last_person_count = data["person_count"]

        if data["cheating_detected"]:
            if self.should_log("OBJECT_WARNING", cooldown_seconds=8):
                object_names = [
                    obj["name"]
                    for obj in data["detected_objects"]
                    if obj.get("is_cheating_object")
                ]

                object_text = ", ".join(object_names)

                if not object_text:
                    object_text = "Suspicious object"

                screenshot_path = self.save_screenshot(
                    frame,
                    "object_warning"
                )

                self.log_alert(
                    "OBJECT_WARNING",
                    f"Detected: {object_text} | Screenshot: {screenshot_path}"
                )

    def _apply_face_result(self, frame):
        """
        Apply latest identity verification result.
        """
        if self.face_worker is None:
            return

        data = self.face_worker.get_latest_result()

        if not data:
            return

        result_time = data.get("time", 0.0)

        if result_time <= self.last_applied_face_time:
            return

        self.last_applied_face_time = result_time

        result = data["result"]

        self.last_face_status = result.get("status", "Face error")
        self.last_identity_distance = result.get("distance")
        self.last_identity_threshold = result.get("threshold")

        if not result.get("verified", False):
            if self.should_log("FACE_WARNING", cooldown_seconds=10):
                screenshot_path = self.save_screenshot(
                    frame,
                    "face_warning"
                )

                self.log_alert(
                    "FACE_WARNING",
                    f"{self.last_face_status} | "
                    f"Dist: {self._fmt(self.last_identity_distance)} | "
                    f"Screenshot: {screenshot_path}"
                )

        self._update_final_face_status()

    def _apply_liveness_result(self, frame):
        """
        Apply latest CNN liveness result.
        """
        if self.liveness_worker is None:
            return

        data = self.liveness_worker.get_latest_result()

        if not data:
            return

        result_time = data.get("time", 0.0)

        if result_time <= self.last_applied_liveness_time:
            return

        self.last_applied_liveness_time = result_time

        result = data["result"]

        self.last_liveness_status = result.get("status", "Liveness error")
        self.last_liveness_warning = bool(result.get("warning", False))
        self.last_live_score = result.get("live_score")
        self.last_spoof_score = result.get("spoof_score")
        self.last_liveness_face_count = result.get("face_count", 0)

        if result.get("warning"):
            if self.should_log("LIVENESS_WARNING", cooldown_seconds=10):
                screenshot_path = self.save_screenshot(
                    frame,
                    "liveness_warning"
                )

                self.log_alert(
                    "LIVENESS_WARNING",
                    f"{self.last_liveness_status} | "
                    f"Live: {self._fmt(self.last_live_score)} | "
                    f"Spoof: {self._fmt(self.last_spoof_score)} | "
                    f"Screenshot: {screenshot_path}"
                )

        self._update_final_face_status()

    def _apply_audio_result(self):
        """
        Apply latest audio/noise detection result.
        """
        if not self.enable_audio or self.audio_detector is None:
            return

        try:
            result = self.audio_detector.get_status()

            self.last_audio_status = result["status"]
            self.last_audio_warning = result["warning"]
            self.last_audio_volume = result["volume"]
            self.last_audio_db = result["db"]
            self.last_audio_threshold = result.get("adaptive_threshold", 0.0)
            self.last_audio_calibrated = result.get("calibrated", False)

            if result["warning"]:
                if self.should_log("AUDIO_WARNING", cooldown_seconds=8):
                    self.log_alert(
                        "AUDIO_WARNING",
                        f"{result['status']} | "
                        f"Vol: {result['volume']:.4f} | "
                        f"dB: {result['db']:.2f} | "
                        f"Threshold: {self.last_audio_threshold:.4f}"
                    )

        except Exception as error:
            self.last_audio_status = f"Audio check error: {error}"
            self.last_audio_warning = True

            if self.should_log("AUDIO_ERROR", cooldown_seconds=10):
                self.log_alert("AUDIO_ERROR", str(error))

    # --------------------------------------------------
    # Final verdict logic
    # --------------------------------------------------

    def _update_final_face_status(self):
        """
        Combine identity verification and liveness into one final verdict.
        """
        face_status = str(self.last_face_status)
        live_status = str(self.last_liveness_status)

        face_ok = face_status == "Student Verified"
        liveness_ok = live_status in {"Live", "Live face detected"}

        if not self.enable_face and not self.enable_liveness:
            self.last_final_face_status = "Face security disabled"

        elif "No face" in live_status:
            self.last_final_face_status = "No face detected"

        elif "Multiple faces" in live_status:
            self.last_final_face_status = "Multiple faces detected"

        elif "Please blink" in live_status:
            self.last_final_face_status = "Please blink once"

        elif "Spoof" in live_status:
            self.last_final_face_status = "Spoof"

        elif "Unknown" in face_status:
            self.last_final_face_status = "Unknown Person"

        elif "Error" in face_status or "error" in face_status:
            self.last_final_face_status = "Face verification error"

        elif face_ok and liveness_ok:
            self.last_final_face_status = "Live"

        elif face_status == "Face not checked":
            self.last_final_face_status = "Waiting for face verification"

        else:
            self.last_final_face_status = "Waiting"

    # --------------------------------------------------
    # Drawing
    # --------------------------------------------------

    def draw_object_boxes(self, frame):
        """
        Draw YOLO object bounding boxes.
        """
        if not self.enable_object:
            return

        for obj in self.last_detected_objects:
            box = obj.get("box")

            if box is None:
                continue

            x1, y1, x2, y2 = box

            name = obj["name"]
            confidence = obj["confidence"]
            is_cheating = obj.get("is_cheating_object", False)

            color = _RED if is_cheating else _GREEN
            prefix = "CHEAT: " if is_cheating else ""
            label = f"{prefix}{name} {confidence:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            text_size, baseline = cv2.getTextSize(
                label,
                _FONT,
                0.55,
                1
            )

            text_w, text_h = text_size
            top = max(y1 - text_h - baseline - 4, 0)

            cv2.rectangle(
                frame,
                (x1, top),
                (x1 + text_w + 4, y1),
                color,
                cv2.FILLED
            )

            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - baseline - 2),
                _FONT,
                0.55,
                _WHITE,
                1,
                _LINE
            )

    def draw_dashboard(self, frame):
        """
        Draw live dashboard information on the camera frame.
        """
        y = 28

        _put(
            frame,
            f"Student: {self.student_id} | Exam: {self.exam_id} | FPS: {self._fps:.1f}",
            y,
            _WHITE,
            0.6
        )

        y += 30

        # Eye status.
        if not self.enable_eye:
            _put(frame, "Eye: disabled", y, _CYAN)

        elif self.last_eye_warning:
            _put(
                frame,
                f"EYE WARNING: {self.last_gaze_direction}",
                y,
                _RED
            )

        else:
            _put(
                frame,
                f"Gaze: {self.last_gaze_direction}",
                y,
                _GREEN
            )

        y += 30

        # Object status.
        if not self.enable_object:
            _put(frame, "Objects: disabled", y, _CYAN)

        elif self.last_object_warning:
            _put(
                frame,
                f"OBJECT WARNING | Persons: {self.last_person_count}",
                y,
                _RED
            )

        else:
            _put(
                frame,
                f"Objects: OK | Persons: {self.last_person_count}",
                y,
                _GREEN
            )

        y += 30

        # Face identity status.
        if not self.enable_face:
            _put(frame, "Face: disabled", y, _CYAN)

        else:
            _put(
                frame,
                f"Face: {_short(self.last_face_status, 55)}",
                y,
                _color_status(self.last_face_status)
            )

        y += 30

        # Liveness status.
        if not self.enable_liveness:
            _put(frame, "Liveness: disabled", y, _CYAN)

        else:
            _put(
                frame,
                f"Liveness: {_short(self.last_liveness_status, 55)}",
                y,
                _color_status(
                    self.last_liveness_status,
                    good_kw=("Live", "Live face", "OK"),
                    bad_kw=("Spoof", "No face", "Multiple", "error")
                )
            )

        y += 30

        # Final verdict.
        _put(
            frame,
            f"Verdict: {_short(self.last_final_face_status, 55)}",
            y,
            _color_status(
                self.last_final_face_status,
                good_kw=("Live",),
                bad_kw=("Spoof", "Unknown", "No face", "Multiple", "error")
            )
        )

        y += 30

        # Scores.
        _put(
            frame,
            f"Live: {self._fmt(self.last_live_score)} | "
            f"Spoof: {self._fmt(self.last_spoof_score)} | "
            f"Dist: {self._fmt(self.last_identity_distance)}",
            y,
            _WHITE,
            0.55
        )

        y += 28

        # Audio status.
        if not self.enable_audio:
            _put(frame, "Audio: disabled", y, _CYAN, 0.6)

        elif self.last_audio_warning:
            _put(
                frame,
                f"AUDIO WARNING | Vol: {self.last_audio_volume:.4f} | "
                f"dB: {self.last_audio_db:.2f}",
                y,
                _RED,
                0.6
            )

        else:
            _put(
                frame,
                f"Audio: {_short(self.last_audio_status, 50)} | "
                f"Vol: {self.last_audio_volume:.4f} | "
                f"dB: {self.last_audio_db:.2f}",
                y,
                _GREEN,
                0.6
            )

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def _fmt(self, value):
        """
        Format dashboard numbers.
        """
        if value is None:
            return "—"

        if isinstance(value, float):
            return f"{value:.3f}"

        return str(value)

    def _update_fps(self):
        """
        Update FPS every 30 frames.
        """
        self._fps_frames += 1

        if self._fps_frames >= 30:
            elapsed = time.perf_counter() - self._fps_t0

            if elapsed > 0:
                self._fps = self._fps_frames / elapsed
            else:
                self._fps = 0.0

            self._fps_frames = 0
            self._fps_t0 = time.perf_counter()

    def should_log(self, alert_type, cooldown_seconds=5):
        """
        Prevent repeated logs for the same alert type.
        """
        now = time.time()
        last = self.last_logged_times.get(alert_type, 0.0)

        if now - last >= cooldown_seconds:
            self.last_logged_times[alert_type] = now
            return True

        return False

    def log_alert(self, alert_type, message):
        """
        Log an alert using the async report worker.
        """
        if self.report_worker is not None:
            self.report_worker.log(
                self.log_path,
                alert_type,
                message
            )
            return

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.log_path, "a", encoding="utf-8") as file:
            file.write(
                f"[{current_time}] {alert_type}: {message}\n"
            )

        print(f"[{current_time}] {alert_type}: {message}")

    def save_screenshot(self, frame, alert_type):
        """
        Save violation screenshot asynchronously.
        """
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{alert_type}_{current_time}.jpg"
        path = os.path.join(self.screenshot_dir, filename)

        if self.report_worker is not None:
            self.report_worker.screenshot(path, frame)

        else:
            cv2.imwrite(
                path,
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 85]
            )

        return path

    def _check_worker_errors(self):
        """
        Check worker thread errors and write them to the report log.
        """
        workers = [
            self.eye_worker,
            self.object_worker,
            self.face_worker,
            self.liveness_worker,
        ]

        for worker in workers:
            if worker is None:
                continue

            error = worker.get_latest_error()

            if not error:
                continue

            alert_type = f"{error.get('worker', 'worker').upper()}_ERROR"

            if self.should_log(alert_type, cooldown_seconds=10):
                self.log_alert(
                    alert_type,
                    error.get("error", "Unknown worker error")
                )

    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------

    def run(self):
        """
        Start the real-time proctoring loop.
        """
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Camera could not be opened.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Start workers.
        self.report_worker.start()

        for worker in (
            self.eye_worker,
            self.object_worker,
            self.face_worker,
            self.liveness_worker,
        ):
            if worker is not None:
                worker.start()

        if self.enable_audio and self.audio_detector is not None:
            self.audio_detector.start()

        print("AI Exam Proctoring System started.")
        print("Press 'q' to quit.")
        print("Press 'r' to reset eye calibration.")
        print(f"Report path: {self.base_report_dir}")

        self.log_alert(
            "SYSTEM_START",
            "Proctoring started."
        )

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("Error: Could not read frame.")
                    break

                self.frame_count += 1
                frame_count = self.frame_count

                # -----------------------------------
                # 1. Submit frames to worker threads
                # -----------------------------------

                # Eye tracking runs on a resized frame.
                if (
                    self.eye_worker is not None
                    and frame_count % self.EYE_EVERY == 0
                ):
                    self.eye_worker.submit(
                        _resize_max(frame, _EYE_RES)
                    )

                # Object detection receives original frame.
                # The detector internally resizes to YOLO input size.
                if (
                    self.object_worker is not None
                    and frame_count % self.OBJECT_EVERY == 0
                ):
                    self.object_worker.submit(frame)

                # Identity verification is expensive, so run it every few seconds.
                if (
                    self.face_worker is not None
                    and time.time() - self.last_identity_check_time >= self.face_check_interval_seconds
                ):
                    self.last_identity_check_time = time.time()
                    self.face_worker.submit(
                        _resize_max(frame, _IDENTITY_RES)
                    )

                # Liveness runs more frequently than DeepFace identity verification.
                if (
                    self.liveness_worker is not None
                    and frame_count % self.LIVENESS_EVERY == 0
                ):
                    self.liveness_worker.submit(
                        _resize_max(frame, _LIVENESS_RES)
                    )

                # -----------------------------------
                # 2. Apply cached worker results
                # -----------------------------------

                self._apply_eye_result(frame)
                self._apply_object_result(frame)

                if self.liveness_available:
                    self._apply_liveness_result(frame)

                if self.face_available:
                    self._apply_face_result(frame)

                if frame_count % self.AUDIO_EVERY == 0:
                    self._apply_audio_result()

                if frame_count % 30 == 0:
                    self._check_worker_errors()

                # -----------------------------------
                # 3. Draw dashboard and boxes
                # -----------------------------------

                self.draw_object_boxes(frame)
                self.draw_dashboard(frame)
                self._update_fps()

                # -----------------------------------
                # 4. Display frame
                # -----------------------------------

                if _DISPLAY_W:
                    display_frame = _resize_max(frame, _DISPLAY_W)
                else:
                    display_frame = frame

                cv2.imshow(
                    "AI Exam Proctoring System",
                    display_frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key == ord("r") and self.eye_tracker is not None:
                    self.eye_tracker.reset_calibration()

                    self.log_alert(
                        "EYE_CALIBRATION_RESET",
                        "Eye calibration reset by user."
                    )

        finally:
            self.log_alert(
                "SYSTEM_END",
                "Proctoring stopped."
            )

            if self.enable_audio and self.audio_detector is not None:
                self.audio_detector.stop()

            for worker in (
                self.eye_worker,
                self.object_worker,
                self.face_worker,
                self.liveness_worker,
            ):
                if worker is not None:
                    worker.stop()

            if self.report_worker is not None:
                self.report_worker.stop()

            cap.release()
            cv2.destroyAllWindows()