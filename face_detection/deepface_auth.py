# # import os
# # import cv2
# # from datetime import datetime
# # from deepface import DeepFace

# # from face_detection.student_manager import (
# #     find_student_by_id,
# #     create_exam_report_folder
# # )


# # MODEL_NAME = "ArcFace"
# # # DETECTOR_BACKEND = "retinaface"
# # DETECTOR_BACKEND = "opencv"


# # class DeepFaceAuthenticator:
# #     def __init__(self, student_id):
# #         self.student_id = student_id
# #         self.student = find_student_by_id(student_id)

# #         if self.student is None:
# #             raise ValueError(f"Student not found with ID: {student_id}")

# #         self.profile_image_path = self.student["profile_image_path"]

# #         if not os.path.exists(self.profile_image_path):
# #             raise FileNotFoundError(
# #                 f"Profile image not found: {self.profile_image_path}"
# #             )

# #         print(f"Loaded student: {self.student['full_name']}")
# #         print(f"Using model: {MODEL_NAME}")
# #         print(f"Using detector: {DETECTOR_BACKEND}")

# #     def verify_frame(self, frame):
# #         try:
# #             result = DeepFace.verify(
# #                 img1_path=self.profile_image_path,
# #                 img2_path=frame,
# #                 model_name=MODEL_NAME,
# #                 detector_backend=DETECTOR_BACKEND,
# #                 enforce_detection=False
# #             )

# #             verified = result["verified"]
# #             distance = result["distance"]
# #             threshold = result["threshold"]

# #             if verified:
# #                 return {
# #                     "status": "Student Verified",
# #                     "verified": True,
# #                     "distance": distance,
# #                     "threshold": threshold
# #                 }

# #             return {
# #                 "status": "Unknown Person",
# #                 "verified": False,
# #                 "distance": distance,
# #                 "threshold": threshold
# #             }

# #         except Exception as error:
# #             return {
# #                 "status": f"Face Verification Error: {str(error)}",
# #                 "verified": False,
# #                 "distance": None,
# #                 "threshold": None
# #             }


# # def save_exam_log(exam_id, student_id, message):
# #     report_folder = create_exam_report_folder(exam_id, student_id)
# #     log_path = os.path.join(report_folder, "log.txt")

# #     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# #     with open(log_path, "a", encoding="utf-8") as file:
# #         file.write(f"[{current_time}] {message}\n")


# # def save_violation_screenshot(exam_id, student_id, frame, violation_type):
# #     report_folder = create_exam_report_folder(exam_id, student_id)
# #     screenshots_folder = os.path.join(report_folder, "screenshots")

# #     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# #     image_name = f"{violation_type}_{current_time}.jpg"
# #     image_path = os.path.join(screenshots_folder, image_name)

# #     cv2.imwrite(image_path, frame)

# #     return image_path


# # import os

# # os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# # os.environ["TF_USE_LEGACY_KERAS"] = "1"

# # import cv2
# # from datetime import datetime
# # from deepface import DeepFace

# # from face_detection.student_manager import (
# #     find_student_by_id,
# #     create_exam_report_folder
# # )

# # from face_detection.cnn_liveness_detector import CNNLivenessDetector


# # MODEL_NAME = "ArcFace"
# # # DETECTOR_BACKEND = "mediapipe"
# # # DETECTOR_BACKEND = "retinaface"
# # DETECTOR_BACKEND = "opencv"


# # class DeepFaceAuthenticator:
# #     def __init__(self, student_id, face_threshold=None):
# #         self.student_id = student_id
# #         self.student = find_student_by_id(student_id)

# #         if self.student is None:
# #             raise ValueError(f"Student not found with ID: {student_id}")

# #         self.profile_image_path = self.student["profile_image_path"]

# #         if not os.path.exists(self.profile_image_path):
# #             raise FileNotFoundError(
# #                 f"Profile image not found: {self.profile_image_path}"
# #             )

# #         self.face_threshold = face_threshold

# #         self.identity_verified = False
# #         self.identity_status = "Face not checked"
# #         self.identity_distance = None
# #         self.identity_threshold = None

# #         self.liveness_detector = CNNLivenessDetector(
# #             threshold=0.70,
# #             sigmoid_live_high=False
# #         )

# #         print(f"Loaded student: {self.student['full_name']}")
# #         print(f"Using model: {MODEL_NAME}")
# #         print(f"Using detector: {DETECTOR_BACKEND}")
# #         print("CNN Liveness enabled.")

# #     def verify_frame(self, frame):
# #         try:
# #             result = DeepFace.verify(
# #                 img1_path=self.profile_image_path,
# #                 img2_path=frame,
# #                 model_name=MODEL_NAME,
# #                 detector_backend=DETECTOR_BACKEND,
# #                 enforce_detection=False,
# #                 align=True
# #             )

# #             distance = result["distance"]
# #             deepface_threshold = result["threshold"]

# #             if self.face_threshold is not None:
# #                 threshold = self.face_threshold
# #                 verified = distance <= threshold
# #             else:
# #                 threshold = deepface_threshold
# #                 verified = result["verified"]

# #             if verified:
# #                 status = "Student Verified"
# #             else:
# #                 status = "Unknown Person"

# #             self.identity_verified = verified
# #             self.identity_status = status
# #             self.identity_distance = distance
# #             self.identity_threshold = threshold

# #             return {
# #                 "status": status,
# #                 "verified": verified,
# #                 "distance": distance,
# #                 "threshold": threshold
# #             }

# #         except Exception as error:
# #             self.identity_verified = False
# #             self.identity_status = f"Face Verification Error: {str(error)}"
# #             self.identity_distance = None
# #             self.identity_threshold = self.face_threshold

# #             return {
# #                 "status": self.identity_status,
# #                 "verified": False,
# #                 "distance": None,
# #                 "threshold": self.face_threshold
# #             }

# #     def check_liveness(self, frame):
# #         return self.liveness_detector.predict(frame)

# #     def verify_frame_with_liveness(self, frame, run_identity_check=True):
# #         liveness_result = self.check_liveness(frame)

# #         if run_identity_check:
# #             identity_result = self.verify_frame(frame)
# #         else:
# #             identity_result = {
# #                 "status": self.identity_status,
# #                 "verified": self.identity_verified,
# #                 "distance": self.identity_distance,
# #                 "threshold": self.identity_threshold
# #             }

# #         if identity_result["status"] == "Face not checked":
# #             final_verified = False
# #             final_status = "Waiting for face verification"

# #         elif identity_result["verified"] and liveness_result["is_live"]:
# #             final_verified = True
# #             final_status = "Verified Live Student"

# #         elif identity_result["verified"] and not liveness_result["is_live"]:
# #             final_verified = False
# #             final_status = "Student Verified but Spoof Detected"

# #         elif not identity_result["verified"]:
# #             final_verified = False
# #             final_status = "Unknown Person"

# #         else:
# #             final_verified = False
# #             final_status = "Suspicious Face"

# #         warning = False

# #         if identity_result["status"] != "Face not checked" and not identity_result["verified"]:
# #             warning = True

# #         if liveness_result["warning"]:
# #             warning = True

# #         return {
# #             "final_verified": final_verified,
# #             "final_status": final_status,
# #             "warning": warning,

# #             "identity_verified": identity_result["verified"],
# #             "identity_status": identity_result["status"],
# #             "identity_distance": identity_result["distance"],
# #             "identity_threshold": identity_result["threshold"],

# #             "is_live": liveness_result["is_live"],
# #             "liveness_status": liveness_result["status"],
# #             "live_score": liveness_result["live_score"],
# #             "spoof_score": liveness_result["spoof_score"],
# #             "face_box": liveness_result["face_box"],
# #             "raw_prediction": liveness_result["raw_prediction"]
# #         }

# #     def draw_liveness_result(self, frame, result):
# #         return self.liveness_detector.draw_result(frame, result)


# # def save_exam_log(exam_id, student_id, message):
# #     report_folder = create_exam_report_folder(exam_id, student_id)
# #     log_path = os.path.join(report_folder, "log.txt")

# #     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# #     with open(log_path, "a", encoding="utf-8") as file:
# #         file.write(f"[{current_time}] {message}\n")


# # def save_violation_screenshot(exam_id, student_id, frame, violation_type):
# #     report_folder = create_exam_report_folder(exam_id, student_id)
# #     screenshots_folder = os.path.join(report_folder, "screenshots")

# #     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# #     image_name = f"{violation_type}_{current_time}.jpg"
# #     image_path = os.path.join(screenshots_folder, image_name)

# #     cv2.imwrite(image_path, frame)

# #     return image_path


# """
# deepface_authenticator.py
# ─────────────────────────
# Real-time face authentication on CPU using three parallel pipelines:

#   Thread 1 – Identity        DeepFace.verify()   (heavy, every N frames)
#   Thread 2 – Liveness        CNNLivenessDetector (lighter, every N frames)
#   Thread 3 – Disk I/O        screenshots + log writes (never blocks camera)

# The camera loop itself never blocks — it always draws the latest cached
# result. This keeps the display at camera FPS even on a slow CPU.

# Architecture:
#   ┌──────────────┐   frames    ┌──────────────────┐
#   │  Camera loop │ ──────────▶ │ Identity thread  │
#   │  (main)      │             └──────────────────┘
#   │              │   frames    ┌──────────────────┐
#   │  draw cached │ ──────────▶ │ Liveness thread  │
#   │  results     │             └──────────────────┘
#   └──────────────┘   jobs      ┌──────────────────┐
#                     ──────────▶ │   I/O thread     │
#                                 └──────────────────┘
# """

# import os
# import threading
# import queue
# import time
# import logging
# from datetime import datetime
# from copy import deepcopy

# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["TF_USE_LEGACY_KERAS"] = "1"

# import cv2
# import numpy as np
# from deepface import DeepFace

# from face_detection.student_manager import find_student_by_id, create_exam_report_folder
# from face_detection.cnn_liveness_detector import CNNLivenessDetector

# logger = logging.getLogger(__name__)

# MODEL_NAME       = "ArcFace"
# DETECTOR_BACKEND = "opencv"   # fastest for CPU; switch to "retinaface" for accuracy


# # ─────────────────────────────────────────────────────────────────────────────
# # Helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def _resize_for_inference(frame: np.ndarray, max_side: int = 320) -> np.ndarray:
#     """Shrink frame before sending to DeepFace/Liveness — big CPU win."""
#     h, w = frame.shape[:2]
#     if max(h, w) <= max_side:
#         return frame
#     scale  = max_side / max(h, w)
#     return cv2.resize(frame, (int(w * scale), int(h * scale)),
#                       interpolation=cv2.INTER_LINEAR)


# def _grey_equalize(frame: np.ndarray) -> np.ndarray:
#     """CLAHE on luminance — helps ArcFace under uneven exam lighting."""
#     lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     lab   = cv2.merge([clahe.apply(l), a, b])
#     return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# # ─────────────────────────────────────────────────────────────────────────────
# # Background worker base
# # ─────────────────────────────────────────────────────────────────────────────

# class _Worker(threading.Thread):
#     """
#     Daemon thread that drains a queue of (frame, ...) jobs.
#     Only keeps the LATEST frame — old ones are dropped so we never
#     fall behind the camera.
#     """
#     def __init__(self, name: str):
#         super().__init__(name=name, daemon=True)
#         # maxsize=1: new frame overwrites the pending one
#         self._queue: queue.Queue = queue.Queue(maxsize=1)
#         self._result = None
#         self._lock   = threading.Lock()
#         self._stop   = threading.Event()

#     def submit(self, *args):
#         """Non-blocking put — drops the old item if the queue is full."""
#         try:
#             self._queue.put_nowait(args)
#         except queue.Full:
#             try:
#                 self._queue.get_nowait()   # discard stale frame
#             except queue.Empty:
#                 pass
#             try:
#                 self._queue.put_nowait(args)
#             except queue.Full:
#                 pass

#     def get_result(self):
#         with self._lock:
#             return deepcopy(self._result)

#     def _set_result(self, result):
#         with self._lock:
#             self._result = result

#     def stop(self):
#         self._stop.set()

#     def run(self):
#         while not self._stop.is_set():
#             try:
#                 args = self._queue.get(timeout=0.1)
#                 self._process(*args)
#             except queue.Empty:
#                 continue
#             except Exception as exc:
#                 logger.warning("[%s] error: %s", self.name, exc)

#     def _process(self, *args):
#         raise NotImplementedError


# # ─────────────────────────────────────────────────────────────────────────────
# # Identity worker  (DeepFace.verify — the slowest call, ~200-800 ms on CPU)
# # ─────────────────────────────────────────────────────────────────────────────

# class _IdentityWorker(_Worker):
#     _DEFAULT_RESULT = {
#         "status": "Face not checked",
#         "verified": False,
#         "distance": None,
#         "threshold": None,
#     }

#     def __init__(self, profile_image_path: str, face_threshold=None):
#         super().__init__("IdentityWorker")
#         self._profile = profile_image_path
#         self._face_threshold = face_threshold
#         self._set_result(self._DEFAULT_RESULT)

#     def _process(self, frame: np.ndarray):
#         small = _resize_for_inference(frame, max_side=320)
#         small = _grey_equalize(small)
#         try:
#             result = DeepFace.verify(
#                 img1_path=self._profile,
#                 img2_path=small,
#                 model_name=MODEL_NAME,
#                 detector_backend=DETECTOR_BACKEND,
#                 enforce_detection=False,
#                 align=True,
#                 silent=True,        # suppress DeepFace console spam
#             )
#             distance          = result["distance"]
#             deepface_threshold = result["threshold"]

#             if self._face_threshold is not None:
#                 threshold = self._face_threshold
#                 verified  = distance <= threshold
#             else:
#                 threshold = deepface_threshold
#                 verified  = result["verified"]

#             self._set_result({
#                 "status":    "Student Verified" if verified else "Unknown Person",
#                 "verified":  verified,
#                 "distance":  distance,
#                 "threshold": threshold,
#             })
#         except Exception as exc:
#             self._set_result({
#                 "status":    f"Verification Error: {exc}",
#                 "verified":  False,
#                 "distance":  None,
#                 "threshold": self._face_threshold,
#             })


# # ─────────────────────────────────────────────────────────────────────────────
# # Liveness worker  (CNN inference — lighter but still ~30-100 ms on CPU)
# # ─────────────────────────────────────────────────────────────────────────────

# class _LivenessWorker(_Worker):
#     _DEFAULT_RESULT = {
#         "is_live": False,
#         "status": "Checking…",
#         "live_score": 0.0,
#         "spoof_score": 1.0,
#         "face_box": None,
#         "raw_prediction": None,
#         "warning": False,
#     }

#     def __init__(self, threshold: float = 0.70):
#         super().__init__("LivenessWorker")
#         self._detector = CNNLivenessDetector(threshold=0.55, sigmoid_live_high=True)
#         self._set_result(self._DEFAULT_RESULT)

#     def _process(self, frame: np.ndarray):
#         small = _resize_for_inference(frame, max_side=320)
#         result = self._detector.predict(small)
#         self._set_result(result)


# # ─────────────────────────────────────────────────────────────────────────────
# # I/O worker  (screenshots + log writes — should NEVER block the camera loop)
# # ─────────────────────────────────────────────────────────────────────────────

# class _IOWorker(threading.Thread):
#     def __init__(self):
#         super().__init__(name="IOWorker", daemon=True)
#         self._queue: queue.Queue = queue.Queue()   # unbounded — every save matters
#         self._stop = threading.Event()

#     def submit_log(self, exam_id: str, student_id: str, message: str):
#         self._queue.put(("log", exam_id, student_id, message))

#     def submit_screenshot(self, exam_id: str, student_id: str,
#                           frame: np.ndarray, violation_type: str):
#         self._queue.put(("screenshot", exam_id, student_id,
#                          frame.copy(), violation_type))

#     def stop(self):
#         self._stop.set()

#     def run(self):
#         while not self._stop.is_set() or not self._queue.empty():
#             try:
#                 job = self._queue.get(timeout=0.2)
#             except queue.Empty:
#                 continue
#             try:
#                 if job[0] == "log":
#                     _, exam_id, student_id, message = job
#                     _write_log(exam_id, student_id, message)
#                 elif job[0] == "screenshot":
#                     _, exam_id, student_id, frame, vtype = job
#                     _write_screenshot(exam_id, student_id, frame, vtype)
#             except Exception as exc:
#                 logger.warning("[IOWorker] %s", exc)


# def _write_log(exam_id: str, student_id: str, message: str):
#     folder   = create_exam_report_folder(exam_id, student_id)
#     log_path = os.path.join(folder, "log.txt")
#     ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     with open(log_path, "a", encoding="utf-8") as f:
#         f.write(f"[{ts}] {message}\n")


# def _write_screenshot(exam_id: str, student_id: str,
#                       frame: np.ndarray, violation_type: str) -> str:
#     folder      = create_exam_report_folder(exam_id, student_id)
#     shots_dir   = os.path.join(folder, "screenshots")
#     os.makedirs(shots_dir, exist_ok=True)
#     ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
#     path        = os.path.join(shots_dir, f"{violation_type}_{ts}.jpg")
#     cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
#     return path


# # ─────────────────────────────────────────────────────────────────────────────
# # Public authenticator
# # ─────────────────────────────────────────────────────────────────────────────

# class DeepFaceAuthenticator:
#     """
#     Drop-in replacement for the original class.

#     Key differences:
#     • DeepFace.verify runs in a background thread — camera never blocks.
#     • Liveness CNN runs in a separate background thread.
#     • Log/screenshot writes are offloaded to a third background thread.
#     • `verify_frame_with_liveness()` returns instantly with cached results.
#     • Call `submit_frame()` each camera tick to feed new frames to workers.
#     • Call `shutdown()` (or use as a context manager) when done.

#     Usage:
#         auth = DeepFaceAuthenticator(student_id="123")
#         with auth:
#             while True:
#                 ret, frame = cap.read()
#                 auth.submit_frame(frame)               # feeds background workers
#                 result = auth.verify_frame_with_liveness(frame, run_identity_check=False)
#                 auth.draw_liveness_result(frame, result)
#                 cv2.imshow("Exam", frame)
#     """

#     # How often to send a frame to each heavy worker (in camera frames).
#     # Identity is much heavier than liveness, so runs less often.
#     IDENTITY_EVERY = 15   # ~0.5 s at 30 fps
#     LIVENESS_EVERY = 5    # ~0.17 s at 30 fps

#     def __init__(self, student_id: str, face_threshold=None):
#         self.student_id = student_id
#         student = find_student_by_id(student_id)
#         if student is None:
#             raise ValueError(f"Student not found: {student_id}")

#         profile = student["profile_image_path"]
#         if not os.path.exists(profile):
#             raise FileNotFoundError(f"Profile image not found: {profile}")

#         self.student = student

#         # Background workers
#         self._id_worker  = _IdentityWorker(profile, face_threshold)
#         self._liv_worker = _LivenessWorker(threshold=0.70)
#         self._io_worker  = _IOWorker()

#         self._id_worker.start()
#         self._liv_worker.start()
#         self._io_worker.start()

#         self._frame_idx = 0

#         print(f"Loaded student : {student['full_name']}")
#         print(f"Model          : {MODEL_NAME} / {DETECTOR_BACKEND}")
#         print(f"Threads        : identity every {self.IDENTITY_EVERY} frames, "
#               f"liveness every {self.LIVENESS_EVERY} frames")

#     # ── Context manager support ───────────────────────────────────────────────

#     def __enter__(self):
#         return self

#     def __exit__(self, *_):
#         self.shutdown()

#     def shutdown(self):
#         """Stop all background threads gracefully."""
#         self._id_worker.stop()
#         self._liv_worker.stop()
#         self._io_worker.stop()
#         self._id_worker.join(timeout=2)
#         self._liv_worker.join(timeout=2)
#         self._io_worker.join(timeout=5)   # let pending I/O drain

#     # ── Main feed method — call once per camera frame ─────────────────────────

#     def submit_frame(self, frame: np.ndarray):
#         """
#         Feed *frame* to the background workers on their schedule.
#         This is non-blocking and returns immediately.
#         """
#         self._frame_idx += 1
#         if self._frame_idx % self.IDENTITY_EVERY == 0:
#             self._id_worker.submit(frame)
#         if self._frame_idx % self.LIVENESS_EVERY == 0:
#             self._liv_worker.submit(frame)

#     # ── Result methods ────────────────────────────────────────────────────────

#     def verify_frame_with_liveness(
#         self,
#         frame: np.ndarray,
#         run_identity_check: bool = True,
#     ) -> dict:
#         """
#         Returns the latest cached result immediately (no blocking).

#         If run_identity_check=True and you want synchronous verification
#         (e.g. for a one-shot check), pass the frame here and it will be
#         submitted to the identity worker. Otherwise just read the cache.
#         """
#         if run_identity_check:
#             self._id_worker.submit(frame)

#         id_result  = self._id_worker.get_result()
#         liv_result = self._liv_worker.get_result()

#         # Combine results
#         status = id_result.get("status", "Face not checked")
#         id_ok  = id_result.get("verified", False)
#         liv_ok = liv_result.get("is_live", False)

#         if status == "Face not checked":
#             final_verified = False
#             final_status   = "Waiting for face verification"
#         elif id_ok and liv_ok:
#             final_verified = True
#             final_status   = "Verified Live Student"
#         elif id_ok and not liv_ok:
#             final_verified = False
#             final_status   = "Student Verified but Spoof Detected"
#         elif not id_ok:
#             final_verified = False
#             final_status   = "Unknown Person"
#         else:
#             final_verified = False
#             final_status   = "Suspicious Face"

#         warning = (status != "Face not checked" and not id_ok) \
#                   or liv_result.get("warning", False)

#         return {
#             "final_verified":    final_verified,
#             "final_status":      final_status,
#             "warning":           warning,

#             "identity_verified": id_ok,
#             "identity_status":   status,
#             "identity_distance": id_result.get("distance"),
#             "identity_threshold":id_result.get("threshold"),

#             "is_live":           liv_ok,
#             "liveness_status":   liv_result.get("status"),
#             "live_score":        liv_result.get("live_score"),
#             "spoof_score":       liv_result.get("spoof_score"),
#             "face_box":          liv_result.get("face_box"),
#             "raw_prediction":    liv_result.get("raw_prediction"),
#         }

#     def draw_liveness_result(self, frame: np.ndarray, result: dict) -> np.ndarray:
#         return self._liv_worker._detector.draw_result(frame, result)

#     # ── Convenience wrappers kept for backward compatibility ──────────────────

#     def verify_frame(self, frame: np.ndarray) -> dict:
#         """Synchronous-style call (submits and returns cached result)."""
#         self._id_worker.submit(frame)
#         return self._id_worker.get_result()

#     def check_liveness(self, frame: np.ndarray) -> dict:
#         """Synchronous-style call (submits and returns cached result)."""
#         self._liv_worker.submit(frame)
#         return self._liv_worker.get_result()

#     # ── I/O helpers ───────────────────────────────────────────────────────────

#     def save_log(self, exam_id: str, message: str):
#         """Non-blocking log write — queued to I/O thread."""
#         self._io_worker.submit_log(exam_id, self.student_id, message)

#     def save_screenshot(self, exam_id: str, frame: np.ndarray, violation_type: str):
#         """Non-blocking screenshot save — queued to I/O thread."""
#         self._io_worker.submit_screenshot(
#             exam_id, self.student_id, frame, violation_type
#         )


# # ─────────────────────────────────────────────────────────────────────────────
# # Module-level helpers (backward-compatible with original API)
# # ─────────────────────────────────────────────────────────────────────────────

# def save_exam_log(exam_id: str, student_id: str, message: str):
#     """Original API — still works, but blocks the caller. Prefer auth.save_log()."""
#     _write_log(exam_id, student_id, message)


# def save_violation_screenshot(exam_id: str, student_id: str,
#                               frame: np.ndarray, violation_type: str) -> str:
#     """Original API — still works, but blocks the caller. Prefer auth.save_screenshot()."""
#     return _write_screenshot(exam_id, student_id, frame, violation_type)

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

        # Stability buffer for liveness result
        self.liveness_history = deque(maxlen=7)

        # H5 liveness model
        # Your model: raw close to 1 = Live
        try:
            self.liveness_detector = CNNLivenessDetector(
                threshold=0.70,
                sigmoid_live_high=True,
                enable_blink_challenge=True,
                blink_valid_seconds=4.0
            )
        except TypeError:
            # If your CNNLivenessDetector does not have blink parameters yet
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
                align=True
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
        """
        Make liveness more stable.
        It returns only:
        Live
        Spoof
        """

        is_live_now = bool(liveness_result.get("is_live", False))

        self.liveness_history.append(1 if is_live_now else 0)

        live_votes = sum(self.liveness_history)
        total_votes = len(self.liveness_history)

        # Need majority vote
        stable_live = live_votes >= max(1, total_votes // 2 + 1)

        if stable_live:
            stable_status = "Live"
        else:
            stable_status = "Spoof"

        return stable_live, stable_status

    def verify_frame_with_liveness(self, frame, run_identity_check=True):
        liveness_result = self.check_liveness(frame)

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

        # Final text should be only Live or Spoof
        if identity_result["verified"] and stable_live:
            final_verified = True
            final_status = "Live"
        else:
            final_verified = False
            final_status = "Spoof"

        warning = False

        if identity_result["status"] != "Face not checked" and not identity_result["verified"]:
            warning = True

        if not stable_live:
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

            # Only show Live or Spoof in text
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
            "requires_action": liveness_result.get("requires_action", False)
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

    cv2.imwrite(image_path, frame)

    return image_path