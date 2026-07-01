# from ultralytics import YOLO


# class CheatingObjectDetector:
#     def __init__(
#         self,
#         model_path="yolo11n.pt",
#         confidence_threshold=0.45,
#         img_size=640
#     ):
#         self.model = YOLO(model_path)
#         self.confidence_threshold = confidence_threshold
#         self.img_size = img_size

#         self.cheating_objects = {
#             "cell phone",
#             "book",
#             "laptop",
#             "remote",
#             "tv",
#             "monitor",
#         }

#     def detect_objects(self, frame):
#         results = self.model.predict(
#             frame,
#             imgsz=self.img_size,
#             conf=self.confidence_threshold,
#             verbose=False
#         )

#         detected_objects = []
#         cheating_detected = False

#         for result in results:
#             boxes = result.boxes

#             for box in boxes:
#                 class_id = int(box.cls[0])
#                 confidence = float(box.conf[0])
#                 object_name = self.model.names[class_id]

#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

#                 is_cheating_object = object_name in self.cheating_objects

#                 if is_cheating_object:
#                     cheating_detected = True

#                 detected_objects.append({
#                     "name": object_name,
#                     "confidence": confidence,
#                     "box": (x1, y1, x2, y2),
#                     "is_cheating_object": is_cheating_object
#                 })

#         return cheating_detected, detected_objects
from ultralytics import YOLO
import numpy as np
import cv2
from collections import deque
import time


class CheatingObjectDetector:
    def __init__(
        self,
        model_path="yolo11n.pt",
        confidence_threshold=0.45,
        img_size=416,           # 416 > 320: much better book recall, still fast on CPU
        skip_frames=2,          # Run inference every N frames
        iou_threshold=0.45,
        warmup=True,
        class_conf_overrides=None,  # dict[str, float] — per-class confidence overrides
    ):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.img_size = img_size
        self.skip_frames = skip_frames
        self.iou_threshold = iou_threshold

        # Export to ONNX or OpenVINO for faster CPU inference (optional)
        # Uncomment the line below to auto-export on first run:
        # self.model = self._try_export_optimized(model_path)

        # Fuse Conv+BN layers — reduces ops with no accuracy loss
        self.model.fuse()

        self._frame_count = 0
        self._last_result = (False, [])

        self.cheating_objects = {
            "cell phone", "book", "laptop", "remote", "tv", "monitor",
        }
        self._cheating_class_ids = {
            cid
            for cid, name in self.model.names.items()
            if name in self.cheating_objects
        }

        # Per-class confidence thresholds (class_id → threshold).
        # "book" is underrepresented in COCO training data so the model
        # assigns it lower scores — lower its bar to 0.25 so it isn't silently
        # dropped. We tell YOLO to run at the minimum threshold so overridden
        # classes pass through; then we apply per-class filtering ourselves.
        defaults = {"book": 0.25}
        overrides = {**defaults, **(class_conf_overrides or {})}
        self._class_conf = {
            cid: overrides[name]
            for cid, name in self.model.names.items()
            if name in overrides
        }
        self._yolo_conf = min(
            confidence_threshold,
            min(self._class_conf.values(), default=confidence_threshold)
        )

        # Optional warm-up pass so the first real frame isn't slow
        if warmup:
            dummy = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            self.model.predict(dummy, imgsz=img_size, conf=self._yolo_conf,
                               iou=iou_threshold, verbose=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_objects(self, frame: np.ndarray) -> tuple[bool, list]:
        """
        Run detection on *frame*.

        Returns (cheating_detected: bool, detected_objects: list[dict]).
        Boxes are always in the coordinate space of the *original* frame.
        Skipped frames reuse the previous result — nearly free on CPU.
        """
        self._frame_count += 1
        if self._frame_count % self.skip_frames != 0:
            return self._last_result

        orig_h, orig_w = frame.shape[:2]

        # Resize + letterbox once; also get the transform so we can invert it
        resized, scale, pad_w, pad_h = self._preprocess(frame)

        results = self.model.predict(
            resized,
            imgsz=self.img_size,
            conf=self._yolo_conf,       # minimum across all per-class thresholds
            iou=self.iou_threshold,
            half=False,         # FP16 unsupported on most CPUs; keep False
            device="cpu",       # Explicit — avoids accidental GPU probe
            verbose=False,
            augment=False,      # TTA off — ~3× slower when enabled
            agnostic_nms=False,
        )

        cheating_detected, detected_objects = self._parse_results(
            results, scale, pad_w, pad_h, orig_w, orig_h
        )
        self._last_result = (cheating_detected, detected_objects)
        return cheating_detected, detected_objects

    def reset(self) -> None:
        """Clear cached state (e.g. when switching camera sources)."""
        self._frame_count = 0
        self._last_result = (False, [])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """
        Letterbox-resize *frame* to (img_size × img_size).

        Returns:
            resized  – the padded image ready for inference
            scale    – how much the original was scaled down
            pad_w    – total horizontal padding added (pixels, before split)
            pad_h    – total vertical padding added (pixels, before split)
        """
        h, w   = frame.shape[:2]
        target = self.img_size

        scale = target / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = target - new_h
        pad_w = target - new_w
        resized = cv2.copyMakeBorder(
            resized,
            pad_h // 2, pad_h - pad_h // 2,
            pad_w // 2, pad_w - pad_w // 2,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        return resized, scale, pad_w, pad_h

    def _parse_results(
        self,
        results,
        scale: float,
        pad_w: int,
        pad_h: int,
        orig_w: int,
        orig_h: int,
    ) -> tuple[bool, list]:
        """
        Convert YOLO boxes from letterboxed-image space → original frame space.

        Letterbox transform:
            x_resized = x_orig * scale + pad_w // 2
            y_resized = y_orig * scale + pad_h // 2

        Inverse:
            x_orig = (x_resized - pad_w // 2) / scale
            y_orig = (y_resized - pad_h // 2) / scale
        """
        cheating_detected = False
        detected_objects  = []

        off_x = pad_w // 2
        off_y = pad_h // 2

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs   = boxes.conf.cpu().numpy()
            coords  = boxes.xyxy.cpu().numpy()          # float, letterbox space

            for class_id, confidence, (x1, y1, x2, y2) in zip(cls_ids, confs, coords):
                # Apply per-class threshold (falls back to global threshold)
                threshold = self._class_conf.get(class_id, self.confidence_threshold)
                if confidence < threshold:
                    continue

                # ── Unproject to original frame coordinates ──────────────
                x1 = int(np.clip((x1 - off_x) / scale, 0, orig_w - 1))
                y1 = int(np.clip((y1 - off_y) / scale, 0, orig_h - 1))
                x2 = int(np.clip((x2 - off_x) / scale, 0, orig_w - 1))
                y2 = int(np.clip((y2 - off_y) / scale, 0, orig_h - 1))

                object_name = self.model.names[class_id]
                is_cheating = class_id in self._cheating_class_ids

                if is_cheating:
                    cheating_detected = True

                detected_objects.append({
                    "name": object_name,
                    "confidence": float(confidence),
                    "box": (x1, y1, x2, y2),
                    "is_cheating_object": is_cheating,
                })

        return cheating_detected, detected_objects

    # ------------------------------------------------------------------
    # Optional: export to ONNX/OpenVINO for a big CPU speed boost
    # ------------------------------------------------------------------

    def _try_export_optimized(self, model_path: str):
        """
        Export the model to OpenVINO IR (best CPU throughput) or ONNX.
        Call once; subsequent runs load the cached export automatically.
        """
        try:
            path = self.model.export(format="openvino", dynamic=False,
                                     imgsz=self.img_size)
            print(f"[CheatingDetector] Loaded OpenVINO model from {path}")
            return YOLO(path)
        except Exception:
            pass
        try:
            path = self.model.export(format="onnx", dynamic=False,
                                     imgsz=self.img_size)
            print(f"[CheatingDetector] Loaded ONNX model from {path}")
            return YOLO(path)
        except Exception:
            print("[CheatingDetector] Could not export; falling back to PyTorch.")
            return self.model