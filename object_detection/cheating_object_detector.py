from ultralytics import YOLO
import numpy as np
import cv2


class CheatingObjectDetector:
    """
    YOLO-based object detector for online exam proctoring.

    This class focuses on exam-relevant classes only. This reduces confusing
    labels such as chair/cup/bottle and improves dashboard clarity.
    """

    def __init__(
        self,
        model_path="yolo11n.pt",
        confidence_threshold=0.40,
        img_size=640,
        skip_frames=1,
        iou_threshold=0.45,
        warmup=True,
        show_only_relevant=True,
        class_conf_overrides=None,
    ):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.img_size = img_size
        self.skip_frames = max(1, int(skip_frames))
        self.iou_threshold = iou_threshold
        self.show_only_relevant = show_only_relevant

        try:
            self.model.fuse()
        except Exception:
            pass

        self._frame_count = 0
        self._last_result = (False, [])

        # Objects that count as cheating evidence.
        self.cheating_objects = {
            "cell phone",
            "book",
            "laptop",
            "remote",
            "tv",
            "monitor",
        }

        # Objects worth drawing on the dashboard.
        # Everything else is ignored to avoid noise like chair/cup/bottle.
        self.relevant_objects = {
            "person",
            "cell phone",
            "book",
            "laptop",
            "remote",
            "tv",
            "monitor",
        }

        self._cheating_class_ids = {
            class_id
            for class_id, name in self.model.names.items()
            if name in self.cheating_objects
        }

        self._relevant_class_ids = {
            class_id
            for class_id, name in self.model.names.items()
            if name in self.relevant_objects
        }

        # Per-class confidence thresholds.
        # Book and phone are often small in webcam frames, so their threshold is lower.
        defaults = {
            "cell phone": 0.30,
            "book": 0.25,
            "laptop": 0.35,
            "remote": 0.30,
            "tv": 0.35,
            "monitor": 0.35,
            "person": 0.40,
        }

        overrides = {**defaults, **(class_conf_overrides or {})}

        self._class_conf = {
            class_id: overrides[name]
            for class_id, name in self.model.names.items()
            if name in overrides
        }

        # YOLO must run at the lowest threshold we want to keep.
        # We apply per-class filtering after prediction.
        self._yolo_conf = min(
            confidence_threshold,
            min(self._class_conf.values(), default=confidence_threshold)
        )

        if warmup:
            try:
                dummy = np.zeros((img_size, img_size, 3), dtype=np.uint8)
                self.model.predict(
                    dummy,
                    imgsz=img_size,
                    conf=self._yolo_conf,
                    iou=iou_threshold,
                    verbose=False,
                    device="cpu"
                )
            except Exception:
                pass

    def detect_objects(self, frame):
        """
        Run object detection on a frame.

        Returns:
            (cheating_detected, detected_objects)
        """
        self._frame_count += 1

        if self._frame_count % self.skip_frames != 0:
            return self._last_result

        if frame is None:
            self._last_result = (False, [])
            return self._last_result

        orig_h, orig_w = frame.shape[:2]
        resized, scale, pad_w, pad_h = self._preprocess(frame)

        predict_kwargs = {
            "source": resized,
            "imgsz": self.img_size,
            "conf": self._yolo_conf,
            "iou": self.iou_threshold,
            "half": False,
            "device": "cpu",
            "verbose": False,
            "augment": False,
            "agnostic_nms": False,
        }

        if self.show_only_relevant and self._relevant_class_ids:
            predict_kwargs["classes"] = list(self._relevant_class_ids)

        results = self.model.predict(**predict_kwargs)

        cheating_detected, detected_objects = self._parse_results(
            results,
            scale,
            pad_w,
            pad_h,
            orig_w,
            orig_h
        )

        self._last_result = (cheating_detected, detected_objects)
        return self._last_result

    def reset(self):
        """
        Clear cached result.
        """
        self._frame_count = 0
        self._last_result = (False, [])

    def _preprocess(self, frame):
        """
        Letterbox-resize frame to YOLO input size.
        """
        h, w = frame.shape[:2]
        target = self.img_size

        scale = target / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(
            frame,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        )

        pad_h = target - new_h
        pad_w = target - new_w

        resized = cv2.copyMakeBorder(
            resized,
            pad_h // 2,
            pad_h - pad_h // 2,
            pad_w // 2,
            pad_w - pad_w // 2,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        return resized, scale, pad_w, pad_h

    def _parse_results(self, results, scale, pad_w, pad_h, orig_w, orig_h):
        """
        Convert YOLO results from letterbox coordinates to original frame coordinates.
        """
        cheating_detected = False
        detected_objects = []

        off_x = pad_w // 2
        off_y = pad_h // 2

        for result in results:
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            coords = boxes.xyxy.cpu().numpy()

            for class_id, confidence, (x1, y1, x2, y2) in zip(cls_ids, confs, coords):
                object_name = self.model.names[class_id]

                if self.show_only_relevant and object_name not in self.relevant_objects:
                    continue

                threshold = self._class_conf.get(
                    class_id,
                    self.confidence_threshold
                )

                if confidence < threshold:
                    continue

                x1 = int(np.clip((x1 - off_x) / scale, 0, orig_w - 1))
                y1 = int(np.clip((y1 - off_y) / scale, 0, orig_h - 1))
                x2 = int(np.clip((x2 - off_x) / scale, 0, orig_w - 1))
                y2 = int(np.clip((y2 - off_y) / scale, 0, orig_h - 1))

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
