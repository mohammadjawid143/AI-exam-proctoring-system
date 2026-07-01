# import cv2
# from object_detection.cheating_object_detector import CheatingObjectDetector


# def run_object_detection_test():
#     detector = CheatingObjectDetector(
#         model_path="yolov8n.pt",
#         confidence_threshold=0.45
#     )

#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Error: Camera could not be opened.")
#         return

#     print("Cheating Object Detection started.")
#     print("Press 'q' to quit.")

#     while True:
#         ret, frame = cap.read()

#         if not ret:
#             print("Error: Could not read frame.")
#             break

#         cheating_detected, detected_objects = detector.detect_objects(frame)

#         for obj in detected_objects:
#             x1, y1, x2, y2 = obj["box"]
#             name = obj["name"]
#             confidence = obj["confidence"]

#             if obj["is_cheating_object"]:
#                 color = (0, 0, 255)
#                 label = f"CHEATING: {name} {confidence:.2f}"
#             else:
#                 color = (0, 255, 0)
#                 label = f"{name} {confidence:.2f}"

#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

#             cv2.putText(
#                 frame,
#                 label,
#                 (x1, y1 - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,
#                 color,
#                 2
#             )

#         if cheating_detected:
#             cv2.putText(
#                 frame,
#                 "WARNING: CHEATING OBJECT DETECTED",
#                 (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 (0, 0, 255),
#                 2
#             )
#         else:
#             cv2.putText(
#                 frame,
#                 "No cheating object",
#                 (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 (0, 255, 0),
#                 2
#             )

#         cv2.imshow("Cheating Object Detection", frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     run_object_detection_test()

import time
import cv2
from object_detection.cheating_object_detector import CheatingObjectDetector

# ── Display constants ────────────────────────────────────────────────────────
FONT          = cv2.FONT_HERSHEY_SIMPLEX
COLOR_CHEAT   = (0, 0, 255)
COLOR_OK      = (0, 255, 0)
COLOR_FPS     = (255, 255, 0)
COLOR_WARN_BG = (0, 0, 180)

WARNING_TEXT  = "WARNING: CHEATING DETECTED"
SAFE_TEXT     = "All clear"


def _draw_box(frame, obj: dict) -> None:
    """Draw a single bounding box + label onto frame (in-place)."""
    x1, y1, x2, y2 = obj["box"]
    name            = obj["name"]
    conf            = obj["confidence"]
    is_cheat        = obj["is_cheating_object"]

    color = COLOR_CHEAT if is_cheat else COLOR_OK
    prefix = "CHEAT: " if is_cheat else ""
    label  = f"{prefix}{name} {conf:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Dark backing strip so text is legible over any background
    (tw, th), baseline = cv2.getTextSize(label, FONT, 0.55, 1)
    top = max(y1 - th - baseline - 4, 0)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 4, y1), color, cv2.FILLED)
    cv2.putText(frame, label, (x1 + 2, y1 - baseline - 2),
                FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_status(frame, cheating_detected: bool, fps: float) -> None:
    """Draw the top-left status banner and FPS counter."""
    h, w = frame.shape[:2]

    # Status banner
    if cheating_detected:
        (tw, th), bl = cv2.getTextSize(WARNING_TEXT, FONT, 0.75, 2)
        cv2.rectangle(frame, (0, 0), (tw + 20, th + bl + 14), COLOR_WARN_BG, cv2.FILLED)
        cv2.putText(frame, WARNING_TEXT, (10, th + 8),
                    FONT, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, SAFE_TEXT, (10, 34),
                    FONT, 0.75, COLOR_OK, 2, cv2.LINE_AA)

    # FPS — top-right corner
    fps_label = f"FPS: {fps:.1f}"
    (fw, _), _ = cv2.getTextSize(fps_label, FONT, 0.55, 1)
    cv2.putText(frame, fps_label, (w - fw - 8, 22),
                FONT, 0.55, COLOR_FPS, 1, cv2.LINE_AA)


def run_object_detection_test(
    camera_index: int = 0,
    model_path: str  = "yolo11n.pt",   # use the nano model for low CPU
    confidence: float = 0.45,
    img_size: int    = 320,            # 320 is much lighter than 640
    skip_frames: int = 2,              # inference every N frames
    display_width: int = 0,            # 0 = keep original; e.g. 640 to shrink
) -> None:
    detector = CheatingObjectDetector(
        model_path=model_path,
        confidence_threshold=confidence,
        img_size=img_size,
        skip_frames=skip_frames,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Camera {camera_index} could not be opened.")
        return

    # Ask the OS for a sane buffer size — reduces latency on slow machines
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Cheating Object Detection started. Press 'q' to quit.")

    # FPS tracking
    fps         = 0.0
    frame_count = 0
    t_start     = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Optional: shrink the display frame to ease OpenCV's imshow draw cost
        if display_width and frame.shape[1] != display_width:
            scale        = display_width / frame.shape[1]
            display_h    = int(frame.shape[0] * scale)
            display_frame = cv2.resize(frame, (display_width, display_h),
                                       interpolation=cv2.INTER_LINEAR)
        else:
            display_frame = frame

        cheating_detected, detected_objects = detector.detect_objects(display_frame)

        for obj in detected_objects:
            _draw_box(display_frame, obj)

        # Update FPS every 15 frames to avoid jitter
        frame_count += 1
        if frame_count % 15 == 0:
            elapsed = time.perf_counter() - t_start
            fps     = frame_count / elapsed if elapsed > 0 else 0.0
            # Reset periodically so the number stays current
            frame_count = 0
            t_start     = time.perf_counter()

        _draw_status(display_frame, cheating_detected, fps)

        cv2.imshow("Cheating Object Detection", display_frame)

        # waitKey(1) burns ~1 ms; on very slow CPUs use waitKey(5) to free more cycles
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")


if __name__ == "__main__":
    run_object_detection_test()