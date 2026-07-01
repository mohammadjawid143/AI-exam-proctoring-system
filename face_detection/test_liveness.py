import cv2
import time

from face_detection.cnn_liveness_detector import CNNLivenessDetector


class LivenessTestRunner:
    """
    Standalone webcam test for the shared CNNLivenessDetector.

    Keys:
    - q: quit
    - b: reset blink state
    - p: reverse sigmoid meaning if your model output is inverted
    - +: increase threshold
    - -: decrease threshold
    """

    def __init__(self):
        self.detector = CNNLivenessDetector(
            threshold=0.60,
            sigmoid_live_high=True,
            enable_blink_challenge=True,
            blink_valid_seconds=12.0,
            blink_challenge_interval_seconds=18.0,
        )

    def draw(self, frame, result):
        self.detector.draw_result(frame, result)

        quality = result.get("quality", {}) or {}
        blur_score = quality.get("blur_score")
        brightness = quality.get("brightness")

        if blur_score is None:
            quality_text = "Quality: —"
        else:
            quality_text = f"Blur: {blur_score:.1f} | Bright: {brightness:.1f}"

        settings_text = (
            f"Threshold: {self.detector.threshold:.2f} | "
            f"Spoof threshold: {self.detector.spoof_threshold:.2f} | "
            f"sigmoid_live_high: {self.detector.sigmoid_live_high}"
        )

        reason_text = f"Reason: {result.get('decision_reason', '')}"

        cv2.putText(
            frame,
            quality_text,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            reason_text,
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            settings_text,
            (20, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


def run_test():
    runner = LivenessTestRunner()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera could not be opened.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("CNN Liveness Test started.")
    print("Press q = quit")
    print("Press b = reset blink")
    print("Press p = reverse prediction meaning")
    print("Press + = increase threshold")
    print("Press - = decrease threshold")
    print("Real test: look at camera and blink once when asked.")

    last_print = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Could not read frame from camera.")
                break

            result = runner.detector.predict(frame)
            runner.draw(frame, result)

            now = time.time()

            if now - last_print >= 1:
                print("--------------------------------")
                print("Status:", result.get("status"))
                print("Is Live:", result.get("is_live"))
                print("Model Live:", result.get("model_live"))
                print("Live Score:", result.get("live_score"))
                print("Spoof Score:", result.get("spoof_score"))
                print("Raw:", result.get("raw_prediction"))
                print("Blink valid:", result.get("blink_valid"))
                print("EAR:", result.get("ear"))
                print("Face area:", result.get("face_area_ratio"))
                print("Face changed:", result.get("face_changed"))
                print("Reason:", result.get("decision_reason"))
                print("Quality:", result.get("quality"))
                last_print = now

            cv2.imshow("CNN Liveness Test", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("b"):
                runner.detector.reset_tracking_state()
                print("Blink/liveness state reset.")

            elif key == ord("p"):
                runner.detector.sigmoid_live_high = not runner.detector.sigmoid_live_high
                runner.detector.score_history.clear()
                print("sigmoid_live_high:", runner.detector.sigmoid_live_high)

            elif key == ord("+") or key == ord("="):
                runner.detector.threshold = min(0.95, runner.detector.threshold + 0.05)
                runner.detector.score_history.clear()
                print("Threshold:", runner.detector.threshold)

            elif key == ord("-"):
                runner.detector.threshold = max(0.05, runner.detector.threshold - 0.05)
                runner.detector.score_history.clear()
                print("Threshold:", runner.detector.threshold)

    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_deepface_authentication():
    run_test()


if __name__ == "__main__":
    run_test()
