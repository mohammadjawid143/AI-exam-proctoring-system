import cv2
from eye_tracking.eye_tracker import EyeTracker


def run_eye_tracking_test():
    # Config هماهنگ با EyeTracker(config)
    config = {
        "detection": {
            "eyes": {
                # حرکت عنبیه برای چپ و راست
                "iris_horizontal_threshold": 0.15,
                "iris_vertical_threshold": 0.14,

                # چرخش سر برای بالا و پایین
                "head_horizontal_threshold": 18,
                "head_vertical_threshold": 15,

                # بسته بودن چشم
                "closed_eye_threshold": 0.18,

                # کالیبراسیون
                "calibration_frames": 50,

                # ضد لرزش
                "smoothing_alpha": 0.20,
                "stable_frames_required": 7,

                # نمایش
                "draw": True,
                "show_debug_text": False,
                "iris_point_radius": 2,

                # اگر چپ و راست تصویر آینه‌ای بود True کن
                "mirror_output": True,

                # اگر بالا/پایین سر برعکس شد، این را True کن
                "invert_head_vertical": False,

                # اگر چپ/راست سر برعکس شد، این را True کن
                "invert_head_horizontal": False
            }
        }
    }

    tracker = EyeTracker(config)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    print("Eye Tracking started.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # این با فایل eye_tracker.py تو هماهنگ است
        gaze_direction, eye_ratio = tracker.track_eyes(frame)

        text = f"Gaze: {gaze_direction} | Eye Ratio: {eye_ratio:.2f}"

        # اگر صورت پیدا نشد، رنگ قرمز شود
        if gaze_direction == "no_face":
            color = (0, 0, 255)
        elif gaze_direction == "center":
            color = (0, 255, 0)
        else:
            color = (0, 255, 255)

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.imshow("Eye Tracking Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_eye_tracking_test()