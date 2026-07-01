import cv2
import time

from face_detection.deepface_auth import (
    DeepFaceAuthenticator,
    save_exam_log,
    save_violation_screenshot,
    MODEL_NAME,
    DETECTOR_BACKEND
)


def format_value(value):
    if value is None:
        return "None"

    if isinstance(value, float):
        return f"{value:.2f}"

    return str(value)


def draw_liveness_box(frame, result):
    face_box = result.get("face_box")

    if face_box is None:
        return frame

    x1, y1, x2, y2 = face_box

    if result["is_live"]:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    live_score = result.get("live_score")

    if live_score is not None:
        label = f"{result['liveness_status']} | Live: {live_score:.2f}"
    else:
        label = result["liveness_status"]

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.putText(
        frame,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

    return frame


def run_deepface_authentication():
    print("===== DeepFace Face Recognition + CNN Liveness Test =====")

    student_id = input("Enter student ID, example STU001: ").strip()
    exam_id = input("Enter exam ID, example exam_001: ").strip()

    if not student_id or not exam_id:
        print("Error: student ID and exam ID are required.")
        return

    try:
        authenticator = DeepFaceAuthenticator(
            student_id=student_id,
            face_threshold=0.55
        )
    except Exception as error:
        print(f"Error: {error}")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    print("Authentication with CNN liveness started.")
    print("Press 'q' to quit.")
    print("Test with real face and then with a printed/mobile photo.")

    frame_count = 0

    last_final_status = "Checking..."
    last_identity_status = "Face not checked"
    last_liveness_status = "Checking liveness"

    last_distance = None
    last_threshold = None

    last_live_score = None
    last_spoof_score = None
    last_raw_prediction = None

    last_logged_status = ""
    last_warning_time = 0
    warning_cooldown = 10

    verified_live_logged = False

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            frame_count += 1

            # CNN liveness checks every frame.
            # DeepFace recognition is heavy, so identity check runs every 60 frames.
            # Also run once at the beginning.
            run_identity_check = frame_count == 1 or frame_count % 60 == 0

            result = authenticator.verify_frame_with_liveness(
                frame,
                run_identity_check=run_identity_check
            )

            last_final_status = result["final_status"]
            last_identity_status = result["identity_status"]
            last_liveness_status = result["liveness_status"]

            last_distance = result["identity_distance"]
            last_threshold = result["identity_threshold"]

            last_live_score = result["live_score"]
            last_spoof_score = result["spoof_score"]
            last_raw_prediction = result["raw_prediction"]

            if run_identity_check:
                print(
                    f"Final: {last_final_status} | "
                    f"Face: {last_identity_status} | "
                    f"Liveness: {last_liveness_status} | "
                    f"Distance: {format_value(last_distance)} | "
                    f"Threshold: {format_value(last_threshold)} | "
                    f"Live Score: {format_value(last_live_score)} | "
                    f"Spoof Score: {format_value(last_spoof_score)} | "
                    f"Raw: {last_raw_prediction}"
                )

            if result["final_verified"] and not verified_live_logged:
                save_exam_log(
                    exam_id,
                    student_id,
                    f"Verified live student using {MODEL_NAME} + {DETECTOR_BACKEND} + CNN Liveness"
                )
                verified_live_logged = True

            current_time = time.time()

            if result["warning"]:
                if current_time - last_warning_time >= warning_cooldown:
                    last_warning_time = current_time

                    warning_message = (
                        f"{last_final_status} | "
                        f"Face: {last_identity_status} | "
                        f"Liveness: {last_liveness_status} | "
                        f"Distance: {format_value(last_distance)} | "
                        f"Threshold: {format_value(last_threshold)} | "
                        f"Live Score: {format_value(last_live_score)} | "
                        f"Spoof Score: {format_value(last_spoof_score)} | "
                        f"Raw: {last_raw_prediction}"
                    )

                    save_exam_log(
                        exam_id,
                        student_id,
                        f"FACE_LIVENESS_WARNING: {warning_message}"
                    )

                    image_path = save_violation_screenshot(
                        exam_id,
                        student_id,
                        frame,
                        "face_liveness_warning"
                    )

                    save_exam_log(
                        exam_id,
                        student_id,
                        f"Screenshot saved: {image_path}"
                    )

                    print("WARNING:", warning_message)

            if last_final_status != last_logged_status:
                save_exam_log(
                    exam_id,
                    student_id,
                    f"Face security status changed: {last_final_status}"
                )
                last_logged_status = last_final_status

            if result["final_verified"]:
                final_color = (0, 255, 0)
            elif "Unknown" in last_final_status or result["warning"]:
                final_color = (0, 0, 255)
            else:
                final_color = (0, 255, 255)

            face_color = (0, 255, 0) if result["identity_verified"] else (0, 0, 255)
            live_color = (0, 255, 0) if result["is_live"] else (0, 0, 255)

            frame = draw_liveness_box(frame, result)

            cv2.putText(
                frame,
                f"Final: {last_final_status}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                final_color,
                2
            )

            cv2.putText(
                frame,
                f"Face: {last_identity_status}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                face_color,
                2
            )

            cv2.putText(
                frame,
                f"Liveness: {last_liveness_status}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                live_color,
                2
            )

            cv2.putText(
                frame,
                f"Live Score: {format_value(last_live_score)} | Spoof Score: {format_value(last_spoof_score)}",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Distance: {format_value(last_distance)} | Threshold: {format_value(last_threshold)}",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Model: {MODEL_NAME} | Detector: {DETECTOR_BACKEND}",
                (20, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            if last_raw_prediction is not None:
                raw_text = f"Raw: {last_raw_prediction}"
                cv2.putText(
                    frame,
                    raw_text[:80],
                    (20, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

            cv2.imshow("Face Recognition + CNN Liveness Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_deepface_authentication()