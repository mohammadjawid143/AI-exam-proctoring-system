import cv2
import time

from face_detection.deepface_auth import DeepFaceAuthenticator


def run_deepface_authentication(student_id=None):
    """
    Standalone webcam test for student identity verification.

    This test does NOT run liveness detection.
    It only checks whether the current face matches the registered student image.
    """
    if student_id is None:
        student_id = input("Enter student ID, example STU001: ").strip()

    if not student_id:
        print("Student ID is required.")
        return

    authenticator = DeepFaceAuthenticator(
        student_id=student_id,
        face_threshold=0.55,
        enforce_detection=False
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera could not be opened.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("DeepFace Identity Test started.")
    print("Press q = quit")
    print("Press v = verify now")
    print("The system also verifies automatically every 5 seconds.")

    last_verify_time = 0.0
    verify_interval = 5.0

    latest_result = {
        "status": "Face not checked",
        "verified": False,
        "distance": None,
        "threshold": 0.55
    }

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Could not read frame from camera.")
                break

            now = time.time()
            key = cv2.waitKey(1) & 0xFF

            should_verify = (now - last_verify_time) >= verify_interval

            if key == ord("v"):
                should_verify = True

            if should_verify:
                latest_result = authenticator.verify_frame(frame)
                last_verify_time = now

                print("--------------------------------")
                print("Status:", latest_result.get("status"))
                print("Verified:", latest_result.get("verified"))
                print("Distance:", latest_result.get("distance"))
                print("Threshold:", latest_result.get("threshold"))

            status = latest_result.get("status", "Face not checked")
            verified = bool(latest_result.get("verified", False))
            distance = latest_result.get("distance")
            threshold = latest_result.get("threshold")

            color = (0, 255, 0) if verified else (0, 0, 255)

            if distance is None:
                distance_text = "Dist: —"
            else:
                distance_text = f"Dist: {distance:.3f}"

            text1 = f"Identity: {status}"
            text2 = f"{distance_text} | Threshold: {threshold}"

            cv2.putText(
                frame,
                text1,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                text2,
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("DeepFace Identity Test", frame)

            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_deepface_authentication()
