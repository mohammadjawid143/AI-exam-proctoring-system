from face_detection.register_student import register_student
from core.proctoring_system import ProctoringSystem


def start_exam():
    print("\n===== Start Full Exam Proctoring =====")

    student_id = input("Enter student ID, example STU001: ").strip()
    exam_id = input("Enter exam ID, example EXAM_001: ").strip()

    if not student_id or not exam_id:
        print("Error: student ID and exam ID are required.")
        return

    try:
        system = ProctoringSystem(
            student_id=student_id,
            exam_id=exam_id,

            # Face Recognition + CNN Liveness
            enable_face=True,

            # Eye Tracking
            enable_eye=True,

            # Object Detection
            enable_object=True,

            # Audio / Voice Detection
            enable_audio=True
        )

        system.run()

    except TypeError as error:
        print("\nError in ProctoringSystem parameters:")
        print(error)
        print("\nMaybe your ProctoringSystem class is old. Replace core/proctoring_system.py with the updated code.")

    except Exception as error:
        print("\nError while starting exam:")
        print(error)


def test_face_liveness():
    try:
        from face_detection.test_liveness import run_deepface_authentication
        run_deepface_authentication()
    except ImportError:
        try:
            from face_detection.test_deepface_authentication import run_deepface_authentication
            run_deepface_authentication()
        except ImportError as error:
            print("Face liveness test file not found.")
            print(error)


def test_eye_tracking():
    try:
        from eye_tracking.test_eye_tracking import run_eye_tracking_test
        run_eye_tracking_test()
    except ImportError as error:
        print("Eye tracking test file not found.")
        print(error)


def test_object_detection():
    try:
        from object_detection.test_object_detection import run_object_detection_test
        run_object_detection_test()
    except ImportError as error:
        print("Object detection test file not found.")
        print(error)


def test_audio_detection():
    try:
        from audio_detection.test_audio_detection import run_audio_detection_test
        run_audio_detection_test()
    except ImportError as error:
        print("Audio detection test file not found.")
        print("Make sure this file exists: audio_detection/test_audio_detection.py")
        print(error)


def main():
    while True:
        print("\n===== AI Exam Proctoring System =====")
        print("1. Register new student")
        print("2. Start full exam proctoring")
        print("3. Test face recognition + CNN liveness")
        print("4. Test eye tracking")
        print("5. Test cheating object detection")
        print("6. Test audio/noise detection")
        print("0. Exit")

        choice = input("Choose option: ").strip()

        if choice == "1":
            register_student()

        elif choice == "2":
            start_exam()

        elif choice == "3":
            test_face_liveness()

        elif choice == "4":
            test_eye_tracking()

        elif choice == "5":
            test_object_detection()

        elif choice == "6":
            test_audio_detection()

        elif choice == "0":
            print("Goodbye.")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()