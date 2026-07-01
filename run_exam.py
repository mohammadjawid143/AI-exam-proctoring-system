import argparse

from core.proctoring_system import ProctoringSystem


def run_full_exam(args):
    """
    Run full proctoring system.
    """
    if not args.student_id:
        raise ValueError("--student-id is required for exam mode.")

    if not args.exam_id:
        raise ValueError("--exam-id is required for exam mode.")

    system = ProctoringSystem(
        student_id=args.student_id,
        exam_id=args.exam_id,
        enable_face=not args.disable_face,
        enable_liveness=not args.disable_liveness,
        enable_eye=not args.disable_eye,
        enable_object=not args.disable_object,
        enable_audio=not args.disable_audio,
    )

    system.run()


def run_liveness_test():
    """
    Run liveness-only test.
    """
    from face_detection.test_liveness import run_test
    run_test()


def run_face_identity_test(args):
    """
    Run face identity-only test.
    """
    from face_detection.test_deepface_authentication import run_deepface_authentication
    run_deepface_authentication(student_id=args.student_id)


def run_eye_test():
    """
    Run eye tracking-only test.
    """
    from eye_tracking.test_eye_tracking import run_eye_tracking_test
    run_eye_tracking_test()


def run_object_test():
    """
    Run object detection-only test.
    """
    from object_detection.test_object_detection import run_object_detection_test
    run_object_detection_test()


def run_audio_test():
    """
    Run audio detection-only test.
    """
    from audio_detection.test_audio_detection import run_audio_detection_test
    run_audio_detection_test()


def build_parser():
    parser = argparse.ArgumentParser(
        description="AI Exam Proctoring System CLI"
    )

    parser.add_argument(
        "--mode",
        choices=[
            "exam",
            "liveness-test",
            "face-test",
            "eye-test",
            "object-test",
            "audio-test",
        ],
        default="exam",
        help="Choose what to run. Default: exam"
    )

    parser.add_argument(
        "--student-id",
        help="Student ID, example STU001. Required for exam mode and face-test mode."
    )

    parser.add_argument(
        "--exam-id",
        help="Exam ID, example EXAM_001. Required for exam mode."
    )

    parser.add_argument("--disable-face", action="store_true")
    parser.add_argument("--disable-liveness", action="store_true")
    parser.add_argument("--disable-eye", action="store_true")
    parser.add_argument("--disable-object", action="store_true")
    parser.add_argument("--disable-audio", action="store_true")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.mode == "exam":
            run_full_exam(args)

        elif args.mode == "liveness-test":
            run_liveness_test()

        elif args.mode == "face-test":
            run_face_identity_test(args)

        elif args.mode == "eye-test":
            run_eye_test()

        elif args.mode == "object-test":
            run_object_test()

        elif args.mode == "audio-test":
            run_audio_test()

    except Exception as error:
        print("Error:", error)


if __name__ == "__main__":
    main()
    
    
    
    
    