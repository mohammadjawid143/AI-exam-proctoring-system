import argparse
from core.proctoring_system import ProctoringSystem


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--student-id", required=True)
    parser.add_argument("--exam-id", required=True)

    parser.add_argument("--disable-face", action="store_true")
    parser.add_argument("--disable-eye", action="store_true")
    parser.add_argument("--disable-object", action="store_true")
    parser.add_argument("--disable-audio", action="store_true")

    args = parser.parse_args()

    system = ProctoringSystem(
        student_id=args.student_id,
        exam_id=args.exam_id,
        enable_face=not args.disable_face,
        enable_eye=not args.disable_eye,
        enable_object=not args.disable_object,
        enable_audio=not args.disable_audio,
    )

    system.run()


if __name__ == "__main__":
    main()