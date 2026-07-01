import argparse
from core.proctoring_system import ProctoringSystem


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--student-id", required=True)
    parser.add_argument("--exam-id", required=True)
    parser.add_argument("--disable-face", action="store_true")

    args = parser.parse_args()

    system = ProctoringSystem(
        student_id=args.student_id,
        exam_id=args.exam_id,
        enable_face=not args.disable_face
    )

    system.run()


if __name__ == "__main__":
    main()