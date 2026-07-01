import cv2

from face_detection.student_manager import (
    create_base_folders,
    create_student_folder,
    register_student_info,
    get_student_profile_path,
    student_exists
)


def register_student():
    create_base_folders()

    print("===== Student Registration =====")

    student_id = input("Enter student ID, example STU001: ").strip()
    full_name = input("Enter full name: ").strip()
    class_name = input("Enter class name, example class_01: ").strip()

    if not student_id or not full_name or not class_name:
        print("Error: student ID, full name, and class name are required.")
        return

    if student_exists(student_id):
        print(f"Error: Student with ID {student_id} already exists.")
        return

    student_folder = create_student_folder(class_name, student_id)
    profile_image_path = get_student_profile_path(class_name, student_id)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    print("Camera opened successfully.")
    print("Look at the camera clearly.")
    print("Press 's' to save profile image.")
    print("Press 'q' to cancel.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame from camera.")
            break

        cv2.putText(
            frame,
            f"Registering: {student_id} | Press s to save",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Register Student", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            cv2.imwrite(profile_image_path, frame)
            print(f"Profile image saved: {profile_image_path}")

            register_student_info(
                student_id=student_id,
                full_name=full_name,
                class_name=class_name,
                profile_image_path=profile_image_path
            )

            print(f"Student folder created: {student_folder}")
            break

        elif key == ord("q"):
            print("Registration cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    register_student()