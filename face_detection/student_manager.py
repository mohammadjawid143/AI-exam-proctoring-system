import os
import json
from datetime import datetime


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASET_DIR = os.path.join(BASE_DIR, "dataset", "students")
DATABASE_DIR = os.path.join(BASE_DIR, "database")
STUDENTS_JSON_PATH = os.path.join(DATABASE_DIR, "students.json")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "exams")


def create_base_folders():
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(DATABASE_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    if not os.path.exists(STUDENTS_JSON_PATH):
        with open(STUDENTS_JSON_PATH, "w", encoding="utf-8") as file:
            json.dump([], file, indent=4, ensure_ascii=False)


def load_students():
    create_base_folders()

    try:
        with open(STUDENTS_JSON_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        return []

    except json.JSONDecodeError:
        return []


def save_students(students):
    create_base_folders()

    with open(STUDENTS_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(students, file, indent=4, ensure_ascii=False)


def student_exists(student_id):
    students = load_students()

    for student in students:
        if student.get("student_id") == student_id:
            return True

    return False


def create_student_folder(class_name, student_id):
    student_folder = os.path.join(DATASET_DIR, class_name, student_id)
    os.makedirs(student_folder, exist_ok=True)
    return student_folder


def get_student_folder(class_name, student_id):
    return os.path.join(DATASET_DIR, class_name, student_id)


def get_student_profile_path(class_name, student_id):
    student_folder = get_student_folder(class_name, student_id)
    return os.path.join(student_folder, "profile.jpg")


def register_student_info(student_id, full_name, class_name, profile_image_path):
    students = load_students()

    if student_exists(student_id):
        print(f"Student with ID {student_id} already exists.")
        return False

    student_data = {
        "student_id": student_id,
        "full_name": full_name,
        "class_name": class_name,
        "profile_image_path": profile_image_path,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    students.append(student_data)
    save_students(students)

    print("Student information saved successfully.")
    return True


def find_student_by_id(student_id):
    students = load_students()

    for student in students:
        if student.get("student_id") == student_id:
            return student

    return None


def create_exam_report_folder(exam_id, student_id):
    folder_path = os.path.join(REPORTS_DIR, exam_id, student_id)
    screenshots_path = os.path.join(folder_path, "screenshots")

    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(screenshots_path, exist_ok=True)

    return folder_path