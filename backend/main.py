import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import sys
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports" / "exams"

exam_process = None


class StartExamRequest(BaseModel):
    student_id: str
    exam_id: str
    enable_face: bool = True

app = FastAPI(
    title="AI Exam Proctoring Backend",
    description="FastAPI backend for exam proctoring reports and dashboard",
    version="1.0.0"
)

templates = Jinja2Templates(directory=str(BASE_DIR / "backend" / "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def safe_path(path: Path) -> Path:
    """
    جلوگیری از دسترسی به فایل‌های بیرون از reports
    """
    resolved_path = path.resolve()
    reports_root = REPORTS_DIR.resolve()

    if not str(resolved_path).startswith(str(reports_root)):
        raise HTTPException(status_code=403, detail="Access denied")

    return resolved_path


def get_log_path(exam_id: str, student_id: str) -> Path:
    return REPORTS_DIR / exam_id / student_id / "log.txt"


def get_screenshots_dir(exam_id: str, student_id: str) -> Path:
    return REPORTS_DIR / exam_id / student_id / "screenshots"


def read_log_file(exam_id: str, student_id: str) -> str:
    log_path = safe_path(get_log_path(exam_id, student_id))

    if not log_path.exists():
        return ""

    with open(log_path, "r", encoding="utf-8") as file:
        return file.read()


def count_alerts(log_text: str) -> dict:
    return {
        "system_start": log_text.count("SYSTEM_START"),
        "system_end": log_text.count("SYSTEM_END"),
        "eye_warnings": log_text.count("EYE_WARNING"),
        "object_warnings": log_text.count("OBJECT_WARNING"),
        "face_warnings": log_text.count("FACE_WARNING"),
        "face_errors": log_text.count("FACE_ERROR"),
        "system_errors": log_text.count("ERROR"),
        "total_warnings": (
            log_text.count("EYE_WARNING")
            + log_text.count("OBJECT_WARNING")
            + log_text.count("FACE_WARNING")
            + log_text.count("FACE_ERROR")
        )
    }


def get_risk_level(total_warnings: int) -> str:
    if total_warnings == 0:
        return "Normal"
    elif total_warnings <= 3:
        return "Low Risk"
    elif total_warnings <= 7:
        return "Medium Risk"
    else:
        return "High Risk"


@app.get("/")
def home():
    return {
        "message": "AI Exam Proctoring FastAPI Backend is running",
        "dashboard": "/dashboard",
        "docs": "/docs"
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={}
    )


@app.get("/api/exams")
def get_exams():
    if not REPORTS_DIR.exists():
        return {
            "exams": []
        }

    exams = []

    for exam in os.listdir(REPORTS_DIR):
        exam_path = REPORTS_DIR / exam

        if exam_path.is_dir():
            exams.append(exam)

    return {
        "exams": exams
    }


@app.get("/api/exams/{exam_id}/students")
def get_students(exam_id: str):
    exam_path = safe_path(REPORTS_DIR / exam_id)

    if not exam_path.exists():
        raise HTTPException(status_code=404, detail="Exam not found")

    students = []

    for student in os.listdir(exam_path):
        student_path = exam_path / student

        if student_path.is_dir():
            students.append(student)

    return {
        "exam_id": exam_id,
        "students": students
    }


@app.get("/api/reports/{exam_id}/{student_id}")
def get_student_report(exam_id: str, student_id: str):
    student_report_path = safe_path(REPORTS_DIR / exam_id / student_id)

    if not student_report_path.exists():
        raise HTTPException(status_code=404, detail="Student report not found")

    log_text = read_log_file(exam_id, student_id)
    alert_counts = count_alerts(log_text)
    screenshots = get_student_screenshots(exam_id, student_id)

    risk_level = get_risk_level(alert_counts["total_warnings"])

    return {
        "exam_id": exam_id,
        "student_id": student_id,
        "risk_level": risk_level,
        "alert_counts": alert_counts,
        "log_text": log_text,
        "screenshots": screenshots["screenshots"]
    }


@app.get("/api/reports/{exam_id}/{student_id}/screenshots")
def get_student_screenshots(exam_id: str, student_id: str):
    screenshots_dir = safe_path(get_screenshots_dir(exam_id, student_id))

    if not screenshots_dir.exists():
        return {
            "exam_id": exam_id,
            "student_id": student_id,
            "screenshots": []
        }

    screenshots = []

    for filename in os.listdir(screenshots_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            screenshots.append({
                "filename": filename,
                "url": f"/api/screenshots/{exam_id}/{student_id}/{filename}"
            })

    return {
        "exam_id": exam_id,
        "student_id": student_id,
        "screenshots": screenshots
    }


@app.get("/api/screenshots/{exam_id}/{student_id}/{filename}")
def get_screenshot(exam_id: str, student_id: str, filename: str):
    image_path = safe_path(REPORTS_DIR / exam_id / student_id / "screenshots" / filename)

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Screenshot not found")

    return FileResponse(image_path)


@app.post("/api/proctoring/start")
def start_proctoring(request: StartExamRequest):
    global exam_process

    if exam_process is not None and exam_process.poll() is None:
        raise HTTPException(
            status_code=400,
            detail="An exam proctoring session is already running."
        )

    command = [
        sys.executable,
        str(BASE_DIR / "run_exam.py"),
        "--student-id",
        request.student_id,
        "--exam-id",
        request.exam_id
    ]

    if not request.enable_face:
        command.append("--disable-face")

    exam_process = subprocess.Popen(
        command,
        cwd=str(BASE_DIR)
    )

    return {
        "message": "Exam proctoring started",
        "student_id": request.student_id,
        "exam_id": request.exam_id,
        "enable_face": request.enable_face
    }


@app.post("/api/proctoring/stop")
def stop_proctoring():
    global exam_process

    if exam_process is None or exam_process.poll() is not None:
        return {
            "message": "No running exam proctoring session."
        }

    exam_process.terminate()
    exam_process = None

    return {
        "message": "Exam proctoring stopped."
    }


@app.get("/api/proctoring/status")
def proctoring_status():
    global exam_process

    if exam_process is not None and exam_process.poll() is None:
        return {
            "running": True
        }

    return {
        "running": False
    }