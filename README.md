# AI Exam Proctoring System

An AI-powered online exam proctoring system designed to monitor students during virtual exams using computer vision, machine learning, and audio analysis.

This project combines face verification, CNN-based liveness detection, eye and gaze tracking, suspicious object detection, audio monitoring, real-time alerts, screenshots, logs, and exam violation reports.

---

## Features

### Face Verification

* Student identity verification using DeepFace.
* ArcFace-based face recognition.
* Compares the live camera frame with the registered student profile image.
* Detects unknown or mismatched faces.

### Liveness Detection

* CNN-based liveness detection using a trained `.h5` model.
* Detects whether the face is real or spoofed.
* Supports blink challenge to reduce photo and phone replay attacks.
* Shows simple liveness status: `Live` or `Spoof`.

### Eye and Gaze Tracking

* Detects student gaze direction.
* Detects suspicious eye movement such as looking left, right, up, or down.
* Detects when the face is missing or the student looks away for too long.

### Object Detection

* YOLO-based object detection.
* Detects suspicious objects such as:

  * Mobile phone
  * Book
  * Multiple persons
  * Other prohibited objects

### Audio Monitoring

* Microphone-based audio monitoring.
* Detects human voice, talking, whispering, or suspicious background noise.
* Uses volume, dB, and voice activity detection logic.

### Alerts and Reports

* Real-time warning system.
* Timestamped logs.
* Violation screenshots.
* Exam report folder for every student and exam session.

---

## Project Structure

```text
ai_exam_proctoring/
│
├── audio_detection/
│   ├── noise_detector.py
│   └── test_audio_detection.py
│
├── backend/
│   └── ...
│
├── core/
│   └── proctoring_system.py
│
├── eye_tracking/
│   ├── eye_tracker.py
│   └── test_eye_tracking.py
│
├── face_detection/
│   ├── cnn_liveness_detector.py
│   ├── deepface_auth.py
│   ├── register_student.py
│   ├── student_manager.py
│   ├── test_liveness.py
│   └── models/
│       └── model.h5
│
├── object_detection/
│   ├── cheating_object_detector.py
│   └── test_object_detection.py
│
├── reports/
│   └── exams/
│
├── main.py
├── run_exam.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Technologies Used

* Python
* OpenCV
* MediaPipe
* DeepFace
* TensorFlow / Keras
* YOLO / Ultralytics
* NumPy
* SoundDevice
* WebRTC VAD
* FastAPI backend
* SQLite / database support
* HTML / PDF-style reporting

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/mohammadjawid143/AI-exam-proctoring-system.git
cd AI-exam-proctoring-system
```

### 2. Create and activate virtual environment

For Windows Git Bash:

```bash
python -m venv env
source env/Scripts/activate
```

For Windows PowerShell:

```powershell
python -m venv env
.\env\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If some packages are missing, install them manually:

```bash
pip install opencv-python numpy tensorflow deepface mediapipe ultralytics sounddevice webrtcvad-wheels
```

---

## Model Files

Model files are not included in the GitHub repository because they are large.

Place the liveness model here:

```text
face_detection/models/model.h5
```

Place YOLO model files in the project root or models directory:

```text
yolo11n.pt
```

Example:

```text
ai_exam_proctoring/
├── face_detection/
│   └── models/
│       └── model.h5
├── yolo11n.pt
```

---

## How to Run

Start the main application:

```bash
python main.py
```

The menu includes:

```text
1. Register new student
2. Start full exam proctoring
3. Test face recognition + CNN liveness
4. Test eye tracking
5. Test cheating object detection
6. Test audio/noise detection
0. Exit
```

---

## Testing Individual Modules

### Test Liveness Detection

```bash
python -m face_detection.test_liveness
```

### Test Eye Tracking

```bash
python -m eye_tracking.test_eye_tracking
```

### Test Object Detection

```bash
python -m object_detection.test_object_detection
```

### Test Audio Detection

```bash
python -m audio_detection.test_audio_detection
```

---

## Exam Workflow

1. Register the student with a profile image.
2. Start the exam proctoring system.
3. Enter student ID and exam ID.
4. The system starts webcam and microphone monitoring.
5. The system checks:

   * Face identity
   * Liveness
   * Eye movement
   * Suspicious objects
   * Audio activity
6. Alerts are logged with timestamps.
7. Screenshots are saved for suspicious activity.
8. Reports are generated inside the `reports/` directory.

---

## Report Output

Exam reports are saved in:

```text
reports/exams/<exam_id>/<student_id>/
```

Example:

```text
reports/exams/EXAM_001/STU001/
```

This folder may contain:

```text
log.txt
screenshots/
violation images
exam activity data
```

---

## Important Notes

This project is designed as an academic and research prototype for AI-based online exam proctoring.

For real university-level deployment, the system should be improved with:

* Better privacy and security controls
* Student consent and policy documentation
* Secure storage of biometric data
* LMS integration
* Stronger liveness detection
* Human review for flagged violations
* Fairness and bias testing
* Production-level authentication and encryption

---

## Privacy Notice

This system uses webcam, microphone, face recognition, and monitoring features. It should only be used with clear user consent and according to institutional rules, privacy laws, and academic integrity policies.

---

## Limitations

* Liveness detection may not be perfect against all spoofing attacks.
* Object detection accuracy depends on the YOLO model.
* Audio detection may be affected by background noise.
* Lighting, camera quality, and face angle can affect recognition accuracy.
* Final exam violation decisions should be reviewed by a human supervisor.

---

## Future Improvements

* LMS integration
* Admin dashboard
* Real-time web backend
* PDF report generation
* Browser lockdown support
* Screen recording
* Better anti-spoofing models
* Multi-student exam session management
* Cloud deployment support

---

## Repository Description

AI-based online exam proctoring system with face verification, liveness detection, gaze tracking, object detection, audio monitoring, and violation reports.

---

## Author

Developed by Mohammad Jawid.

---

## License

This project is for educational and academic use.
