import cv2
import queue
import threading
import traceback
from datetime import datetime


class LatestFrameWorker:
    def __init__(self, name, process_func):
        self.name = name
        self.process_func = process_func

        self.frame_queue = queue.Queue(maxsize=1)
        self.result_lock = threading.Lock()

        self.latest_result = None
        self.latest_error = None

        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        if not self.thread.is_alive():
            self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)

    def submit(self, frame):
        if frame is None:
            return

        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            self.frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    def get_latest_result(self):
        with self.result_lock:
            return self.latest_result

    def get_latest_error(self):
        with self.result_lock:
            return self.latest_error

    def _run(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                result = self.process_func(frame)

                with self.result_lock:
                    self.latest_result = result
                    self.latest_error = None

            except Exception as error:
                with self.result_lock:
                    self.latest_error = {
                        "worker": self.name,
                        "error": str(error),
                        "traceback": traceback.format_exc()
                    }


class AsyncReportWorker:
    def __init__(self):
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        if not self.thread.is_alive():
            self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)

    def log(self, log_path, alert_type, message):
        self.queue.put({
            "type": "log",
            "log_path": log_path,
            "alert_type": alert_type,
            "message": message,
            "time": datetime.now()
        })

    def screenshot(self, image_path, frame):
        self.queue.put({
            "type": "screenshot",
            "image_path": image_path,
            "frame": frame.copy()
        })

    def _run(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if item["type"] == "log":
                    current_time = item["time"].strftime("%Y-%m-%d %H:%M:%S")

                    with open(item["log_path"], "a", encoding="utf-8") as file:
                        file.write(
                            f"[{current_time}] "
                            f"{item['alert_type']}: {item['message']}\n"
                        )

                    print(
                        f"[{current_time}] "
                        f"{item['alert_type']}: {item['message']}"
                    )

                elif item["type"] == "screenshot":
                    cv2.imwrite(
                        item["image_path"],
                        item["frame"],
                        [cv2.IMWRITE_JPEG_QUALITY, 85]
                    )

            except Exception as error:
                print("AsyncReportWorker error:", error)