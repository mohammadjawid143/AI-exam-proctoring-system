import time
import threading
from collections import deque

import numpy as np
import sounddevice as sd
import webrtcvad


class NoiseDetector:
    def __init__(
        self,
        sample_rate=48000,
        frame_duration_ms=30,
        vad_mode=2,
        calibration_seconds=3,
        warning_seconds=2.5,
        speech_ratio_threshold=0.55,
        min_volume_threshold=0.015,
        device=None
    ):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.device = device

        self.block_size = int(self.sample_rate * self.frame_duration_ms / 1000)

        self.vad = webrtcvad.Vad(vad_mode)

        self.calibration_seconds = calibration_seconds
        self.warning_seconds = warning_seconds
        self.speech_ratio_threshold = speech_ratio_threshold
        self.min_volume_threshold = min_volume_threshold

        self.current_volume = 0.0
        self.current_db = -100.0

        self.background_samples = []
        self.calibrated = False
        self.calibration_start_time = None
        self.adaptive_volume_threshold = min_volume_threshold

        self.speech_frames = deque(maxlen=50)

        self.speech_start_time = None
        self.warning = False
        self.status = "Audio not started"

        self.running = False
        self.stream = None
        self.lock = threading.Lock()

    def _float_to_pcm16(self, audio_float):
        audio_float = np.clip(audio_float, -1.0, 1.0)
        audio_int16 = (audio_float * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def _calculate_volume(self, audio_float):
        volume = float(np.sqrt(np.mean(audio_float ** 2)))
        db = 20 * np.log10(volume + 1e-6)
        return volume, db

    def _reset_state(self):
        self.background_samples = []
        self.calibrated = False
        self.calibration_start_time = None
        self.adaptive_volume_threshold = self.min_volume_threshold

        self.speech_frames.clear()
        self.speech_start_time = None

        self.current_volume = 0.0
        self.current_db = -100.0

        self.warning = False
        self.status = "Audio not started"

    def _calibrate(self, volume):
        if self.calibration_start_time is None:
            self.calibration_start_time = time.time()

        self.background_samples.append(volume)

        elapsed = time.time() - self.calibration_start_time

        if elapsed >= self.calibration_seconds:
            bg_mean = float(np.mean(self.background_samples))
            bg_std = float(np.std(self.background_samples))

            self.adaptive_volume_threshold = max(
                self.min_volume_threshold,
                bg_mean + (3.0 * bg_std),
                bg_mean * 2.5
            )

            self.calibrated = True
            self.status = "Audio OK"

            print("Audio calibration completed.")
            print("Background mean:", bg_mean)
            print("Adaptive threshold:", self.adaptive_volume_threshold)

    def audio_callback(self, indata, frames, time_info, status):
        audio_float = indata[:, 0].astype(np.float32)

        volume, db = self._calculate_volume(audio_float)

        try:
            pcm_bytes = self._float_to_pcm16(audio_float)
            is_speech = self.vad.is_speech(pcm_bytes, self.sample_rate)
        except Exception:
            is_speech = False

        with self.lock:
            self.current_volume = volume
            self.current_db = db

            if not self.calibrated:
                self._calibrate(volume)
                self.warning = False
                self.status = "Calibrating audio... keep silent"
                return

            volume_is_enough = volume >= self.adaptive_volume_threshold
            self.speech_frames.append(is_speech and volume_is_enough)

            if len(self.speech_frames) == 0:
                speech_ratio = 0.0
            else:
                speech_ratio = sum(self.speech_frames) / len(self.speech_frames)

            speech_detected = speech_ratio >= self.speech_ratio_threshold

            if speech_detected:
                if self.speech_start_time is None:
                    self.speech_start_time = time.time()

                elapsed = time.time() - self.speech_start_time

                if elapsed >= self.warning_seconds:
                    self.warning = True
                    self.status = "Audio warning: human voice detected"
                else:
                    self.warning = False
                    self.status = "Voice detected"

            else:
                self.speech_start_time = None
                self.warning = False

                if volume >= self.adaptive_volume_threshold * 1.8:
                    self.status = "Loud noise detected"
                else:
                    self.status = "Audio OK"

    def start(self):
        try:
            self._reset_state()

            self.stream = sd.InputStream(
                device=self.device,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                dtype="float32",
                callback=self.audio_callback
            )

            self.stream.start()

            self.running = True
            self.status = "Calibrating audio... keep silent"

            print("Audio monitoring started.")
            print("Audio device:", self.device)
            print("Sample rate:", self.sample_rate)
            print("Please stay silent for calibration...")

        except Exception as error:
            self.running = False
            self.warning = False
            self.status = f"Audio error: {str(error)}"

            print("Audio monitoring error:", error)

    def stop(self):
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            self.running = False
            self.status = "Audio monitoring stopped"

            print("Audio monitoring stopped.")

        except Exception as error:
            print("Audio stop error:", error)

    def get_status(self):
        with self.lock:
            return {
                "running": self.running,
                "volume": self.current_volume,
                "db": self.current_db,
                "warning": self.warning,
                "status": self.status,
                "calibrated": self.calibrated,
                "adaptive_threshold": self.adaptive_volume_threshold
            }