import time

from audio_detection.noise_detector import NoiseDetector


def run_audio_detection_test():
    detector = NoiseDetector(
        sample_rate=16000,
        frame_duration_ms=30,
        vad_mode=2,
        calibration_seconds=3,
        warning_seconds=3,
        speech_ratio_threshold=0.60,
        min_volume_threshold=0.015
    )

    detector.start()

    print("Audio detection test started.")
    print("Please stay silent for first 3 seconds for calibration.")
    print("Then speak near the microphone.")
    print("Press Ctrl + C to stop.")

    try:
        while True:
            result = detector.get_status()

            print(
                f"Status: {result['status']} | "
                f"Volume: {result['volume']:.4f} | "
                f"dB: {result['db']:.2f} | "
                f"Warning: {result['warning']} | "
                f"Calibrated: {result['calibrated']} | "
                f"Threshold: {result['adaptive_threshold']:.4f}"
            )

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping audio test...")

    finally:
        detector.stop()


if __name__ == "__main__":
    run_audio_detection_test()