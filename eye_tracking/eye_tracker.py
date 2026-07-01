import cv2
import mediapipe as mp
import numpy as np


class EyeTracker:
    def __init__(self, config):
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.config = config
        eyes_config = config.get("detection", {}).get("eyes", {})

        # آستانه تشخیص حرکت عنبیه
        self.iris_horizontal_threshold = eyes_config.get("iris_horizontal_threshold", 0.15)
        self.iris_vertical_threshold = eyes_config.get("iris_vertical_threshold", 0.14)

        # آستانه تشخیص چرخش سر نسبت به بینی
        self.head_horizontal_threshold = eyes_config.get("head_horizontal_threshold", 18)
        self.head_vertical_threshold = eyes_config.get("head_vertical_threshold", 15)

        # تشخیص بسته بودن چشم
        self.closed_eye_threshold = eyes_config.get("closed_eye_threshold", 0.18)

        # تعداد فریم برای کالیبراسیون
        self.calibration_frames = eyes_config.get("calibration_frames", 50)

        # تنظیمات نمایش
        self.draw = eyes_config.get("draw", True)
        self.show_debug_text = eyes_config.get("show_debug_text", False)
        self.iris_point_radius = eyes_config.get("iris_point_radius", 2)

        # اگر جهت‌ها برعکس شد، این‌ها را تغییر بده
        self.mirror_output = eyes_config.get("mirror_output", False)
        self.invert_head_vertical = eyes_config.get("invert_head_vertical", False)
        self.invert_head_horizontal = eyes_config.get("invert_head_horizontal", False)

        # تنظیمات ضد لرزش
        self.smoothing_alpha = eyes_config.get("smoothing_alpha", 0.20)
        self.stable_frames_required = eyes_config.get("stable_frames_required", 7)

        self.gaze_direction = "center"
        self.iris_direction = "center"
        self.head_direction = "center"
        self.eye_ratio = 0.3
        self.alert_logger = None

        # نقاط برای محاسبه باز یا بسته بودن چشم
        self.LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]

        # نقاط کامل دور چشم
        self.LEFT_EYE_FULL = [
            362, 382, 381, 380, 374, 373, 390, 249,
            263, 466, 388, 387, 386, 385, 384, 398
        ]

        self.RIGHT_EYE_FULL = [
            33, 7, 163, 144, 145, 153, 154, 155,
            133, 173, 157, 158, 159, 160, 161, 246
        ]

        # نقاط عنبیه در MediaPipe
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]

        # نقطه بینی
        self.NOSE_TIP = 4

        # مقادیر کالیبراسیون عنبیه
        self.calibration_count = 0

        self.iris_horizontal_values = []
        self.iris_vertical_values = []

        self.head_horizontal_values = []
        self.head_vertical_values = []

        self.base_iris_horizontal = None
        self.base_iris_vertical = None

        self.base_head_horizontal = None
        self.base_head_vertical = None

        # متغیرهای نرم‌سازی
        self.smoothed_iris_horizontal = None
        self.smoothed_iris_vertical = None

        self.smoothed_head_horizontal = None
        self.smoothed_head_vertical = None

        self.left_iris_center_smooth = None
        self.right_iris_center_smooth = None

        # متغیرهای پایدارسازی متن
        self.candidate_gaze = "center"
        self.candidate_count = 0

    def set_alert_logger(self, alert_logger):
        self.alert_logger = alert_logger

    def reset_calibration(self):
        """
        با زدن کلید r می‌توانی دوباره کالیبراسیون کنی.
        """
        self.calibration_count = 0

        self.iris_horizontal_values = []
        self.iris_vertical_values = []

        self.head_horizontal_values = []
        self.head_vertical_values = []

        self.base_iris_horizontal = None
        self.base_iris_vertical = None

        self.base_head_horizontal = None
        self.base_head_vertical = None

        self.smoothed_iris_horizontal = None
        self.smoothed_iris_vertical = None

        self.smoothed_head_horizontal = None
        self.smoothed_head_vertical = None

        self.left_iris_center_smooth = None
        self.right_iris_center_smooth = None

        self.gaze_direction = "center"
        self.iris_direction = "center"
        self.head_direction = "center"

        self.candidate_gaze = "center"
        self.candidate_count = 0

    def _landmark_to_point(self, landmark, frame_w, frame_h):
        return np.array([
            int(landmark.x * frame_w),
            int(landmark.y * frame_h)
        ])

    def _get_points(self, face_landmarks, indices, frame_w, frame_h):
        return np.array([
            self._landmark_to_point(face_landmarks.landmark[i], frame_w, frame_h)
            for i in indices
        ])

    def _calculate_ear(self, eye_points):
        """
        EAR = Eye Aspect Ratio
        برای تشخیص باز یا بسته بودن چشم.
        """
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])

        if C == 0:
            return 0.0

        return (A + B) / (2.0 * C)

    def _get_iris_position_ratio(self, eye_points, iris_points):
        """
        موقعیت عنبیه داخل محدوده چشم را حساب می‌کند.
        """
        x, y, w, h = cv2.boundingRect(eye_points)

        if w == 0 or h == 0:
            return 0.5, 0.5, None

        iris_center = np.mean(iris_points, axis=0).astype(int)

        x_ratio = (iris_center[0] - x) / w
        y_ratio = (iris_center[1] - y) / h

        return x_ratio, y_ratio, iris_center

    def _smooth_value(self, new_value, old_value):
        """
        نرم کردن عددها برای کم شدن لرزش.
        """
        if old_value is None:
            return new_value

        return (
            self.smoothing_alpha * new_value
            + (1 - self.smoothing_alpha) * old_value
        )

    def _smooth_point(self, new_point, old_point):
        """
        نرم کردن نقطه عنبیه.
        """
        if new_point is None:
            return None

        if old_point is None:
            return new_point.astype(int)

        smooth = (
            self.smoothing_alpha * new_point
            + (1 - self.smoothing_alpha) * old_point
        )

        return smooth.astype(int)

    def _make_gaze_stable(self, raw_gaze):
        """
        پایدار کردن متن.
        یعنی اگر فقط یک یا دو فریم اشتباه شد، متن سریع تغییر نکند.
        """
        if raw_gaze == self.candidate_gaze:
            self.candidate_count += 1
        else:
            self.candidate_gaze = raw_gaze
            self.candidate_count = 1

        if self.candidate_count >= self.stable_frames_required:
            self.gaze_direction = self.candidate_gaze

        return self.gaze_direction

    def _get_head_values(self, face_landmarks, frame_w, frame_h, left_eye_points, right_eye_points):
        """
        چرخش سر را با فاصله مرکز چشم‌ها تا نوک بینی حساب می‌کند.
        این همان منطق کد قبلی تو است.
        """
        left_eye_center = np.mean(left_eye_points, axis=0)
        right_eye_center = np.mean(right_eye_points, axis=0)

        eye_center = (left_eye_center + right_eye_center) / 2.0

        nose_tip = self._landmark_to_point(
            face_landmarks.landmark[self.NOSE_TIP],
            frame_w,
            frame_h
        )

        head_horizontal = eye_center[0] - nose_tip[0]
        head_vertical = eye_center[1] - nose_tip[1]

        return head_horizontal, head_vertical, eye_center, nose_tip

    def _detect_iris_direction(self, iris_horizontal_change, iris_vertical_change):
        """
        تشخیص جهت نگاه از روی عنبیه.
        در این نسخه عنبیه بیشتر برای left/right استفاده می‌شود.
        """
        direction = "center"

        if iris_horizontal_change > self.iris_horizontal_threshold:
            direction = "right"
        elif iris_horizontal_change < -self.iris_horizontal_threshold:
            direction = "left"

        # بالا/پایین از عنبیه دقیق نبود، پس اولویت اصلی را به سر می‌دهیم.
        # این بخش را نگه داشتیم، ولی در تصمیم نهایی اولویت پایین‌تر دارد.
        elif iris_vertical_change > self.iris_vertical_threshold:
            direction = "down"
        elif iris_vertical_change < -self.iris_vertical_threshold:
            direction = "up"

        if self.mirror_output:
            if direction == "left":
                direction = "right"
            elif direction == "right":
                direction = "left"

        return direction

    def _detect_head_direction(self, head_horizontal_change, head_vertical_change):
        """
        تشخیص جهت سر با توجه به فاصله مرکز چشم‌ها تا بینی.
        """
        direction = "center"

        # بالا و پایین را از چرخش سر می‌گیریم
        if head_vertical_change < -self.head_vertical_threshold:
            direction = "down"
        elif head_vertical_change > self.head_vertical_threshold:
            direction = "up"

        # چپ و راست سر
        elif head_horizontal_change > self.head_horizontal_threshold:
            direction = "left"
        elif head_horizontal_change < -self.head_horizontal_threshold:
            direction = "right"

        if self.invert_head_vertical:
            if direction == "up":
                direction = "down"
            elif direction == "down":
                direction = "up"

        if self.invert_head_horizontal:
            if direction == "left":
                direction = "right"
            elif direction == "right":
                direction = "left"

        if self.mirror_output:
            if direction == "left":
                direction = "right"
            elif direction == "right":
                direction = "left"

        return direction

    def _combine_iris_and_head(self, iris_direction, head_direction):
        """
        ترکیب حرکت عنبیه و چرخش سر.

        منطق:
        - بالا/پایین را بیشتر از چرخش سر می‌گیریم.
        - چپ/راست را اول از عنبیه می‌گیریم.
        - اگر عنبیه center بود ولی سر چرخید، از جهت سر استفاده می‌کنیم.
        """

        # بالا و پایین با چرخش سر دقیق‌تر است
        if head_direction in ["up", "down"]:
            return head_direction

        # چپ و راست از عنبیه
        if iris_direction in ["left", "right"]:
            return iris_direction

        # اگر سر چپ یا راست چرخید
        if head_direction in ["left", "right"]:
            return head_direction

        # اگر خواستی بالا/پایین عنبیه هم لحاظ شود
        if iris_direction in ["up", "down"]:
            return iris_direction

        return "center"

    def _draw_iris_point(self, frame, iris_center):
        """
        فقط نقطه کوچک عنبیه را رسم می‌کند.
        """
        if iris_center is not None:
            cv2.circle(
                frame,
                tuple(iris_center),
                self.iris_point_radius,
                (0, 0, 255),
                -1
            )

    def track_eyes(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if not results.multi_face_landmarks:
                self.gaze_direction = "no_face"

                if self.draw:
                    self._draw_text(frame, self.gaze_direction, self.eye_ratio)

                return self.gaze_direction, self.eye_ratio

            face_landmarks = results.multi_face_landmarks[0]
            frame_h, frame_w = frame.shape[:2]

            if len(face_landmarks.landmark) < 478:
                self.gaze_direction = "iris_not_found"

                if self.draw:
                    self._draw_text(frame, self.gaze_direction, self.eye_ratio)

                return self.gaze_direction, self.eye_ratio

            # نقاط چشم برای EAR
            left_eye_ear_points = self._get_points(
                face_landmarks,
                self.LEFT_EYE_EAR,
                frame_w,
                frame_h
            )

            right_eye_ear_points = self._get_points(
                face_landmarks,
                self.RIGHT_EYE_EAR,
                frame_w,
                frame_h
            )

            left_ear = self._calculate_ear(left_eye_ear_points)
            right_ear = self._calculate_ear(right_eye_ear_points)

            self.eye_ratio = (left_ear + right_ear) / 2.0

            if self.eye_ratio < self.closed_eye_threshold:
                stable_gaze = self._make_gaze_stable("eyes_closed")

                if self.draw:
                    self._draw_text(frame, stable_gaze, self.eye_ratio)

                return stable_gaze, self.eye_ratio

            # نقاط کامل چشم
            left_eye_full_points = self._get_points(
                face_landmarks,
                self.LEFT_EYE_FULL,
                frame_w,
                frame_h
            )

            right_eye_full_points = self._get_points(
                face_landmarks,
                self.RIGHT_EYE_FULL,
                frame_w,
                frame_h
            )

            # نقاط عنبیه
            left_iris_points = self._get_points(
                face_landmarks,
                self.LEFT_IRIS,
                frame_w,
                frame_h
            )

            right_iris_points = self._get_points(
                face_landmarks,
                self.RIGHT_IRIS,
                frame_w,
                frame_h
            )

            left_x_ratio, left_y_ratio, left_iris_center = self._get_iris_position_ratio(
                left_eye_full_points,
                left_iris_points
            )

            right_x_ratio, right_y_ratio, right_iris_center = self._get_iris_position_ratio(
                right_eye_full_points,
                right_iris_points
            )

            iris_horizontal = (left_x_ratio + right_x_ratio) / 2.0
            iris_vertical = (left_y_ratio + right_y_ratio) / 2.0

            # گرفتن مقدار چرخش سر نسبت به بینی
            head_horizontal, head_vertical, eye_center, nose_tip = self._get_head_values(
                face_landmarks,
                frame_w,
                frame_h,
                left_eye_ear_points,
                right_eye_ear_points
            )

            # نرم کردن مقدار عنبیه
            iris_horizontal = self._smooth_value(
                iris_horizontal,
                self.smoothed_iris_horizontal
            )

            iris_vertical = self._smooth_value(
                iris_vertical,
                self.smoothed_iris_vertical
            )

            self.smoothed_iris_horizontal = iris_horizontal
            self.smoothed_iris_vertical = iris_vertical

            # نرم کردن مقدار سر
            head_horizontal = self._smooth_value(
                head_horizontal,
                self.smoothed_head_horizontal
            )

            head_vertical = self._smooth_value(
                head_vertical,
                self.smoothed_head_vertical
            )

            self.smoothed_head_horizontal = head_horizontal
            self.smoothed_head_vertical = head_vertical

            # نرم کردن نقطه عنبیه
            left_iris_center = self._smooth_point(
                left_iris_center,
                self.left_iris_center_smooth
            )

            right_iris_center = self._smooth_point(
                right_iris_center,
                self.right_iris_center_smooth
            )

            self.left_iris_center_smooth = left_iris_center
            self.right_iris_center_smooth = right_iris_center

            # کالیبراسیون
            if self.calibration_count < self.calibration_frames:
                self.iris_horizontal_values.append(iris_horizontal)
                self.iris_vertical_values.append(iris_vertical)

                self.head_horizontal_values.append(head_horizontal)
                self.head_vertical_values.append(head_vertical)

                self.calibration_count += 1

                self.gaze_direction = (
                    f"calibrating {self.calibration_count}/{self.calibration_frames}"
                )

                if self.draw:
                    self._draw_iris_point(frame, left_iris_center)
                    self._draw_iris_point(frame, right_iris_center)
                    self._draw_text(frame, self.gaze_direction, self.eye_ratio)

                return self.gaze_direction, self.eye_ratio

            # ساخت مقدار پایه بعد از کالیبراسیون
            if (
                self.base_iris_horizontal is None
                or self.base_iris_vertical is None
                or self.base_head_horizontal is None
                or self.base_head_vertical is None
            ):
                self.base_iris_horizontal = sum(self.iris_horizontal_values) / len(self.iris_horizontal_values)
                self.base_iris_vertical = sum(self.iris_vertical_values) / len(self.iris_vertical_values)

                self.base_head_horizontal = sum(self.head_horizontal_values) / len(self.head_horizontal_values)
                self.base_head_vertical = sum(self.head_vertical_values) / len(self.head_vertical_values)

                self.gaze_direction = "center"

                if self.draw:
                    self._draw_iris_point(frame, left_iris_center)
                    self._draw_iris_point(frame, right_iris_center)
                    self._draw_text(frame, self.gaze_direction, self.eye_ratio)

                return self.gaze_direction, self.eye_ratio

            # تغییرات نسبت به حالت مستقیم
            iris_horizontal_change = iris_horizontal - self.base_iris_horizontal
            iris_vertical_change = iris_vertical - self.base_iris_vertical

            head_horizontal_change = head_horizontal - self.base_head_horizontal
            head_vertical_change = head_vertical - self.base_head_vertical

            # تشخیص جداگانه
            self.iris_direction = self._detect_iris_direction(
                iris_horizontal_change,
                iris_vertical_change
            )

            self.head_direction = self._detect_head_direction(
                head_horizontal_change,
                head_vertical_change
            )

            # ترکیب عنبیه و چرخش سر
            raw_gaze = self._combine_iris_and_head(
                self.iris_direction,
                self.head_direction
            )

            # پایدارسازی نهایی
            stable_gaze = self._make_gaze_stable(raw_gaze)

            if self.draw:
                self._draw_iris_point(frame, left_iris_center)
                self._draw_iris_point(frame, right_iris_center)
                self._draw_text(frame, stable_gaze, self.eye_ratio)

            return stable_gaze, self.eye_ratio

        except Exception as e:
            if self.alert_logger:
                self.alert_logger.log_alert(
                    "EYE_TRACKING_ERROR",
                    f"Error in eye tracking: {str(e)}"
                )

            return self.gaze_direction, self.eye_ratio