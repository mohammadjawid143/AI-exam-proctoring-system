import cv2

# باز کردن وب‌کم (0 معمولاً وب‌کم پیش‌فرض است)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame (stream end?). Exiting ...")
        break

    # نمایش تصویر وب‌کم
    cv2.imshow('Webcam Test', frame)

    # خروج با زدن دکمه q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# آزاد کردن منابع
cap.release()
cv2.destroyAllWindows()