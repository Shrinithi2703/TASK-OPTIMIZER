import cv2

def capture_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise Exception("Failed to capture frame from webcam.")
    return frame