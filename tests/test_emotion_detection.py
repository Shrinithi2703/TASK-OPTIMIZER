import unittest
from app.emotion_detection import detect_emotion
import cv2

class TestEmotionDetection(unittest.TestCase):
    def test_detect_emotion(self):
        frame = cv2.imread("test_image.jpg")  # Add a test image
        emotion = detect_emotion(frame)
        self.assertIn(emotion, ["happy", "sad", "angry", "neutral", "surprise", "fear", "disgust"])

if __name__ == "__main__":
    unittest.main()