from emotion_detection import detect_emotion
from recommendation import recommend_tasks
from utils import capture_frame

def main():
    frame = capture_frame()
    emotion = detect_emotion(frame)
    tasks = recommend_tasks(emotion)
    print(f"Detected Emotion: {emotion}")
    print("Recommended Tasks:")
    for task in tasks:
        print(f"- {task}")

if __name__ == "__main__":
    main()