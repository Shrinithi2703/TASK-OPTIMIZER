from config import emotion_tasks

def recommend_tasks(emotion):
    return emotion_tasks.get(emotion, ["No recommendations available"])