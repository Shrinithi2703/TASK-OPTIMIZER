import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime

# Define file path
USER_HISTORY_FILE = "data/user_history.csv"

# Ensure the data directory exists
os.makedirs(os.path.dirname(USER_HISTORY_FILE), exist_ok=True)

# Load existing data (handle missing columns)
if os.path.exists(USER_HISTORY_FILE) and os.path.getsize(USER_HISTORY_FILE) > 0:
    try:
        # Read without specifying columns to avoid errors
        user_history = pd.read_csv(USER_HISTORY_FILE)

        # Ensure all required columns exist
        required_columns = ["timestamp", "emotion", "task"]
        for col in required_columns:
            if col not in user_history.columns:
                user_history[col] = None  # Add missing columns

        user_history = user_history[required_columns]  # Keep only necessary columns

    except pd.errors.EmptyDataError:
        user_history = pd.DataFrame(columns=["timestamp", "emotion", "task"])
else:
    user_history = pd.DataFrame(columns=["timestamp", "emotion", "task"])

# Emotion-based task recommendations
emotion_tasks = {
    "happy": ["Brainstorm new ideas", "Lead a team meeting", "Work on a creative project"],
     "sad": ["Engage in light social interactions", "Work on familiar tasks", "Seek HR support if needed"],
    "angry": ["Practice deep breathing", "Meditate", "Take a break"],
     "neutral": ["Proceed with regular tasks", "Attend meetings", "Review work progress"],
    "surprise": ["Explore something new", "Take a photo", "Share your experience"],
    "fear": ["Face your fear gradually", "Talk to a trusted person", "Practice relaxation techniques"],
    "disgust": ["Clean your space", "Try a new recipe", "Change your environment"]
}

# Initialize session state variables
if "running" not in st.session_state:
    st.session_state.running = False

# Streamlit UI
st.title("Emotion-Based Task Recommender")
st.write("Real-time emotion detection with personalized task recommendations!")

# Start/Stop button logic
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Emotion Detection"):
        st.session_state.running = True
with col2:
    if st.button("Stop Emotion Detection"):
        st.session_state.running = False

# Webcam & Emotion Detection Loop
video_placeholder = st.empty()
emotion_placeholder = st.empty()
recommendation_placeholder = st.empty()

cap = cv2.VideoCapture(1)

while st.session_state.running:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture image from webcam.")
        break

    # Convert frame to RGB for DeepFace
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    # Analyze emotion
    try:
        result = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        emotion_placeholder.write(f"**Detected Emotion:** {emotion}")

        # Get recommended tasks
        tasks = emotion_tasks.get(emotion, ["No recommendations available"])

        # Display recommendations
        recommendation_placeholder.markdown("### Recommended Tasks:\n" + "\n".join([f"- {task}" for task in tasks]))

        # Add new tasks to history with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entries = [{"timestamp": timestamp, "emotion": emotion, "task": task} for task in tasks]

        # Convert to DataFrame and append to CSV
        new_df = pd.DataFrame(new_entries)
        new_df.to_csv(USER_HISTORY_FILE, mode='a', header=not os.path.exists(USER_HISTORY_FILE), index=False)

    except Exception as e:
        st.error(f"Error: {e}")

    # Wait for 3 seconds before detecting again
    time.sleep(2)

cap.release()
