import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import threading
import pygame

# Streamlit page settings
st.set_page_config(page_title="Fatigue Detection", page_icon="🚥", layout="wide", initial_sidebar_state="collapsed")

# Background CSS
background_css = """
<style>
    .stApp {
        background-image: url('https://i.pinimg.com/originals/6d/46/f9/6d46f977733e6f9a9fa8f356e2b3e0fa.gif');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    header {
        visibility: hidden;
    }
</style>
"""
st.markdown(background_css, unsafe_allow_html=True)

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Constants
EYE_AR_THRESHOLD = 0.25
YAWN_AR_THRESHOLD = 0.6
HEAD_BEND_THRESHOLD = 15
EYE_AR_CONSEC_FRAMES = 15
MOUTH_OPEN_CONSEC_FRAMES = 7
HEAD_BEND_CONSEC_FRAMES = 10

# Sound function
def play_sound(sound_file, volume):
    pygame.mixer.init()
    pygame.mixer.music.set_volume(volume)
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

# Helper functions
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(landmarks, left=True):
    if left:
        eye_indices = [33, 160, 158, 133, 153, 144]
    else:
        eye_indices = [362, 385, 387, 263, 373, 380]
    points = [landmarks[i] for i in eye_indices]
    A = euclidean(points[1], points[5])
    B = euclidean(points[2], points[4])
    C = euclidean(points[0], points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(landmarks):
    mouth_indices = [78, 308, 14, 13, 87, 317]
    points = [landmarks[i] for i in mouth_indices]
    A = euclidean(points[2], points[3])
    B = euclidean(points[0], points[1])
    mar = A / B
    return mar

def head_bend_distance(landmarks):
    nose_tip = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    eyes_midpoint = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    vertical_distance = nose_tip[1] - eyes_midpoint[1]
    return vertical_distance

# Title
st.title("🚗 Driving Fatigue Management")
st.title("")

# Sidebar
with st.sidebar:
    st.header("Alert Settings")
    volume = st.slider("Volume", 0.0, 1.0, 0.5)
    st.session_state.eye_alert = st.checkbox("Detect Eyes Closure", value=st.session_state.get("eye_alert", False))
    st.session_state.head_alert = st.checkbox("Detect Head Down", value=st.session_state.get("head_alert", False))
    st.session_state.yawn_alert = st.checkbox("Detect Yawning", value=st.session_state.get("yawn_alert", False))
    st.session_state.all_alert = st.checkbox("Detect All", value=True)
    sound_option = st.radio("Select Alert Sound", ["beep", "buzzer", "horn"])

# Initialize session state
if "running" not in st.session_state:
    st.session_state.running = False

# Main detection function
def detect_fatigue():
    cap = cv2.VideoCapture(0)
    frame_window = st.empty()
    eye_counter = 0
    mouth_open_counter = 0
    head_bend_counter = 0

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            ih, iw, _ = frame.shape
            landmark_points = [(int(pt.x * iw), int(pt.y * ih)) for pt in landmarks.landmark]

            avg_ear = (eye_aspect_ratio(landmark_points, left=True) + eye_aspect_ratio(landmark_points, left=False)) / 2.0
            mar = mouth_aspect_ratio(landmark_points)
            vertical_distance = head_bend_distance(landmark_points)

            color = (0, 255, 0)
            alert_text = ""

            # Detect Eyes Closure
            if (st.session_state.eye_alert or st.session_state.all_alert) and avg_ear < EYE_AR_THRESHOLD:
                eye_counter += 1
                if eye_counter >= EYE_AR_CONSEC_FRAMES:
                    threading.Thread(target=play_sound, args=(f"{sound_option}.wav", volume), daemon=True).start()
                    alert_text = "Eyes Closed!"
                    color = (0, 0, 255)
                    eye_counter = 0
            else:
                eye_counter = 0

            # Detect Yawning
            if (st.session_state.yawn_alert or st.session_state.all_alert) and mar > YAWN_AR_THRESHOLD:
                mouth_open_counter += 1
                if mouth_open_counter >= MOUTH_OPEN_CONSEC_FRAMES:
                    threading.Thread(target=play_sound, args=(f"{sound_option}.wav", volume), daemon=True).start()
                    alert_text = "Yawning Detected!"
                    color = (0, 0, 255)
                    mouth_open_counter = 0
            else:
                mouth_open_counter = 0

            # Detect Head Down
            if (st.session_state.head_alert or st.session_state.all_alert) and vertical_distance > HEAD_BEND_THRESHOLD:
                head_bend_counter += 1
                if head_bend_counter >= HEAD_BEND_CONSEC_FRAMES:
                    threading.Thread(target=play_sound, args=(f"{sound_option}.wav", volume), daemon=True).start()
                    alert_text = "Head Down!"
                    color = (0, 0, 255)
                    head_bend_counter = 0
            else:
                head_bend_counter = 0

            # Draw face landmarks
            for idx, point in enumerate(landmark_points):
                cv2.circle(frame, point, 1, color, -1)

            cv2.putText(frame, alert_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        frame_window.image(frame, channels="BGR", use_column_width=True)
    cap.release()
    st.write("Fatigue detection stopped.")

# Start/Stop button
if st.button("Start / Stop"):
    if not st.session_state.running:
        st.session_state.running = True
        detect_fatigue()
    else:
        st.session_state.running = False
