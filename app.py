import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np
import threading
import av
import streamlit.components.v1 as components
import base64

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

# Hide "Select Device" button
hide_webrtc_css = """
<style>
    button[aria-label="Select device"] {
        display: none !important;
    }
</style>
"""
st.markdown(hide_webrtc_css, unsafe_allow_html=True)

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
def play_sound():
    with open("beep.wav", "rb") as f:
        data = f.read()
        b64_encoded = base64.b64encode(data).decode()

    components.html(f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{b64_encoded}" type="audio/mp3">
        </audio>
    """, height=0)

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

# Fatigue Detection Model
class FatigueDetectionModel:
    def __init__(self):
        self.eye_counter = 0
        self.mouth_open_counter = 0
        self.head_bend_counter = 0

    def detect(self, landmark_points):
        fatigue_event = None

        avg_ear = (eye_aspect_ratio(landmark_points, left=True) + eye_aspect_ratio(landmark_points, left=False)) / 2.0
        mar = mouth_aspect_ratio(landmark_points)
        vertical_distance = head_bend_distance(landmark_points)

        # Eyes Closure Detection
        if avg_ear < EYE_AR_THRESHOLD:
            self.eye_counter += 1
            if self.eye_counter >= EYE_AR_CONSEC_FRAMES:
                fatigue_event = "Eyes Closed"
                self.eye_counter = 0
        else:
            self.eye_counter = 0

        # Yawning Detection
        if mar > YAWN_AR_THRESHOLD:
            self.mouth_open_counter += 1
            if self.mouth_open_counter >= MOUTH_OPEN_CONSEC_FRAMES:
                fatigue_event = "Yawning"
                self.mouth_open_counter = 0
        else:
            self.mouth_open_counter = 0

        # Head Down Detection
        if vertical_distance > HEAD_BEND_THRESHOLD:
            self.head_bend_counter += 1
            if self.head_bend_counter >= HEAD_BEND_CONSEC_FRAMES:
                fatigue_event = "Head Down"
                self.head_bend_counter = 0
        else:
            self.head_bend_counter = 0

        return fatigue_event

# Webcam Transformer
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = FatigueDetectionModel()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        alert_text = ""
        color = (0, 255, 0)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            ih, iw, _ = img.shape
            landmark_points = [(int(pt.x * iw), int(pt.y * ih)) for pt in landmarks.landmark]

            fatigue_event = self.model.detect(landmark_points)

            if fatigue_event:
                threading.Thread(target=play_sound, daemon=True).start()
                alert_text = fatigue_event
                color = (0, 0, 255)

            if alert_text:
                cv2.putText(img, alert_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Title
st.title("🚗 Driving Fatigue Detection (Simple)")
st.title("")  # Empty line for spacing

# Start webcam
webrtc_streamer(
    key="example",
    video_transformer_factory=VideoTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
