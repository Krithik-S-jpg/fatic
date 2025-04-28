import streamlit as st
import cv2
import dlib
from scipy.spatial import distance
import numpy as np
import threading
import pygame

st.set_page_config(page_title="Fatigue Detection", page_icon="🚥", layout="wide", initial_sidebar_state="collapsed")

background_css = """
<style>
    /* Background Image */
    .stApp {
        background-image: url('https://i.pinimg.com/originals/6d/46/f9/6d46f977733e6f9a9fa8f356e2b3e0fa.gif');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Hide default Streamlit header */
    header {
        visibility: hidden;
    }
</style>
"""

# Inject the CSS into the page
st.markdown(background_css, unsafe_allow_html=True)

# Constants
EYE_AR_THRESHOLD = 0.25  # Adjusted threshold for eye aspect ratio
YAWN_AR_THRESHOLD = 0.2
HEAD_BEND_THRESHOLD = 20
EYE_AR_CONSEC_FRAMES = 15
MOUTH_OPEN_CONSEC_FRAMES = 7
HEAD_BEND_CONSEC_FRAMES = 10

# Load dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate mouth aspect ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[13], mouth[19])
    B = distance.euclidean(mouth[14], mouth[18])
    C = distance.euclidean(mouth[15], mouth[17])
    D = distance.euclidean(mouth[12], mouth[16])
    mar = (A + B + C) / (2.0 * D)
    return mar

# Function to calculate vertical distance between nose tip and eyes
def head_bend_distance(landmarks):
    nose_tip = landmarks[30]
    left_eye = landmarks[36]
    right_eye = landmarks[45]
    eyes_midpoint = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    vertical_distance = nose_tip[1] - eyes_midpoint[1]
    return vertical_distance

# Function to play sound
def play_sound(sound_file, volume):
    pygame.mixer.init()
    pygame.mixer.music.set_volume(volume)
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

# Streamlit app
st.title("🚗 Driving Fatigue Management")
st.title("")
# Sidebar elements
with st.sidebar:
    st.header("Alert Settings")

    # Volume slider
    volume = st.slider("Volume", 0.0, 1.0, 0.5)

    # Checkboxes
    st.session_state.eye_alert = st.checkbox("Detect Eyes Closure", value=st.session_state.get("eye_alert", False))
    st.session_state.head_alert = st.checkbox("Detect Head Down", value=st.session_state.get("head_alert", False))
    st.session_state.yawn_alert = st.checkbox("Detect Yawning", value=st.session_state.get("yawn_alert", False))
    st.session_state.all_alert = st.checkbox("Detect All", value=True)

    # Radio button for sound option
    sound_option = st.radio("Select Alert Sound", ["beep", "buzzer", "horn"])

# Initialize session state
if "running" not in st.session_state:
    st.session_state.running = False

def detect_fatigue():
    cap = cv2.VideoCapture(0)
    eye_counter = 0
    mouth_open_counter = 0
    head_bend_counter = 0
    frame_window = st.empty()

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
            left_eye, right_eye = landmarks[42:48], landmarks[36:42]
            mouth = landmarks[48:68]

            # Calculate the eye aspect ratio
            avg_ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            mar = mouth_aspect_ratio(mouth)
            vertical_distance = head_bend_distance(landmarks)

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

            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)
            cv2.putText(frame, alert_text, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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