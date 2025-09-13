import streamlit as st
import cv2
import numpy as np
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
from utils.face_utils import train_face_recognition_model, recognize_face, detect_faces
from utils.video_utils import initialize_video_writer, save_frame_as_image

# Initialize session state variables
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'face_trained' not in st.session_state:
    st.session_state.face_trained = False
if 'recognizer' not in st.session_state:
    st.session_state.recognizer = None
if 'label_map' not in st.session_state:
    st.session_state.label_map = None
if 'video_writer' not in st.session_state:
    st.session_state.video_writer = None
if 'recording_started' not in st.session_state:
    st.session_state.recording_started = False
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'last_face_detected' not in st.session_state:
    st.session_state.last_face_detected = False
if 'recording_paused' not in st.session_state:
    st.session_state.recording_paused = False


# Create necessary directories
os.makedirs('face_data', exist_ok=True)
os.makedirs('recordings', exist_ok=True)
os.makedirs('models', exist_ok=True)

st.title("Face-Activated Video Recorder")
st.write("This app records video only when YOUR face is detected.")

# Sidebar for user registration
with st.sidebar:
    st.header("User Registration")
    user_name = st.text_input("Enter your name", key="user_name_input")
    
    # Face registration using webcam
    st.subheader("Register Your Face")
    st.write("Please record a short video of your face for registration. Move your head slightly to capture different angles.")
    
    reg_cam = st.camera_input("Record your face", key="reg_cam")
    
    if reg_cam and user_name:
        # Save the registration image
        img = Image.open(reg_cam)
        img_array = np.array(img)
        
        # Convert RGB to BGR for OpenCV
        img_array_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Save multiple frames for training
        for i in range(5):
            save_frame_as_image(img_array_bgr, 'face_data', user_name)
        
        st.success(f"Face samples captured for {user_name}!")
        
        # Train the face recognition model
        recognizer, label_map = train_face_recognition_model('face_data', user_name)
        
        if recognizer and label_map:
            st.session_state.recognizer = recognizer
            st.session_state.label_map = label_map
            st.session_state.face_trained = True
            st.session_state.user_name = user_name
            st.success("Face recognition model trained successfully!")
        else:
            st.error("Failed to train face recognition model. Please try again.")

# Main application
if st.session_state.face_trained and st.session_state.recognizer:
    st.header("Face-Activated Recording")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Recording", disabled=st.session_state.recording, key="start_btn"):
            st.session_state.recording = True
            st.session_state.recording_started = False
            st.session_state.recording_paused = False
    with col2:
        if st.button("Stop Recording", disabled=not st.session_state.recording, key="stop_btn"):
            st.session_state.recording = False
            if st.session_state.video_writer:
                st.session_state.video_writer.release()
                st.session_state.video_writer = None
            st.success("Recording saved!")

    # Real-time video recording using OpenCV
    frame_placeholder = st.empty()
    if st.session_state.recording:
        cap = cv2.VideoCapture(0)
        fps = 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        filename = f"recordings/{st.session_state.user_name}_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        st.info(f"Recording started: {filename}")
        while st.session_state.recording and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video")
                break
            frame_bgr = frame
            # Recognize face
            face_detected, recognized_name, face_coords = recognize_face(
                st.session_state.recognizer,
                st.session_state.label_map,
                frame_bgr
            )
            # Status text
            rec_text = "RECORDING"
            rec_color = (0, 0, 255)
            cv2.putText(frame_bgr, rec_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, rec_color, 2, cv2.LINE_AA)
            if face_detected and recognized_name == st.session_state.user_name:
                x, y, w, h = face_coords
                cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame_bgr, recognized_name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame_bgr, "FACE DETECTED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                st.success("Your face detected - Recording")
            else:
                cv2.putText(frame_bgr, "FACE NOT DETECTED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                st.warning("Face not detected - Still recording")
            # Only write frames when face is detected and recognized
            if face_detected and recognized_name == st.session_state.user_name:
                video_writer.write(frame_bgr)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, caption="Live Feed", use_column_width=True)
        video_writer.release()
        cap.release()
        st.success("Recording saved in 'recordings' folder!")
    else:
        st.info("Click 'Start Recording' to begin")
else:
    st.info("Please register your face using the sidebar to begin.")

# Display instructions
with st.expander("How to use this app"):
    st.markdown("""
    1. Enter your name in the sidebar
    2. Record a short video of your face for registration
    3. Click 'Start Recording' to begin face-activated recording
    4. The app will only record when YOUR face is detected
    5. If someone else comes in front of the camera, recording will stop immediately
    6. Click 'Stop Recording' to save the video
    
    Recordings are saved in the 'recordings' folder.
    """)

# Footer
st.markdown("---")
st.markdown("Face recognition powered by OpenCV LBPH")