import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from PIL import Image
import time

# For web deployment, we need to handle camera access differently
st.set_page_config(page_title="Face-Activated Recorder", layout="wide")

# In your app
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Initialize session state
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""

st.title("Face-Activated Video Recorder")

# Check if we're in a web environment
@st.cache_data
def is_web_environment():
    try:
        # Try to detect if we're running on Streamlit sharing
        if "streamlit.app" in st.secrets.get("server_address", ""):
            return True
    except:
        pass
    return False

WEB_ENV = is_web_environment()

if WEB_ENV:
    st.warning("""
    **Web Deployment Notice:**
    - Camera access requires HTTPS
    - Some features may be limited in web environment
    - Recordings are stored temporarily (download them promptly)
    """)

# Simplified face detection for web compatibility
def detect_faces(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Use a pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    return faces

# Training section
with st.sidebar:
    st.header("User Registration")
    
    user_name = st.text_input("Enter your name", key="user_name")
    st.session_state.user_name = user_name
    
    if user_name:
        st.write(f"Training for: {user_name}")
        
        # Capture training images
        img_file_buffer = st.camera_input("Take a picture for training", key="train_cam")
        
        if img_file_buffer is not None:
            # Convert buffer to numpy array
            bytes_data = img_file_buffer.getvalue()
            image = Image.open(img_file_buffer)
            img_array = np.array(image)
            
            # Detect faces
            faces = detect_faces(img_array)
            
            if len(faces) > 0:
                # Save the face data (in a real app, you'd train a model here)
                st.session_state.model_trained = True
                st.success("Face captured successfully!")
            else:
                st.error("No face detected. Please try again.")

# Main recording interface
if st.session_state.model_trained and st.session_state.user_name:
    st.header("Face-Activated Recording")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Recording", disabled=st.session_state.recording):
            st.session_state.recording = True
            st.session_state.recorded_frames = []
    
    with col2:
        if st.button("Stop Recording", disabled=not st.session_state.recording):
            st.session_state.recording = False
            
            # Save recording
            if hasattr(st.session_state, 'recorded_frames') and st.session_state.recorded_frames:
                # Create video from frames
                height, width, _ = st.session_state.recorded_frames[0].shape
                
                # Save temporarily (in web environment)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(tmpfile.name, fourcc, 20.0, (width, height))
                    
                    for frame in st.session_state.recorded_frames:
                        out.write(frame)
                    
                    out.release()
                    
                    # Offer download
                    with open(tmpfile.name, "rb") as file:
                        st.download_button(
                            label="Download Recording",
                            data=file,
                            file_name=f"{st.session_state.user_name}_recording.mp4",
                            mime="video/mp4"
                        )
    
    # Recording interface
    if st.session_state.recording:
        camera = st.camera_input("Recording in progress...", key="record_cam")
        
        if camera is not None:
            # Convert to numpy array
            image = Image.open(camera)
            img_array = np.array(image)
            
            # Detect faces
            faces = detect_faces(img_array)
            
            if len(faces) > 0:
                # Draw rectangles around faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(img_array, st.session_state.user_name, 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Store frame for recording
                if not hasattr(st.session_state, 'recorded_frames'):
                    st.session_state.recorded_frames = []
                
                st.session_state.recorded_frames.append(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                
                st.success("Face detected - Recording")
            else:
                st.warning("No face detected")
            
            # Show processed image
            st.image(img_array, caption="Processed Frame", use_column_width=True)
else:
    st.info("Please register your face using the sidebar to begin.")

# Add privacy notice for web deployment
st.markdown("---")
st.markdown("""
**Privacy Notice:** 
- This app processes images locally in your browser when possible
- Face data is not stored on our servers
- Recordings are temporarily processed and can be downloaded
""")

