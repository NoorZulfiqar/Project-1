import cv2
import os
from datetime import datetime

def initialize_video_writer(width, height, fps, recordings_dir, user_name):
    """Initialize video writer for recording"""
    if not os.path.exists(recordings_dir):
        os.makedirs(recordings_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(recordings_dir, f"{user_name}_recording_{timestamp}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    return out, filename

def save_frame_as_image(frame, face_data_dir, user_name):
    """Save frame as image for training"""
    user_dir = os.path.join(face_data_dir, user_name)
    os.makedirs(user_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(user_dir, f"{user_name}_{timestamp}.jpg")
    
    cv2.imwrite(filename, frame)
    return filename