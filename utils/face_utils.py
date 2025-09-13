import cv2
import numpy as np
import os
import pickle

def train_face_recognition_model(face_data_dir, user_name):
    """Train a simple face recognition model using LBPH"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    faces = []
    labels = []
    label_id = 0
    label_map = {}
    
    user_dir = os.path.join(face_data_dir, user_name)
    if not os.path.exists(user_dir):
        return None, None
    
    # Process all images for this user
    for image_file in os.listdir(user_dir):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(user_dir, image_file)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            for (x, y, w, h) in detected_faces:
                face_roi = gray[y:y+h, x:x+w]
                faces.append(face_roi)
                labels.append(label_id)
    
    if not faces:
        return None, None
    
    # Train the recognizer
    recognizer.train(faces, np.array(labels))
    
    # Save the model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{user_name}_model.yml")
    recognizer.save(model_path)
    
    # Save label mapping
    label_map[label_id] = user_name
    with open(os.path.join(model_dir, f"{user_name}_labels.pickle"), 'wb') as f:
        pickle.dump(label_map, f)
    
    return recognizer, label_map

def recognize_face(recognizer, label_map, frame, confidence_threshold=70):
    """Recognize face in the frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # Predict using the recognizer
        label, confidence = recognizer.predict(roi_gray)
        
        if confidence < confidence_threshold:
            user_name = label_map.get(label, "Unknown")
            return True, user_name, (x, y, w, h)
    
    return False, None, None

def detect_faces(frame):
    """Detect faces in the frame without recognition"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces