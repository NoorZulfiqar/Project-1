import cv2
import os
import numpy as np
import dlib
import face_recognition_models

# Load pretrained model paths
shape_predictor_path = face_recognition_models.pose_predictor_model_location()
face_encoder_path = face_recognition_models.face_recognition_model_location()

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(shape_predictor_path)
face_encoder = dlib.face_recognition_model_v1(face_encoder_path)

def detect_faces(image):
    """Detect faces in an image (returns list of rectangles)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return detector(gray, 1)

def encode_faces(image, detections):
    """Generate encodings for all detected faces"""
    encodings = []
    for det in detections:
        shape = shape_predictor(image, det)
        encoding = np.array(face_encoder.compute_face_descriptor(image, shape))
        encodings.append((encoding, det))
    return encodings

def train_face_recognition_model(face_data_dir, user_name):
    """Train a simple face recognizer from saved images"""
    encodings = []
    labels = []
    label_map = {}

    user_dir = os.path.join(face_data_dir, user_name)
    if not os.path.exists(user_dir):
        return None, None

    label_id = 0
    label_map[label_id] = user_name

    for img_file in os.listdir(user_dir):
        path = os.path.join(user_dir, img_file)
        img = cv2.imread(path)
        detections = detect_faces(img)
        face_encs = encode_faces(img, detections)
        for enc, _ in face_encs:
            encodings.append(enc)
            labels.append(label_id)

    if len(encodings) == 0:
        return None, None

    # Store known encodings
    recognizer = {"encodings": np.array(encodings), "labels": np.array(labels)}
    return recognizer, label_map

def recognize_face(recognizer, label_map, frame, threshold=0.6):
    """Recognize face in a frame using Euclidean distance"""
    detections = detect_faces(frame)
    if len(detections) == 0:
        return False, None, None

    encs = encode_faces(frame, detections)
    for enc, det in encs:
        distances = np.linalg.norm(recognizer["encodings"] - enc, axis=1)
        idx = np.argmin(distances)
        if distances[idx] < threshold:
            (x, y, w, h) = (det.left(), det.top(), det.width(), det.height())
            return True, label_map[recognizer["labels"][idx]], (x, y, w, h)

    return False, None, None
