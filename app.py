import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import model_from_json

from utils import load_params

# Helper functions
def load_model(model_json_path, model_weights_path):
    """Load the model architecture and weights from disk."""
    with open(model_json_path, 'r') as json_file:
        model = model_from_json(json_file.read())
    model.load_weights(model_weights_path)
    return model


def preprocess_frame(frame, face_detector, target_size):
    """Convert frame to grayscale and detect faces."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    return gray_frame, faces


def predict_emotion(model, roi_gray, target_size, emotion_dict):
    """Predict the emotion for the given region of interest (ROI)."""
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, tuple(target_size)), -1), 0)
    prediction = model.predict(cropped_img)
    emotion = emotion_dict[int(np.argmax(prediction))]
    return emotion


# Streamlit app functions
def emotion_detection_app(params):
    """Streamlit application for real-time emotion detection."""
    st.title("Real-Time Emotion Detection")
    st.sidebar.header("Options")
    use_webcam = st.sidebar.radio("Input Source", ["Webcam", "Upload Video"])

    # Load the model and face detector
    st.sidebar.write("Loading model...")
    model = load_model(params['model']['json_path'], params['model']['weights_path'])
    face_detector = cv2.CascadeClassifier(params['cascade_classifier'])
    st.sidebar.write("Model loaded successfully!")

    target_size = tuple(params['target_size'])
    emotion_dict = params['emotion_dict']

    if use_webcam == "Webcam":
        # Webcam option
        st.write("Starting webcam...")
        run_webcam(model, face_detector, target_size, emotion_dict)

    elif use_webcam == "Upload Video":
        # File upload option
        video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if video_file is not None:
            process_uploaded_video(video_file, model, face_detector, target_size, emotion_dict)


def run_webcam(model, face_detector, target_size, emotion_dict):
    """Real-time webcam emotion detection."""
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Unable to access the webcam.")
            break

        frame = cv2.resize(frame, (1280, 720))
        gray_frame, faces = preprocess_frame(frame, face_detector, target_size)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
            roi_gray = gray_frame[y:y + h, x:x + w]
            emotion = predict_emotion(model, roi_gray, target_size, emotion_dict)
            cv2.putText(frame, emotion, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        if st.button("Stop Webcam"):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_uploaded_video(video_file, model, face_detector, target_size, emotion_dict):
    """Process an uploaded video file for emotion detection."""
    st.write("Processing uploaded video...")

    # Save the uploaded file temporarily
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture(temp_video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        gray_frame, faces = preprocess_frame(frame, face_detector, target_size)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
            roi_gray = gray_frame[y:y + h, x:x + w]
            emotion = predict_emotion(model, roi_gray, target_size, emotion_dict)
            cv2.putText(frame, emotion, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    os.remove(temp_video_path)


# Main entry point for the Streamlit app
if __name__ == "__main__":
    params = load_params("params.yaml")
    emotion_detection_app(params)
