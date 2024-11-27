import sys
import os
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Add the parent directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import setup_logger, load_params


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


def detect_emotions(params, logger):
    """Main function for real-time emotion detection."""
    logger.info("Loading the model...")
    model = load_model(params['model']['json_path'], params['model']['weights_path'])

    logger.info("Initializing video capture and face detector...")
    cap = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier(params['cascade_classifier'])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        gray_frame, faces = preprocess_frame(frame, face_detector, params['target_size'])

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
            roi_gray = gray_frame[y:y + h, x:x + w]
            emotion = predict_emotion(model, roi_gray, params['target_size'], params['emotion_dict'])
            cv2.putText(frame, emotion, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Terminating video capture...")
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    params = load_params("params.yaml")
    logger = setup_logger("logs/testing.log")
    detect_emotions(params, logger)


if __name__ == "__main__":
    main()
