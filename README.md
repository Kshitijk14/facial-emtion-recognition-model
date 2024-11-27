# Real-Time Emotion Detection with Streamlit

This Streamlit application provides real-time emotion detection using a webcam or uploaded video. It utilizes a pre-trained emotion detection model built using TensorFlow/Keras and OpenCV for face detection. The model recognizes emotions such as happiness, sadness, surprise, etc., from faces detected in the video stream.

## Features
- **Webcam-based Emotion Detection:** Stream live video from your webcam and detect emotions in real time.
- **Upload Video for Processing:** Upload a video file (e.g., MP4, AVI, MOV) for emotion detection on each frame.
- **Live Video Display:** Visualize detected emotions in real-time with bounding boxes around faces and emotion labels.

## FER2013 Dataset
- https://www.kaggle.com/msambare/fer2013

## Installation

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/yourusername/emotion-detection-streamlit.git](https://github.com/Kshitijk14/facial-emtion-recognition-model.git)
   ```
2. **Create and activate a virtual environment**:
   ```bash
   python -m venv env
   .\env\Scripts\activate
   ```
3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Download the pre-trained model files**: You will need to have a trained emotion detection model (`model.json`, `weights.h5`) and a `cascade_classifier.xml` for face detection.
   * Place these files in the appropriate directories as specified in your `params.yaml`.
   * Alternatively, you can train your own model or download a pre-trained one and update the paths accordingly.

## Configuration

 **`params.yaml`**
 The application expects a `params.yaml` configuration file with the following structure:
 
 ```bash
 model:
  json_path: "path/to/your/model.json"  # Path to the model architecture file (JSON)
  weights_path: "path/to/your/weights.h5"  # Path to the model weights file (H5)
  keras_path: "artifacts/models/model_2/emotion_model.keras"

 target_size: [48, 48]  # Input size of the images (width, height)
 batch_size: 64
 epochs: 50
 learning_rate: 0.0001

 emotion_dict:
   0: "Angry"
   1: "Disgust"
   2: "Fear"
   3: "Happy"
   4: "Sad"
   5: "Surprise"
   6: "Neutral"

 cascade_classifier: "path/to/haarcascade_frontalface_default.xml"  # Path to the OpenCV Haar Cascade face detector
 ```

* `json_path`: Path to the saved model's architecture (JSON file).
* `weights_path`: Path to the saved model's weights (H5 file).
* `target_size`: Size to which each face image is resized before feeding it into the model.
* `emotion_dict`: Mapping of predicted class indices to human-readable emotion labels.
* `cascade_classifier`: Path to the Haar Cascade classifier XML for detecting faces.


## Usage

1. **Start the Streamlit app**: After setting up the environment and configuring the `params.yaml` file, run the following command to start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. **Interact with the app**:
   * Once the app is running, it will open in your default browser (`http://localhost:8501`).
   * On the sidebar, you can choose between **Webcam** or **Upload Video**:
     - **Webcam**: Stream live video and detect emotions in real-time from your webcam.
     - **Upload Video**: Upload a pre-recorded video file for emotion detection on each frame.
3. **Stop the Webcam**: If you are using the webcam, click the **Stop Webcam** button to end the video stream.


## Notes

* The model requires grayscale images of faces, so the webcam stream and uploaded video are automatically converted to grayscale.
* The Haar Cascade Classifier is used to detect faces in the video frames before passing them to the emotion detection model.
* The app supports common video file formats like MP4, AVI, and MOV.


## Acknowledgements

* OpenCV: For real-time face detection.
* TensorFlow/Keras: For emotion detection model.
* Streamlit: For building the web interface.
* Haar Cascade Classifier: For detecting faces in video frames.


### Explanation:

- The `README.md` provides details about setting up and running the application.
- The **Configuration** section explains how to set up the `params.yaml` file, which is required for loading the model and other configurations.
- The **Usage** section provides instructions on how to interact with the app.
- The **License** section and **Acknowledgments** section provide credits to the libraries used and any other relevant information.
