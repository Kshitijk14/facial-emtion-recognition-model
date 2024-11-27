import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add the parent directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import setup_logger, load_params


def load_model(model_json_path, model_weights_path):
    """Load the model architecture and weights from disk."""
    with open(model_json_path, 'r') as json_file:
        model = model_from_json(json_file.read())
    model.load_weights(model_weights_path)
    return model


def prepare_data_generator(test_data_path, target_size, batch_size):
    """Prepare the test data generator."""
    test_data_gen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_data_gen.flow_from_directory(
        test_data_path,
        target_size=tuple(target_size),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical'
    )
    return test_generator


def evaluate_and_report(model, test_generator, emotion_labels, logger):
    """Evaluate the model and generate metrics."""
    logger.info("Predicting test data...")
    predictions = model.predict(test_generator, steps=len(test_generator))

    logger.info("Generating confusion matrix...")
    c_matrix = confusion_matrix(test_generator.classes, predictions.argmax(axis=1))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=emotion_labels.values())
    cm_display.plot(cmap=plt.cm.Blues)
    plt.show()

    logger.info("Generating classification report...")
    report = classification_report(test_generator.classes, predictions.argmax(axis=1), target_names=list(emotion_labels.values()))
    print(report)


def main():
    params = load_params("params.yaml")
    logger = setup_logger("logs/evaluation.log")

    logger.info("Loading the model...")
    model = load_model(params['model']['json_path'], params['model']['weights_path'])

    logger.info("Preparing test data...")
    test_generator = prepare_data_generator(
        params['test_data_path'],
        params['target_size'],
        params['batch_size']
    )

    evaluate_and_report(model, test_generator, params['emotion_dict'], logger)


if __name__ == "__main__":
    main()