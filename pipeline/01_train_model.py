import sys
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add the parent directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import setup_logger, load_params


def build_model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def train_model(params, logger):
    try:
        logger.info("Initializing data generators...")
        
        # Data generators for training and validation
        train_data_gen = ImageDataGenerator(rescale=1.0 / 255)
        validation_data_gen = ImageDataGenerator(rescale=1.0 / 255)
        
        # Train generator
        train_generator = train_data_gen.flow_from_directory(
            params['train_data_path'],
            target_size=tuple(params['target_size']),
            batch_size=params['batch_size'],
            color_mode="grayscale",
            class_mode='categorical'
        )
        logger.info(f"Train data generator initialized with {len(train_generator)} batches.")

        # Validation generator
        validation_generator = validation_data_gen.flow_from_directory(
            params['test_data_path'],
            target_size=tuple(params['target_size']),
            batch_size=params['batch_size'],
            color_mode="grayscale",
            class_mode='categorical'
        )
        logger.info(f"Validation data generator initialized with {len(validation_generator)} batches.")
        logger.info("************************************")

        logger.info("Building the model...")
        input_shape = (*params['target_size'], 1)
        num_classes = len(params['emotion_dict'])
        model = build_model(input_shape, num_classes)
        logger.info(f"Model architecture: {model.summary()}")
        print(model.summary())
        logger.info("Model built successfully.")
        logger.info("************************************")

        logger.info("Compiling the model...")
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=params['learning_rate']),
            metrics=['accuracy']
        )
        logger.info(f"Compilation completed. Optimizer: Adam, Learning rate: {params['learning_rate']}.")
        logger.info("************************************")

        logger.info("Starting training...")
        history = model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=params['epochs'],
            validation_data=validation_generator,
            validation_steps=len(validation_generator)
        )
        logger.info("Training completed successfully.")
        logger.info("************************************")

        logger.info("Saving the model...")
        model_dir = os.path.dirname(params['model']['json_path'])
        os.makedirs(model_dir, exist_ok=True)

        # Save model JSON
        model_json = model.to_json()
        with open(params['model']['json_path'], "w") as json_file:
            json_file.write(model_json)
        logger.info(f"Model architecture saved to {params['model']['json_path']}.")

        # Save model weights
        model.save_weights(params['model']['weights_path'])
        logger.info(f"Model weights saved to {params['model']['weights_path']}.")

        # Save the entire model in Keras format
        model.save(params['model']['keras_path'])
        logger.info(f"Entire model saved in Keras format to {params['model']['keras_path']}.")
        logger.info("************************************")
        
        # Logging final metrics
        logger.info("Final training metrics:")
        logger.info(f"Training Accuracy: {history.history['accuracy'][-1]:.4f}")
        logger.info(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
        logger.info("Model training and saving completed successfully.")
        logger.info("************************************")

    except Exception as e:
        logger.error("An error occurred during training.", exc_info=True)
        raise


if __name__ == "__main__":
    params = load_params("params.yaml")
    logger = setup_logger("logs/training.log")
    train_model(params, logger)
