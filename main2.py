import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import random

# Suppress TensorFlow macOS warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def load_and_preprocess_image(image_path):
    """Load and preprocess the input image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_image = tf.image.resize(img, (256, 256))
    preprocessed_image = resized_image.numpy() / 255.0
    return preprocessed_image

def make_prediction(model, preprocessed_image):
    """Make a prediction using the model."""
    yhat = model.predict(np.expand_dims(preprocessed_image, axis=0))
    return "Good" if yhat[0][0] > 0.5 else "Bad"

def classify_images(model, folder, label, num_samples=5):
    """Randomly pick files from a folder, classify them, and display results."""
    files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    selected_files = random.sample(files, min(num_samples, len(files)))
    print(f"Selected {len(selected_files)} files from {label} folder.")

    results = []
    for file_path in selected_files:
        try:
            preprocessed_image = load_and_preprocess_image(file_path)
            prediction = make_prediction(model, preprocessed_image)
            results.append((file_path, prediction))
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    return results

if __name__ == "__main__":
    model_path = 'models/imageclassifier.keras'
    bad_folder = 'converted_data/bad'
    good_folder = 'converted_data/good'

    try:
        model = load_model(model_path)
        print("Model loaded successfully.")

        # Classify images in the 'bad' folder
        bad_results = classify_images(model, bad_folder, label="Bad")
        for file_path, prediction in bad_results:
            print(f"File: {file_path} - Prediction: {prediction}")

        # Classify images in the 'good' folder
        good_results = classify_images(model, good_folder, label="Good")
        for file_path, prediction in good_results:
            print(f"File: {file_path} - Prediction: {prediction}")
    except Exception as e:
        print(f"An error occurred: {e}")
