import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import os

# Suppress TensorFlow macOS warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Optional: Suppress matplotlib GUI interaction
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def load_and_preprocess_image(image_path):
    """Load and preprocess the input image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_image = tf.image.resize(img, (256, 256))
    preprocessed_image = resized_image.numpy() / 255.0
    # Optional: Debug display
    # plt.imshow(resized_image.numpy().astype(int))
    # plt.title("Resized Image")
    # plt.show(block=False)
    # plt.pause(2)
    # plt.close()
    return preprocessed_image

def make_prediction(model, preprocessed_image):
    print("About to predict...")
    yhat = model.predict(np.expand_dims(preprocessed_image, axis=0))
    print(f"Raw model output: {yhat}")
    return "Good" if yhat[0][0] > 0.5 else "Bad"

if __name__ == "__main__":
    model_path = 'models/imageclassifier.keras'
    test_image_path = 'g01142.jpg'
    try:
        model = load_model(model_path)
        print("Model loaded successfully.")
        print(f"Model input shape: {model.input_shape}")
        preprocessed_image = load_and_preprocess_image(test_image_path)
        prediction = make_prediction(model, preprocessed_image)
        print(f"Predicted class: {prediction}")
    except Exception as e:
        print(f"An error occurred: {e}")
