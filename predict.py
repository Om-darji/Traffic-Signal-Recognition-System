import tensorflow as tf
import numpy as np
import cv2
import os

# --- MODEL CONSTANTS (MUST MATCH traffic.py) ---
# Ensure these match the values used when training your model!
IMG_WIDTH = 30  
IMG_HEIGHT = 30 
NUM_CATEGORIES = 43 # Based on the dataset size

# --- INPUT FILES ---
MODEL_FILENAME = 'model.h5'  # <-- UPDATE THIS to your saved model file
IMAGE_TO_PREDICT = 'gtsrb/19/00004_00018.ppm'    # <-- UPDATE THIS to a sample image file

def load_and_predict():
    """Loads a saved model, processes a single image, and prints the prediction."""

    # 1. Load the Model
    if not os.path.exists(MODEL_FILENAME):
        print(f"Error: Model file '{MODEL_FILENAME}' not found.")
        print("Please train your model in traffic.py first and ensure the name matches.")
        return

    print(f"Loading model: {MODEL_FILENAME}...")
    model = tf.keras.models.load_model(MODEL_FILENAME)

    # 2. Load and Preprocess the New Image
    if not os.path.exists(IMAGE_TO_PREDICT):
        print(f"Error: Image file '{IMAGE_TO_PREDICT}' not found.")
        return

    print(f"Processing image: {IMAGE_TO_PREDICT}...")

    # Read the image
    image_array = cv2.imread(IMAGE_TO_PREDICT)

    if image_array is None:
        print("Error: Failed to read image using cv2.imread. Check file path/integrity.")
        return

    # Resize the image to match model input
    resized_array = cv2.resize(image_array, (IMG_WIDTH, IMG_HEIGHT))

    # Add the batch dimension: (W, H, 3) -> (1, W, H, 3)
    input_data = np.expand_dims(resized_array, axis=0)

    # 3. Make the Prediction
    print("Making prediction...")
    # model.predict returns an array of shape (1, NUM_CATEGORIES)
    probabilities = model.predict(input_data)[0]

    # 4. Interpret the Result

    # Find the category ID with the highest probability
    predicted_category_id = np.argmax(probabilities)

    # Get the confidence score for that prediction
    confidence = probabilities[predicted_category_id]

    print("-" * 30)
    print("✅ Prediction Result:")
    print(f"Input Image Size: ({IMG_WIDTH}, {IMG_HEIGHT})")
    print(f"Predicted Category ID (Label): **{predicted_category_id}**")
    print(f"Confidence: **{confidence * 100:.2f}%**")
    print("-" * 30)

    # Optional: Print the top 3 predictions for comparison
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    print("\nTop 3 Predictions:")
    for i in top_3_indices:
        print(f"  Category {i}: {probabilities[i] * 100:.2f}%")


if __name__ == "__main__":
    load_and_predict()
