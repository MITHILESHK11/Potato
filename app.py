import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
MODEL_PATH = 'model.weights.h5'
model = load_model(MODEL_PATH)

# Define class names
CLASS_NAMES = ["Class 1", "Class 2", "Class 3"]  # Replace with your actual class names

# Preprocessing function
def preprocess_image(image):
    image = image.resize((256, 256))  # Resize to match model input
    image_array = np.array(image) / 255.0  # Rescale to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Streamlit App
st.title("Potato Disease Classification")
st.write("Upload an image of a potato plant to classify its disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(preprocessed_image)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    # Display the prediction
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence}%")
