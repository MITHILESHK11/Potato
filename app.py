import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer
from PIL import Image
import numpy as np

# Path to the model or weights
FULL_MODEL_PATH = "trainedmodel.h5"
WEIGHTS_PATH = "model.weights.h5"

# Function to create the model architecture if only weights are available
def create_model():
    model = Sequential([
        InputLayer(input_shape=(256, 256, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')  # Adjust number of classes if needed
    ])
    return model

# Load the model
def load_trained_model():
    try:
        # First, try to load the full model
        model = load_model(FULL_MODEL_PATH)
        st.success("Loaded full model successfully.")
    except Exception as e:
        # If full model is not available, load weights
        st.warning("Full model not found. Attempting to load weights...")
        model = create_model()
        model.load_weights(WEIGHTS_PATH)
        st.success("Loaded weights successfully.")
    return model

# Load the model
model = load_trained_model()

# Define class names (adjust based on your dataset)
class_names = ["Healthy", "Early Blight", "Late Blight"]

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((256, 256))  # Resize image to match model input
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Streamlit app UI
st.title("Potato Leaf Disease Classification")
st.write("Upload an image of a potato leaf to predict its condition.")

# File uploader for the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make prediction
    predictions = model.predict(preprocessed_image)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    # Display prediction results
    st.write(f"### Prediction: {predicted_class}")
    st.write(f"### Confidence: {confidence}%")
