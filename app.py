import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Path to the .keras directory
MODEL_DIR = "trainedmodel.keras"

# Load the model
@st.cache_resource
def load_trained_model():
    try:
        model = load_model(MODEL_DIR)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        return None

# Load the model
model = load_trained_model()

# Define class names (adjust based on your dataset)
class_names = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]  # Update with your actual class names

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

    if model is not None:
        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make prediction
        predictions = model.predict(preprocessed_image)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100 * np.max(predictions[0]), 2)

        # Display prediction results
        st.write(f"### Prediction: {predicted_class}")
        st.write(f"### Confidence: {confidence}%")
    else:
        st.error("Model could not be loaded. Please check the model directory.")

