import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('models/1.keras')  # Path to your saved model
    return model

model = load_model()

# Class names (same as your dataset)
class_names = ['Class1', 'Class2', 'Class3']  # Replace with actual class names

# Function to make prediction
def predict(image):
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    
    return predicted_class, confidence

# Streamlit UI
st.title("Potato Disease Classification")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image = image.resize((256, 256))  # Resize to match model's input size

    # Prediction button
    if st.button('Predict'):
        predicted_class, confidence = predict(image)
        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Confidence: {confidence}%")
