import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Function to load and make predictions using the selected model
def model_prediction(test_image, model_path):
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar setup
st.sidebar.title("Plant Disease System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Disease Recognition'])

# Image and title for the app
img = Image.open('Diseases.png')
st.image(img, use_column_width=True)

# Default homepage
if app_mode == 'Home':
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

# Disease recognition page
elif app_mode == 'Disease Recognition':
    st.header('Plant Disease Detection System for Sustainable Agriculture')

    # Model selection
    model_choice = st.selectbox('Choose the Model:', 
                                ['Model 1: AdamW (10 Epochs, 98% Train, 97% Val)', 
                                 'Model 2: Nadam (10 Epochs, 99% Train, 98% Val)', 
                                 'Model 3: Nadam (12 Epochs, 99.99% Train, 98.9% Val)'], 
                                index=2)

    # Set model paths based on user selection
    model_paths = {
        'Model 1: AdamW (10 Epochs, 98% Train, 97% Val)': "trained_plant_disease_model_2.keras",
        'Model 2: Nadam (10 Epochs, 99% Train, 98% Val)': "trained_plant_disease_model_3.keras",
        'Model 3: Nadam (12 Epochs, 99.99% Train, 98.9% Val)': "trained_plant_disease_model_4.keras"
    }
    selected_model_path = model_paths[model_choice]

    # Upload and display the test image
    test_image = st.file_uploader('Choose an Image:')
    if test_image is not None and st.button('Show Image'):
        st.image(test_image, use_column_width=True)

    # Perform prediction
    if test_image is not None and st.button('Predict'):
        st.snow()
        result_index = model_prediction(test_image, selected_model_path)
        class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
        st.success(f'Model Prediction: It\'s {class_names[result_index]}')
