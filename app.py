import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import os

# Constants
IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 10
CHANNELS = 3

# Dataset Path (pointing to your GitHub folder structure)
DATASET_PATH = "potato"  # Adjust this if the folder name is different

# Streamlit Title
st.title("Potato Leaf Disease Prediction")
st.write("An AI model to classify potato leaf diseases: Early Blight, Late Blight, and Healthy.")

# Load dataset
@st.cache_data  # Cache dataset loading for performance
def load_dataset():
    return tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_PATH,
        shuffle=True,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE
    )

st.write("Loading dataset...")
datasets = load_dataset()

# Class Names
class_names = datasets.class_names
st.write(f"Class Names: {class_names}")

# Split dataset into training, validation, and testing
def split_dataset(dataset, train_split=0.8, val_split=0.1, test_split=0.1):
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)

    dataset = dataset.shuffle(dataset_size, seed=123)
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size).take(val_size)
    test_ds = dataset.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = split_dataset(datasets)

# Prefetch datasets for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data preprocessing
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1.0 / 255)
])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2)
])

# Build the model
@st.cache_resource  # Cache the model to avoid reloading during app use
def build_model():
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    n_classes = len(class_names)

    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        resize_and_rescale,
        data_augmentation,
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(n_classes, activation='softmax'),
    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model()
st.write("Model architecture built.")

# Train the model
if st.button("Train Model"):
    st.write("Training the model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )
    st.write("Model training completed.")
    
    # Display training accuracy/loss curves
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(acc, label='Training Accuracy')
    ax[0].plot(val_acc, label='Validation Accuracy')
    ax[0].set_title('Accuracy')
    ax[0].legend()

    ax[1].plot(loss, label='Training Loss')
    ax[1].plot(val_loss, label='Validation Loss')
    ax[1].set_title('Loss')
    ax[1].legend()

    st.pyplot(fig)

# Evaluate the model
if st.button("Evaluate Model"):
    st.write("Evaluating the model...")
    test_loss, test_acc = model.evaluate(test_ds)
    st.write(f"Test Accuracy: {test_acc * 100:.2f}%")

# Upload and predict
uploaded_file = st.file_uploader("Upload a potato leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.write("Processing uploaded image...")
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = round(100 * np.max(predictions), 2)

    st.write(f"Predicted Class: **{predicted_class}**")
    st.write(f"Confidence: **{confidence}%**")
