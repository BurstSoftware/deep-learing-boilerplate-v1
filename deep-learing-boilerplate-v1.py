import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image
import numpy as np

# Set page configuration
st.set_page_config(page_title="Deep Learning Image Classifier", page_icon="🧠")

# Title and description
st.title("Deep Learning Image Classifier")
st.write("Upload an image to classify it as either a **Cat** or **Dog** using a pre-trained CNN model.")

# Function to create and compile a simple CNN model
@st.cache_resource
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to match model input
    image_array = np.array(image) / 255.0  # Normalize pixel values
    if image_array.shape[-1] != 3:  # Convert grayscale to RGB if needed
        image_array = np.stack([image_array] * 3, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Load model
model = create_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    with st.spinner("Classifying..."):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        probability = prediction[0][0]
        
        # Display result
        if probability > 0.5:
            st.success(f"This is a **Dog**! (Confidence: {probability:.2%})")
        else:
            st.success(f"This is a **Cat**! (Confidence: {1 - probability:.2%})")

# Optional: Add a button to retrain or fine-tune (placeholder)
if st.button("Train Model (Placeholder)"):
    st.info("Training functionality is a placeholder. In a real app, you would implement model training here.")

# Footer
st.markdown("---")
st.write("Built with Streamlit and TensorFlow. Model is a simple CNN for demonstration purposes.")
