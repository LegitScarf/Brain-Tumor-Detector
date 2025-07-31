# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 21:13:06 2025

@author: KIIT
"""

import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import os
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto",
)

# Class labels
CLASS_LABELS = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
IMG_SIZE = (150, 150)

# Add custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;  /* Light background */
        color: #333;
        font-family: 'Arial', sans-serif;
    }
    .title {
        color: #4c1130; /* Darker, more professional title color */
        text-align: center;
        margin-bottom: 2rem;
        font-size: 2.5rem; /* Increased font size for title */
    }
    .header {
        color: #2c3e50; /* Dark gray header */
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-size: 1.8rem; /* Increased font size for header */
    }
    .upload-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* Subtle shadow */
        margin-bottom: 2rem;
    }
    .prediction-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* Subtle shadow */
        margin-top: 2rem;
    }
    .image-container {
        border: 1px solid #e1e1e1;
        border-radius: 5px;
        margin-bottom: 1rem;
        overflow: hidden; /* To contain rounded corners of image */
    }
    .report-container {
        border: 1px solid #e1e1e1;
        border-radius: 5px;
        margin-top: 1rem;
    }
    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 12px 24px; /* Slightly larger button */
        border: none;
        border-radius: 8px; /* More rounded corners */
        cursor: pointer;
        font-size: 1.1rem; /* Increased font size */
        transition: background-color 0.3s ease;
        width: 100%; /* Make button full width */
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); /* Add shadow to button */
    }
    .stButton>button:hover {
        background-color: #45a049; /* Darker green */
        box-shadow: 0 3px 6px rgba(0,0,0,0.2); /* Increase shadow on hover */
    }
    .stSuccess {
        color: green;
        font-size: 1.2rem; /* Increase font size for success message */
        font-weight: bold;
    }
    .stError {
        color: red;
        font-size: 1.2rem; /* Increase font size for error message */
        font-weight: bold;
    }

    /* Improve layout for image and prediction */
    .image-and-prediction {
        display: flex;
        flex-direction: column;
        align-items: center; /* Center items horizontally */
        gap: 2rem; /* Add some gap between image and prediction */
    }
    
    @media (min-width: 768px) {
        .image-and-prediction {
            flex-direction: row; /* Arrange side-by-side on wider screens */
            justify-content: space-between; /* Space them out */
            align-items: flex-start; /* Align items to the start (top) */
        }
        .image-container, .prediction-section {
            width: 45%; /* Give them a percentage of the width */
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to load the TFLite model
@st.cache_resource
def load_tflite_model():
    try:
        # Get the model path relative to the script
        model_path = "brain_tumor_model_quantized.tflite"
        
        # Load TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        return interpreter, input_details, output_details
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Function to preprocess the image
def preprocess_image(img):
    # Convert to RGB if grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Resize to the expected input size
    img = cv2.resize(img, IMG_SIZE)
    
    # Normalize pixel values
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

# Function to make prediction using TFLite model
def predict_tumor(img, interpreter, input_details, output_details):
    # Preprocess the image
    processed_img = preprocess_image(img)
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], processed_img)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    # Get the predicted class and confidence
    predicted_class_index = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class_index]
    
    return CLASS_LABELS[predicted_class_index], confidence

# Main function
def main():
    # Load model
    interpreter, input_details, output_details = load_tflite_model()
    
    if interpreter is None:
        st.error("Failed to load the model. Please check if the model file exists in the repository.")
        st.stop()
    
    # App title and description
    st.title('Brain Tumor Detection 🧠')
    st.markdown("Upload an MRI image for brain tumor detection. 🔍")
    
    # Upload section
    uploaded_file = st.file_uploader("Choose an MRI image...", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        # Convert to numpy array for cv2 processing
        image_np = np.array(image)
        
        # Ensure the image is in RGB format
        if image_np.ndim == 3:
            if image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)  # Convert RGBA to RGB
            elif image_np.shape[2] == 1:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB) # Convert Gray to RGB
        elif image_np.ndim == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        else:
            st.error("Invalid image format. Please upload a valid image. ❌")
            return
        
        # Image display
        st.subheader("Uploaded MRI Image", anchor=False)
        st.image(image_np, use_column_width=True)
        
        if st.button('Predict'):
            # Use columns to better organize layout
            image_col, prediction_col = st.columns(2)
            
            with image_col:
                st.subheader("Uploaded MRI Image", anchor=False)
                st.image(image_np, use_column_width=True)
            
            with prediction_col:
                with st.spinner("Processing..."):
                    # Make prediction
                    predicted_class, confidence = predict_tumor(image_np, interpreter, input_details, output_details)
                
                # Display the result
                st.subheader('Prediction', anchor=False)
                st.write(f'Predicted Class: {predicted_class}')
                st.write(f'Confidence: {confidence:.2f}')
                
                # Visualize the prediction
                if predicted_class == 'no_tumor':
                    st.success("No tumor detected. ✅")
                else:
                    st.error(f"Tumor detected: {predicted_class} 😔")

if __name__ == '__main__':
    main()
