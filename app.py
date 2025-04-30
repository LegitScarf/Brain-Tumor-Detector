# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 13:35:46 2025

@author: KIIT
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
from PIL import Image
from io import BytesIO

# Load the model
try:
    model = load_model('D:/Machine Learning/Model Deployment/brain_tumor_dataset/brain_tumor_model.h5')
except Exception as e:
    st.error(f"Error loading the model: {e}. Please make sure the model file 'brain_tumor_model.h5' is in the same directory as the script, or provide the correct path. 😕")
    # Stop the app if the model fails to load.  Important for Streamlit.
    st.stop()

# Class labels
class_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Function to preprocess the image
def preprocess_image(img):
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to make prediction
def predict_tumor(img):
    img = preprocess_image(img)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] # Get confidence for the predicted class.
    return class_labels[predicted_class], confidence

# Streamlit app
def main():
    # Apply a custom theme
    st.set_page_config(
        page_title="Brain Tumor Detection",
        page_icon=":brain:",  # You can use an emoji here
        layout="centered",
        initial_sidebar_state="auto",
    )

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

    # --- Main App Content ---
    st.title('Brain Tumor Detection 🧠') # Added emoji to the title
    st.markdown("Upload an MRI image for brain tumor detection. 🔍") # Added emoji to the markdown

    # Upload section
    with st.container():
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
                st.error("Invalid image format. Please upload a valid image. ❌") # Added emoji
                return

            # Image display
            st.subheader("Uploaded MRI Image", anchor=False)
            st.image(image_np,  use_column_width=True)

            if st.button('Predict'):
                # Use columns to better organize layout
                image_col, prediction_col = st.columns(2)
                
                with image_col:
                    st.subheader("Uploaded MRI Image", anchor=False)
                    st.image(image_np,  use_column_width=True)
                
                with prediction_col:
                    # Make prediction
                    predicted_class, confidence = predict_tumor(image_np)
        
                    # Display the result
                    st.subheader('Prediction', anchor=False)
                    st.write(f'Predicted Class: {predicted_class}')
                    st.write(f'Confidence: {confidence:.2f}')
        
                    # Visualize the prediction.  Make this fancier.
                    if predicted_class == 'no_tumor':
                        st.success("No tumor detected. ✅") # Added emoji
                    else:
                        st.error(f"Tumor detected: {predicted_class} 😔") # Added emoji

if __name__ == '__main__':
    main()
