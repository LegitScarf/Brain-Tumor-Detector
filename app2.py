# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 21:13:06 2025

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
import gdown

# --- Configuration ---
MODEL_FILE = "brain_tumor_model_quantized.tflite"  # OR "brain_tumor_model.h5" or "models/brain_tumor_model.h5"
GOOGLE_DRIVE_LINK = "https://drive.google.com/file/d/1Qa_w2eeBpPByx8lftwsuLLfSeSk68-yH/view?usp=sharing"  # Replace with your Drive link

# --- Download Model (Improved) ---
@st.cache_resource
def load_model_from_drive(drive_url, local_path):
    if not os.path.exists(local_path):
        try:
            gdown.download(drive_url, local_path, quiet=True)
            st.info(f"Model downloaded to: {local_path}")
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            return None

    #  Important:  Load the correct type of model based on the file extension
    if local_path.endswith(".h5") or local_path.endswith(".keras"):  # Add .keras support
        return load_model(local_path)
    elif local_path.endswith(".tflite"):
        interpreter = tf.lite.Interpreter(model_path=local_path)
        interpreter.allocate_tensors()
        return interpreter  # Return the interpreter, not the model itself
    else:
        st.error(f"Unsupported model format: {local_path}")
        return None

model = load_model_from_drive(GOOGLE_DRIVE_LINK, MODEL_FILE)

if model is None:
    st.error("Failed to load model. App cannot run.")
    st.stop()


# --- Constants ---
CLASS_LABELS = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
IMG_SIZE = (150, 150)

# --- Functions ---
def preprocess_image(img):
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_tumor(img_array):
    img_array = preprocess_image(img_array)  # Preprocess here

    if MODEL_FILE.endswith(".h5") or MODEL_FILE.endswith(".keras"):
        prediction = model.predict(img_array)
    elif MODEL_FILE.endswith(".tflite"):
        interpreter = model  # 'model' is the interpreter in this case
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    else:
        raise ValueError("Unsupported model format")

    predicted_class_index = np.argmax(prediction)
    confidence = prediction[0][predicted_class_index]
    return CLASS_LABELS[predicted_class_index], confidence


# --- Streamlit UI (same as before) ---
def main():
    # ... (Your Streamlit UI code)
    pass  # Replace 'pass' with your UI code


if __name__ == '__main__':
    main()