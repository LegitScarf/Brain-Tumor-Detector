import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Class labels
CLASS_LABELS = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
IMG_SIZE = (150, 150)

# Enhanced custom CSS for modern, sleek styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main > div {
        padding-top: 2rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 3rem;
        margin: 2rem auto;
        max-width: 1200px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .hero-section {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem 0;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: #64748b;
        font-weight: 400;
        margin-bottom: 2rem;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    }
    
    .upload-card {
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        border-radius: 16px;
        padding: 2.5rem;
        border: 2px dashed #e2e8f0;
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .upload-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 16px 16px 0 0;
    }
    
    .upload-card:hover {
        border-color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.1);
    }
    
    .result-card {
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        border-radius: 16px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #10b981, #059669);
        border-radius: 16px 16px 0 0;
    }
    
    .image-container {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    
    .prediction-container {
        background: linear-gradient(135deg, #f8fafc, #ffffff);
        border-radius: 12px;
        padding: 2rem;
        border: 1px solid #e2e8f0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.8rem 2rem;
        border: none;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a67d8, #6b46c1);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    .success-message {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        margin-top: 1rem;
    }
    
    .error-message {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
        margin-top: 1rem;
    }
    
    .prediction-label {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .confidence-label {
        font-size: 1.2rem;
        color: #64748b;
        font-weight: 500;
    }
    
    .confidence-bar {
        background: #e2e8f0;
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    .stats-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 1.5rem;
    }
    
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        display: block;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
        margin: 2rem 0;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .main-container {
            margin: 1rem;
            padding: 2rem;
        }
        
        .stats-container {
            grid-template-columns: 1fr;
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
    
    return CLASS_LABELS[predicted_class_index], confidence, prediction[0]

# Function to format tumor type for display
def format_tumor_type(tumor_type):
    type_mapping = {
        'glioma_tumor': 'Glioma Tumor',
        'meningioma_tumor': 'Meningioma Tumor',
        'no_tumor': 'No Tumor',
        'pituitary_tumor': 'Pituitary Tumor'
    }
    return type_mapping.get(tumor_type, tumor_type)

# Main function
def main():
    # Load model
    interpreter, input_details, output_details = load_tflite_model()
    
    if interpreter is None:
        st.error("Failed to load the model. Please check if the model file exists in the repository.")
        st.stop()
    
    # Main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Hero section
    st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">Brain Tumor Detection</h1>
            <p class="hero-subtitle">Advanced AI-powered analysis for medical imaging. Upload an MRI scan to detect and classify brain tumors with high precision.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Upload section
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'], help="Upload a clear MRI brain scan image")
    
    if uploaded_file is None:
        st.markdown("""
            <div style="text-align: center; padding: 2rem 0;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📁</div>
                <h3 style="color: #64748b; font-weight: 500;">Upload MRI Image</h3>
                <p style="color: #94a3b8;">Drag and drop or click to select an MRI brain scan</p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        # Convert to numpy array for cv2 processing
        image_np = np.array(image)
        
        # Ensure the image is in RGB format
        if image_np.ndim == 3:
            if image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            elif image_np.shape[2] == 1:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.ndim == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        else:
            st.error("Invalid image format. Please upload a valid image.")
            return
        
        # Create two columns for layout
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.markdown("**Uploaded MRI Scan**")
            st.image(image_np, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
            st.markdown("**Analysis**")
            
            if st.button('🔍 Analyze Image', use_container_width=True):
                with st.spinner("Processing image..."):
                    # Make prediction
                    predicted_class, confidence, all_predictions = predict_tumor(image_np, interpreter, input_details, output_details)
                
                # Display results
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                
                formatted_class = format_tumor_type(predicted_class)
                st.markdown(f'<div class="prediction-label">Diagnosis: {formatted_class}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence-label">Confidence: {confidence:.1%}</div>', unsafe_allow_html=True)
                
                # Confidence bar
                st.markdown(f"""
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence*100:.1f}%;"></div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Result message
                if predicted_class == 'no_tumor':
                    st.markdown(f"""
                        <div class="success-message">
                            ✓ No tumor detected. The scan appears normal.
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="error-message">
                            ⚠ Tumor detected: {formatted_class}
                        </div>
                    """, unsafe_allow_html=True)
                
                # Statistics
                st.markdown('<div class="stats-container">', unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="stat-card">
                        <span class="stat-value">{confidence:.1%}</span>
                        <span class="stat-label">Confidence</span>
                    </div>
                """, unsafe_allow_html=True)
                
                # Find second highest prediction for comparison
                sorted_predictions = np.sort(all_predictions)[::-1]
                second_confidence = sorted_predictions[1] if len(sorted_predictions) > 1 else 0
                
                st.markdown(f"""
                    <div class="stat-card">
                        <span class="stat-value">{second_confidence:.1%}</span>
                        <span class="stat-label">Alt. Possibility</span>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Disclaimer
                st.markdown("""
                    <div style="margin-top: 1.5rem; padding: 1rem; background: #fef3c7; border-radius: 8px; border-left: 4px solid #f59e0b;">
                        <small style="color: #92400e;"><strong>Disclaimer:</strong> This is an AI-assisted tool for educational purposes. Always consult with qualified medical professionals for proper diagnosis and treatment.</small>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
