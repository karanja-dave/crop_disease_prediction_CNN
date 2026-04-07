"""
NeuralNest: Crop Disease Classification - Deployment App
AI-Powered Crop Disease Detection for Kenyan Agriculture
"""

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="NeuralNest - Crop Disease Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #558B2F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disease-card {
        background-color: #F1F8E9;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #2E7D32;
    }
    .confidence-high { color: #2E7D32; font-weight: bold; }
    .confidence-medium { color: #F9A825; font-weight: bold; }
    .confidence-low { color: #C62828; font-weight: bold; }
    .stButton>button {
        background-color: #2E7D32;
        color: white;
        font-weight: bold;
        padding: 0.75rem 2rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model with caching for performance."""
    model_path = "best_model.h5"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    return keras.models.load_model(model_path)

@st.cache_data
def load_class_names():
    """Load class names from JSON."""
    with open("class_names.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_advisory_rules():
    """Load treatment recommendations."""
    with open("advisory_rules.json", "r") as f:
        return json.load(f)

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction."""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def get_confidence_color(confidence):
    """Return color class based on confidence level."""
    if confidence >= 0.85:
        return "confidence-high"
    elif confidence >= 0.70:
        return "confidence-medium"
    else:
        return "confidence-low"

def create_confidence_chart(predictions, class_names, top_n=5):
    """Create a horizontal bar chart of top predictions."""
    # Get top N predictions
    top_indices = np.argsort(predictions[0])[-top_n:][::-1]
    top_probs = predictions[0][top_indices] * 100
    top_classes = [class_names[i].replace('_', ' ') for i in top_indices]
    
    # Create color scale based on confidence
    colors = ['#2E7D32' if p >= 85 else '#F9A825' if p >= 70 else '#C62828' for p in top_probs]
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_probs,
            y=top_classes,
            orientation='h',
            marker_color=colors,
            text=[f'{p:.1f}%' for p in top_probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Top Disease Predictions",
        xaxis_title="Confidence (%)",
        yaxis_title="",
        height=400,
        template="plotly_white",
        showlegend=False
    )
    
    return fig

def main():
    # Header
    st.markdown('<p class="main-header">🌾 NeuralNest</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Crop Disease Detection for Kenyan Agriculture</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        **NeuralNest** uses deep learning to identify crop diseases 
        from leaf images. Upload a photo of your crop leaf to get:
        
        - Disease identification
        - Confidence score
        - Treatment recommendations
        - Prevention strategies
        """)
        
        st.header("Supported Crops")
        crops = ["🌽 Corn (Maize)", "🥔 Potato", "🌾 Wheat"]
        for crop in crops:
            st.write(crop)
        
        st.header("Model Info")
        st.write("Architecture: MobileNetV2")
        st.write("Input Size: 224×224 pixels")
        st.write("Classes: 10 disease categories")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Upload Crop Leaf Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo of a crop leaf showing visible symptoms"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Add analyze button
            analyze_button = st.button("🔍 Analyze Disease", type="primary")
        else:
            # Show sample images
            st.info("👆 Upload a leaf image to begin analysis")
            st.markdown("""
            **Tips for best results:**
            - Use natural lighting
            - Focus on affected leaf area
            - Avoid shadows and glare
            - Include clear symptom boundaries
            """)
    
    # Analysis results
    with col2:
        if uploaded_file is not None and analyze_button:
            with st.spinner("Analyzing image with Neural Network..."):
                # Load resources
                model = load_model()
                class_names = load_class_names()
                advisory_rules = load_advisory_rules()
                
                # Preprocess and predict
                processed_image = preprocess_image(image)
                predictions = model.predict(processed_image, verbose=0)
                
                # Get top prediction
                predicted_class_idx = np.argmax(predictions[0])
                predicted_class = class_names[predicted_class_idx]
                confidence = float(predictions[0][predicted_class_idx])
                
                # Display results
                st.subheader("📊 Analysis Results")
                
                # Confidence indicator
                conf_color = get_confidence_color(confidence)
                st.markdown(f"""
                    <div class="disease-card">
                        <h3>Predicted: {predicted_class.replace('_', ' ')}</h3>
                        <p class="{conf_color}">Confidence: {confidence*100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Confidence chart
                fig = create_confidence_chart(predictions, class_names)
                st.plotly_chart(fig, use_container_width=True)
                
                # Advisory information
                if predicted_class in advisory_rules:
                    info = advisory_rules[predicted_class]
                    
                    st.subheader("🩺 Disease Information")
                    
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.metric("Crop Type", info['crop'])
                    with col_info2:
                        severity_color = {
                            "Critical": "🔴", "High": "🟠", 
                            "Medium": "🟡", "Low": "🟢", "None": "✅"
                        }
                        st.metric("Severity", f"{severity_color.get(info['severity'], '⚪')} {info['severity']}")
                    
                    # Treatment recommendations
                    st.subheader("💊 Recommended Treatment")
                    for i, treatment in enumerate(info['treatment'], 1):
                        st.write(f"{i}. {treatment}")
                    
                    # Prevention strategies
                    st.subheader("🛡️ Prevention Strategies")
                    for i, prevention in enumerate(info['prevention'], 1):
                        st.write(f"{i}. {prevention}")
                    
                    # Confidence warning if low
                    if confidence < info.get('confidence_threshold', 0.70):
                        st.warning(f"""
                            ⚠️ Low confidence prediction ({confidence*100:.1f}%). 
                            Please consult an agricultural expert for confirmation.
                        """)
                else:
                    st.warning("No advisory information available for this prediction.")
                
                # Timestamp
                st.caption(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666;">
            <p>NeuralNest - Developed for Kenyan Agriculture | 
            <a href="https://github.com/karanja-dave/crop_disease_prediction_CNN.git">GitHub</a> | 
            <a href="mailto:prollyjunior@gmail.com">Contact Support</a></p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()