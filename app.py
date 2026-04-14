"""
NeuralNest: Crop Disease Classification
"""

# Set environment variables BEFORE importing TensorFlow
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN warnings

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json
import plotly.graph_objects as go
from datetime import datetime
import time

# Print debug info to verify correct file is running
print(f"🔥 RUNNING FILE: {os.path.abspath(__file__)}")
print(f"🔥 FILE SIZE: {os.path.getsize(__file__)} bytes")

logo_path = os.path.join(os.getcwd(), "assets", "logo.png")

# ===============
# SESSION STATE
# ===============
if "theme" not in st.session_state:
    st.session_state.theme = "light"

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "history" not in st.session_state:
    st.session_state.history = []

def set_page(page_name):
    st.session_state.page = page_name
    st.rerun()


# =========================
# THEME VARIABLES
# =========================
if st.session_state.theme == "dark":
    sidebar_text = "white"
    sidebar_subtext = "#cfcfcf"
    social_bg = "rgba(255,255,255,0.1)"
else:
    sidebar_text = "#1a1a2e"
    sidebar_subtext = "#444"
    social_bg = "rgba(0,0,0,0.08)"
# =========================
# BUILT-IN ADVISORY DATABASE (DEPLOYMENT FALLBACK)
# =========================
BUILT_IN_ADVISORY = {
    # ==================== CORN ====================
    "Corn___Common_Rust": {
        "crop": "Corn", "severity": "Medium",
        "treatment": ["Apply fungicides at silk stage", "Use mancozeb products"],
        "prevention": ["Plant resistant hybrids", "Avoid late planting"],
        "confidence_threshold": 0.70
    },
    "Corn___Gray_Leaf_Spot": {
        "crop": "Corn", "severity": "High",
        "treatment": ["Apply fungicides with azoxystrobin", "Remove infected debris", "Rotate crops 2-3 years"],
        "prevention": ["Plant resistant varieties", "Avoid overhead irrigation", "Monitor regularly"],
        "confidence_threshold": 0.70
    },
    "Corn___Northern_Leaf_Blight": {
        "crop": "Corn", "severity": "High",
        "treatment": ["Apply pyraclostrobin", "Remove crop residue"],
        "prevention": ["Use resistant hybrids", "Practice crop rotation"],
        "confidence_threshold": 0.70
    },
    "Corn___Healthy": {
        "crop": "Corn", "severity": "None",
        "treatment": ["No treatment needed"],
        "prevention": ["Continue good practices", "Monitor regularly"],
        "confidence_threshold": 0.85
    },
    
    # ==================== POTATO ====================
    "Potato___Early_Blight": {
        "crop": "Potato", "severity": "Medium",
        "treatment": ["Apply chlorothalonil", "Remove lower leaves", "Maintain soil moisture"],
        "prevention": ["Rotate 3+ years", "Use certified seed", "Proper hilling"],
        "confidence_threshold": 0.70
    },
    "Potato___Late_Blight": {
        "crop": "Potato", "severity": "Critical",
        "treatment": ["Apply mefenoxam immediately", "Destroy infected plants", "Harvest tubers"],
        "prevention": ["Use certified seed", "Avoid poorly drained areas", "Monitor weather"],
        "confidence_threshold": 0.75
    },
    "Potato___Healthy": {
        "crop": "Potato", "severity": "None",
        "treatment": ["No treatment needed"],
        "prevention": ["Continue monitoring", "Proper irrigation"],
        "confidence_threshold": 0.85
    },
    
    # ==================== WHEAT ====================
    "Wheat___Brown_Rust": {
        "crop": "Wheat", "severity": "High",
        "treatment": ["Apply fungicides with propiconazole", "Remove infected leaves", "Improve air circulation"],
        "prevention": ["Plant resistant varieties", "Avoid dense planting", "Monitor in humid weather"],
        "confidence_threshold": 0.70
    },
    "Wheat___Yellow_Rust": {
        "crop": "Wheat", "severity": "High",
        "treatment": ["Apply fungicides with tebuconazole", "Remove infected plant debris"],
        "prevention": ["Use resistant cultivars", "Early planting", "Avoid excessive nitrogen"],
        "confidence_threshold": 0.70
    },
    "Wheat___Healthy": {
        "crop": "Wheat", "severity": "None",
        "treatment": ["No treatment needed"],
        "prevention": ["Continue good practices", "Regular monitoring"],
        "confidence_threshold": 0.85
    }
}

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="NeuralNest AI | Crop Disease Detection", 
    layout="wide", 
    page_icon="🌾",
    initial_sidebar_state="expanded"
)

if st.session_state.theme == "dark":
    sidebar_bg = "linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)"
else:
    sidebar_bg = "#f5f5f5"
# =========================
# CUSTOM CSS
# =========================
# =========================
# THEME VARIABLES
# =========================
if st.session_state.theme == "dark":
    main_bg = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    card_bg = "#1e1e2f"
    text_color = "white"
    subtext_color = "rgba(255,255,255,0.6)"
    box_bg = "rgba(255,255,255,0.95)"
else:
    main_bg = "#f5f5f5"
    card_bg = "white"
    text_color = "#1a1a2e"
    subtext_color = "#666"
    box_bg = "white"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

* {{
    font-family: 'Poppins', sans-serif;
}}

/* Hide default Streamlit elements */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {sidebar_bg} !important;
    border-right: none !important;
}}

[data-testid="stSidebar"] > div:first-child {{
    padding-top: 0 !important;
}}

.page-subtitle,
.sidebar-section,
.sidebar-text {{
    color: {sidebar_subtext} !important;
    opacity: 1 !important;
}}

/* Logo Section */
.logo-container {{
    text-align: center;
    padding: 30px 20px 20px 20px;
    background: rgba(255,255,255,0.05);
    margin: -20px -20px 20px -20px;
}}

.logo-circle {{
    width: 100px;
    height: 100px;
    border-radius: 50%;
    margin: 0 auto 15px auto;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 4px solid rgba(255,255,255,0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    background: transparent;
}}

.logo-img {{
    width: 70px;
    height: 70px;
    object-fit: contain;
    border-radius: 50%;
}}

.brand-title {{
    color: white;
    font-size: 24px;
    font-weight: 700;
    margin: 0;
    letter-spacing: 1px;
}}

.brand-subtitle {{
    color: rgba(255,255,255,0.6);
    font-size: 12px;
    margin-top: 5px;
    text-transform: uppercase;
    letter-spacing: 2px;
}}
            
.sidebar-section {{
    margin-bottom: 25px;
}}
            
/* Social Icons */
.social-bar {{
    display: flex;
    justify-content: center;
    gap: 12px;
    margin: 20px 0;
    padding: 0 20px;
}}

.social-btn {{
    width: 36px;
    height: 36px;
    background: {social_bg};
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: {sidebar_text};
    font-size: 14px;
    transition: all 0.3s ease;
    cursor: pointer;
}}

.social-btn:hover {{
    background: #4CAF50;
    color: white;
    transform: translateY(-3px);
}}

/* Main Content Background */
.main {{
    background: {main_bg};
    background-attachment: fixed;
}}

/* Content Container with Glass Effect */
.content-wrapper {{
    background: {box_bg};
    color: {text_color};
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 40px;
    margin: 20px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    min-height: 80vh;
}}

/* Page Header */
.page-header {{
    text-align: center;
    margin-bottom: 40px;
}}

.page-title {{
    color: {text_color};
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 10px;
}}

.page-subtitle {{
    color: {subtext_color};
    font-size: 16px;
}}

.underline {{
    width: 80px;
    height: 4px;
    background: linear-gradient(90deg, #4CAF50, #2E7D32);
    margin: 20px auto;
    border-radius: 2px;
}}

/* Input Cards */
.input-card {{
    background: {card_bg};
    color: {text_color};
    border-radius: 15px;
    padding: 30px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    border: 1px solid rgba(0,0,0,0.05);
    margin-bottom: 20px;
}}

/* Upload Area Styling */
.upload-container {{
    border: 3px dashed #4CAF50;
    border-radius: 15px;
    padding: 50px 30px;
    text-align: center;
    background: linear-gradient(135deg, #f1f8e9 0%, #dcedc8 100%);
    transition: all 0.3s ease;
}}

.upload-container:hover {{
    border-color: #2E7D32;
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
}}

.upload-icon {{
    font-size: 60px;
    margin-bottom: 15px;
}}

/* Result Display */
.result-box {{
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 15px;
    padding: 30px;
    color: white;
    text-align: center;
    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
}}

.result-label {{
    color: rgba(255,255,255,0.7);
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 10px;
}}

.result-value {{
    color: #4CAF50;
    font-size: 28px;
    font-weight: 700;
}}

.feature-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    margin-top: 30px;
}}

.feature-box {{
    background: {card_bg} !important;
    border-radius: 15px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    border-bottom: 4px solid #4CAF50;
    transition: all 0.3s ease;
    color: {text_color} !important;
}}

.feature-box:hover {{
    transform: translateY(-5px);
    box-shadow: 0 10px 35px rgba(0,0,0,0.25);
}}

/* FORCE EVERYTHING INSIDE BOX */
.feature-box div,
.feature-box span,
.feature-box p {{
    color: {text_color} !important;
}}

.feature-emoji {{
    font-size: 48px;
    margin-bottom: 15px;
    color: inherit !important;
}}

.feature-title {{
    font-weight: 600;
    font-size: 16px;
    margin-bottom: 8px;
    color: {text_color} !important;
}}

.feature-desc {{
    font-size: 13px;
    color: {subtext_color} !important;
}}

.confidence-high {{
    background: rgba(76,175,80,0.2);
    color: #2E7D32;
}}

.confidence-medium {{
    background: rgba(255,193,7,0.2);
    color: #F57F17;
}}

.confidence-low {{
    background: rgba(244,67,54,0.2);
    color: #C62828;
}}

/* Chart Container */
.chart-box {{
    background: {card_bg};
    color: {text_color};
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin-top: 20px;
}}

/* Footer */
.app-footer {{
    text-align: center;
    padding: 20px;
    color: #666;
    font-size: 12px;
    margin-top: 40px;
    border-top: 1px solid #eee;
}}

/* Responsive */
@media (max-width: 768px) {{
    .content-wrapper {{
        margin: 10px;
        padding: 20px;
    }}
    
    .page-title {{
        font-size: 28px;
    }}
}}
</style>
""", unsafe_allow_html=True)
# =========================
# LOAD RESOURCES WITH PROPER FALLBACK - KEEP ONLY THIS ONE
# =========================
@st.cache_resource
def load_model_cached():
    for path in ["models/deployment/NeuralNest_MobileNetV2.keras", "models/deployment/NeuralNest_MobileNetV2.h5"]:
        if os.path.exists(path):
            try:
                return tf.keras.models.load_model(path, compile=False) if path.endswith('.keras') else load_model(path, compile=False)
            except Exception as e:
                print(f"Model load error: {e}")
                continue
    return None

@st.cache_data
def load_class_names():
    path = "models/deployment/class_names.json"
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Class names load error: {e}")
    return list(BUILT_IN_ADVISORY.keys())

def load_advisory_with_fallback():
    """Load advisory from file with validation, fallback to built-in"""
    path = "models/deployment/advisory_rules.json"
    
    if not os.path.exists(path):
        print(f"❌ Advisory file not found: {path}")
        return BUILT_IN_ADVISORY, "built-in (file missing)"
    
    try:
        with open(path, "r") as f:
            data = json.load(f)
        
        print(f"📄 Advisory file entries: {len(data)}")
        
        # Check for placeholder-only content
        if not data or (len(data) == 1 and "class_name_example" in data):
            print("⚠️ Using built-in (file has no real data)")
            return BUILT_IN_ADVISORY, "built-in (placeholder only)"
        
        # Validate entries have required fields
        valid = {}
        for k, v in data.items():
            if v and isinstance(v, dict) and 'treatment' in v and 'prevention' in v:
                valid[k] = v
        
        if len(valid) > 0:
            print(f"✅ Using file advisory: {len(valid)} valid entries")
            return valid, "file"
        else:
            print("⚠️ Using built-in (no valid entries in file)")
            return BUILT_IN_ADVISORY, "built-in (no valid entries)"
            
    except Exception as e:
        print(f"❌ Using built-in (error: {e})")
        return BUILT_IN_ADVISORY, f"built-in (error: {e})"

# Loading block
print("=" * 50)
print("LOADING RESOURCES...")
model = load_model_cached()
class_names = load_class_names()
advisory_rules, advisory_source = load_advisory_with_fallback()

print(f"✅ Model: {'Loaded' if model else 'Not found'}")
print(f"✅ Classes: {len(class_names)}")
print(f"✅ Advisory: {len(advisory_rules)} rules from {advisory_source}")
print(f"📊 Sample keys: {list(advisory_rules.keys())[:3]}")
print("=" * 50)

# =========================
# HELPER FUNCTIONS
# =========================
def preprocess(image):
    """Preprocess image for model"""
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def get_confidence_class(conf):
    if conf >= 0.85:
        return "confidence-high", "Excellent"
    elif conf >= 0.70:
        return "confidence-medium", "Good"
    return "confidence-low", "Low"

def get_severity_color(severity):
    """Get color class for severity level"""
    severity_map = {
        "Critical": "severity-critical",
        "High": "severity-high",
        "Medium": "severity-medium",
        "Low": "severity-low",
        "None": "severity-none"
    }
    return severity_map.get(severity, "severity-medium")

def get_severity_emoji(severity):
    """Get emoji for severity level"""
    emoji_map = {
        "Critical": "🔴",
        "High": "🟠",
        "Medium": "🟡",
        "Low": "🟢",
        "None": "✅"
    }
    return emoji_map.get(severity, "⚪")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image("assets/logo.png", width=100)

    st.markdown(
        """
        <div style="text-align:center;">
    <div style="color:{sidebar_text}; font-size:24px; font-weight:700;">
        NeuralNest
    </div>
    <div style="color:{sidebar_subtext}; font-size:12px;">
        CNN CROP DISEASE PREDICTION FOR KENYA'S FARMERS
    </div>
</div>
        """,
        unsafe_allow_html=True
    )
    
    # Social Icons - Kaggle, LinkedIn, GitHub, Twitter
    st.markdown("""
    <div class="social-bar">
        <a href="https://x.com/Instructure" target="_blank" class="social-btn" title="Twitter">𝕏</a>
        <a href="https://www.linkedin.com/company/ngao-labs" target="_blank" class="social-btn" title="LinkedIn">in</a>
        <a href="https://github.com/karanja-dave/crop_disease_prediction_CNN.git" target="_blank" class="social-btn" title="GitHub">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="white"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
        </a>
        <a href="https://www.kaggle.com/datasets/shubham2703/five-crop-diseases-dataset" target="_blank" class="social-btn" title="Kaggle">k</a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Navigation Buttons
    nav_items = [
        ("🏠", "Home"),
        ("🔬", "Disease Detection"),
        ("📊", "Reports"),
        ("⚙️", "Settings")
    ]
    
    for icon, label in nav_items:
        if st.button(f"{icon} {label}", key=f"nav_{label}", use_container_width=True):
            set_page(label)
    
    st.markdown("---")
    
    # About Section from your code
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-header">About</div>
        <div class="sidebar-text">
            <strong>NeuralNest</strong> is powered by Convolutional Neural Networks (CNNs) to help Kenyan farmers
        detect crop diseases instantly, accurately, and with confidence.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Supported Crops from your code
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-header">Supported Crops :</div>
        <div class="sidebar-list">🌽 Corn (Maize)</div>
        <div class="sidebar-list">🥔 Potato</div>
        <div class="sidebar-list">🌾 Wheat</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Info
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-header">Model Info</div>
        <div class="sidebar-list">Architecture: MobileNetV2</div>
        <div class="sidebar-list">Input Size: 224×224 pixels</div>
        <div class="sidebar-list">Classes: {} disease categories</div>
    </div>
    """.format(len(class_names)), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Status
    if model is not None:
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; background: rgba(76,175,80,0.2); border-radius: 10px;">
            <div style="color: #4CAF50; font-weight: 600;">AI for Smarter Farming</div>
            <div style="color: rgba(255,255,255,0.6); font-size: 12px;">{len(class_names)} classes loaded</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 15px; background: rgba(244,67,54,0.2); border-radius: 10px;">
            <div style="color: #f44336; font-weight: 600;">❌ Model Not Found</div>
        </div>
        """, unsafe_allow_html=True)
    
# =========================
# PAGE: HOME
# =========================
if st.session_state.page == "Home":
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">NeuralNest AI</h1>
        <div class="underline"></div>
        <p class="page-subtitle">Your intelligent crop disease detection assistant powered by Convolutional Neural Networks (CNNs).
        NeuralNest analyzes leaf images of corn, wheat, and potato crops to accurately identify diseases
        and provide real-time diagnosis with confidence scores and actionable treatment recommendations.
        Built to support Kenyan farmers, the system enhances early detection, reduces crop losses,
        and improves agricultural decision-making through accessible AI technology.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-box">
            <div class="feature-emoji">🔬</div>
            <div class="feature-title">Disease Detection</div>
            <div class="feature-desc">AI-powered identification with 90%+ accuracy using deep learning</div>
        </div>
        <div class="feature-box">
            <div class="feature-emoji">💊</div>
            <div class="feature-title">Treatment Plans</div>
            <div class="feature-desc">Personalized recommendations for crop disease management</div>
        </div>
        <div class="feature-box">
            <div class="feature-emoji">📱</div>
            <div class="feature-title">Farmer Friendly</div>
            <div class="feature-desc">Simple interface designed for ease of use in the field</div>
        </div>
        <div class="feature-box">
            <div class="feature-emoji">⚡</div>
            <div class="feature-title">Real-time Analysis</div>
            <div class="feature-desc">Instant results from leaf image uploads</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Start
    st.markdown("""
    <div style="margin-top: 40px; text-align: center;">
        <h3 style="color: #1a1a2e; margin-bottom: 20px;">Get Started</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🔬 Start Disease Detection", use_container_width=True):
            set_page("Disease Detection")

# =========================
# PAGE: DISEASE DETECTION
# =========================
elif st.session_state.page == "Disease Detection":
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">🔬 Disease Detection</h1>
        <div class="underline"></div>
        <p class="page-subtitle">Upload a photo of your crop leaf to get disease identification, 
            confidence scores, and treatment recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if model is None:
        st.error("❌ Model not loaded. Please check if model files exist in `models/deployment/`")
        st.stop()
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "📤 Upload Leaf Image", 
            type=["jpg", "png", "jpeg"],
            help="Upload a clear photo of a crop leaf showing visible symptoms"
        )
        
        if uploaded_file is None:
            st.markdown("""
            <div class="upload-container">
                <div class="upload-icon">📤</div>
                <div style="font-size: 18px; font-weight: 600; color: #2E7D32; margin-bottom: 8px;">
                    Drop your image here
                </div>
                <div style="color: #666; font-size: 14px;">
                    Supports: JPG, PNG, JPEG (Max 10MB)
                </div>
            </div>
            """, unsafe_allow_html=True)
            # Tips from your code
            st.markdown("""
            <div style="margin-top: 30px;">
            <div class="tips-box">
                <div class="tips-title">💡 Tips for best results:</div>
                <div class="tips-list">• Use natural lighting</div>
                <div class="tips-list">• Focus on affected leaf area</div>
                <div class="tips-list">• Avoid shadows and glare</div>
                <div class="tips-list">• Include clear symptom boundaries</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            image = Image.open(uploaded_file)
            st.image(image, caption="📸 Uploaded Image", use_container_width=True)
            
            width, height = image.size
            quality_status = "✅ Good Quality" if min(width, height) >= 224 else "⚠️ Low Resolution"
            st.info(f"{quality_status} ({width}×{height}px)")
        
        st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
        analyze = st.button("🔍 Analyze Disease", use_container_width=True, 
                          disabled=(uploaded_file is None),
                          type="primary")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if uploaded_file and analyze:
            with st.spinner("🧠 Analyzing image with Neural Network..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)
            
            try:
                processed = preprocess(image)
                preds = model.predict(processed, verbose=0)
                
                idx = np.argmax(preds[0])
                label = class_names[idx]
                confidence = float(preds[0][idx])
                
                st.session_state.history.append({
    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "disease": label,
    "confidence": round(confidence * 100, 2)
})
                disease_name = label.replace('___', ' - ').replace('_', ' ')
                conf_class, conf_text = get_confidence_class(confidence)
                
                # Result Display - Enhanced from your code
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-label">Predicted Disease</div>
                    <div class="result-value">{disease_name}</div>
                    <div style="margin-top: 15px;">
                        <span class="confidence-badge {conf_class}">
                            {conf_text} Confidence: {confidence*100:.1f}%
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence Chart
                st.markdown('<div class="chart-box">', unsafe_allow_html=True)
                st.subheader("📊 Confidence Analysis")
                
                top_5_idx = np.argsort(preds[0])[-5:][::-1]
                top_5_labels = [class_names[i].replace('___', ' - ').replace('_', ' ')[:25] + "..." 
                               if len(class_names[i]) > 25 
                               else class_names[i].replace('___', ' - ').replace('_', ' ') 
                               for i in top_5_idx]
                top_5_scores = preds[0][top_5_idx] * 100
                
                fig = go.Figure(go.Bar(
                    x=top_5_scores,
                    y=top_5_labels,
                    orientation='h',
                    marker=dict(
                        color=top_5_scores,
                        colorscale=[[0, '#f44336'], [0.5, '#ffc107'], [1, '#4caf50']],
                        line=dict(color='rgba(0,0,0,0.1)', width=1)
                    ),
                    text=[f"{s:.1f}%" for s in top_5_scores],
                    textposition='outside',
                ))
                fig.update_layout(
                    height=300,
                    xaxis=dict(range=[0, 105], title="Confidence %"),
                    yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
# ============================================
# ADVISORY SECTION
# ============================================
                
                # First check if advisory exists
                if label in advisory_rules and advisory_rules[label]:
                    info = advisory_rules[label]
                    
                    st.markdown('<div class="disease-info-card">', unsafe_allow_html=True)
                    st.subheader("📋 Disease Information")
                    
                    # Disease Info Grid
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.markdown("""
                        <div class="info-item">
                            <div class="info-label">Crop Type</div>
                            <div class="info-value">{}</div>
                        </div>
                        """.format(info.get('crop', 'Unknown')), unsafe_allow_html=True)
                    
                    with col_info2:
                        severity = info.get('severity', 'Unknown')
                        severity_class = get_severity_color(severity)
                        severity_emoji = get_severity_emoji(severity)
                        st.markdown(f"""
                        <div class="info-item">
                            <div class="info-label">Severity</div>
                            <div class="info-value {severity_class}">{severity_emoji} {severity}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Treatment Recommendations
                    st.markdown("""
                    <div class="recommendation-section">
                        <div class="recommendation-title">💊 Recommended Treatment</div>
                    """, unsafe_allow_html=True)
                    
                    for i, treatment in enumerate(info.get('treatment', ['No specific treatment available']), 1):
                        st.markdown(f'<div class="recommendation-list">{i}. {treatment}</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Prevention Strategies
                    st.markdown("""
                    <div class="recommendation-section">
                        <div class="recommendation-title">🛡️ Prevention Strategies</div>
                    """, unsafe_allow_html=True)
                    
                    for i, prevention in enumerate(info.get('prevention', ['No specific prevention available']), 1):
                        st.markdown(f'<div class="recommendation-list">{i}. {prevention}</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Low confidence warning - INSIDE the if block
                    confidence_threshold = info.get('confidence_threshold', 0.70)
                    if confidence < confidence_threshold:
                        st.warning(f"""
                            ⚠️ Low confidence prediction ({confidence*100:.1f}%). 
                            The model is uncertain about this diagnosis. 
                            Please consult an agricultural expert for confirmation.
                        """)
                    
                    # CLOSE THE DISEASE INFO CARD HERE
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                else:
                    # This else matches "if label in advisory_rules and advisory_rules[label]"
                    st.error(f"❌ No advisory found for: `{label}`")
                    st.info("💡 No advisory information available for this prediction. Please consult your local agricultural extension officer.")
                
                st.caption(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")

# =========================
# PAGE: REPORTS
# =========================
elif st.session_state.page == "Reports":
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">📊 Reports</h1>
        <div class="underline"></div>
        <p class="page-subtitle">Your crop disease detection analytics dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    history = st.session_state.history

    if len(history) == 0:
        st.info("📋 No scans yet. Start analyzing crop images to generate reports.")
    else:
        # =========================
        # METRICS CALCULATION
        # =========================
        total_scans = len(history)

        diseases_found = sum(
            1 for h in history 
            if "healthy" not in h["disease"].lower()
        )

        healthy_crops = total_scans - diseases_found

        today_date = datetime.now().strftime("%Y-%m-%d")
        today_scans = sum(
            1 for h in history 
            if h["time"].startswith(today_date)
        )

        # =========================
        # METRICS UI
        # =========================
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Scans", total_scans, f"+{today_scans} today")

        with col2:
            st.metric("Diseases Found", diseases_found)

        with col3:
            st.metric("Healthy Crops", healthy_crops)

        # =========================
        # HISTORY TABLE
        # =========================
        st.markdown("### 🧾 Scan History")

        for i, record in enumerate(reversed(history), 1):
            st.markdown(f"""
            <div class="input-card">
                <b>Scan {i}</b><br>
                🕒 {record['time']}<br>
                🌿 Disease: {record['disease']}<br>
                📊 Confidence: {record['confidence']}%
            </div>
            """, unsafe_allow_html=True)

# =========================
# PAGE: SETTINGS
# =========================
elif st.session_state.page == "Settings":
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">⚙️ Settings</h1>
        <div class="underline"></div>
        <p class="page-subtitle">Configure your NeuralNest AI preferences</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # =========================
    # LEFT COLUMN
    # =========================
    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)

        st.subheader("🔧 Detection Settings")

        threshold = st.select_slider(
            "Confidence Threshold",
            options=["Low (50%)", "Medium (70%)", "High (85%)"],
            value="High (85%)"
        )

        animations = st.toggle("Enable Animations", value=True)

        # 🌗 THEME TOGGLE (WORKING)
        theme = st.radio(
            "Theme Mode",
            ["🌙 Dark Mode", "☀️ Light Mode"],
            index=0 if st.session_state.get("theme", "dark") == "dark" else 1
        )

        new_theme = "dark" if "Dark" in theme else "light"

        if "theme" not in st.session_state:
            st.session_state.theme = "dark"

        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # RIGHT COLUMN
    # =========================
    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)

        st.subheader("📊 Model Information")

        if model is not None:
            st.success("✅ Model Loaded Successfully")
            st.write(f"**Classes:** {len(class_names)}")
            st.write(f"**TensorFlow:** {tf.__version__}")
            st.write(f"**Input Size:** 224×224 pixels")
            st.write(f"**Advisory Rules:** {len(advisory_rules)}")
            st.write(f"**Advisory Source:** {advisory_source}")
        else:
            st.error("❌ No model loaded")

        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# FALLBACK: Always render something if page state is unknown
# =========================
else:
    # This prevents the blank white box!
    st.error(f"⚠️ Unknown page state: '{st.session_state.page}'")
    st.info("Resetting to Home page...")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🏠 Go to Home", use_container_width=True, key="fallback_home_btn"):
            st.session_state.page = "Home"
            st.rerun()

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="app-footer">
    <p style="font-weight: 600; color: #1a1a2e;"> NeuralNest AI</p>
    <p>Developed for Kenyan Agriculture | 
    <a href="https://github.com/karanja-dave/crop_disease_prediction_CNN.git" target="_blank" style="color: #4CAF50;">GitHub</a> • 
    <a href="mailto:prollyjunior@gmail.com" style="color: #4CAF50;">Contact</a></p>
    <p style="color: #999; font-size: 11px;">© 2026 NeuralNest | Ngao Labs Cohort II</p>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)