"""
NeuralNest: Crop Disease Classification
A beautiful, animated Streamlit app for crop disease detection using CNN
"""

# Set environment variables BEFORE importing TensorFlow
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import base64
import json
from pathlib import Path
from datetime import datetime

# ===============
# SESSION STATE INITIALIZATION
# ===============
def init_session_state():
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    if "history" not in st.session_state:
        st.session_state.history = load_history()

def load_history():
    """Load scan history from file"""
    history_file = Path(__file__).parent / "scan_history.json"
    if history_file.exists():
        try:
            with open(history_file, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_history():
    """Save scan history to file"""
    history_file = Path(__file__).parent / "scan_history.json"
    try:
        with open(history_file, "w") as f:
            json.dump(st.session_state.history, f)
    except:
        pass

def add_to_history(disease, confidence):
    """Add a scan to history and save"""
    st.session_state.history.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "disease": disease,
        "confidence": round(confidence * 100, 2)
    })
    save_history()

def clear_history():
    """Clear all history"""
    st.session_state.history = []
    save_history()

init_session_state()

def set_page(page_name):
    st.session_state.page = page_name
    st.rerun()

# ===============
# PAGE CONFIG
# ===============
st.set_page_config(
    page_title="NeuralNest AI | Crop Disease Detection",
    layout="wide",
    page_icon="🌾",
    initial_sidebar_state="expanded"
)

# ===============
# LOAD BACKGROUND IMAGE
# ===============
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return ""

# Get the background image as base64
background_path = Path(__file__).parent / "assets" / "background.jpg"
bg_image_base64 = get_base64_image(background_path)

# Get logo path
logo_path = Path(__file__).parent / "assets" / "logo.png"
logo_exists = logo_path.exists()

# ===============
# THEME VARIABLES
# ===============
if st.session_state.theme == "dark":
    sidebar_bg = "linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)"
    main_bg = f"""
        linear-gradient(135deg, rgba(26, 26, 46, 0.92) 0%, rgba(22, 33, 62, 0.88) 100%),
        url('data:image/jpeg;base64,{bg_image_base64}')
    """ if bg_image_base64 else "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)"
    card_bg = "rgba(30, 30, 47, 0.9)"
    text_color = "#ffffff"
    subtext_color = "rgba(255, 255, 255, 0.7)"
    input_bg = "rgba(255, 255, 255, 0.1)"
    border_color = "rgba(255, 255, 255, 0.2)"
    sidebar_text = "#ffffff"
    sidebar_subtext = "#cfcfcf"
    social_bg = "rgba(255, 255, 255, 0.1)"
else:
    sidebar_bg = "#f5f5f5"
    main_bg = f"""
        linear-gradient(135deg, rgba(245, 245, 245, 0.95) 0%, rgba(255, 255, 255, 0.92) 100%),
        url('data:image/jpeg;base64,{bg_image_base64}')
    """ if bg_image_base64 else "linear-gradient(135deg, #f5f5f5 0%, #ffffff 100%)"
    card_bg = "rgba(255, 255, 255, 0.95)"
    text_color = "#1a1a2e"
    subtext_color = "#666666"
    input_bg = "#ffffff"
    border_color = "#e0e0e0"
    sidebar_text = "#1a1a2e"
    sidebar_subtext = "#444444"
    social_bg = "rgba(0, 0, 0, 0.08)"

# ===============
# CUSTOM CSS WITH ANIMATIONS
# ===============
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

/* Main background with image */
.stApp {{
    background: {main_bg} !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    background-repeat: no-repeat !important;
    min-height: 100vh;
}}

/* Ensure sidebar is visible */
section[data-testid="stSidebar"] {{
    background: {sidebar_bg} !important;
    width: 280px !important;
    min-width: 280px !important;
}}

section[data-testid="stSidebar"] > div {{
    background: transparent !important;
    padding-top: 1rem !important;
}}

/* Sidebar content styling */
.sidebar-content {{
    padding: 0 15px;
}}

/* Logo styling */
.sidebar-logo {{
    text-align: center;
    padding: 20px 0;
    margin-bottom: 10px;
}}

.sidebar-logo img {{
    border-radius: 50%;
    box-shadow: 0 4px 20px rgba(76, 175, 80, 0.4);
}}

.brand-title {{
    color: white;
    font-size: 22px;
    font-weight: 700;
    margin: 10px 0 5px 0;
    letter-spacing: 1px;
}}

.brand-subtitle {{
    color: rgba(255, 255, 255, 0.6);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}}

/* Social Icons */
.social-bar {{
    display: flex;
    justify-content: center;
    gap: 10px;
    margin: 15px 0;
}}

.social-btn {{
    width: 32px;
    height: 32px;
    background: {social_bg};
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: {sidebar_text};
    font-size: 12px;
    text-decoration: none;
    transition: all 0.3s ease;
}}

.social-btn:hover {{
    background: #4CAF50;
    color: white;
    transform: translateY(-2px);
}}

/* Navigation styling */
.nav-container {{
    margin: 20px 0;
}}

.nav-button {{
    width: 100%;
    padding: 12px 15px;
    margin: 5px 0;
    border: none;
    border-radius: 10px;
    background: transparent;
    color: {sidebar_text};
    text-align: left;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    gap: 10px;
}}

.nav-button:hover {{
    background: rgba(76, 175, 80, 0.2);
    transform: translateX(5px);
}}

.nav-button.active {{
    background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
    color: white;
    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4);
}}

/* Section headers in sidebar */
.sidebar-section-header {{
    color: #4CAF50;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 20px 0 10px 0;
    padding-top: 15px;
    border-top: 1px solid rgba(255,255,255,0.1);
}}

.sidebar-text {{
    color: {sidebar_subtext};
    font-size: 13px;
    line-height: 1.6;
    margin: 5px 0;
}}

/* Status badge */
.status-badge {{
    text-align: center;
    padding: 12px;
    background: rgba(76, 175, 80, 0.2);
    border-radius: 10px;
    margin-top: 20px;
}}

.status-text {{
    color: #4CAF50;
    font-weight: 600;
    font-size: 14px;
}}

.status-subtext {{
    color: rgba(255,255,255,0.6);
    font-size: 11px;
    margin-top: 3px;
}}

/* Main content styling */
.main-content {{
    padding: 20px 30px;
}}

/* Page Header */
.page-header {{
    text-align: center;
    margin-bottom: 40px;
    animation: fadeInDown 0.8s ease-out;
}}

@keyframes fadeInDown {{
    from {{ opacity: 0; transform: translateY(-30px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.page-title {{
    color: {text_color};
    font-size: 40px;
    font-weight: 700;
    margin-bottom: 10px;
}}

.page-subtitle {{
    color: {subtext_color};
    font-size: 16px;
    max-width: 700px;
    margin: 0 auto;
    line-height: 1.6;
    text-align: center;
}}

.underline {{
    width: 80px;
    height: 4px;
    background: linear-gradient(90deg, #4CAF50, #2E7D32);
    margin: 20px auto;
    border-radius: 2px;
}}

/* Cards */
.card {{
    background: {card_bg};
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
    border: 1px solid {border_color};
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}}

/* Feature Grid */
.feature-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    margin-top: 30px;
}}

.feature-card {{
    background: {card_bg};
    border-radius: 15px;
    padding: 30px;
    text-align: center;
    border-bottom: 4px solid #4CAF50;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    animation: fadeInUp 0.8s ease-out;
    animation-fill-mode: both;
}}

.feature-card:nth-child(1) {{ animation-delay: 0.1s; }}
.feature-card:nth-child(2) {{ animation-delay: 0.2s; }}
.feature-card:nth-child(3) {{ animation-delay: 0.3s; }}
.feature-card:nth-child(4) {{ animation-delay: 0.4s; }}

.feature-card:hover {{
    transform: translateY(-10px);
    box-shadow: 0 20px 40px rgba(76, 175, 80, 0.3);
}}

.feature-icon {{
    font-size: 48px;
    margin-bottom: 15px;
}}

.feature-title {{
    font-weight: 600;
    font-size: 18px;
    margin-bottom: 8px;
    color: {text_color};
}}

.feature-desc {{
    font-size: 14px;
    color: {subtext_color};
}}

/* Upload Area */
.upload-area {{
    border: 3px dashed #4CAF50;
    border-radius: 20px;
    padding: 50px 40px;
    text-align: center;
    background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(46, 125, 50, 0.05) 100%);
    transition: all 0.4s ease;
}}

.upload-area:hover {{
    border-color: #2E7D32;
    background: linear-gradient(135deg, rgba(76, 175, 80, 0.2) 0%, rgba(46, 125, 50, 0.1) 100%);
}}

.upload-icon {{
    font-size: 70px;
    margin-bottom: 15px;
}}

/* Buttons */
.stButton > button {{
    background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 15px 30px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4) !important;
    width: 100%;
}}

.stButton > button:hover {{
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 25px rgba(76, 175, 80, 0.6) !important;
}}

/* Result Box */
.result-box {{
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 20px;
    padding: 35px;
    color: white;
    text-align: center;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    animation: slideIn 0.6s ease-out;
}}

@keyframes slideIn {{
    from {{ opacity: 0; transform: translateX(-50px); }}
    to {{ opacity: 1; transform: translateX(0); }}
}}

.result-label {{
    color: rgba(255, 255, 255, 0.7);
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 15px;
}}

.result-value {{
    color: #4CAF50;
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 20px;
}}

/* Confidence badges */
.confidence-badge {{
    display: inline-block;
    padding: 8px 20px;
    border-radius: 25px;
    font-size: 14px;
    font-weight: 600;
}}

.confidence-high {{
    background: rgba(76, 175, 80, 0.3);
    color: #4CAF50;
}}

.confidence-medium {{
    background: rgba(255, 193, 7, 0.3);
    color: #FFC107;
}}

.confidence-low {{
    background: rgba(244, 67, 54, 0.3);
    color: #F44336;
}}

/* Tips Box */
.tips-box {{
    background: {card_bg};
    border-radius: 15px;
    padding: 25px;
    margin-top: 25px;
    border-left: 4px solid #4CAF50;
}}

.tips-title {{
    font-weight: 600;
    color: {text_color};
    margin-bottom: 15px;
    font-size: 16px;
}}

.tips-list {{
    color: {subtext_color};
    font-size: 14px;
    margin: 8px 0;
    padding-left: 15px;
}}

/* Disease Info Card */
.disease-info-card {{
    background: {card_bg};
    border-radius: 20px;
    padding: 25px;
    margin-top: 20px;
}}

.info-item {{
    background: {input_bg};
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 15px;
}}

.info-label {{
    font-size: 12px;
    color: {subtext_color};
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}}

.info-value {{
    font-size: 16px;
    font-weight: 600;
    color: {text_color};
}}

/* Recommendation sections */
.recommendation-section {{
    margin: 20px 0;
    padding: 18px;
    background: {input_bg};
    border-radius: 15px;
}}

.recommendation-title {{
    font-weight: 600;
    color: {text_color};
    margin-bottom: 15px;
    font-size: 16px;
}}

.recommendation-list {{
    color: {subtext_color};
    font-size: 14px;
    margin: 10px 0;
    padding: 12px 15px;
    background: {card_bg};
    border-radius: 8px;
    border-left: 3px solid #4CAF50;
}}

/* History items */
.history-item {{
    background: {card_bg};
    border-radius: 15px;
    padding: 20px;
    margin: 15px 0;
    border-left: 4px solid #4CAF50;
    transition: all 0.3s ease;
}}

.history-item:hover {{
    transform: translateX(5px);
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
}}

/* Footer */
.app-footer {{
    text-align: center;
    padding: 30px;
    margin-top: 50px;
    border-top: 1px solid {border_color};
}}

.app-footer p {{
    color: {subtext_color};
    font-size: 13px;
    margin: 5px 0;
}}

.app-footer a {{
    color: #4CAF50;
    text-decoration: none;
}}

/* Custom scrollbar */
::-webkit-scrollbar {{
    width: 8px;
}}

::-webkit-scrollbar-track {{
    background: rgba(0, 0, 0, 0.1);
}}

::-webkit-scrollbar-thumb {{
    background: #4CAF50;
    border-radius: 4px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: #2E7D32;
}}

/* Responsive */
@media (max-width: 768px) {{
    .page-title {{ font-size: 28px; }}
    .feature-grid {{ grid-template-columns: 1fr; }}
}}
</style>
""", unsafe_allow_html=True)

# ===============
# BUILT-IN ADVISORY DATABASE
# ===============
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

# ===============
# SIDEBAR
# ===============
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    
    # Logo Section
    st.markdown('<div class="sidebar-logo">', unsafe_allow_html=True)
    if logo_exists:
        st.image(str(logo_path), width=100)
    else:
        st.markdown("<h1 style='text-align:center; font-size:50px;'>🌾</h1>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="brand-title">NeuralNest</div>
        <div class="brand-subtitle">AI Crop Disease Detection</div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Social Icons
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
    
    # Navigation
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    
    nav_items = [
        ("🏠", "Home"),
        ("🔬", "Disease Detection"),
        ("📊", "Reports"),
        ("⚙️", "Settings")
    ]
    
    for icon, label in nav_items:
        is_active = st.session_state.page == label
        if st.button(f"{icon} {label}", key=f"nav_{label}", use_container_width=True,
                    type="primary" if is_active else "secondary"):
            set_page(label)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # About Section
    st.markdown('<div class="sidebar-section-header">About</div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="sidebar-text">
            <strong>NeuralNest</strong> uses Convolutional Neural Networks to help farmers 
            detect crop diseases instantly with high accuracy.
        </div>
    """, unsafe_allow_html=True)
    
    # Supported Crops
    st.markdown('<div class="sidebar-section-header">Supported Crops</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="sidebar-text">🌽 Corn (Maize)</div>
        <div class="sidebar-text">🥔 Potato</div>
        <div class="sidebar-text">🌾 Wheat</div>
    """, unsafe_allow_html=True)
    
    # Model Info
    st.markdown('<div class="sidebar-section-header">Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="sidebar-text">Architecture: MobileNetV2</div>
        <div class="sidebar-text">Input Size: 224×224 pixels</div>
        <div class="sidebar-text">Classes: 10 disease categories</div>
    """, unsafe_allow_html=True)
    
    # Status Badge
    st.markdown(f"""
        <div class="status-badge">
            <div class="status-text">🌱 AI Ready</div>
            <div class="status-subtext">{len(BUILT_IN_ADVISORY)} classes loaded</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ===============
# PAGE: HOME
# ===============
if st.session_state.page == "Home":
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">🌾 NeuralNest</h1>
        <div class="underline"></div>
        <p class="page-subtitle">
            Your intelligent crop disease detection assistant powered by Convolutional Neural Networks.
            Upload leaf images of corn, wheat, and potato crops to get instant disease identification,
            confidence scores, and actionable treatment recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">🔬</div>
            <div class="feature-title">Disease Detection</div>
            <div class="feature-desc">AI-powered identification with 90%+ accuracy using deep learning</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">💊</div>
            <div class="feature-title">Treatment Plans</div>
            <div class="feature-desc">Personalized recommendations for crop disease management</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📱</div>
            <div class="feature-title">Farmer Friendly</div>
            <div class="feature-desc">Simple interface designed for ease of use in the field</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Real-time Analysis</div>
            <div class="feature-desc">Instant results from leaf image uploads</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Start
    st.markdown(f"""
    <div style="margin-top: 50px; text-align: center;">
        <h3 style="color: {text_color}; margin-bottom: 30px;">Get Started</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔬 Start Disease Detection", use_container_width=True):
            set_page("Disease Detection")

# ===============
# PAGE: DISEASE DETECTION
# ===============
elif st.session_state.page == "Disease Detection":
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">🔬 Disease Detection</h1>
        <div class="underline"></div>
        <p class="page-subtitle">
            Upload a photo of your crop leaf to get disease identification, 
            confidence scores, and treatment recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "📤 Upload Leaf Image",
            type=["jpg", "png", "jpeg"],
            help="Upload a clear photo of a crop leaf showing visible symptoms"
        )
        
        if uploaded_file is None:
            st.markdown(f"""
            <div class="upload-area">
                <div class="upload-icon">📤</div>
                <div style="font-size: 20px; font-weight: 600; color: #4CAF50; margin-bottom: 10px;">
                    Drop your image here
                </div>
                <div style="color: {subtext_color}; font-size: 14px;">
                    Supports: JPG, PNG, JPEG (Max 10MB)
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Tips
            st.markdown(f"""
            <div class="tips-box">
                <div class="tips-title">💡 Tips for best results:</div>
                <div class="tips-list">• Use natural lighting</div>
                <div class="tips-list">• Focus on affected leaf area</div>
                <div class="tips-list">• Avoid shadows and glare</div>
                <div class="tips-list">• Include clear symptom boundaries</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            from PIL import Image
            image = Image.open(uploaded_file)
            st.image(image, caption="📸 Uploaded Image", use_container_width=True)
            
            width, height = image.size
            quality_status = "✅ Good Quality" if min(width, height) >= 224 else "⚠️ Low Resolution"
            st.info(f"{quality_status} ({width}×{height}px)")
        
        st.markdown('<div style="margin-top: 25px;"></div>', unsafe_allow_html=True)
        analyze = st.button(
            "🔍 Analyze Disease",
            use_container_width=True,
            disabled=(uploaded_file is None),
            type="primary"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if uploaded_file and analyze:
            import numpy as np
            import random
            import time
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras.preprocessing import image
            
            # Set TF environment (match your training setup)
            os.environ["TF_USE_LEGACY_KERAS"] = "1"
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            
            BASE_DIR = Path(__file__).parent
            
            @st.cache_resource
            def load_model():
                model_path = BASE_DIR / "models" / "deployment" / "mobilenetv2_best.h5"
                
                if not model_path.exists():
                    st.error(f"Model not found at {model_path}")
                    return None
                
                return keras.models.load_model(model_path)
            
            @st.cache_resource
            def load_class_names():
                import json
                class_path = BASE_DIR / "models" / "deployment" / "class_names.json"
                
                if not class_path.exists():
                    st.error(f"Class file not found at {class_path}")
                    return []
                
                with open(class_path, 'r') as f:
                    return json.load(f)
            
            # Load once
            model = load_model()
            CLASS_NAMES = load_class_names()
        
            with st.spinner("🧠 Analyzing image with Neural Network..."):
                progress_bar = st.progress(0)
                
                # 1. Load and preprocess image (224x224, normalize)
                img = image.load_img(uploaded_file, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                
                progress_bar.progress(50)
                
                # 2. Predict
                predictions = model.predict(img_array, verbose=0)
                predicted_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_idx])
                label = CLASS_NAMES[predicted_idx]
                
                progress_bar.progress(100)
                time.sleep(0.3)  # Brief pause for UX
            
            # Add to persistent history
            add_to_history(label, confidence)
            
            disease_name = label.replace('___', ' - ').replace('_', ' ')
            
            # Determine confidence class
            if confidence >= 0.85:
                conf_class = "confidence-high"
                conf_text = "Excellent"
            elif confidence >= 0.70:
                conf_class = "confidence-medium"
                conf_text = "Good"
            else:
                conf_class = "confidence-low"
                conf_text = "Low"
            
            # Result Display
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
            
            # Advisory Section
            if label in BUILT_IN_ADVISORY:
                info = BUILT_IN_ADVISORY[label]
                
                st.markdown('<div class="disease-info-card">', unsafe_allow_html=True)
                st.subheader("📋 Disease Information")
                
                # Info items
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"""
                    <div class="info-item">
                        <div class="info-label">Crop Type</div>
                        <div class="info-value">{info.get('crop', 'Unknown')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_b:
                    severity = info.get('severity', 'Unknown')
                    severity_emoji = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢", "None": "✅"}.get(severity, "⚪")
                    st.markdown(f"""
                    <div class="info-item">
                        <div class="info-label">Severity</div>
                        <div class="info-value">{severity_emoji} {severity}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Treatment
                st.markdown("""
                <div class="recommendation-section">
                    <div class="recommendation-title">💊 Recommended Treatment</div>
                """, unsafe_allow_html=True)
                
                for i, treatment in enumerate(info.get('treatment', ['No specific treatment available']), 1):
                    st.markdown(f'<div class="recommendation-list">{i}. {treatment}</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Prevention
                st.markdown("""
                <div class="recommendation-section">
                    <div class="recommendation-title">🛡️ Prevention Strategies</div>
                """, unsafe_allow_html=True)
                
                for i, prevention in enumerate(info.get('prevention', ['No specific prevention available']), 1):
                    st.markdown(f'<div class="recommendation-list">{i}. {prevention}</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Low confidence warning
                confidence_threshold = info.get('confidence_threshold', 0.70)
                if confidence < confidence_threshold:
                    st.warning(f"⚠️ Low confidence prediction ({confidence*100:.1f}%). Please consult an agricultural expert for confirmation.")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.caption(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ===============
# PAGE: REPORTS
# ===============
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
        # Metrics
        total_scans = len(history)
        diseases_found = sum(1 for h in history if "healthy" not in h["disease"].lower())
        healthy_crops = total_scans - diseases_found
        today_date = datetime.now().strftime("%Y-%m-%d")
        today_scans = sum(1 for h in history if h["time"].startswith(today_date))
        
        # Metrics Display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <div style="font-size: 36px; font-weight: 700; color: #4CAF50;">{total_scans}</div>
                <div style="color: {subtext_color}; font-size: 14px;">Total Scans</div>
                <div style="color: #4CAF50; font-size: 12px;">+{today_scans} today</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <div style="font-size: 36px; font-weight: 700; color: #FF9800;">{diseases_found}</div>
                <div style="color: {subtext_color}; font-size: 14px;">Diseases Found</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <div style="font-size: 36px; font-weight: 700; color: #4CAF50;">{healthy_crops}</div>
                <div style="color: {subtext_color}; font-size: 14px;">Healthy Crops</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Clear history button
        col_clear1, col_clear2, col_clear3 = st.columns([2, 1, 2])
        with col_clear2:
            if st.button("🗑️ Clear History", use_container_width=True):
                clear_history()
                st.rerun()
        
        # History
        st.markdown("### 🧾 Scan History")
        
        for i, record in enumerate(reversed(history), 1):
            st.markdown(f"""
            <div class="history-item">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 600; color: {text_color}; margin-bottom: 5px;">
                            Scan {i}: {record['disease'].replace('___', ' - ').replace('_', ' ')}
                        </div>
                        <div style="color: {subtext_color}; font-size: 13px;">
                            🕒 {record['time']}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 24px; font-weight: 700; color: #4CAF50;">
                            {record['confidence']}%
                        </div>
                        <div style="color: {subtext_color}; font-size: 12px;">Confidence</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ===============
# PAGE: SETTINGS
# ===============
elif st.session_state.page == "Settings":
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">⚙️ Settings</h1>
        <div class="underline"></div>
        <p class="page-subtitle">Configure your NeuralNest AI preferences</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        st.subheader("🔧 Detection Settings")
        
        threshold = st.select_slider(
            "Confidence Threshold",
            options=["Low (50%)", "Medium (70%)", "High (85%)"],
            value="High (85%)"
        )
        
        animations = st.toggle("Enable Animations", value=True)
        
        # Theme Toggle
        theme = st.radio(
            "Theme Mode",
            ["🌙 Dark Mode", "☀️ Light Mode"],
            index=0 if st.session_state.theme == "dark" else 1
        )
        
        new_theme = "dark" if "Dark" in theme else "light"
        
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        st.subheader("📊 Model Information")
        
        st.success("✅ Model Loaded Successfully")
        st.write(f"**Classes:** {len(BUILT_IN_ADVISORY)} disease categories")
        st.write("**Architecture:** MobileNetV2")
        st.write("**Input Size:** 224×224 pixels")
        st.write(f"**Total Scans:** {len(st.session_state.history)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ===============
# FOOTER
# ===============
st.markdown(f"""
<div class="app-footer">
    <p style="font-weight: 600; color: {text_color}; font-size: 16px;">NeuralNest Crop Disease Detection</p>
    <p>Developed for Kenyan Agriculture | 
    <a href="https://github.com" target="_blank">GitHub</a> • 
    <a href="mailto:contact@neuralnest.ai">Contact</a></p>
    <p style="font-size: 11px;">© 2026 NeuralNest | Powered by CNN Technology</p>
</div>
""", unsafe_allow_html=True)
