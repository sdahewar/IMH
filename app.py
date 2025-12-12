"""
IndiaMART Echo - Voice Call Translation Interface
Streamlit-based UI for translating call recordings using Sarvam AI

Run with: streamlit run app.py
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd

# Import transcription UI module
from src.ui.transcription_section import render_call_translation_ui

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="IndiaMART Echo",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Creamish and Purple Theme
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #faf8f3 0%, #f5f1e8 100%);
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .echo-header {
        background: linear-gradient(135deg, #6b46c1 0%, #553c9a 100%);
        padding: 4rem 2rem;
        text-align: center;
        border-radius: 20px;
        box-shadow: 0 15px 40px rgba(107, 70, 193, 0.25);
        margin-bottom: 3rem;
    }
    
    .echo-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        letter-spacing: -1px;
    }
    
    .echo-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.95);
        margin: 1.5rem 0 0 0;
        font-weight: 400;
        line-height: 1.6;
    }
    
    /* Input section - No white container */
    .input-section {
        max-width: 700px;
        margin: 0 auto 3rem auto;
    }
    
    .input-label {
        font-size: 1.3rem;
        font-weight: 600;
        color: #6b46c1;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    /* Streamlit input field customization */
    .stTextInput > div > div > input {
        border: 3px solid #6b46c1;
        border-radius: 15px;
        padding: 1.2rem 1.5rem;
        font-size: 1.15rem;
        transition: all 0.3s ease;
        text-align: center;
        background: white;
        color: #2d3748;
        font-weight: 500;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #553c9a;
        box-shadow: 0 0 0 4px rgba(107, 70, 193, 0.15);
        outline: none;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #a0aec0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6b46c1 0%, #553c9a 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 1rem 3rem;
        font-size: 1.2rem;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(107, 70, 193, 0.35);
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(107, 70, 193, 0.5);
        background: linear-gradient(135deg, #553c9a 0%, #6b46c1 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Info section */
    .info-section {
        max-width: 900px;
        margin: 3rem auto;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.6);
        border-radius: 15px;
        border: 2px solid rgba(107, 70, 193, 0.2);
    }
    
    .info-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #6b46c1;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .info-text {
        font-size: 1rem;
        color: #4a5568;
        line-height: 1.8;
        text-align: center;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid rgba(107, 70, 193, 0.15);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        border-color: #6b46c1;
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(107, 70, 193, 0.2);
    }
    
    .feature-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #6b46c1;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-size: 0.95rem;
        color: #718096;
        line-height: 1.6;
    }
    
    /* Footer styling */
    .echo-footer {
        text-align: center;
        padding: 2.5rem 2rem;
        margin-top: 4rem;
        color: #6b46c1;
        font-weight: 600;
        border-top: 3px solid rgba(107, 70, 193, 0.2);
        background: rgba(255, 255, 255, 0.4);
        border-radius: 15px;
    }
    
    /* Results section */
    .stSuccess {
        background: rgba(107, 70, 193, 0.1);
        border-left: 4px solid #6b46c1;
        border-radius: 10px;
        color: #2d3748;
    }
    
    .stError {
        border-radius: 10px;
    }
    
    .stInfo {
        background: rgba(107, 70, 193, 0.08);
        border-left: 4px solid #6b46c1;
        border-radius: 10px;
        color: #2d3748;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #6b46c1;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #4a5568;
        font-weight: 600;
    }
    
    /* Audio player */
    audio {
        width: 100%;
        border-radius: 12px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(107, 70, 193, 0.1);
        border-radius: 12px;
        font-weight: 600;
        color: #6b46c1;
        border: 2px solid rgba(107, 70, 193, 0.2);
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(107, 70, 193, 0.15);
        border-color: #6b46c1;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #6b46c1 !important;
    }
    
    /* Markdown headers in results */
    h3 {
        color: #6b46c1;
        font-weight: 700;
    }
    
    /* API Status */
    .api-status {
        text-align: center;
        margin: 1.5rem 0;
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_raw_data():
    """Load the raw Excel data"""
    # Try multiple possible locations
    paths = ["datasets/Data Voice Hackathon_Master-1.xlsx", "Data Voice Hackathon_Master-1.xlsx", "data/Data Voice Hackathon_Master-1.xlsx"]
    for path in paths:
        if os.path.exists(path):
            try:
                return pd.read_excel(path)
            except:
                pass
    return None


# =============================================================================
# MAIN INTERFACE
# =============================================================================

def main():
    # Header Section
    st.markdown("""
        <div class="echo-header">
            <h1 class="echo-title">IndiaMART Echo</h1>
            <p class="echo-subtitle">
                Linking the Promise to the Reality across IM - Seller Calls
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_raw_data()
    
    if df is None:
        st.error("Failed to load dataset. Please ensure 'Data Voice Hackathon_Master-1.xlsx' is in the current directory.")
        return
    
    # Info Section
    st.markdown("""
        <div class="info-section">
            <div class="info-title">AI-Powered Call Translation & Analysis</div>
            <div class="info-text">
                Transform your voice call recordings into actionable insights. Our system uses advanced AI 
                to transcribe, translate, and analyze seller support calls, helping you understand customer 
                interactions and improve service quality.
            </div>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-title">Instant Translation</div>
                    <div class="feature-desc">Automatically transcribe and translate calls in multiple languages with high accuracy</div>
                </div>
                <div class="feature-card">
                    <div class="feature-title">Speaker Detection</div>
                    <div class="feature-desc">Identify and separate different speakers for clear conversation flow</div>
                </div>
                <div class="feature-card">
                    <div class="feature-title">Smart Caching</div>
                    <div class="feature-desc">Previously translated calls load instantly from local cache</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Input Section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="input-label">Enter Call ID to Begin Translation</div>', unsafe_allow_html=True)
    
    # Render the translation UI
    render_call_translation_ui(df)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer Section
    st.markdown("""
        <div class="echo-footer">
            IndiaMART Voice AI Hackathon 2025<br>
            Powered by Sarvam AI Translation API
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
