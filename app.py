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
    page_title="VyaparEcho - IndiaMART Voice Insights Engine",
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
    .stApp {
        background: #0e1117 !important;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .echo-header {
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 50%, #5b21b6 100%);
        padding: 3.5rem 2rem;
        text-align: center;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
        margin-bottom: 3rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
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
    
    /* Input label */
    .input-label {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e0e0e0;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    /* Streamlit input field customization */
    .stTextInput > div > div > input {
        border: 3px solid #7c3aed;
        border-radius: 15px;
        padding: 1.2rem 1.5rem;
        font-size: 1.15rem;
        transition: all 0.3s ease;
        text-align: center;
        background: #1e1e1e;
        color: #ffffff;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #8b5cf6;
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1);
        outline: none;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #a0aec0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 1rem 2.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(124, 58, 237, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(124, 58, 237, 0.4);
    }    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Info section */
    .info-section {
        max-width: 900px;
        margin: 3rem auto;
        padding: 2rem;
        background: rgba(30, 30, 30, 0.8);
        border-radius: 15px;
        border: 2px solid rgba(124, 58, 237, 0.3);
    }
    
    /* Info text colors */
    .info-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e0e0e0;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .info-text {
        text-align: center;
        max-width: 800px;
        margin: 0 auto 3rem auto;
        color: #b0b0b0;
        font-size: 1.15rem;
        line-height: 1.6;
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
        border-color: #7c3aed;
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(107, 70, 193, 0.2);
    }
    
    .feature-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #7c3aed;
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
        color: #e0e0e0;
        font-weight: 600;
        border-top: 3px solid rgba(124, 58, 237, 0.3);
        background: rgba(30, 30, 30, 0.8);
        border-radius: 15px;
    }
    
    /* Results section */
    .stSuccess {
        background: rgba(107, 70, 193, 0.1);
        border-left: 4px solid #7c3aed;
        border-radius: 10px;
        color: #2d3748;
    }
    
    .stError {
        border-radius: 10px;
    }
    
    .stInfo {
        background: rgba(107, 70, 193, 0.08);
        border-left: 4px solid #7c3aed;
        border-radius: 10px;
        color: #2d3748;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #7c3aed;
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
        color: #7c3aed;
        border: 2px solid rgba(107, 70, 193, 0.2);
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(107, 70, 193, 0.15);
        border-color: #7c3aed;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #7c3aed !important;
    }
    
    /* Markdown headers in results */
    h3 {
        color: #7c3aed;
        font-weight: 700;
    }
    
    /* API Status */
    .api-status {
        text-align: center;
        margin: 1.5rem 0;
        font-size: 1rem;
        font-weight: 600;
    }
    
    /* Flow Container Styles */
    .flow-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1.5rem;
        margin-top: 2.5rem;
        flex-wrap: wrap;
    }
    
    .flow-step {
        flex: 1;
        min-width: 220px;
        background: white;
        padding: 1.8rem 1.5rem;
        border-radius: 15px;
        border: 2px solid rgba(107, 70, 193, 0.15);
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(107,70,193,0.12);
    }

    .flow-step:hover {
        transform: translateY(-5px);
        border-color: #7c3aed;
        box-shadow: 0 12px 25px rgba(107, 70, 193, 0.25);
    }
    
    .flow-step-icon {
        font-size: 2.5rem;
        color: #7c3aed;
        margin-bottom: 1rem;
    }
    
    .flow-step-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #7c3aed;
        margin-bottom: 0.5rem;
    }
    
    .flow-step-desc {
        font-size: 0.95rem;
        color: #4a5568;
        line-height: 1.6;
    }
    
    .arrow {
        font-size: 2.2rem;
        color: #7c3aed;
        margin: 0 1rem;
    }
    
    @media(max-width: 900px) {
        .arrow {
            display: none;
        }
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

import base64

def get_logo_base64():
    """Load and encode the IndiaMART logo as base64"""
    try:
        with open("indiamart_logo.png", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

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
    # Header Section with Logo
    st.markdown("""
        <div class="echo-header">
            <div style="display: flex; align-items: center; justify-content: center; gap: 2.5rem;">
                <img src="data:image/png;base64,{logo_base64}" 
                style="height: 120px; width: auto;" 
                alt="IndiaMART Logo">
                <div style="text-align: left;">
                    <h1 class="echo-title" style="margin: 0; line-height: 1.2; color: white;">VyaparEcho - IndiaMART Voice Insights Engine</h1>
                    <p class="echo-subtitle" style="margin: 0.8rem 0 0 0; color: rgba(255, 255, 255, 0.95);">
                        Transform voice calls into actionable insights
                    </p>
                </div>
            </div>
        </div>
    """.format(logo_base64=get_logo_base64()), unsafe_allow_html=True)
    
    # Load data
    df = load_raw_data()
    
    if df is None:
        st.error("Failed to load dataset. Please ensure 'Data Voice Hackathon_Master-1.xlsx' is in the current directory.")
        return
    
    # # Info Section
    # st.markdown("""
    #     <div class="info-section">
    #         <div class="info-title">AI-Powered Call Translation & Analysis</div>
    #         <div class="info-text">
    #             Transform your voice call recordings into actionable insights. Our system uses advanced AI 
    #             to transcribe, translate, and analyze seller support calls, helping you understand customer 
    #             interactions and improve service quality.
    #         </div>
    #         <div class="feature-grid">
    #             <div class="feature-card">
    #                 <div class="feature-title">Instant Translation</div>
    #                 <div class="feature-desc">Automatically transcribe and translate calls in multiple languages with high accuracy</div>
    #             </div>
    #             <div class="feature-card">
    #                 <div class="feature-title">Speaker Detection</div>
    #                 <div class="feature-desc">Identify and separate different speakers for clear conversation flow</div>
    #             </div>
    #             <div class="feature-card">
    #                 <div class="feature-title">Smart Caching</div>
    #                 <div class="feature-desc">Previously translated calls load instantly from local cache</div>
    #             </div>
    #         </div>
    # """, unsafe_allow_html=True)

    # Info Section (Updated Flow UI)
    st.html("""
        <div class="info-section">
            
            <div class="info-title">How VyaparEcho Works</div>
            <div class="info-text">
                The system processes your call step-by-step to convert raw audio into structured, actionable insights.
            </div>

            <div class="flow-container">
                
                <div class="flow-step">
                    <div class="flow-step-icon">🎧</div>
                    <div class="flow-step-title">1. Audio Ingestion</div>
                    <div class="flow-step-desc">
                        The system fetches the voice recording using the Call ID and prepares it for processing.
                    </div>
                </div>

                <div class="arrow">➡️</div>

                <div class="flow-step">
                    <div class="flow-step-icon">📝</div>
                    <div class="flow-step-title">2. AI Transcription</div>
                    <div class="flow-step-desc">
                        Speech is converted into precise text with speaker-wise diarization and timestamps.
                    </div>
                </div>

                <div class="arrow">➡️</div>

                <div class="flow-step">
                    <div class="flow-step-icon">🌐</div>
                    <div class="flow-step-title">3. Translation Engine</div>
                    <div class="flow-step-desc">
                        Text is translated into English or any supported language using Sarvam AI's models.
                    </div>
                </div>

                <div class="arrow">➡️</div>

                <div class="flow-step">
                    <div class="flow-step-icon">💡</div>
                    <div class="flow-step-title">4. Insight Generation</div>
                    <div class="flow-step-desc">
                        The translated call is analyzed for sentiment, intent, issues, and actionable insights.
                    </div>
                </div>

            </div>
        </div>
    """)

    
    # Input Section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    # Render the translation UI
    render_call_translation_ui(df)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer Section
    st.markdown("""
        <div class="echo-footer">
            IndiaMART Voice AI Hackathon 2025<br>
            Powered by Sarvam AI Translation API and Gemini 2.5 Flash
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
