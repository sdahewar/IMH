#!/bin/bash

# Streamlit Setup and Launch Script
# This script installs Streamlit if needed and launches the unified dashboard

echo "=============================================="
echo "  IndiaMART Insights Engine - Streamlit"
echo "=============================================="
echo ""

# Check if Streamlit is installed
if python3 -c "import streamlit" 2>/dev/null; then
    echo "✅ Streamlit is already installed"
else
    echo "📦 Installing Streamlit..."
    
    # Try pip install with --break-system-packages (for externally managed environments)
    if pip3 install streamlit --break-system-packages 2>/dev/null; then
        echo "✅ Streamlit installed successfully"
    else
        echo "⚠️  Could not install Streamlit automatically"
        echo ""
        echo "Please install Streamlit manually using one of these methods:"
        echo "  1. Using system package manager: sudo apt install python3-streamlit"
        echo "  2. Using virtual environment:"
        echo "     python3 -m venv venv"
        echo "     source venv/bin/activate"
        echo "     pip install streamlit"
        echo "  3. Using pipx: pipx install streamlit"
        echo ""
        exit 1
    fi
fi

echo ""
echo "=============================================="
echo "  Starting Streamlit Dashboard"
echo "=============================================="
echo ""
echo "📊 Dashboard will be available at: http://localhost:8501"
echo "🔊 Transcription tab includes Sarvam AI integration"
echo ""

# Start Streamlit
streamlit run app.py
