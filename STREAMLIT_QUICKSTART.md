# Streamlit Integration - Quick Start Guide

## Overview

The Translation API has been **fully integrated** into the Streamlit dashboard. You now have a **unified interface** with a single server running.

## What Changed

### ✅ Removed
- Flask server (`translation_server.py`)
- Flask templates and static files
- Flask dependencies

### ✅ Added
- New Streamlit tab: **"🔊 Call Transcription"**
- Transcription module: `src/ui/transcription_section.py`
- Local transcript caching in `output/transcripts/`

## Quick Start

### Option 1: Automatic Setup (Recommended)

```bash
./start_streamlit.sh
```

This script will:
- Check and install Streamlit if needed
- Launch the unified dashboard

### Option 2: Manual Start

If Streamlit is already installed:

```bash
streamlit run app.py
```

### Option 3: Install Streamlit First

If you need to install Streamlit manually:

```bash
# Method A: Break system packages
pip3 install streamlit --break-system-packages

# Method B: Virtual environment
python3 -m venv venv
source venv/bin/activate
pip install streamlit

# Then run
streamlit run app.py
```

## Using the Transcription Feature

1. **Start Streamlit**: Run `streamlit run app.py`

2. **Open Browser**: Navigate to `http://localhost:8501`

3. **Go to Transcription Tab**: Click on "🔊 Call Transcription"

4. **Enter GLID**: Type a seller's Global ID (e.g., `16939`)

5. **Click "Transcribe Calls"**: The system will:
   - Fetch all calls for that GLID
   - Sort by call date
   - Call Sarvam API for each recording
   - Display speaker-wise transcriptions
   - Cache results locally

## Features

### 🎯 GLID-Based Filtering
- Enter any seller GLID
- Automatically fetches all associated calls
- Sorted chronologically by `call_entered_on`

### 🔊 Sarvam AI Integration
- Calls `http://localhost:8888/translate` for each recording
- Displays language detected
- Shows speaker diarization (speaker_1, speaker_2, etc.)
- Includes timestamps and full text

### 💾 Local Caching
- Transcripts saved to `output/transcripts/<glid>/<call_date>_<call_id>.json`
- Avoids repeated API calls
- Instant loading for cached transcripts

### 📊 Rich Display
- Call metadata (ID, duration, direction, city)
- Audio player for each call
- Expandable sections per call
- Speaker-wise conversation view
- Raw JSON viewer

## Architecture

```
┌─────────────────────────────────────────────────┐
│         Streamlit Dashboard (Port 8501)         │
│                                                 │
│  ┌─────────────┐  ┌──────────────────────────┐ │
│  │  Insights   │  │  Call Transcription      │ │
│  │  Analytics  │  │  (Sarvam AI)             │ │
│  └─────────────┘  └──────────────────────────┘ │
│                            │                    │
└────────────────────────────┼────────────────────┘
                             │
                             ▼
                   ┌─────────────────┐
                   │  Sarvam API     │
                   │  (Port 8888)    │
                   └─────────────────┘
                             │
                             ▼
                   ┌─────────────────┐
                   │  XLSX Dataset   │
                   └─────────────────┘
```

## File Structure

```
IMH-LATEST/
├── app.py                              # Main Streamlit dashboard (UPDATED)
├── src/
│   └── ui/
│       ├── __init__.py                 # UI package init
│       └── transcription_section.py    # Transcription module (NEW)
├── output/
│   └── transcripts/                    # Cached transcripts (NEW)
│       └── <glid>/
│           └── <date>_<call_id>.json
├── start_streamlit.sh                  # Setup script (NEW)
├── requirements.txt                    # Updated (Flask removed)
└── Copy of Data Voice Hackathon_Master.xlsx
```

## Code Structure

### Main Dashboard (`app.py`)
- Imports transcription module
- Adds 5th tab for transcription
- Passes full dataset to transcription UI

### Transcription Module (`src/ui/transcription_section.py`)

**Functions:**
- `render_transcription_ui(df)` - Main UI rendering
- `fetch_calls_for_glid(df, glid)` - Filter and sort calls
- `transcribe_call(audio_url)` - Call Sarvam API
- `render_call_result(call_data)` - Display results
- `save_transcript_locally()` - Cache to JSON
- `load_cached_transcript()` - Load from cache

## Prerequisites

1. **Sarvam API Running**: Ensure `http://localhost:8888/translate` is accessible
2. **Dataset**: XLSX file in project directory
3. **Streamlit**: Install via script or manually

## Troubleshooting

### Issue: "Streamlit not found"
**Solution**: Run `./start_streamlit.sh` or install manually

### Issue: "Cannot connect to Sarvam API"
**Solution**: 
1. Check if Sarvam API is running: `curl http://localhost:8888/translate`
2. Start the API server
3. Verify port 8888 is accessible

### Issue: "No calls found for GLID"
**Solution**:
1. Verify GLID exists in dataset
2. Try a different GLID (e.g., 16939)
3. Check dataset is loaded correctly

## Testing

### Test the Dashboard

```bash
# Start Streamlit
streamlit run app.py

# Open browser
http://localhost:8501

# Navigate to "Call Transcription" tab
# Enter GLID: 16939
# Click "Transcribe Calls"
```

### Verify Features

- ✅ GLID filtering works
- ✅ Calls sorted by date
- ✅ Sarvam API called for each recording
- ✅ Speaker diarization displayed
- ✅ Results cached locally
- ✅ Audio player works
- ✅ Expandable sections work

## Benefits of Integration

### Before (Flask + Streamlit)
- 2 separate servers
- 2 different UIs
- Port conflicts possible
- Duplicate data loading

### After (Streamlit Only)
- ✅ Single unified dashboard
- ✅ One server (port 8501)
- ✅ Consistent UI/UX
- ✅ Shared data loader
- ✅ Simpler deployment

## Next Steps

1. **Start Sarvam API** (if not running)
2. **Run Streamlit**: `./start_streamlit.sh` or `streamlit run app.py`
3. **Test Transcription**: Use GLID 16939 or any other
4. **Explore Features**: Try different GLIDs, check caching, view speakers

---

**Created for**: IndiaMART Insights Engine  
**Powered by**: Sarvam AI Translation API  
**Framework**: Streamlit
