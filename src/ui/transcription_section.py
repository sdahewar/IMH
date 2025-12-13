"""
Streamlit Transcription Section - Sarvam AI Integration
Handles call transcription using Sarvam AI Translation API
"""

import os
import json
import requests
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

# Sarvam AI API Configuration
SARVAM_API_URL = "http://localhost:8888/translate"
TRANSCRIPTS_DIR = "output/transcripts"

# Load enhanced segments dataset for BusinessSegment lookup
@st.cache_data
def load_enhanced_segments():
    """Load enhanced segments dataset"""
    try:
        return pd.read_excel("datasets/enhanced_segments.xlsx")
    except Exception as e:
        st.warning(f"Could not load enhanced_segments.xlsx: {e}")
        return None

def get_business_segment(glid: int) -> str:
    """
    Get BusinessSegment for a given GLID from enhanced_segments.xlsx
    
    Args:
        glid: Global ID to lookup
        
    Returns:
        BusinessSegment string or 'N/A' if not found
    """
    segments_df = load_enhanced_segments()
    if segments_df is None:
        return 'N/A'
    
    try:
        # Find the row with matching GLID
        segment_row = segments_df[segments_df['GLID'] == glid]
        if not segment_row.empty:
            return str(segment_row.iloc[0].get('BusinessSegment', 'N/A'))
        return 'N/A'
    except Exception as e:
        return 'N/A'

def generate_insights_report(insights: Dict, call_id: str, glid: str = None) -> str:
    """
    Generate a formatted text report from insights data
    
    Args:
        insights: Dictionary containing all insights data
        call_id: The call ID for reference
        glid: GLID if available
        
    Returns:
        Formatted text report string
    """
    report = []
    report.append("=" * 60)
    report.append("VYAPARECHO - INDIAMART VOICE INSIGHTS REPORT")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Call ID: {call_id}")
    if glid:
        report.append(f"GLID: {glid}")
    report.append("")
    
    # Problems & Actionables
    report.append("-" * 60)
    report.append("IDENTIFIED PROBLEMS & ACTIONABLES")
    report.append("-" * 60)
    problems = insights.get('problems_identified', [])
    if problems:
        for i, problem in enumerate(problems, 1):
            report.append(f"\n{i}. {problem.get('problem_name', 'Unknown')}")
            report.append("   Actionables:")
            for action in problem.get('actionables', []):
                report.append(f"   - {action}")
            action_im = problem.get('action_for_indiamart', '')
            if action_im and str(action_im).strip() and 'no mapped' not in str(action_im).lower():
                report.append(f"   Action for IndiaMART: {action_im}")
    else:
        report.append("No problems identified.")
    report.append("")
    
    # Seller Tone Analysis
    report.append("-" * 60)
    report.append("SELLER TONE ANALYSIS")
    report.append("-" * 60)
    seller_tone = insights.get('seller_tone', {})
    report.append(f"Sentiment: {seller_tone.get('sentiment', 'N/A')}")
    report.append(f"Churn Risk: {seller_tone.get('churn_risk', 'N/A')}")
    report.append(f"Issue Severity: {seller_tone.get('issue_severity', 'N/A')}")
    report.append(f"Engagement Level: {seller_tone.get('engagement_level', 'N/A')}")
    report.append(f"Outcome Prediction: {seller_tone.get('outcome_prediction', 'N/A')}")
    if seller_tone.get('actionables_if_any'):
        report.append("\nRecommended Actions:")
        for action in seller_tone.get('actionables_if_any', []):
            report.append(f"- {action}")
    report.append("")
    
    # Executive Tone Analysis
    report.append("-" * 60)
    report.append("EXECUTIVE PERFORMANCE ANALYSIS")
    report.append("-" * 60)
    exec_tone = insights.get('executive_tone', {})
    report.append(f"Empathy & Professionalism: {exec_tone.get('empathy_and_professionalism', 'N/A')}")
    report.append(f"Clarity & Confidence: {exec_tone.get('clarity_and_confidence', 'N/A')}")
    report.append(f"Persuasion Effectiveness: {exec_tone.get('persuasion_effectiveness', 'N/A')}")
    report.append(f"Problem-Solving: {exec_tone.get('problem_solving_orientation', 'N/A')}")
    report.append(f"Rapport-Building: {exec_tone.get('rapport_building_strength', 'N/A')}")
    if exec_tone.get('actionables_if_any'):
        report.append("\nImprovement Suggestions:")
        for action in exec_tone.get('actionables_if_any', []):
            report.append(f"- {action}")
    
    report.append("")
    report.append("=" * 60)
    report.append("End of Report")
    report.append("=" * 60)
    
    return "\n".join(report)


def fetch_calls_for_glid(df: pd.DataFrame, glid: str) -> pd.DataFrame:
    """
    Fetch all calls for a specific GLID, sorted by call_entered_on
    
    Args:
        df: Full dataset DataFrame
        glid: Global ID to filter by
        
    Returns:
        Filtered and sorted DataFrame
    """
    try:
        glid_int = int(glid)
        calls = df[df['glid'] == glid_int].copy()
        
        if len(calls) > 0:
            # Sort by call_entered_on
            calls = calls.sort_values('call_entered_on', ascending=True)
        
        return calls
    except ValueError:
        st.error(f"Invalid GLID format: {glid}. Please enter a numeric value.")
        return pd.DataFrame()


def fetch_call_by_id(df: pd.DataFrame, click_to_call_id: str) -> Optional[pd.Series]:
    """
    Fetch a single call by click_to_call_id
    
    Args:
        df: Full dataset DataFrame
        click_to_call_id: Call ID to fetch
        
    Returns:
        Single row as Series or None if not found
    """
    try:
        # Try to match as string or int
        call = df[df['click_to_call_id'].astype(str) == str(click_to_call_id)]
        
        if len(call) > 0:
            return call.iloc[0]
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching call: {str(e)}")
        return None


def transcribe_call(audio_url: str, timeout: int = 120) -> Dict[str, Any]:
    """
    Call Sarvam AI Translation API to transcribe audio
    
    Args:
        audio_url: URL of the audio file
        timeout: Request timeout in seconds
        
    Returns:
        API response dictionary
    """
    try:
        response = requests.post(
            SARVAM_API_URL,
            json={'audio_url': audio_url},
            timeout=timeout
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                'status': 'error',
                'message': f'API returned status {response.status_code}',
                'error_details': response.text
            }
            
    except requests.exceptions.Timeout:
        return {
            'status': 'error',
            'message': f'Request timeout after {timeout} seconds'
        }
    except requests.exceptions.ConnectionError:
        return {
            'status': 'error',
            'message': 'Cannot connect to Sarvam API. Is it running on port 8888?'
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


def save_transcript_locally(glid: str, call_date: str, call_id: str, transcript_data: Dict) -> str:
    """
    Save transcript to local JSON file
    
    Args:
        glid: Global ID
        call_date: Call date
        call_id: Call ID
        transcript_data: Transcript data to save
        
    Returns:
        Path to saved file
    """
    # Create directory structure
    glid_dir = os.path.join(TRANSCRIPTS_DIR, str(glid))
    os.makedirs(glid_dir, exist_ok=True)
    
    # Create filename from call date and ID
    safe_date = call_date.replace(':', '-').replace(' ', '_')
    filename = f"{safe_date}_{call_id}.json"
    filepath = os.path.join(glid_dir, filename)
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(transcript_data, f, indent=2, ensure_ascii=False)
    
    return filepath


def load_cached_transcript(glid: str, call_id: str) -> Optional[Dict]:
    """
    Load cached transcript if it exists
    
    Args:
        glid: Global ID
        call_id: Call ID
        
    Returns:
        Cached transcript data or None
    """
    glid_dir = os.path.join(TRANSCRIPTS_DIR, str(glid))
    
    if not os.path.exists(glid_dir):
        return None
    
    # Look for file with matching call_id
    for filename in os.listdir(glid_dir):
        if filename.endswith(f"_{call_id}.json"):
            filepath = os.path.join(glid_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return None
    
    return None


def render_call_result(call_idx: int, call_row: pd.Series, transcript_data: Dict):
    """
    Render a single call's transcription result
    
    Args:
        call_idx: Call index (1-based)
        call_row: Row from DataFrame with call metadata
        transcript_data: Transcription data from API
    """
    # Format call date
    call_date = pd.to_datetime(call_row['call_entered_on'])
    formatted_date = call_date.strftime('%Y-%m-%d %H:%M:%S')
    
    # Create expander for this call
    status_emoji = "✅" if transcript_data.get('status') == 'success' else "❌"
    with st.expander(f"{status_emoji} Call {call_idx} - {formatted_date}", expanded=(call_idx == 1)):
        
        # Call metadata
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Call ID", call_row.get('click_to_call_id', 'N/A'))
        with col2:
            st.metric("Duration", f"{call_row.get('call_duration', 0)}s")
        with col3:
            st.metric("Direction", call_row.get('FLAG_IN_OUT', 'N/A'))
        with col4:
            # Get BusinessSegment from enhanced_segments.xlsx based on GLID
            glid = call_row.get('glid', None)
            business_segment = get_business_segment(glid) if glid else 'N/A'
            # Custom CSS to reduce font size for longer text
            st.markdown("""
                <style>
                [data-testid="stMetricValue"] {
                    font-size: 18px !important;
                }
                </style>
            """, unsafe_allow_html=True)
            st.metric("Business Segment", business_segment)
        
        # Audio player
        audio_url = call_row.get('call_recording_url', '')
        if audio_url and audio_url != 'nan' and str(audio_url) != 'nan':
            st.audio(audio_url)
        else:
            st.warning("No audio URL available")
        
        # Transcription results
        if transcript_data.get('status') == 'success':
            # Language detected
            language_code = transcript_data.get('language_code', 'Unknown')
            st.success(f"🌐 Language Detected: **{language_code}**")
            
            # Speaker-wise transcription
            speakers = transcript_data.get('speakers', [])
            
            if speakers:
                st.markdown("### 💬 Conversation Transcript")
                
                for speaker in speakers:
                    speaker_id = speaker.get('speaker_id', 'Unknown')
                    text = speaker.get('text', '')
                    
                    # Display speaker with colored badge
                    if 'speaker_1' in speaker_id.lower():
                        st.markdown(f"**🔵 {speaker_id}**")
                    else:
                        st.markdown(f"**🟢 {speaker_id}**")
                    
                    st.markdown(f"> {text}")
                    st.markdown("---")
            else:
                st.info("No speaker data available in transcription")
            
        else:
            # Error case
            error_msg = transcript_data.get('message', 'Unknown error')
            st.error(f"**Transcription Failed:** {error_msg}")
            
            if transcript_data.get('error_details'):
                with st.expander("Error Details"):
                    st.code(transcript_data['error_details'])
        
        # Insights Generation Section
        if transcript_data.get('status') == 'success':
            st.markdown("---")
            st.markdown("### 🧠 AI-Powered Insights")
            
            # Get Insights button
            if st.button("🔍 Generate Insights", type="primary", use_container_width=True):
                with st.spinner("Analyzing call with AI... This may take a few seconds"):
                    try:
                        from src.insights_engine import generate_insights
                        
                        # Prepare segment info
                        segment_info = {
                            'glid': call_row.get('glid', 'N/A'),
                            'segment': call_row.get('customer_type', 'N/A'),
                            'business_segment': get_business_segment(call_row.get('glid', None)) if call_row.get('glid') else 'N/A'
                        }
                        
                        # Generate insights
                        insights_result = generate_insights(transcript_data, segment_info)
                        
                        if insights_result.get('status') == 'success':
                            insights = insights_result.get('insights', {})
                            
                            # Display insights in organized sections
                            st.success("✅ Insights generated successfully!")
                            
                            # Problems Identified
                            with st.expander("📋 **Identified Problems & Actionables**", expanded=True):
                                problems = insights.get('problems_identified', [])
                                if problems:
                                    for i, problem in enumerate(problems, 1):
                                        # Problem card with styled HTML
                                        problem_name = problem.get('problem_name', 'Unknown')
                                        actionables = problem.get('actionables', [])
                                        action_im = problem.get('action_for_indiamart', '')
                                        
                                        actionable_html = ''.join([f'<li style="margin: 8px 0; padding-left: 10px; border-left: 3px solid #7c3aed;">{action}</li>' for action in actionables])
                                        
                                        im_action_html = f'<div style="margin-top: 15px; padding: 12px; background: rgba(124, 58, 237, 0.15); border-radius: 8px; border-left: 4px solid #7c3aed;"><strong>🎯 Action for IndiaMART:</strong> {action_im}</div>' if action_im and action_im.strip() and 'no mapped' not in action_im.lower() else ''
                                        
                                        st.markdown(f'''
                                        <div style="background: rgba(30, 30, 30, 0.6); padding: 20px; border-radius: 12px; margin-bottom: 20px; border: 1px solid rgba(124, 58, 237, 0.3);">
                                            <h4 style="color: #a78bfa; margin: 0 0 15px 0; font-size: 1.2rem;">🔹 {i}. {problem_name}</h4>
                                            <div style="color: #9ca3af; font-size: 0.95rem; margin-bottom: 10px;"><strong>Actionables:</strong></div>
                                            <ul style="list-style: none; padding: 0; margin: 0; color: #e5e7eb;">
                                                {actionable_html}
                                            </ul>
                                            {im_action_html}
                                        </div>
                                        ''', unsafe_allow_html=True)
                                else:
                                    st.info("No mapped problems found in this call.")
                            
                            # Seller Tone Analysis
                            with st.expander("😊 **Seller Tone Analysis**"):
                                seller_tone = insights.get('seller_tone', {})
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Sentiment", seller_tone.get('sentiment', 'N/A'))
                                    st.metric("Churn Risk", seller_tone.get('churn_risk', 'N/A'))
                                    st.metric("Issue Severity", seller_tone.get('issue_severity', 'N/A'))
                                
                                with col2:
                                    st.metric("Upsell Readiness", seller_tone.get('upsell_readiness', 'N/A'))
                                    st.metric("Engagement Level", seller_tone.get('engagement_level', 'N/A'))
                                    st.metric("Trust & Rapport", seller_tone.get('trust_and_rapport', 'N/A'))
                                
                                with col3:
                                    st.metric("Outcome Prediction", seller_tone.get('outcome_prediction', 'N/A'))
                                    st.metric("Follow-up Willingness", seller_tone.get('follow_up_willingness', 'N/A'))
                                
                                if seller_tone.get('emotional_trigger_points'):
                                    st.markdown(f"**Emotional Triggers:** {seller_tone.get('emotional_trigger_points')}")
                                
                                if seller_tone.get('actionables_if_any'):
                                    st.markdown("**Recommended Actions:**")
                                    for action in seller_tone.get('actionables_if_any', []):
                                        st.markdown(f"• {action}")
                            
                            # Executive Tone Analysis
                            with st.expander("👨‍💼 **Executive Performance Analysis**"):
                                exec_tone = insights.get('executive_tone', {})
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**Empathy & Professionalism:** {exec_tone.get('empathy_and_professionalism', 'N/A')}")
                                    st.markdown(f"**Clarity & Confidence:** {exec_tone.get('clarity_and_confidence', 'N/A')}")
                                    st.markdown(f"**Persuasion Effectiveness:** {exec_tone.get('persuasion_effectiveness', 'N/A')}")
                                
                                with col2:
                                    st.markdown(f"**Problem-Solving:** {exec_tone.get('problem_solving_orientation', 'N/A')}")
                                    st.markdown(f"**Rapport-Building:** {exec_tone.get('rapport_building_strength', 'N/A')}")
                                
                                if exec_tone.get('actionables_if_any'):
                                    st.markdown("**Improvement Suggestions:**")
                                    for action in exec_tone.get('actionables_if_any', []):
                                        st.markdown(f"• {action}")
                                        
                            # Download Report Button (OUTSIDE the expander)
                            st.markdown("---")
                            report_text = generate_insights_report(
                                insights, 
                                call_id, 
                                str(call_row.get('glid', ''))
                            )
                            st.download_button(
                                label="📥 Download Insights Report",
                                data=report_text,
                                file_name=f"insights_report_{call_id}.txt",
                                mime="text/plain",
                                help="Download a text file containing all insights"
                            )
                        
                        else:
                            st.error(f"❌ Insights generation failed: {insights_result.get('message', 'Unknown error')}")
                            if insights_result.get('raw_response'):
                                with st.expander("Raw LLM Response"):
                                    st.code(insights_result['raw_response'])
                    
                    except ImportError:
                        st.error("❌ Insights engine not available. Please install required dependencies: `pip install google-generativeai python-dotenv`")
                    except Exception as e:
                        st.error(f"❌ Error generating insights: {str(e)}")


def render_transcription_ui(df: pd.DataFrame):
    """
    Main UI rendering function for the transcription section
    
    Args:
        df: Full dataset DataFrame
    """
    st.header("🔊 Call Transcription (Sarvam AI)")
    st.markdown("Transcribe and analyze call recordings using Sarvam AI Translation API")
    
    # Check if Sarvam API is accessible
    try:
        health_check = requests.get(SARVAM_API_URL.replace('/translate', '/health'), timeout=2)
        api_status = "🟢 Online"
    except:
        api_status = "🔴 Offline"
    
    st.info(f"**Sarvam API Status:** {api_status} | **Endpoint:** {SARVAM_API_URL}")
    
    # GLID input
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        glid_input = st.text_input(
            "Enter Seller GLID (Global ID)",
            placeholder="e.g., 16939",
            help="Enter the Global ID of the seller to fetch all their call recordings"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        transcribe_button = st.button("🎙️ Transcribe Calls", type="primary", use_container_width=True)
    
    # Process transcription
    if transcribe_button:
        if not glid_input or not glid_input.strip():
            st.error("Please enter a GLID")
            return
        
        glid = glid_input.strip()
        
        # Fetch calls
        with st.spinner(f"Fetching calls for GLID {glid}..."):
            calls = fetch_calls_for_glid(df, glid)
        
        if len(calls) == 0:
            st.warning(f"No calls found for GLID {glid}")
            return
        
        # Display call count
        st.success(f"Found **{len(calls)}** call(s) for GLID {glid}")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Results container
        results_container = st.container()
        
        # Transcribe each call
        transcription_results = []
        
        for idx, (_, call_row) in enumerate(calls.iterrows(), 1):
            # Update progress
            progress = idx / len(calls)
            progress_bar.progress(progress)
            status_text.text(f"Transcribing call {idx} of {len(calls)}...")
            
            call_id = str(call_row.get('click_to_call_id', ''))
            audio_url = str(call_row.get('call_recording_url', ''))
            call_date = str(call_row.get('call_entered_on', ''))
            
            # Check cache first
            cached_transcript = load_cached_transcript(glid, call_id)
            
            if cached_transcript:
                st.info(f"Using cached transcript for Call {idx}")
                transcript_data = cached_transcript
            else:
                # Skip if no audio URL
                if not audio_url or audio_url == 'nan':
                    transcript_data = {
                        'status': 'error',
                        'message': 'No audio URL available'
                    }
                else:
                    # Call Sarvam API
                    transcript_data = transcribe_call(audio_url)
                    
                    # Save to cache if successful
                    if transcript_data.get('status') == 'success':
                        save_transcript_locally(glid, call_date, call_id, transcript_data)
            
            transcription_results.append({
                'call_row': call_row,
                'transcript': transcript_data
            })
        
        # Clear progress
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 Transcription Results")
        
        # Summary stats
        successful = sum(1 for r in transcription_results if r['transcript'].get('status') == 'success')
        failed = len(transcription_results) - successful
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Calls", len(transcription_results))
        with col2:
            st.metric("Successful", successful, delta=None)
        with col3:
            st.metric("Failed", failed, delta=None)
        
        st.markdown("---")
        
        # Render each call result
        with results_container:
            for idx, result in enumerate(transcription_results, 1):
                render_call_result(idx, result['call_row'], result['transcript'])
        
        st.success("✅ Transcription complete!")
    
    # Show cached transcripts info
    st.markdown("---")
    st.markdown("### 💾 Cached Transcripts")
    
    if os.path.exists(TRANSCRIPTS_DIR):
        total_cached = 0
        for glid_folder in os.listdir(TRANSCRIPTS_DIR):
            glid_path = os.path.join(TRANSCRIPTS_DIR, glid_folder)
            if os.path.isdir(glid_path):
                count = len([f for f in os.listdir(glid_path) if f.endswith('.json')])
                total_cached += count
        
        st.info(f"**{total_cached}** transcripts cached locally in `{TRANSCRIPTS_DIR}/`")
    else:
        st.info("No cached transcripts yet")


def render_call_translation_ui(df: pd.DataFrame):
    """
    Simplified UI for single call translation using click_to_call_id
    
    Args:
        df: Full dataset DataFrame
    """
    
    # Check if Sarvam API is accessible
    try:
        health_check = requests.get(SARVAM_API_URL.replace('/translate', '/health'), timeout=2)
        api_status = "Online"
        api_color = "#28a745"
    except:
        api_status = "Offline"
        api_color = "#dc3545"
    
    # API Status indicator
    st.markdown(f"""
        <div style='text-align: center; margin: 20px 0;'>
            <span style='color: {api_color}; font-weight: 600;'>Sarvam API Status: {api_status}</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Input section
    call_id_input = st.text_input(
        "",
        placeholder="Enter Call ID (e.g., 12345678)",
        key="call_id_input",
        label_visibility="collapsed"
    )
    
    # Center the button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        translate_button = st.button(
            "Generate Translation",
            type="primary",
            use_container_width=True,
            key="translate_btn"
        )
    
    # Initialize session state for translation
    if 'current_call_id' not in st.session_state:
        st.session_state.current_call_id = None
    if 'current_call_row' not in st.session_state:
        st.session_state.current_call_row = None
    if 'current_transcript' not in st.session_state:
        st.session_state.current_transcript = None
    if 'current_audio_url' not in st.session_state:
        st.session_state.current_audio_url = None
    
    # Process translation
    if translate_button:
        if not call_id_input or not call_id_input.strip():
            st.error("Please enter a Call ID")
        else:
            call_id = call_id_input.strip()
            
            # Fetch call from dataset
            with st.spinner(f"Fetching call {call_id}..."):
                call_row = fetch_call_by_id(df, call_id)
            
            if call_row is None:
                st.error(f"No call found with ID: {call_id}")
                st.info("Tip: Make sure you're entering a valid click_to_call_id from the dataset")
            else:
                # Get audio URL
                audio_url = str(call_row.get('call_recording_url', ''))
                
                if not audio_url or audio_url == 'nan':
                    st.error("No audio recording URL found for this call")
                else:
                    # Check cache first
                    cached_transcript = load_cached_transcript(
                        str(call_row.get('glid', '')), 
                        call_id
                    )
                    
                    if cached_transcript:
                        st.success("Using cached translation")
                        transcript_data = cached_transcript
                    else:
                        # Call Sarvam API
                        with st.spinner("Translating audio... This may take up to few seconds"):
                            transcript_data = transcribe_call(audio_url)
                        
                        # Save to cache if successful
                        if transcript_data.get('status') == 'success':
                            save_transcript_locally(
                                str(call_row.get('glid', '')),
                                str(call_row.get('call_entered_on', '')),
                                call_id,
                                transcript_data
                            )
                    
                    # Store in session state
                    st.session_state.current_call_id = call_id
                    st.session_state.current_call_row = call_row
                    st.session_state.current_transcript = transcript_data
                    st.session_state.current_audio_url = audio_url
    
    # Display translation results if available in session state
    if st.session_state.current_transcript is not None:
        call_id = st.session_state.current_call_id
        call_row = st.session_state.current_call_row
        transcript_data = st.session_state.current_transcript
        audio_url = st.session_state.current_audio_url
        
        # Display results in an expander (popup-like)
        st.markdown("---")
        
        if transcript_data.get('status') == 'success':
            # Success case
            st.success("Translation Complete!")
            
            # Call metadata
            st.markdown("### Call Details")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("GLID", call_row.get('glid', 'N/A'))
            with col2:
                st.metric("Duration", f"{call_row.get('call_duration', 0)}s")
            with col3:
                st.metric("Direction", call_row.get('FLAG_IN_OUT', 'N/A'))
            with col4:
                # Get BusinessSegment from enhanced_segments.xlsx based on GLID
                glid = call_row.get('glid', None)
                business_segment = get_business_segment(glid) if glid else 'N/A'
                # Custom CSS to reduce font size for longer text
                st.markdown("""
                    <style>
                    [data-testid="stMetricValue"] {
                        font-size: 18px !important;
                    }
                    </style>
                """, unsafe_allow_html=True)
                st.metric("Business Segment", business_segment)
            
            # Audio player
            st.markdown("### Audio Recording")
            st.audio(audio_url)
            
            # Language detected
            language_code = transcript_data.get('language_code', 'Unknown')
            st.info(f"**Detected Language:** {language_code}")
            
            # Speaker-wise transcription in expandable box
            speakers = transcript_data.get('speakers', [])
            
            if speakers:
                # Calculate total text length for dynamic sizing
                total_text_length = sum(len(speaker.get('text', '')) for speaker in speakers)
                
                # Determine font size based on text length
                if total_text_length < 500:
                    font_size = '1rem'
                elif total_text_length < 1500:
                    font_size = '0.95rem'
                elif total_text_length < 3000:
                    font_size = '0.9rem'
                else:
                    font_size = '0.85rem'
                
                # Build consolidated transcript text
                transcript_lines = []
                for speaker in speakers:
                    speaker_id = speaker.get('speaker_id', 'Unknown')
                    text = speaker.get('text', '')
                    transcript_lines.append(f"<strong>{speaker_id}:</strong> {text}")
                
                consolidated_text = "<br><br>".join(transcript_lines)
                
                with st.expander("Show Transcripted Text", expanded=False):
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #6b46c1 0%, #553c9a 100%); 
                                    color: white; 
                                    padding: 20px; 
                                    border-radius: 12px; 
                                    max-height: 600px;
                                    overflow-y: auto;
                                    font-size: {font_size};
                                    line-height: 1.8;
                                    box-shadow: 0 4px 12px rgba(107, 70, 193, 0.3);'>
                            {consolidated_text}
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No speaker data available in transcription")
            
            # Insights Generation Section
            st.markdown("---")
            st.markdown("### 🧠 AI-Powered Insights")
            
            # Initialize session state for insights
            if 'insights_data' not in st.session_state:
                st.session_state.insights_data = None
            if 'insights_call_id' not in st.session_state:
                st.session_state.insights_call_id = None
            
            # Get Insights button
            if st.button("🔍 Generate Insights", type="primary", use_container_width=True, key="insights_btn_main"):
                with st.spinner("Analyzing call with AI... This may take a few seconds"):
                    try:
                        from src.insights_engine import generate_insights
                        
                        # Prepare segment info
                        segment_info = {
                            'glid': call_row.get('glid', 'N/A'),
                            'segment': call_row.get('customer_type', 'N/A'),
                            'business_segment': get_business_segment(call_row.get('glid', None)) if call_row.get('glid') else 'N/A'
                        }
                        
                        # Generate insights
                        insights_result = generate_insights(transcript_data, segment_info)
                        
                        # Store in session state
                        st.session_state.insights_data = insights_result
                        st.session_state.insights_call_id = call_id
                        
                        if insights_result.get('status') != 'success':
                            st.error(f"❌ Insights generation failed: {insights_result.get('message', 'Unknown error')}")
                    
                    except ImportError:
                        st.error("❌ Insights engine not available. Please install required dependencies: `pip install google-generativeai python-dotenv`")
                    except Exception as e:
                        st.error(f"❌ Error generating insights: {str(e)}")
            
            # Display insights if available in session state (outside button handler)
            if st.session_state.insights_data is not None and st.session_state.insights_call_id == call_id:
                insights_result = st.session_state.insights_data
                
                if insights_result.get('status') == 'success':
                    insights = insights_result.get('insights', {})
                    
                    # Display insights in organized sections
                    st.success("✅ Insights generated successfully!")
                    
                    # Problems Identified
                    with st.expander("📋 **Identified Problems & Actionables**", expanded=True):
                        problems = insights.get('problems_identified', [])
                        if problems:
                            for i, problem in enumerate(problems, 1):
                                # Problem card with styled HTML
                                problem_name = problem.get('problem_name', 'Unknown')
                                actionables = problem.get('actionables', [])
                                action_im = problem.get('action_for_indiamart', '')
                                
                                actionable_html = ''.join([f'<li style="margin: 8px 0; padding-left: 10px; border-left: 3px solid #7c3aed;">{action}</li>' for action in actionables])
                                
                                im_action_html = f'<div style="margin-top: 15px; padding: 12px; background: rgba(124, 58, 237, 0.15); border-radius: 8px; border-left: 4px solid #7c3aed;"><strong>🎯 Action for IndiaMART:</strong> {action_im}</div>' if action_im and action_im.strip() and 'no mapped' not in action_im.lower() else ''
                                
                                st.markdown(f'''
                                <div style="background: rgba(30, 30, 30, 0.6); padding: 20px; border-radius: 12px; margin-bottom: 20px; border: 1px solid rgba(124, 58, 237, 0.3);">
                                    <h4 style="color: #a78bfa; margin: 0 0 15px 0; font-size: 1.2rem;">🔹 {i}. {problem_name}</h4>
                                    <div style="color: #9ca3af; font-size: 0.95rem; margin-bottom: 10px;"><strong>Actionables:</strong></div>
                                    <ul style="list-style: none; padding: 0; margin: 0; color: #e5e7eb;">
                                        {actionable_html}
                                    </ul>
                                    {im_action_html}
                                </div>
                                ''', unsafe_allow_html=True)
                        else:
                            st.info("No mapped problems found in this call.")
                    
                    # Seller Tone Analysis
                    with st.expander("😊 **Seller Tone Analysis**"):
                        seller_tone = insights.get('seller_tone', {})
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Sentiment", seller_tone.get('sentiment', 'N/A'))
                            st.metric("Churn Risk", seller_tone.get('churn_risk', 'N/A'))
                            st.metric("Issue Severity", seller_tone.get('issue_severity', 'N/A'))
                        
                        with col2:
                            st.metric("Upsell Readiness", seller_tone.get('upsell_readiness', 'N/A'))
                            st.metric("Engagement Level", seller_tone.get('engagement_level', 'N/A'))
                            st.metric("Trust & Rapport", seller_tone.get('trust_and_rapport', 'N/A'))
                        
                        with col3:
                            st.metric("Outcome Prediction", seller_tone.get('outcome_prediction', 'N/A'))
                            st.metric("Follow-up Willingness", seller_tone.get('follow_up_willingness', 'N/A'))
                        
                        if seller_tone.get('emotional_trigger_points'):
                            st.markdown(f"**Emotional Triggers:** {seller_tone.get('emotional_trigger_points')}")
                        
                        if seller_tone.get('actionables_if_any'):
                            st.markdown("**Recommended Actions:**")
                            for action in seller_tone.get('actionables_if_any', []):
                                st.markdown(f"• {action}")
                    
                    # Executive Tone Analysis
                    with st.expander("👨‍💼 **Executive Performance Analysis**"):
                        exec_tone = insights.get('executive_tone', {})
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Empathy & Professionalism:** {exec_tone.get('empathy_and_professionalism', 'N/A')}")
                            st.markdown(f"**Clarity & Confidence:** {exec_tone.get('clarity_and_confidence', 'N/A')}")
                            st.markdown(f"**Persuasion Effectiveness:** {exec_tone.get('persuasion_effectiveness', 'N/A')}")
                        
                        with col2:
                            st.markdown(f"**Problem-Solving:** {exec_tone.get('problem_solving_orientation', 'N/A')}")
                            st.markdown(f"**Rapport-Building:** {exec_tone.get('rapport_building_strength', 'N/A')}")
                        
                        if exec_tone.get('actionables_if_any'):
                            st.markdown("**Improvement Suggestions:**")
                            for action in exec_tone.get('actionables_if_any', []):
                                st.markdown(f"• {action}")
                    
                    # Download Report Button (OUTSIDE the expander)
                    st.markdown("---")
                    report_text = generate_insights_report(
                        insights, 
                        call_id, 
                        str(call_row.get('glid', ''))
                    )
                    st.download_button(
                        label="📥 Download Insights Report",
                        data=report_text,
                        file_name=f"insights_report_{call_id}.txt",
                        mime="text/plain",
                        help="Download a text file containing all insights"
                    )
                
                else:
                    if insights_result.get('raw_response'):
                        with st.expander("Raw LLM Response"):
                            st.code(insights_result['raw_response'])
        else:
            # Error case
            error_msg = transcript_data.get('message', 'Unknown error')
            st.error(f"**Translation Failed:** {error_msg}")
            
            if transcript_data.get('error_details'):
                with st.expander("Error Details"):
                    st.code(transcript_data['error_details'])
