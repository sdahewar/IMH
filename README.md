# IndiaMART Echo - Call Translation & Insights

AI-powered call translation and insights engine for IndiaMART's Voice Call Hackathon.

## Features

- **Call Translation**: Translate call recordings using Sarvam AI
- **Business Segment Detection**: Automatically fetch business segments
- **AI Insights**: Generate actionable insights using Google Gemini LLM
- **Problem Identification**: Map calls to known problems with actionables
- **Tone Analysis**: Analyze both seller and executive tones
- **Segment-Aware**: Provide segment-specific recommendations

## Setup

### 1. Install Dependencies

```bash
pip install streamlit pandas openpyxl requests google-generativeai python-dotenv
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```bash
# Google Gemini API Key for Insights Engine
GEMINI_API_KEY=your_gemini_api_key_here

# Custom LLM API Endpoint (for IndiaMART's internal LLM)
API_BASE_URL_FOR_CLIENT=https://imllm.intermesh.net
```

**Get a Gemini API key:**
1. Go to https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key and paste it in `.env`

### 3. Ensure Data Files

Make sure these files exist in the `datasets/` folder:
- `datasets/Data Voice Hackathon_Master-1.xlsx` - Main dataset
- `datasets/enhanced_segments.xlsx` - Business segment data
- `datasets/Problem_Actionable_Final.xlsx` - Problem definitions with actionables

### 4. Start the Application

```bash
bash start_streamlit.sh
```

Or directly:

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Usage

### Translate a Call

1. Enter a **Call ID** (click_to_call_id)
2. Click **"Fetch & Translate Call"**
3. View the translated conversation

### Generate Insights

1. After translation completes, click **"🔍 Generate Insights"**
2. Wait for AI analysis (5-10 seconds)
3. Review insights in expandable sections:
   - **Identified Problems & Actionables**
   - **Seller Tone Analysis**
   - **Executive Performance Analysis**
   - **Segment-Specific Insights**

## Insights Output Structure

The insights engine provides:

### Problems Identified
- Problem name from reference list
- Evidence from transcript
- Specific actionables

### Seller Tone Analysis
- Sentiment (Positive/Neutral/Negative)
- Churn Risk Signal
- Upsell Readiness
- Issue Severity
- Engagement Level
- Trust & Rapport Score
- Emotional Triggers
- Outcome Prediction
- Follow-up Willingness

### Executive Tone Analysis
- Empathy & Professionalism
- Clarity & Confidence
- Persuasion Effectiveness
- Problem-Solving Orientation
- Rapport-Building Strength

### Segment-Specific Insights
- Segment relevance
- Common patterns for this segment
- Segment-specific recommendations

## Problem Reference List

The system includes 13 pre-defined problem categories:

1. Low BuyLead Quality
2. Payment & Billing Issues
3. Catalog Ranking Issues
4. Subscription Renewal Concerns
5. Technical Issues
6. BuyLead Consumption Issues
7. Seller Education Needed
8. Onboarding & Verification Issues
9. Churn Risk Signal
10. Service Escalation
11. Upsell Opportunity
12. Positive Feedback
13. Follow-up Required

Each problem has specific actionables for the servicing team.

## Architecture

```
app.py                          # Main Streamlit interface
├── src/
│   ├── insights_engine.py      # LLM insights generation
│   ├── config.py               # Configuration
│   └── ui/
│       └── transcription_section.py  # Translation UI
├── datasets/
│   ├── Problem_Actionable_Final.xlsx  # Problem definitions
│   ├── enhanced_segments.xlsx         # Business segments
│   └── Data Voice Hackathon_Master-1.xlsx  # Main dataset
└── output/
    └── transcripts/            # Cached translations
```

## Anti-Hallucination Measures

The insights engine is designed to prevent hallucinations:

- ✅ Uses only information from transcript, segment, and problem reference
- ✅ Returns "insufficient evidence" when uncertain
- ✅ Never invents problems or actionables
- ✅ Strictly follows structured JSON output
- ✅ Evidence-based insights only

## Customization

### Add New Problems

Edit `problem_reference.json`:

```json
{
  "problem_name": "Your Problem",
  "description": "Description",
  "keywords": ["keyword1", "keyword2"],
  "actionables": ["Action 1", "Action 2"]
}
```

### Change LLM Model

Edit `src/insights_engine.py`:

```python
# Change from gemini-pro to another model
self.model = genai.GenerativeModel('gemini-1.5-pro')
```

## Troubleshooting

### "Insights engine not available"
- Install dependencies: `pip install google-generativeai python-dotenv`

### "API key not found"
- Create `.env` file with `GEMINI_API_KEY`

### "Failed to parse LLM response"
- LLM returned invalid JSON
- Check raw response in error details
- May need to refine prompt

### "Sarvam API Offline"
- Ensure Sarvam translation server is running on port 8888

## License

MIT License - IndiaMART Voice Call Hackathon 2025
