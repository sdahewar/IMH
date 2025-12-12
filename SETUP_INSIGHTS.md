# Quick Setup Guide - Insights Engine

## Step 1: Get Gemini API Key

1. Visit: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the generated key

## Step 2: Create .env File

Create a file named `.env` in the project root:

```bash
# Google Gemini API Key for Insights Engine
GEMINI_API_KEY=paste_your_key_here

# Custom LLM API Endpoint (for IndiaMART's internal LLM)
API_BASE_URL_FOR_CLIENT=https://imllm.intermesh.net
```

## Step 3: Verify Installation

Dependencies are already installed:
- ✅ google-generativeai
- ✅ python-dotenv

## Step 4: Test the Insights Engine

1. Start the app: `bash start_streamlit.sh`
2. Enter a Call ID
3. Click "Fetch & Translate Call"
4. Click "🔍 Generate Insights"
5. Review the AI-generated insights!

## Example Call IDs to Test

Try these call IDs from the dataset:
- 16939
- 17234
- 18456

## Troubleshooting

### "API key not found"
- Make sure `.env` file exists in project root
- Check that GEMINI_API_KEY is set correctly
- Restart the Streamlit app

### "Insights engine not available"
- Dependencies are already installed
- Just need to add API key

### "Failed to parse LLM response"
- Check raw response in error details
- May need to adjust prompt
- Try again (LLM responses can vary)

## What You'll Get

The insights engine provides:

1. **Identified Problems**
   - Matched from 13 pre-defined categories
   - Evidence from transcript
   - Specific actionables

2. **Seller Tone Analysis**
   - Sentiment, churn risk, upsell readiness
   - Engagement level, trust score
   - Emotional triggers
   - Outcome prediction

3. **Executive Performance**
   - Empathy & professionalism
   - Clarity & confidence
   - Problem-solving orientation
   - Improvement suggestions

4. **Segment Insights**
   - Business segment relevance
   - Segment-specific patterns
   - Tailored recommendations

## Cost

Google Gemini API:
- **Free tier:** 60 requests per minute
- **Cost:** Free for testing
- Perfect for hackathon use!

## Ready to Go!

Just add your Gemini API key to `.env` and you're ready to generate insights! 🚀
