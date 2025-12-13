"""
Insights Engine for IndiaMART Voice Call Hackathon
Uses custom IndiaMART LLM API to analyze call transcripts and generate actionable insights
"""

import os
import json
import pandas as pd
import requests
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API endpoints
API_BASE_URL_FOR_CLIENT = os.getenv("API_BASE_URL_FOR_CLIENT", "https://imllm.intermesh.net")
API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = 'google/gemini-2.5-flash-lite'  # Using allowed model for your team


class InsightsEngine:
    """
    LLM-based insights engine for analyzing call transcripts
    """
    
    def __init__(self, problem_actionable_path: str = "datasets/Problem_Actionable_Final.xlsx"):
        """
        Initialize the insights engine
        
        Args:
            problem_actionable_path: Path to datasets/Problem_Actionable_Final.xlsx file
        """
        self.problem_actionable = self._load_problem_actionable(problem_actionable_path)
        self.api_base_url = API_BASE_URL_FOR_CLIENT
        self.api_key = API_KEY
        self.model_name = MODEL_NAME
    
    def _load_problem_actionable(self, path: str) -> pd.DataFrame:
        """Load Problem_Actionable_Final sheet"""
        try:
            # Read the Excel file
            df = pd.read_excel(path)
            # First row contains the actual headers
            headers = df.iloc[0].tolist()
            # Reload skipping the first row and using those headers
            df = pd.read_excel(path, skiprows=1, names=headers)
            return df
        except Exception as e:
            print(f"Warning: Could not load Problem_Actionable_Final: {e}")
            return pd.DataFrame()
    
    def _build_prompt(self, transcript: Dict, segment_info: Dict) -> str:
        """
        Build the LLM prompt with all required inputs
        
        Args:
            transcript: Translation data with speakers
            segment_info: Segment metadata (GLID, Segment, BusinessSegment)
        
        Returns:
            Formatted prompt string
        """
        # Extract speaker conversation
        conversation = ""
        if transcript.get('status') == 'success':
            speakers = transcript.get('speakers', [])
            for speaker in speakers:
                speaker_id = speaker.get('speaker_id', 'Unknown')
                text = speaker.get('text', '')
                conversation += f"{speaker_id}: {text}\n\n"
        
        # Format problem actionable reference
        problems_text = ""
        if not self.problem_actionable.empty:
            for _, row in self.problem_actionable.iterrows():
                problem_statement = row.get('Problem Statement', 'N/A')
                description = row.get('Description', 'N/A')
                
                # Collect action items
                action_items = []
                for i in range(1, 5):
                    action = row.get(f'Action Item {i}', None)
                    if pd.notna(action) and str(action).strip():
                        action_items.append(str(action))
                
                action_for_im = row.get('Action for IM', None)
                
                problems_text += f"**{problem_statement}**\n"
                problems_text += f"Description: {description}\n"
                if action_items:
                    problems_text += f"Action Items: {', '.join(action_items)}\n"
                if pd.notna(action_for_im) and str(action_for_im).strip():
                    problems_text += f"Action for IndiaMART: {action_for_im}\n"
                problems_text += "\n"
        
        prompt = f"""You are an Insight Engine designed for IndiaMART's Voice Call Hackathon.

Your task:  
Given (1) the call transcript, (2) seller's segment information, and (3) the problem–actionable reference list, you must produce ONLY factual, evidence-based insights that satisfy IndiaMART's judging criteria: accuracy, actionability, coverage, and innovation.

------------------------------------------
INPUT 1: CALL TRANSCRIPT
------------------------------------------
{conversation}

------------------------------------------
INPUT 2: SEGMENT INFORMATION
------------------------------------------
GLID: {segment_info.get('glid', 'N/A')}
Segment: {segment_info.get('segment', 'N/A')}
BusinessSegment: {segment_info.get('business_segment', 'N/A')}

------------------------------------------
INPUT 3: PROBLEM-ACTIONABLE REFERENCE LIST
------------------------------------------
{problems_text}

------------------------------------------
YOUR OBJECTIVES:
Extract insights from this specific call ONLY.  
Your output must strictly contain:

A. IDENTIFIED PROBLEM + ACTIONABLES
1. Identify which problem(s) from the reference list match the transcript.
   - Use only evidence present in the transcript.
   - If multiple problems match, list all.
   - If none match, output "No mapped problem found".

2. For each identified problem:
   - Provide the exact actionable(s) from the reference list.
   - Include "Action for IndiaMART" ONLY if it exists in the reference data. If not available, do not print .
   - Do NOT invent new actionables or actions.

B. SELLER TONE ANALYSIS + ACTIONABLES
Analyze seller tone across:
- Sentiment (Positive / Neutral / Negative)
  Positive: cooperative, appreciative, agreeable, willing to proceed, problem resolved.
  Neutral: factual, procedural, reserved, no strong emotion, mild hesitation.
  Negative: frustrated, dissatisfied, rejecting, annoyed, complaining, wanting to end the call.
- Churn Risk Signal
- Upsell Readiness
- Issue Severity (High / Medium / Low)
- Engagement Level
- Trust & Rapport Score
- Emotional Trigger Points
- Outcome Prediction (Resolution / Escalation likelihood)
- Follow-up Willingness

If any tone category suggests risk or negative behavior:
→ Provide a clear actionable for the servicing team.

C. EXECUTIVE TONE ANALYSIS + ACTIONABLES
Evaluate executive tone on:
- Empathy & Professionalism
- Clarity & Confidence
- Persuasion / Upsell Effectiveness
- Problem-Solving Orientation
- Rapport-Building Strength

If the executive tone is weak in any category:
→ Suggest actionable improvement points.

D. SEGMENT-AWARE INSIGHT ENHANCEMENT
Using the seller's BusinessSegment & Segment metadata:
- Highlight if this problem is common for this segment.
- Mention segment-specific considerations ONLY if supported by transcript patterns.
Do NOT generalize beyond what is known.

------------------------------------------
STRICT OUTPUT FORMAT (MANDATORY)
Provide the final result in EXACTLY this structured JSON:

{{
  "problems_identified": [
     {{
       "problem_name": "",
       "evidence_from_transcript": "",
       "actionables": [""],
       "action_for_indiamart": ""
     }}
  ],
  "seller_tone": {{
     "sentiment": "",
     "churn_risk": "",
     "upsell_readiness": "",
     "issue_severity": "",
     "engagement_level": "",
     "trust_and_rapport": "",
     "emotional_trigger_points": "",
     "outcome_prediction": "",
     "follow_up_willingness": "",
     "actionables_if_any": [""]
  }},
  "executive_tone": {{
     "empathy_and_professionalism": "",
     "clarity_and_confidence": "",
     "persuasion_effectiveness": "",
     "problem_solving_orientation": "",
     "rapport_building_strength": "",
     "actionables_if_any": [""]
  }},
  "segment_notes": {{
     "segment": "{segment_info.get('segment', '')}",
     "business_segment": "{segment_info.get('business_segment', '')}",
     "relevance_to_segment": ""
  }}
}}

------------------------------------------
ANTI-HALLUCINATION RULES (MANDATORY)
- Use ONLY information present in transcript, segment_info, and problem_reference_list.
- Do NOT invent problems, actions, sentiments, or facts.
- If uncertain, respond with: "insufficient evidence".
- Never guess seller intent, tone, emotions, or problem unless explicitly shown.
- Never fabricate missing fields—use null or empty string if not applicable.

Begin analysis now. Output ONLY the JSON, no additional text.
"""
        return prompt
    
    def analyze_call(self, transcript: Dict, segment_info: Dict) -> Dict[str, Any]:
        """
        Analyze a call transcript and generate insights
        
        Args:
            transcript: Translation data from Sarvam API
            segment_info: Segment metadata
        
        Returns:
            Structured insights dictionary
        """
        try:
            # Build prompt
            prompt = self._build_prompt(transcript, segment_info)
            
            # Call custom LLM API
            api_url = f"{self.api_base_url}/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            response_text = response_data['choices'][0]['message']['content'].strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Try to extract JSON if there's extra text
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                response_text = json_match.group()
            
            # Sanitize common JSON issues
            # Replace single quotes with double quotes (careful with apostrophes in text)
            # Remove control characters that break JSON
            response_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', response_text)
            
            # Parse JSON
            insights = json.loads(response_text)
            
            return {
                'status': 'success',
                'insights': insights
            }
            
        except json.JSONDecodeError as e:
            return {
                'status': 'error',
                'message': f'Failed to parse LLM response as JSON: {str(e)}',
                'raw_response': response_text if 'response_text' in locals() else None
            }
        except requests.exceptions.RequestException as e:
            return {
                'status': 'error',
                'message': f'API request failed: {str(e)}'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Insights generation failed: {str(e)}'
            }
    
    def format_insights_for_display(self, insights: Dict) -> str:
        """
        Format insights dictionary into readable text for display
        
        Args:
            insights: Insights dictionary from analyze_call
        
        Returns:
            Formatted string for display
        """
        if insights.get('status') != 'success':
            return f"Error: {insights.get('message', 'Unknown error')}"
        
        data = insights.get('insights', {})
        output = []
        
        # Problems Identified
        output.append("=" * 80)
        output.append("IDENTIFIED PROBLEMS & ACTIONABLES")
        output.append("=" * 80)
        
        problems = data.get('problems_identified', [])
        if problems:
            for i, problem in enumerate(problems, 1):
                output.append(f"\n{i}. {problem.get('problem_name', 'Unknown')}")
                output.append(f"   Evidence: {problem.get('evidence_from_transcript', 'N/A')}")
                output.append(f"   Actionables:")
                for action in problem.get('actionables', []):
                    output.append(f"   • {action}")
                if problem.get('action_for_indiamart'):
                    output.append(f"   Action for IndiaMART: {problem.get('action_for_indiamart')}")
        else:
            output.append("\nNo mapped problems found.")
        
        # Seller Tone
        output.append("\n" + "=" * 80)
        output.append("SELLER TONE ANALYSIS")
        output.append("=" * 80)
        
        seller_tone = data.get('seller_tone', {})
        output.append(f"\nSentiment: {seller_tone.get('sentiment', 'N/A')}")
        output.append(f"Churn Risk: {seller_tone.get('churn_risk', 'N/A')}")
        output.append(f"Upsell Readiness: {seller_tone.get('upsell_readiness', 'N/A')}")
        output.append(f"Issue Severity: {seller_tone.get('issue_severity', 'N/A')}")
        output.append(f"Engagement Level: {seller_tone.get('engagement_level', 'N/A')}")
        output.append(f"Trust & Rapport: {seller_tone.get('trust_and_rapport', 'N/A')}")
        output.append(f"Emotional Triggers: {seller_tone.get('emotional_trigger_points', 'N/A')}")
        output.append(f"Outcome Prediction: {seller_tone.get('outcome_prediction', 'N/A')}")
        output.append(f"Follow-up Willingness: {seller_tone.get('follow_up_willingness', 'N/A')}")
        
        if seller_tone.get('actionables_if_any'):
            output.append(f"\nActionables:")
            for action in seller_tone.get('actionables_if_any', []):
                output.append(f"• {action}")
        
        # Executive Tone
        output.append("\n" + "=" * 80)
        output.append("EXECUTIVE TONE ANALYSIS")
        output.append("=" * 80)
        
        exec_tone = data.get('executive_tone', {})
        output.append(f"\nEmpathy & Professionalism: {exec_tone.get('empathy_and_professionalism', 'N/A')}")
        output.append(f"Clarity & Confidence: {exec_tone.get('clarity_and_confidence', 'N/A')}")
        output.append(f"Persuasion Effectiveness: {exec_tone.get('persuasion_effectiveness', 'N/A')}")
        output.append(f"Problem-Solving Orientation: {exec_tone.get('problem_solving_orientation', 'N/A')}")
        output.append(f"Rapport-Building Strength: {exec_tone.get('rapport_building_strength', 'N/A')}")
        
        if exec_tone.get('actionables_if_any'):
            output.append(f"\nActionables:")
            for action in exec_tone.get('actionables_if_any', []):
                output.append(f"• {action}")
        
        # Segment Notes
        output.append("\n" + "=" * 80)
        output.append("SEGMENT-SPECIFIC INSIGHTS")
        output.append("=" * 80)
        
        segment = data.get('segment_notes', {})
        output.append(f"\nSegment: {segment.get('segment', 'N/A')}")
        output.append(f"Business Segment: {segment.get('business_segment', 'N/A')}")
        output.append(f"Relevance: {segment.get('relevance_to_segment', 'N/A')}")
        
        return "\n".join(output)


# Convenience function for direct use
def generate_insights(transcript: Dict, segment_info: Dict) -> Dict[str, Any]:
    """
    Generate insights for a call transcript
    
    Args:
        transcript: Translation data from Sarvam API
        segment_info: Dict with keys: glid, segment, business_segment
    
    Returns:
        Insights dictionary
    """
    engine = InsightsEngine()
    return engine.analyze_call(transcript, segment_info)
