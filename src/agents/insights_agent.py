"""
InsightsAgent - IndiaMART Customer Service Insights Extraction
AI-Assisted Sales & Servicing Enhancement (NOT replacement)

Uses NVIDIA NIM Nemotron-4-Mini-Hindi for Hinglish text analysis
"""

import json
import time
from typing import Dict, Any, List, Optional
from openai import OpenAI

from src.config import (
    NVIDIA_BASE_URL,
    NVIDIA_MODEL,
    NVIDIA_API_KEY,
    MODEL_TEMPERATURE,
    MODEL_TOP_P,
    MODEL_MAX_TOKENS,
    ISSUE_CATEGORIES,
    SELLER_UNDERTONES
)

# =============================================================================
# INDIAMART INSIGHTS EXTRACTION PROMPT
# =============================================================================

INSIGHTS_PROMPT = """You are an expert analyst for IndiaMART's customer service team.
Your role is to ASSIST sales and servicing teams (NOT replace them) by extracting actionable insights.

TRANSCRIPT:
{transcript}

CALL METADATA:
- Seller Type: {customer_type}
- City: {city}
- Call Direction: {call_direction}
- Is Repeat Ticket: {is_repeat}
- Call Duration: {duration} seconds

Analyze this call and provide comprehensive insights in JSON format:

{{
    "primary_category": "<LEAD_QUALITY|PAYMENT_BILLING|CATALOG_MANAGEMENT|SUBSCRIPTION_RENEWAL|TECHNICAL_ISSUES|BUYLEAD_CONSUMPTION|SELLER_EDUCATION|ONBOARDING_ISSUES|CHURN_RISK|RETENTION_OPPORTUNITY|SERVICE_ESCALATION|PRODUCTION_SUPPORT|UPSELL_OPPORTUNITY|POSITIVE_FEEDBACK|FOLLOW_UP_REQUIRED|MISCELLANEOUS>",
    
    "seller_pain_points": {{
        "listing_issues": "<any listing/catalog problems mentioned>",
        "payment_delays": "<payment related issues>",
        "bl_quality_problems": "<BuyLead quality concerns>",
        "verification_hurdles": "<OVP/verification issues>",
        "other_pain_points": ["<list of other specific pain points>"]
    }},
    
    "seller_undertone": "<ANGRY|IRRITATED|DISSATISFIED|DISENGAGED|CONFUSED|HESITANT|NEUTRAL|INTERESTED|SATISFIED|ENTHUSIASTIC>",
    
    "sentiment_details": {{
        "urgency_level": "<HIGH|MEDIUM|LOW>",
        "frustration_indicators": ["<repeated phrases or frustration signals>"],
        "engagement_quality": "<HIGH|MEDIUM|LOW>",
        "conversation_complexity": "<SIMPLE|MODERATE|COMPLEX>"
    }},
    
    "churn_risk_assessment": {{
        "risk_level": "<HIGH|MEDIUM|LOW|NONE>",
        "churn_signals": ["<specific signals indicating churn risk>"],
        "competitor_mentions": "<any competitor mentioned>",
        "discontinuation_intent": <true/false>,
        "winback_opportunity": <true/false>
    }},
    
    "seller_understanding": {{
        "understands_products": "<YES|PARTIAL|NO>",
        "understands_seller_panel": "<YES|PARTIAL|NO>",
        "understands_lms": "<YES|PARTIAL|NO>",
        "understands_bl_shortlisting": "<YES|PARTIAL|NO>",
        "understands_diy_catalog": "<YES|PARTIAL|NO>",
        "needs_base_education": <true/false>,
        "education_topics_needed": ["<specific topics seller needs training on>"]
    }},
    
    "catalog_insights": {{
        "cqs_issues": "<any CQS related mentions>",
        "rank_issues": "<A Rank/D Rank issues>",
        "mcat_issues": "<MCAT category issues>",
        "product_addition_needed": <true/false>,
        "isq_improvement_needed": <true/false>,
        "needs_production_support": <true/false>,
        "production_priority": "<HIGH|MEDIUM|LOW|NONE>"
    }},
    
    "opportunities": {{
        "upsell_opportunity": <true/false>,
        "upsell_type": "<what can be upsold>",
        "renewal_opportunity": <true/false>,
        "engaged_but_not_renewing": <true/false>,
        "cross_sell_potential": ["<other products/services>"]
    }},
    
    "executive_performance": {{
        "objection_handling": "<EXCELLENT|GOOD|NEEDS_IMPROVEMENT|POOR>",
        "product_knowledge_displayed": <true/false>,
        "empathy_shown": <true/false>,
        "solution_provided": <true/false>,
        "app_sharing_compliance_checked": <true/false>,
        "followed_process": <true/false>
    }},
    
    "action_items": {{
        "immediate_actions": ["<actions needed right now>"],
        "follow_up_needed": <true/false>,
        "follow_up_reason": "<why follow-up is needed>",
        "wc_wm_needed": <true/false>,
        "ticket_required": <true/false>,
        "escalation_needed": <true/false>,
        "production_work_order": <true/false>
    }},
    
    "best_reach_method": "<Email|WhatsApp|Seller Panel|IM App|DIR|Help Page|Phone Call>",
    
    "top_5_talking_points": [
        "<point 1 for executive to remember>",
        "<point 2>",
        "<point 3>",
        "<point 4>",
        "<point 5>"
    ],
    
    "executive_learnings": [
        "<learning 1 from this call>",
        "<learning 2>",
        "<learning 3>"
    ],
    
    "issue_summary": "<1-2 sentence summary of the main issue>",
    
    "proactive_recommendation": "<specific actionable recommendation for the team>"
}}

Respond ONLY with valid JSON."""


class InsightsAgent:
    """
    IndiaMART Insights Agent - Assists sales/servicing teams with AI-powered insights
    Uses NVIDIA NIM for analysis
    """
    
    def __init__(self, api_key: str = None, verbose: bool = True):
        self.api_key = api_key or NVIDIA_API_KEY
        self.client = OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=self.api_key
        )
        self.model = NVIDIA_MODEL
        self.verbose = verbose
        
        self._log(f"✅ InsightsAgent initialized (NVIDIA NIM)")
        
    def _log(self, message: str):
        if self.verbose:
            print(message)
    
    def _call_llm(self, prompt: str) -> str:
        """Call NVIDIA NIM API"""
        response_text = ""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=MODEL_TEMPERATURE,
            top_p=MODEL_TOP_P,
            max_tokens=MODEL_MAX_TOKENS,
            stream=True
        )
        
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                response_text += chunk.choices[0].delta.content
        
        return response_text.strip()
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from response"""
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            start_idx = 1
            end_idx = len(lines)
            for i, line in enumerate(lines):
                if i > 0 and line.strip() == "```":
                    end_idx = i
                    break
            response = "\n".join(lines[start_idx:end_idx])
            if response.startswith("json"):
                response = response[4:].strip()
        return json.loads(response)
    
    def analyze_transcript(self, transcript: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract comprehensive IndiaMART-specific insights from a transcript
        """
        metadata = metadata or {}
        
        self._log(f"\n🔍 Analyzing transcript for IndiaMART insights...")
        self._log(f"   Length: {len(transcript)} characters")
        
        prompt = INSIGHTS_PROMPT.format(
            transcript=transcript[:8000],
            customer_type=metadata.get('customer_type', 'Unknown'),
            city=metadata.get('city', 'Unknown'),
            call_direction=metadata.get('call_direction', 'Unknown'),
            is_repeat=metadata.get('is_repeat', 'Unknown'),
            duration=metadata.get('duration', 'Unknown')
        )
        
        try:
            start_time = time.time()
            response = self._call_llm(prompt)
            elapsed = time.time() - start_time
            
            result = self._parse_json_response(response)
            result['analysis_success'] = True
            result['processing_time'] = round(elapsed, 2)
            
            # Log key findings
            self._log(f"   ✅ Category: {result.get('primary_category', 'N/A')}")
            self._log(f"   🎭 Undertone: {result.get('seller_undertone', 'N/A')}")
            self._log(f"   ⚠️  Churn Risk: {result.get('churn_risk_assessment', {}).get('risk_level', 'N/A')}")
            self._log(f"   ⏱️  Time: {elapsed:.2f}s")
            
            if result.get('opportunities', {}).get('upsell_opportunity'):
                self._log(f"   💰 UPSELL OPPORTUNITY DETECTED!")
            
            if result.get('seller_understanding', {}).get('needs_base_education'):
                self._log(f"   📚 Seller needs BASE EDUCATION")
            
            return result
            
        except json.JSONDecodeError as e:
            self._log(f"   ⚠️ JSON Parse Error: {str(e)}")
            return {'analysis_success': False, 'error': str(e), 'primary_category': 'MISCELLANEOUS'}
        except Exception as e:
            self._log(f"   ❌ Error: {str(e)}")
            return {'analysis_success': False, 'error': str(e), 'primary_category': 'MISCELLANEOUS'}
    
    def get_executive_popup(self, transcript: str) -> Dict[str, Any]:
        """
        Generate real-time popup insights for executives during calls
        """
        self._log("\n📌 Generating executive popup...")
        
        prompt = f"""Analyze this ongoing IndiaMART call and provide REAL-TIME guidance for the executive.

TRANSCRIPT SO FAR:
{transcript[:5000]}

Provide a JSON response for the executive popup:
{{
    "seller_mood": "<current mood of seller>",
    "key_concern": "<main issue seller is facing>",
    "top_5_talking_points": [
        "<point 1 - most important>",
        "<point 2>",
        "<point 3>",
        "<point 4>",
        "<point 5>"
    ],
    "avoid_saying": ["<things NOT to say>"],
    "upsell_hint": "<if there's an upsell opportunity, mention it>",
    "immediate_action": "<what to do right now>"
}}"""

        try:
            response = self._call_llm(prompt)
            return self._parse_json_response(response)
        except:
            return {"error": "Could not generate popup"}
    
    def get_daily_learnings(self, transcripts: List[str]) -> Dict[str, Any]:
        """
        Generate Top 5 learnings from today's calls for executives
        """
        self._log("\n📚 Generating daily learnings...")
        
        combined = "\n---\n".join(transcripts[:10])
        
        prompt = f"""Analyze these IndiaMART customer service calls from today and provide learnings for executives.

TODAY'S CALLS:
{combined[:15000]}

Provide a JSON response:
{{
    "top_5_learnings": [
        {{
            "learning": "<key learning>",
            "why_important": "<why this matters>",
            "action_to_take": "<what to do differently>"
        }}
    ],
    "common_mistakes": ["<mistakes to avoid>"],
    "best_practices_observed": ["<what worked well>"],
    "training_recommendations": ["<suggested training topics>"]
}}"""

        try:
            response = self._call_llm(prompt)
            return self._parse_json_response(response)
        except:
            return {"error": "Could not generate learnings"}
    
    def ask_question(self, transcript: str, question: str) -> str:
        """Ask a specific question about the transcript"""
        prompt = f"""Based on this IndiaMART call transcript, answer the question.

TRANSCRIPT:
{transcript[:6000]}

QUESTION: {question}

Provide a clear, actionable answer."""

        try:
            return self._call_llm(prompt)
        except Exception as e:
            return f"Error: {str(e)}"
