"""
AggregationAgent - LLM-powered agent for aggregating insights across multiple transcripts
Supports aggregation by customer, vendor, location, or custom grouping
"""

import json
import time
from typing import Dict, Any, List, Optional
from collections import Counter
import pandas as pd
from openai import OpenAI

from src.config import (
    NVIDIA_BASE_URL,
    NVIDIA_MODEL,
    NVIDIA_API_KEY,
    MODEL_TEMPERATURE,
    MODEL_TOP_P,
    MODEL_MAX_TOKENS,
    ISSUE_CATEGORIES
)
from src.agents.insights_agent import InsightsAgent


class AggregationAgent:
    """
    Agent for aggregating insights across multiple transcripts
    Can group by customer, vendor, location, or any custom attribute
    """
    
    def __init__(self, api_key: str = None, verbose: bool = True):
        """
        Initialize the AggregationAgent
        
        Args:
            api_key: NVIDIA NIM API key
            verbose: Print detailed status messages
        """
        self.api_key = api_key or NVIDIA_API_KEY
        self.client = OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=self.api_key
        )
        self.model = NVIDIA_MODEL
        self.verbose = verbose
        self.insights_agent = InsightsAgent(api_key=self.api_key, verbose=False)
        
    def _log(self, message: str):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            print(message)
    
    def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Make a call to the LLM"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response_text = ""
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=MODEL_TEMPERATURE,
                top_p=MODEL_TOP_P,
                max_tokens=MODEL_MAX_TOKENS,
                stream=True
            )
            
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    response_text += chunk.choices[0].delta.content
            
            return response_text.strip()
            
        except Exception as e:
            self._log(f"❌ LLM Error: {str(e)}")
            raise
    
    def analyze_multiple_transcripts(self, 
                                     transcripts: List[Dict[str, Any]], 
                                     show_individual: bool = True) -> Dict[str, Any]:
        """
        Analyze multiple transcripts and return individual + aggregated insights
        
        Args:
            transcripts: List of dicts with 'transcript' and optional 'metadata' keys
            show_individual: Show status for each transcript
            
        Returns:
            Dict with individual results and aggregated insights
        """
        self._log(f"\n{'=' * 80}")
        self._log(f"🔍 ANALYZING {len(transcripts)} TRANSCRIPTS")
        self._log(f"{'=' * 80}")
        
        results = []
        
        for i, item in enumerate(transcripts):
            transcript = item.get('transcript', '')
            metadata = item.get('metadata', {})
            
            if show_individual:
                self._log(f"\n[{i+1}/{len(transcripts)}] Processing transcript...")
                if metadata:
                    self._log(f"   Customer: {metadata.get('customer_type', 'N/A')} | City: {metadata.get('city', 'N/A')}")
            
            result = self.insights_agent.analyze_transcript(transcript, metadata)
            result['metadata'] = metadata
            results.append(result)
            
            if show_individual and result.get('analysis_success'):
                self._log(f"   ✅ {result.get('primary_category', 'N/A')} | {result.get('sentiment', 'N/A')}")
            
            time.sleep(0.3)  # Rate limiting
        
        # Generate aggregated insights
        aggregated = self._aggregate_results(results)
        
        return {
            'individual_results': results,
            'aggregated_insights': aggregated,
            'total_analyzed': len(results),
            'successful': sum(1 for r in results if r.get('analysis_success', False))
        }
    
    def _aggregate_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from multiple transcript analyses"""
        
        # Count categories
        categories = Counter(r.get('primary_category', 'UNKNOWN') for r in results if r.get('analysis_success'))
        
        # Count sentiments
        sentiments = Counter(r.get('sentiment', 'UNKNOWN') for r in results if r.get('analysis_success'))
        
        # Count churn risks
        churn_risks = Counter(r.get('churn_risk', 'UNKNOWN') for r in results if r.get('analysis_success'))
        
        # Count resolution status
        resolutions = Counter(r.get('resolution_status', 'UNKNOWN') for r in results if r.get('analysis_success'))
        
        # Collect all pain points
        all_pain_points = []
        for r in results:
            if r.get('analysis_success') and r.get('customer_pain_points'):
                all_pain_points.extend(r['customer_pain_points'])
        pain_point_counts = Counter(all_pain_points)
        
        # Collect all keywords
        all_keywords = []
        for r in results:
            if r.get('analysis_success') and r.get('keywords'):
                all_keywords.extend(r['keywords'])
        keyword_counts = Counter(all_keywords)
        
        # Executive performance
        exec_stats = {
            'empathy_shown': sum(1 for r in results if r.get('executive_performance', {}).get('empathy_shown')),
            'solution_offered': sum(1 for r in results if r.get('executive_performance', {}).get('solution_offered')),
            'followed_process': sum(1 for r in results if r.get('executive_performance', {}).get('followed_process')),
            'escalation_needed': sum(1 for r in results if r.get('executive_performance', {}).get('escalation_needed'))
        }
        
        total = len([r for r in results if r.get('analysis_success')])
        
        return {
            'category_distribution': dict(categories.most_common()),
            'sentiment_distribution': dict(sentiments.most_common()),
            'churn_risk_distribution': dict(churn_risks.most_common()),
            'resolution_distribution': dict(resolutions.most_common()),
            'top_pain_points': dict(pain_point_counts.most_common(10)),
            'top_keywords': dict(keyword_counts.most_common(15)),
            'executive_performance': {
                'empathy_rate': round(exec_stats['empathy_shown'] / total * 100, 1) if total > 0 else 0,
                'solution_rate': round(exec_stats['solution_offered'] / total * 100, 1) if total > 0 else 0,
                'process_compliance': round(exec_stats['followed_process'] / total * 100, 1) if total > 0 else 0,
                'escalation_rate': round(exec_stats['escalation_needed'] / total * 100, 1) if total > 0 else 0
            },
            'high_churn_risk_count': churn_risks.get('HIGH', 0),
            'follow_up_required_count': sum(1 for r in results if r.get('requires_follow_up'))
        }
    
    def aggregate_by_customer(self, df: pd.DataFrame, customer_id: Any) -> Dict[str, Any]:
        """
        Aggregate all transcripts for a specific customer
        
        Args:
            df: DataFrame with transcript data
            customer_id: Customer ID (glid) to filter by
            
        Returns:
            Aggregated insights for the customer
        """
        customer_df = df[df['glid'] == customer_id]
        
        if len(customer_df) == 0:
            return {'error': f'No records found for customer {customer_id}'}
        
        self._log(f"\n{'=' * 80}")
        self._log(f"👤 CUSTOMER ANALYSIS: {customer_id}")
        self._log(f"{'=' * 80}")
        self._log(f"📊 Found {len(customer_df)} call records")
        
        # Prepare transcripts
        transcripts = []
        for _, row in customer_df.iterrows():
            transcripts.append({
                'transcript': row.get('transcript', ''),
                'metadata': {
                    'customer_type': row.get('customer_type', ''),
                    'city': row.get('city_name', ''),
                    'call_direction': row.get('FLAG_IN_OUT', ''),
                    'is_repeat': row.get('is_ticket_repeat60d', ''),
                    'duration': row.get('call_duration', ''),
                    'call_id': row.get('click_to_call_id', '')
                }
            })
        
        # Analyze all transcripts
        results = self.analyze_multiple_transcripts(transcripts, show_individual=True)
        
        # Add customer-specific summary
        results['customer_id'] = customer_id
        results['customer_type'] = customer_df['customer_type'].iloc[0]
        results['city'] = customer_df['city_name'].iloc[0]
        results['total_calls'] = len(customer_df)
        results['repeat_calls'] = len(customer_df[customer_df['is_ticket_repeat60d'] == 'Yes'])
        
        # Generate customer-specific recommendations
        results['customer_recommendations'] = self._generate_customer_recommendations(results)
        
        return results
    
    def aggregate_by_location(self, df: pd.DataFrame, city: str) -> Dict[str, Any]:
        """
        Aggregate all transcripts for a specific city/location
        
        Args:
            df: DataFrame with transcript data
            city: City name to filter by
            
        Returns:
            Aggregated insights for the location
        """
        city_df = df[df['city_name'] == city]
        
        if len(city_df) == 0:
            return {'error': f'No records found for city {city}'}
        
        self._log(f"\n{'=' * 80}")
        self._log(f"📍 LOCATION ANALYSIS: {city}")
        self._log(f"{'=' * 80}")
        self._log(f"📊 Found {len(city_df)} call records")
        
        # Sample if too many (for efficiency)
        if len(city_df) > 50:
            self._log(f"   ⚠️ Sampling 50 records from {len(city_df)} for analysis")
            city_df = city_df.sample(n=50, random_state=42)
        
        # Prepare transcripts
        transcripts = []
        for _, row in city_df.iterrows():
            transcripts.append({
                'transcript': row.get('transcript', ''),
                'metadata': {
                    'customer_type': row.get('customer_type', ''),
                    'city': city,
                    'customer_id': row.get('glid', ''),
                    'call_direction': row.get('FLAG_IN_OUT', ''),
                    'is_repeat': row.get('is_ticket_repeat60d', ''),
                    'duration': row.get('call_duration', '')
                }
            })
        
        results = self.analyze_multiple_transcripts(transcripts, show_individual=True)
        
        results['city'] = city
        results['total_calls_in_city'] = len(df[df['city_name'] == city])
        results['unique_customers'] = df[df['city_name'] == city]['glid'].nunique()
        
        # Generate location-specific insights
        results['location_insights'] = self._generate_location_insights(results, city)
        
        return results
    
    def aggregate_by_customer_type(self, df: pd.DataFrame, customer_type: str, sample_size: int = 50) -> Dict[str, Any]:
        """
        Aggregate transcripts for a specific customer type
        
        Args:
            df: DataFrame with transcript data
            customer_type: Customer type to filter by (CATALOG, TSCATALOG, STAR, etc.)
            sample_size: Number of samples to analyze
            
        Returns:
            Aggregated insights for the customer type
        """
        type_df = df[df['customer_type'] == customer_type]
        
        if len(type_df) == 0:
            return {'error': f'No records found for customer type {customer_type}'}
        
        self._log(f"\n{'=' * 80}")
        self._log(f"👥 CUSTOMER TYPE ANALYSIS: {customer_type}")
        self._log(f"{'=' * 80}")
        self._log(f"📊 Found {len(type_df)} call records")
        
        # Sample for efficiency
        if len(type_df) > sample_size:
            self._log(f"   📊 Sampling {sample_size} records for analysis")
            type_df = type_df.sample(n=sample_size, random_state=42)
        
        transcripts = []
        for _, row in type_df.iterrows():
            transcripts.append({
                'transcript': row.get('transcript', ''),
                'metadata': {
                    'customer_type': customer_type,
                    'city': row.get('city_name', ''),
                    'customer_id': row.get('glid', ''),
                    'is_repeat': row.get('is_ticket_repeat60d', ''),
                    'duration': row.get('call_duration', '')
                }
            })
        
        results = self.analyze_multiple_transcripts(transcripts, show_individual=True)
        
        results['customer_type'] = customer_type
        results['total_calls_for_type'] = len(df[df['customer_type'] == customer_type])
        results['unique_customers'] = df[df['customer_type'] == customer_type]['glid'].nunique()
        
        # Generate segment-specific recommendations
        results['segment_recommendations'] = self._generate_segment_recommendations(results, customer_type)
        
        return results
    
    def _generate_customer_recommendations(self, results: Dict) -> str:
        """Generate personalized recommendations for a customer"""
        
        agg = results.get('aggregated_insights', {})
        
        prompt = f"""Based on this customer's call history analysis, provide personalized recommendations.

CUSTOMER ANALYSIS:
- Total Calls: {results.get('total_calls', 0)}
- Repeat Calls: {results.get('repeat_calls', 0)}
- Top Issues: {list(agg.get('category_distribution', {}).keys())[:3]}
- Sentiment Pattern: {list(agg.get('sentiment_distribution', {}).keys())}
- Churn Risk: {agg.get('high_churn_risk_count', 0)} high-risk calls
- Top Pain Points: {list(agg.get('top_pain_points', {}).keys())[:5]}

Provide:
1. **Customer Health Assessment**: Overall assessment of this customer's experience
2. **Immediate Actions**: What should be done right away for this customer
3. **Long-term Recommendations**: How to improve their experience
4. **Retention Risk**: Is this customer at risk of churning and why

Keep it concise and actionable."""

        try:
            return self._call_llm(prompt)
        except:
            return "Unable to generate recommendations"
    
    def _generate_location_insights(self, results: Dict, city: str) -> str:
        """Generate location-specific insights"""
        
        agg = results.get('aggregated_insights', {})
        
        prompt = f"""Based on call analysis for {city}, provide location-specific insights.

LOCATION ANALYSIS:
- City: {city}
- Total Calls Analyzed: {results.get('total_analyzed', 0)}
- Unique Customers: {results.get('unique_customers', 0)}
- Top Issues: {list(agg.get('category_distribution', {}).keys())[:3]}
- Common Pain Points: {list(agg.get('top_pain_points', {}).keys())[:5]}
- Churn Risk: {agg.get('high_churn_risk_count', 0)} high-risk calls

Provide:
1. **Regional Trends**: What patterns are specific to this location?
2. **Common Issues**: What are the most common problems in this area?
3. **Recommendations**: How can service be improved for this location?
4. **Resource Allocation**: Any suggestions for support staffing?

Keep it concise and actionable."""

        try:
            return self._call_llm(prompt)
        except:
            return "Unable to generate location insights"
    
    def _generate_segment_recommendations(self, results: Dict, customer_type: str) -> str:
        """Generate segment-specific recommendations"""
        
        agg = results.get('aggregated_insights', {})
        
        prompt = f"""Based on call analysis for {customer_type} customers, provide segment recommendations.

SEGMENT ANALYSIS:
- Customer Type: {customer_type}
- Total Calls Analyzed: {results.get('total_analyzed', 0)}
- Unique Customers: {results.get('unique_customers', 0)}
- Top Issues: {list(agg.get('category_distribution', {}).keys())[:3]}
- Sentiment: {list(agg.get('sentiment_distribution', {}).keys())}
- Pain Points: {list(agg.get('top_pain_points', {}).keys())[:5]}
- Executive Performance: {agg.get('executive_performance', {})}

Provide:
1. **Segment Characteristics**: What defines this customer segment's needs?
2. **Service Gaps**: Where is service failing for this segment?
3. **Training Needs**: What training would help executives serve this segment better?
4. **Process Improvements**: What process changes would help?
5. **Priority Actions**: Top 3 actions to improve this segment's experience

Keep it concise and actionable."""

        try:
            return self._call_llm(prompt)
        except:
            return "Unable to generate segment recommendations"
    
    def generate_executive_summary(self, results: Dict) -> str:
        """
        Generate an executive summary from aggregated results
        
        Args:
            results: Results from aggregate analysis
            
        Returns:
            Executive summary text
        """
        self._log("\n📋 Generating executive summary...")
        
        agg = results.get('aggregated_insights', {})
        
        prompt = f"""Create an executive summary of this customer service call analysis.

ANALYSIS RESULTS:
- Total Calls Analyzed: {results.get('total_analyzed', 0)}
- Successful Analysis: {results.get('successful', 0)}

CATEGORY DISTRIBUTION:
{json.dumps(agg.get('category_distribution', {}), indent=2)}

SENTIMENT DISTRIBUTION:
{json.dumps(agg.get('sentiment_distribution', {}), indent=2)}

CHURN RISK:
{json.dumps(agg.get('churn_risk_distribution', {}), indent=2)}

TOP PAIN POINTS:
{json.dumps(agg.get('top_pain_points', {}), indent=2)}

EXECUTIVE PERFORMANCE:
{json.dumps(agg.get('executive_performance', {}), indent=2)}

Create an executive summary including:
1. **Overview**: Key findings at a glance
2. **Critical Issues**: What needs immediate attention
3. **Positive Trends**: What's working well
4. **Risk Assessment**: Overall customer health
5. **Top 5 Recommendations**: Prioritized action items
6. **KPIs to Monitor**: Key metrics to track

Format professionally for management review."""

        try:
            summary = self._call_llm(prompt)
            self._log("   ✅ Summary generated")
            return summary
        except Exception as e:
            self._log(f"   ❌ Error: {str(e)}")
            return f"Error generating summary: {str(e)}"

