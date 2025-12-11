"""
Insights Aggregator
Aggregates classified call data to produce actionable business insights
"""

import os
import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple

from src.config import ISSUE_CATEGORIES

# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================

class InsightsAggregator:
    """
    Aggregates classified call data to produce business insights
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with classified DataFrame
        
        Args:
            df: DataFrame with classification results (ai_* columns)
        """
        self.df = df
        self.insights = {}
        
    def aggregate_by_category(self) -> Dict[str, Dict]:
        """Aggregate insights by issue category"""
        category_stats = {}
        
        if 'ai_primary_category' in self.df.columns:
            category_counts = self.df['ai_primary_category'].value_counts()
            total = len(self.df)
            
            for category, count in category_counts.items():
                category_stats[category] = {
                    'count': int(count),
                    'percentage': round(count / total * 100, 2),
                    'category_name': ISSUE_CATEGORIES.get(category, {}).get('name', category)
                }
        
        self.insights['category_distribution'] = category_stats
        return category_stats
    
    def aggregate_by_geography(self) -> Dict[str, Dict]:
        """Aggregate insights by city/geography"""
        geo_insights = {}
        
        if 'city_name' in self.df.columns and 'ai_primary_category' in self.df.columns:
            city_counts = self.df['city_name'].value_counts().head(20)
            
            for city in city_counts.index:
                city_df = self.df[self.df['city_name'] == city]
                
                geo_insights[city] = {
                    'total_calls': int(len(city_df)),
                    'top_issues': city_df['ai_primary_category'].value_counts().head(3).to_dict() if 'ai_primary_category' in city_df else {},
                    'churn_risk_high': int(len(city_df[city_df.get('ai_churn_risk', '') == 'HIGH'])) if 'ai_churn_risk' in city_df else 0,
                    'avg_call_duration': round(city_df['call_duration'].mean(), 2) if 'call_duration' in city_df else 0
                }
        
        self.insights['geography_insights'] = geo_insights
        return geo_insights
    
    def aggregate_by_customer_type(self) -> Dict[str, Dict]:
        """Aggregate insights by customer type"""
        customer_insights = {}
        
        if 'customer_type' in self.df.columns:
            for ctype in self.df['customer_type'].unique():
                ctype_df = self.df[self.df['customer_type'] == ctype]
                
                customer_insights[ctype] = {
                    'total_calls': int(len(ctype_df)),
                    'top_issues': ctype_df['ai_primary_category'].value_counts().head(5).to_dict() if 'ai_primary_category' in ctype_df else {},
                    'repeat_ticket_rate': round(len(ctype_df[ctype_df['is_ticket_repeat60d'] == 'Yes']) / len(ctype_df) * 100, 2) if 'is_ticket_repeat60d' in ctype_df and len(ctype_df) > 0 else 0,
                    'sentiment_distribution': ctype_df['ai_sentiment'].value_counts().to_dict() if 'ai_sentiment' in ctype_df else {},
                    'churn_risk_distribution': ctype_df['ai_churn_risk'].value_counts().to_dict() if 'ai_churn_risk' in ctype_df else {}
                }
        
        self.insights['customer_type_insights'] = customer_insights
        return customer_insights
    
    def identify_systemic_issues(self) -> List[Dict]:
        """Identify systemic issues based on patterns"""
        systemic_issues = []
        
        if 'ai_primary_category' in self.df.columns:
            category_counts = self.df['ai_primary_category'].value_counts()
            for category, count in category_counts.head(5).items():
                if count > len(self.df) * 0.1:
                    systemic_issues.append({
                        'type': 'HIGH_FREQUENCY_ISSUE',
                        'category': category,
                        'count': int(count),
                        'percentage': round(count / len(self.df) * 100, 2),
                        'recommendation': f"Implement proactive solution for {ISSUE_CATEGORIES.get(category, {}).get('name', category)} - affects {round(count / len(self.df) * 100, 1)}% of calls"
                    })
        
        if 'is_ticket_repeat60d' in self.df.columns:
            repeat_rate = len(self.df[self.df['is_ticket_repeat60d'] == 'Yes']) / len(self.df) * 100
            if repeat_rate > 30:
                systemic_issues.append({
                    'type': 'HIGH_REPEAT_RATE',
                    'rate': round(repeat_rate, 2),
                    'recommendation': "Focus on first-call resolution - high repeat rate indicates unresolved issues"
                })
        
        if 'ai_churn_risk' in self.df.columns and 'customer_type' in self.df.columns:
            for ctype in self.df['customer_type'].unique():
                ctype_df = self.df[self.df['customer_type'] == ctype]
                if len(ctype_df) > 50:
                    high_churn = len(ctype_df[ctype_df['ai_churn_risk'] == 'HIGH'])
                    if high_churn / len(ctype_df) > 0.15:
                        systemic_issues.append({
                            'type': 'HIGH_CHURN_SEGMENT',
                            'customer_type': ctype,
                            'churn_rate': round(high_churn / len(ctype_df) * 100, 2),
                            'recommendation': f"Implement retention program for {ctype} customers - {round(high_churn / len(ctype_df) * 100, 1)}% show high churn risk"
                        })
        
        self.insights['systemic_issues'] = systemic_issues
        return systemic_issues
    
    def extract_pain_points(self) -> Dict[str, int]:
        """Extract common customer pain points"""
        pain_points = Counter()
        
        if 'ai_customer_pain_points' in self.df.columns:
            for points in self.df['ai_customer_pain_points'].dropna():
                if isinstance(points, list):
                    for point in points:
                        pain_points[point.lower().strip()] += 1
                elif isinstance(points, str):
                    try:
                        points_list = eval(points) if points.startswith('[') else [points]
                        for point in points_list:
                            pain_points[point.lower().strip()] += 1
                    except:
                        pass
        
        self.insights['top_pain_points'] = dict(pain_points.most_common(20))
        return dict(pain_points.most_common(20))
    
    def analyze_resolution_patterns(self) -> Dict[str, Any]:
        """Analyze resolution patterns and effectiveness"""
        resolution_analysis = {}
        
        if 'ai_resolution_status' in self.df.columns:
            resolution_dist = self.df['ai_resolution_status'].value_counts().to_dict()
            total = sum(resolution_dist.values())
            
            resolution_analysis = {
                'distribution': {k: {'count': int(v), 'percentage': round(v/total*100, 2)} for k, v in resolution_dist.items()},
                'resolution_rate': round(resolution_dist.get('RESOLVED', 0) / total * 100, 2) if total > 0 else 0,
                'escalation_rate': round(resolution_dist.get('ESCALATED', 0) / total * 100, 2) if total > 0 else 0
            }
            
            if 'ai_primary_category' in self.df.columns:
                category_resolution = {}
                for category in self.df['ai_primary_category'].unique():
                    cat_df = self.df[self.df['ai_primary_category'] == category]
                    resolved = len(cat_df[cat_df['ai_resolution_status'] == 'RESOLVED'])
                    category_resolution[category] = round(resolved / len(cat_df) * 100, 2) if len(cat_df) > 0 else 0
                
                resolution_analysis['resolution_by_category'] = category_resolution
        
        self.insights['resolution_analysis'] = resolution_analysis
        return resolution_analysis
    
    def generate_actionable_recommendations(self) -> List[Dict]:
        """Generate top actionable recommendations based on all insights"""
        recommendations = []
        
        if 'category_distribution' in self.insights:
            top_category = max(self.insights['category_distribution'].items(), 
                             key=lambda x: x[1]['count'], default=(None, {}))
            if top_category[0]:
                recommendations.append({
                    'priority': 1,
                    'category': 'Process Improvement',
                    'issue': f"High volume of {ISSUE_CATEGORIES.get(top_category[0], {}).get('name', top_category[0])} issues",
                    'action': f"Create dedicated playbook and training for handling {top_category[0]} issues. Consider self-service options.",
                    'impact': f"Could reduce {top_category[1]['percentage']}% of call volume",
                    'effort': 'Medium'
                })
        
        if 'is_ticket_repeat60d' in self.df.columns:
            repeat_rate = len(self.df[self.df['is_ticket_repeat60d'] == 'Yes']) / len(self.df) * 100
            if repeat_rate > 25:
                recommendations.append({
                    'priority': 2,
                    'category': 'First Call Resolution',
                    'issue': f"High repeat ticket rate: {round(repeat_rate, 1)}%",
                    'action': "Implement post-call verification, improve resolution documentation, and empower executives with more authority",
                    'impact': f"Reducing repeat rate by 10% could save ~{int(len(self.df) * 0.1 * 0.37)} calls",
                    'effort': 'High'
                })
        
        if 'ai_churn_risk' in self.df.columns:
            high_churn = len(self.df[self.df['ai_churn_risk'] == 'HIGH'])
            if high_churn > 0:
                recommendations.append({
                    'priority': 3,
                    'category': 'Retention',
                    'issue': f"{high_churn} customers showing high churn risk",
                    'action': "Implement proactive outreach program for high-risk customers within 24 hours of flagged calls",
                    'impact': f"Potential to save {high_churn} customers from churning",
                    'effort': 'Medium'
                })
        
        if 'geography_insights' in self.insights:
            geo = self.insights['geography_insights']
            high_volume_cities = [city for city, data in geo.items() if data['total_calls'] > 50]
            if high_volume_cities:
                recommendations.append({
                    'priority': 4,
                    'category': 'Geographic Strategy',
                    'issue': f"High call volume from {len(high_volume_cities)} cities",
                    'action': f"Consider dedicated support teams for top cities: {', '.join(high_volume_cities[:5])}",
                    'impact': "Improved regional support and faster resolution",
                    'effort': 'High'
                })
        
        if 'ai_executive_performance' in self.df.columns:
            recommendations.append({
                'priority': 5,
                'category': 'Training',
                'issue': "Opportunity to improve executive performance",
                'action': "Analyze executive performance metrics and create targeted training programs",
                'impact': "Better customer experience and higher satisfaction",
                'effort': 'Medium'
            })
        
        self.insights['recommendations'] = recommendations
        return recommendations
    
    def generate_full_report(self) -> Dict[str, Any]:
        """Generate comprehensive insights report"""
        print("Aggregating by category...")
        self.aggregate_by_category()
        
        print("Aggregating by geography...")
        self.aggregate_by_geography()
        
        print("Aggregating by customer type...")
        self.aggregate_by_customer_type()
        
        print("Identifying systemic issues...")
        self.identify_systemic_issues()
        
        print("Extracting pain points...")
        self.extract_pain_points()
        
        print("Analyzing resolution patterns...")
        self.analyze_resolution_patterns()
        
        print("Generating recommendations...")
        self.generate_actionable_recommendations()
        
        self.insights['summary'] = {
            'total_calls': int(len(self.df)),
            'unique_customers': int(self.df['glid'].nunique()) if 'glid' in self.df else 0,
            'unique_cities': int(self.df['city_name'].nunique()) if 'city_name' in self.df else 0,
            'date_range': {
                'start': str(self.df['call_entered_on'].min()) if 'call_entered_on' in self.df else 'N/A',
                'end': str(self.df['call_entered_on'].max()) if 'call_entered_on' in self.df else 'N/A'
            },
            'avg_call_duration': round(self.df['call_duration'].mean(), 2) if 'call_duration' in self.df else 0,
            'repeat_ticket_rate': round(len(self.df[self.df['is_ticket_repeat60d'] == 'Yes']) / len(self.df) * 100, 2) if 'is_ticket_repeat60d' in self.df else 0
        }
        
        return self.insights
    
    def save_report(self, filepath: str):
        """Save insights report to JSON file"""
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            else:
                return obj
        
        clean_insights = convert_types(self.insights)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(clean_insights, f, ensure_ascii=False, indent=2)
        
        print(f"Report saved to {filepath}")


# =============================================================================
# QUICK INSIGHTS FROM EXISTING SUMMARY
# =============================================================================

def quick_insights_from_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract quick insights from existing summary column without Gemini
    Uses pattern matching on the already available summaries
    """
    insights = {
        'total_calls': len(df),
        'sentiment_distribution': {},
        'key_topics': Counter(),
        'concerns_patterns': Counter(),
        'alert_calls': 0
    }
    
    for idx, row in df.iterrows():
        summary = str(row.get('summary', ''))
        
        if '@@@Sentiment:' in summary:
            try:
                sentiment_part = summary.split('@@@Sentiment:')[1].split('@@@')[0]
                if 'Positive' in sentiment_part:
                    insights['sentiment_distribution']['Positive'] = insights['sentiment_distribution'].get('Positive', 0) + 1
                elif 'Negative' in sentiment_part:
                    insights['sentiment_distribution']['Negative'] = insights['sentiment_distribution'].get('Negative', 0) + 1
                else:
                    insights['sentiment_distribution']['Neutral'] = insights['sentiment_distribution'].get('Neutral', 0) + 1
            except:
                pass
        
        if '@@@Key Topics:' in summary:
            try:
                topics_part = summary.split('@@@Key Topics:')[1].split('\n')[1]
                topics = [t.strip() for t in topics_part.split(',')]
                for topic in topics:
                    if topic and len(topic) > 2:
                        insights['key_topics'][topic.lower()] += 1
            except:
                pass
        
        if 'Alert (If Any):' in summary:
            alert_part = summary.split('Alert (If Any):')[1].split('@@@')[0]
            if 'None' not in alert_part and alert_part.strip():
                insights['alert_calls'] += 1
    
    insights['key_topics'] = dict(insights['key_topics'].most_common(30))
    
    return insights
