"""
IndiaMART Insights Engine - Demo with 5 Use Cases
Demonstrates end-to-end flow from audio/transcript to actionable insights

Usage: python scripts/run_demo.py
"""

import os
import sys
import json
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from dotenv import load_dotenv

from src.classifiers import NvidiaClassifier
from src.utils.helpers import print_header, print_section

load_dotenv()


def demo_use_case(idx: int, row: pd.Series, result: dict):
    """Display a single use case demo"""
    print_header(f"USE CASE {idx}: {result.get('primary_category', 'UNKNOWN')}", "▓")
    
    # Call Metadata
    print(" CALL METADATA")
    print(f"   • Customer Type: {row['customer_type']}")
    print(f"   • City: {row['city_name']}")
    print(f"   • Call Direction: {row['FLAG_IN_OUT']}")
    print(f"   • Duration: {row['call_duration']} seconds")
    print(f"   • Repeat Ticket: {row['is_ticket_repeat60d']}")
    print(f"   • Vertical: {row['iil_vertical_name']}")
    
    # Transcript Preview
    print_section("TRANSCRIPT PREVIEW (First 500 chars)")
    transcript = row['transcript'][:500] if pd.notna(row['transcript']) else "N/A"
    print(f"   {transcript}...")
    
    # AI Classification Results
    print_section("🤖 AI CLASSIFICATION RESULTS")
    print(f"   Primary Category: {result.get('primary_category', 'N/A')}")
    print(f"   Secondary Categories: {result.get('secondary_categories', [])}")
    
    print_section("📝 ISSUE SUMMARY")
    print(f"   {result.get('issue_summary', 'N/A')}")
    
    print_section("😊 SENTIMENT ANALYSIS")
    print(f"   Sentiment: {result.get('sentiment', 'N/A')}")
    print(f"   Sentiment Shift: {result.get('sentiment_shift', 'N/A')}")
    
    print_section("⚠️ RISK ASSESSMENT")
    print(f"   Churn Risk: {result.get('churn_risk', 'N/A')}")
    print(f"   Urgency: {result.get('urgency', 'N/A')}")
    print(f"   Resolution Status: {result.get('resolution_status', 'N/A')}")
    
    print_section("😟 CUSTOMER PAIN POINTS")
    pain_points = result.get('customer_pain_points', [])
    if pain_points:
        for pp in pain_points:
            print(f"   • {pp}")
    else:
        print("   None identified")
    
    print_section("👨‍💼 EXECUTIVE PERFORMANCE")
    exec_perf = result.get('executive_performance', {})
    print(f"   • Empathy Shown: {'✅' if exec_perf.get('empathy_shown') else '❌'}")
    print(f"   • Solution Offered: {'✅' if exec_perf.get('solution_offered') else '❌'}")
    print(f"   • Followed Process: {'✅' if exec_perf.get('followed_process') else '❌'}")
    print(f"   • Escalation Needed: {'⚠️ Yes' if exec_perf.get('escalation_needed') else 'No'}")
    
    print_section("💡 ACTIONABLE INSIGHT")
    print(f"   {result.get('actionable_insight', 'N/A')}")
    
    print_section("🔄 FOLLOW-UP")
    if result.get('requires_follow_up'):
        print(f"   ⚠️ REQUIRES FOLLOW-UP: {result.get('follow_up_reason', 'N/A')}")
    else:
        print("   ✅ No follow-up required")
    
    print_section("🏷️ KEYWORDS")
    keywords = result.get('keywords', [])
    print(f"   {', '.join(keywords) if keywords else 'N/A'}")
    
    print()


def run_demo():
    """Run the full demo with 5 use cases"""
    print("""
    
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║            🎯 IndiaMART INSIGHTS ENGINE - DEMO                               ║
║            Voice Call Analysis Pipeline                                      ║
║            Data Voice Hackathon 2024                                         ║
║                                                                              ║
║            Demonstrating End-to-End Flow:                                    ║
║            Audio → Transcript → Classification → Actionable Insights        ║
║                                                                              ║
║            Powered by: NVIDIA NIM (Nemotron-4-Mini-Hindi)                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Get API key from config
    from src.config import NVIDIA_API_KEY
    api_key = os.getenv("NVIDIA_API_KEY") or NVIDIA_API_KEY
    
    if not api_key:
        print("❌ NVIDIA API key not found!")
        print("   Please check src/config.py or set NVIDIA_API_KEY environment variable")
        return
    
    print("✅ NVIDIA NIM API configured")
    print("   Model: nvidia/nemotron-4-mini-hindi-4b-instruct (optimized for Hinglish)")
    
    # Load data
    print("\n📂 Loading dataset...")
    df = pd.read_excel("Data Voice Hackathon_Master-1.xlsx")
    print(f"   Loaded {len(df):,} call records")
    
    # Select 5 diverse use cases
    print("\n🎯 Selecting 5 diverse use cases for demonstration...")
    
    use_cases = []
    
    # 1. High repeat ticket case
    repeat_cases = df[df['is_ticket_repeat60d'] == 'Yes']
    if len(repeat_cases) > 0:
        use_cases.append(repeat_cases.sample(1).iloc[0])
    
    # 2. Long duration call
    long_calls = df[df['call_duration'] > 300]
    if len(long_calls) > 0:
        use_cases.append(long_calls.sample(1).iloc[0])
    
    # 3. New customer
    new_customers = df[df['vintage_months'] <= 6]
    if len(new_customers) > 0:
        use_cases.append(new_customers.sample(1).iloc[0])
    
    # 4. Premium customer
    premium = df[df['customer_type'].isin(['STAR', 'LEADER'])]
    if len(premium) > 0:
        use_cases.append(premium.sample(1).iloc[0])
    
    # 5. Random sample
    use_cases.append(df.sample(1).iloc[0])
    
    while len(use_cases) < 5:
        use_cases.append(df.sample(1).iloc[0])
    
    print(f"   Selected {len(use_cases)} diverse use cases")
    
    # Classify
    classifier = NvidiaClassifier(api_key=api_key)
    
    print("\n🔄 Running NVIDIA NIM classification on each use case...")
    print("   This may take a few seconds per call...\n")
    
    results = []
    for i, row in enumerate(use_cases[:5], 1):
        print(f"   Processing Use Case {i}/5... ", end="", flush=True)
        
        metadata = {
            'customer_type': row.get('customer_type', ''),
            'city': row.get('city_name', ''),
            'call_direction': row.get('FLAG_IN_OUT', ''),
            'is_repeat': row.get('is_ticket_repeat60d', ''),
            'duration': row.get('call_duration', ''),
            'summary': row.get('summary', '')
        }
        
        result = classifier.classify_single(row['transcript'], metadata)
        results.append(result)
        
        if result.get('classification_success', False):
            print(f"✅ Category: {result.get('primary_category', 'N/A')}")
        else:
            print(f"⚠️ Classification issue: {result.get('error', 'Unknown')}")
        
        time.sleep(0.5)
    
    # Display all use cases
    print("\n\n" + "█" * 80)
    print(" DETAILED USE CASE ANALYSIS")
    print("█" * 80)
    
    for i, (row, result) in enumerate(zip(use_cases[:5], results), 1):
        demo_use_case(i, row, result)
    
    # Summary
    print_header("📊 AGGREGATED INSIGHTS FROM 5 USE CASES", "█")
    
    categories = [r.get('primary_category', 'UNKNOWN') for r in results]
    print("\n📋 Category Distribution:")
    for cat in set(categories):
        count = categories.count(cat)
        print(f"   • {cat}: {count} ({count/len(categories)*100:.0f}%)")
    
    sentiments = [r.get('sentiment', 'UNKNOWN') for r in results]
    print("\n😊 Sentiment Distribution:")
    for sent in set(sentiments):
        count = sentiments.count(sent)
        print(f"   • {sent}: {count}")
    
    churn_risks = [r.get('churn_risk', 'UNKNOWN') for r in results]
    high_churn = churn_risks.count('HIGH')
    print(f"\n⚠️ High Churn Risk Cases: {high_churn}")
    
    follow_ups = sum(1 for r in results if r.get('requires_follow_up', False))
    print(f"\n🔄 Cases Requiring Follow-up: {follow_ups}")
    
    print_header("💡 KEY ACTIONABLE INSIGHTS", "█")
    
    for i, result in enumerate(results, 1):
        insight = result.get('actionable_insight', 'N/A')
        print(f"\n   Use Case {i}: {insight}")
    
    # Create merged DataFrame with all original columns + AI metrics
    os.makedirs("output", exist_ok=True)
    
    # Build merged DataFrame
    rows_data = []
    for i, (row, result) in enumerate(zip(use_cases[:5], results)):
        # Start with original row data
        row_dict = row.to_dict()
        
        # Flatten and add AI results with 'ai_' prefix
        for key, value in result.items():
            if key == 'executive_performance' and isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    row_dict[f'ai_exec_{sub_key}'] = sub_value
            elif isinstance(value, list):
                row_dict[f'ai_{key}'] = ', '.join(str(v) for v in value)
            else:
                row_dict[f'ai_{key}'] = value
        
        rows_data.append(row_dict)
    
    merged_df = pd.DataFrame(rows_data)
    
    # Save as CSV
    output_file = f"output/demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ Demo results saved to {output_file}")
    print(f"   Total columns: {len(merged_df.columns)}")
    
    # List AI columns
    ai_cols = [c for c in merged_df.columns if c.startswith('ai_')]
    print(f"   AI-derived columns: {len(ai_cols)}")
    
    print("\n" + "═" * 80)
    print(" DEMO COMPLETED SUCCESSFULLY!")
    print("═" * 80)


if __name__ == "__main__":
    run_demo()

