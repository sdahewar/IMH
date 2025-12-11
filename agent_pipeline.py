"""
IndiaMART Insights Engine - Agent Pipeline
End-to-end LLM-powered analysis using NVIDIA NIM Nemotron-4-Mini-Hindi

Usage:
    python agent_pipeline.py                    # Interactive mode
    python agent_pipeline.py --transcript       # Analyze a single transcript
    python agent_pipeline.py --customer <id>   # Analyze by customer
    python agent_pipeline.py --city <name>     # Analyze by city
    python agent_pipeline.py --type <type>     # Analyze by customer type
"""

import os
import sys
import json
import argparse
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents import InsightsAgent, AggregationAgent
from src.config import NVIDIA_MODEL, OUTPUT_DIR


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              🤖 IndiaMART INSIGHTS ENGINE - AGENT PIPELINE                   ║
║                                                                              ║
║              Powered by: NVIDIA NIM (Nemotron-4-Mini-Hindi)                  ║
║              Optimized for Hinglish Transcripts                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


def load_data(filepath: str = "Data Voice Hackathon_Master.xlsx") -> pd.DataFrame:
    """Load the dataset"""
    paths_to_try = [filepath, f"data/{filepath}"]
    
    for path in paths_to_try:
        if os.path.exists(path):
            print(f"📂 Loading data from {path}...")
            df = pd.read_excel(path)
            print(f"✅ Loaded {len(df):,} records")
            return df
    
    print(f"❌ Could not find data file")
    return None


def analyze_single_transcript_interactive(agent: InsightsAgent):
    """Interactive mode for analyzing a single transcript"""
    print("\n" + "=" * 80)
    print("📝 SINGLE TRANSCRIPT ANALYSIS")
    print("=" * 80)
    
    print("\nPaste your transcript below (press Enter twice to finish):")
    print("-" * 40)
    
    lines = []
    empty_count = 0
    while True:
        try:
            line = input()
            if line == "":
                empty_count += 1
                if empty_count >= 2:
                    break
            else:
                empty_count = 0
            lines.append(line)
        except EOFError:
            break
    
    transcript = "\n".join(lines).strip()
    
    if not transcript:
        print("❌ No transcript provided")
        return
    
    print(f"\n📊 Transcript length: {len(transcript)} characters")
    
    # Get optional metadata
    print("\n📋 Optional: Enter metadata (press Enter to skip)")
    customer_type = input("   Customer Type (e.g., CATALOG, TSCATALOG): ").strip() or "Unknown"
    city = input("   City: ").strip() or "Unknown"
    
    metadata = {
        'customer_type': customer_type,
        'city': city
    }
    
    # Analyze
    print("\n🔍 Analyzing transcript...")
    result = agent.analyze_transcript(transcript, metadata)
    
    # Display results
    print("\n" + "=" * 80)
    print("📊 ANALYSIS RESULTS")
    print("=" * 80)
    
    if result.get('analysis_success'):
        print(f"\n🏷️  Primary Category: {result.get('primary_category', 'N/A')}")
        print(f"📝 Issue Summary: {result.get('issue_summary', 'N/A')}")
        print(f"\n😊 Sentiment: {result.get('sentiment', 'N/A')}")
        print(f"📈 Sentiment Shift: {result.get('sentiment_shift', 'N/A')}")
        print(f"⚠️  Churn Risk: {result.get('churn_risk', 'N/A')}")
        print(f"🚨 Urgency: {result.get('urgency', 'N/A')}")
        print(f"✅ Resolution: {result.get('resolution_status', 'N/A')}")
        
        print(f"\n😟 Pain Points:")
        for pp in result.get('customer_pain_points', []):
            print(f"   • {pp}")
        
        print(f"\n👨‍💼 Executive Performance:")
        exec_perf = result.get('executive_performance', {})
        print(f"   Empathy: {'✅' if exec_perf.get('empathy_shown') else '❌'}")
        print(f"   Solution Offered: {'✅' if exec_perf.get('solution_offered') else '❌'}")
        print(f"   Followed Process: {'✅' if exec_perf.get('followed_process') else '❌'}")
        
        print(f"\n💡 Actionable Insight:")
        print(f"   {result.get('actionable_insight', 'N/A')}")
        
        if result.get('requires_follow_up'):
            print(f"\n🔄 Follow-up Required: {result.get('follow_up_reason', 'N/A')}")
    else:
        print(f"\n❌ Analysis failed: {result.get('error', 'Unknown error')}")
    
    # Ask follow-up questions
    print("\n" + "-" * 40)
    print("💬 Ask follow-up questions about this transcript (type 'exit' to quit)")
    
    while True:
        question = input("\n❓ Your question: ").strip()
        if question.lower() in ['exit', 'quit', 'q', '']:
            break
        
        answer = agent.ask_question(transcript, question)
        print(f"\n💡 Answer:\n{answer}")
    
    return result


def analyze_by_customer(df: pd.DataFrame, customer_id: int, agent: AggregationAgent):
    """Analyze all transcripts for a specific customer"""
    print(f"\n👤 Analyzing customer: {customer_id}")
    
    result = agent.aggregate_by_customer(df, customer_id)
    
    if 'error' in result:
        print(f"❌ {result['error']}")
        return
    
    # Print aggregated insights
    print_aggregated_results(result)
    
    # Generate and print recommendations
    if 'customer_recommendations' in result:
        print("\n" + "=" * 80)
        print("💡 CUSTOMER-SPECIFIC RECOMMENDATIONS")
        print("=" * 80)
        print(result['customer_recommendations'])
    
    # Save results
    save_results(result, f"customer_{customer_id}")
    
    return result


def analyze_by_city(df: pd.DataFrame, city: str, agent: AggregationAgent):
    """Analyze all transcripts for a specific city"""
    print(f"\n📍 Analyzing city: {city}")
    
    result = agent.aggregate_by_location(df, city)
    
    if 'error' in result:
        print(f"❌ {result['error']}")
        return
    
    print_aggregated_results(result)
    
    if 'location_insights' in result:
        print("\n" + "=" * 80)
        print("📍 LOCATION-SPECIFIC INSIGHTS")
        print("=" * 80)
        print(result['location_insights'])
    
    save_results(result, f"city_{city.replace(' ', '_')}")
    
    return result


def analyze_by_customer_type(df: pd.DataFrame, customer_type: str, agent: AggregationAgent, sample_size: int = 30):
    """Analyze transcripts for a specific customer type"""
    print(f"\n👥 Analyzing customer type: {customer_type}")
    
    result = agent.aggregate_by_customer_type(df, customer_type, sample_size=sample_size)
    
    if 'error' in result:
        print(f"❌ {result['error']}")
        return
    
    print_aggregated_results(result)
    
    if 'segment_recommendations' in result:
        print("\n" + "=" * 80)
        print("👥 SEGMENT-SPECIFIC RECOMMENDATIONS")
        print("=" * 80)
        print(result['segment_recommendations'])
    
    save_results(result, f"type_{customer_type}")
    
    return result


def print_aggregated_results(result: dict):
    """Print aggregated analysis results"""
    print("\n" + "=" * 80)
    print("📊 AGGREGATED ANALYSIS RESULTS")
    print("=" * 80)
    
    print(f"\n📈 Overview:")
    print(f"   Total Analyzed: {result.get('total_analyzed', 0)}")
    print(f"   Successful: {result.get('successful', 0)}")
    
    agg = result.get('aggregated_insights', {})
    
    print(f"\n🏷️  Category Distribution:")
    for cat, count in agg.get('category_distribution', {}).items():
        pct = count / result.get('total_analyzed', 1) * 100
        bar = "█" * int(pct / 5)
        print(f"   {cat:25} {bar} {count} ({pct:.1f}%)")
    
    print(f"\n😊 Sentiment Distribution:")
    for sent, count in agg.get('sentiment_distribution', {}).items():
        print(f"   • {sent}: {count}")
    
    print(f"\n⚠️  Churn Risk:")
    for risk, count in agg.get('churn_risk_distribution', {}).items():
        print(f"   • {risk}: {count}")
    
    print(f"\n😟 Top Pain Points:")
    for pp, count in list(agg.get('top_pain_points', {}).items())[:5]:
        print(f"   • {pp}: {count}")
    
    print(f"\n👨‍💼 Executive Performance:")
    exec_perf = agg.get('executive_performance', {})
    print(f"   Empathy Rate: {exec_perf.get('empathy_rate', 0)}%")
    print(f"   Solution Rate: {exec_perf.get('solution_rate', 0)}%")
    print(f"   Process Compliance: {exec_perf.get('process_compliance', 0)}%")
    print(f"   Escalation Rate: {exec_perf.get('escalation_rate', 0)}%")


def save_results(result: dict, prefix: str):
    """Save analysis results to file"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Remove non-serializable items
    save_result = {k: v for k, v in result.items() if k != 'individual_results'}
    
    filename = f"{OUTPUT_DIR}/agent_analysis_{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(save_result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n💾 Results saved to {filename}")


def interactive_mode(df: pd.DataFrame):
    """Run interactive mode"""
    insights_agent = InsightsAgent(verbose=True)
    aggregation_agent = AggregationAgent(verbose=True)
    
    while True:
        print("\n" + "=" * 80)
        print("🎯 MAIN MENU")
        print("=" * 80)
        print("""
Select an option:

    1. 📝 Analyze a single transcript (paste text)
    2. 👤 Analyze by Customer ID
    3. 📍 Analyze by City/Location
    4. 👥 Analyze by Customer Type
    5. 📊 Quick dataset overview
    6. 🔍 Search transcripts by keyword
    7. ❌ Exit

        """)
        
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == "1":
            analyze_single_transcript_interactive(insights_agent)
        
        elif choice == "2":
            customer_id = input("\nEnter Customer ID (glid): ").strip()
            try:
                customer_id = int(customer_id)
                analyze_by_customer(df, customer_id, aggregation_agent)
            except ValueError:
                print("❌ Invalid customer ID")
        
        elif choice == "3":
            print("\n📍 Available cities (top 20):")
            top_cities = df['city_name'].value_counts().head(20)
            for i, (city, count) in enumerate(top_cities.items(), 1):
                print(f"   {i:2}. {city}: {count} calls")
            
            city = input("\nEnter city name: ").strip()
            if city:
                analyze_by_city(df, city, aggregation_agent)
        
        elif choice == "4":
            print("\n👥 Available customer types:")
            for ctype, count in df['customer_type'].value_counts().items():
                print(f"   • {ctype}: {count} calls")
            
            ctype = input("\nEnter customer type: ").strip()
            if ctype:
                sample = input("Sample size (default 30): ").strip()
                sample_size = int(sample) if sample else 30
                analyze_by_customer_type(df, ctype, aggregation_agent, sample_size)
        
        elif choice == "5":
            print("\n📊 DATASET OVERVIEW")
            print("-" * 40)
            print(f"Total Records: {len(df):,}")
            print(f"Unique Customers: {df['glid'].nunique():,}")
            print(f"Unique Cities: {df['city_name'].nunique()}")
            print(f"Date Range: {df['call_entered_on'].min()} to {df['call_entered_on'].max()}")
            print(f"\nCustomer Types:")
            for ctype, count in df['customer_type'].value_counts().head(10).items():
                print(f"   {ctype}: {count}")
            print(f"\nTop Cities:")
            for city, count in df['city_name'].value_counts().head(10).items():
                print(f"   {city}: {count}")
        
        elif choice == "6":
            keyword = input("\nEnter keyword to search: ").strip()
            if keyword:
                matches = df[df['transcript'].str.contains(keyword, case=False, na=False)]
                print(f"\n🔍 Found {len(matches)} transcripts containing '{keyword}'")
                
                if len(matches) > 0:
                    analyze = input("Analyze these transcripts? (y/n): ").strip().lower()
                    if analyze == 'y':
                        sample_size = min(30, len(matches))
                        print(f"\n📊 Analyzing {sample_size} matching transcripts...")
                        
                        sample_df = matches.sample(n=sample_size, random_state=42)
                        transcripts = []
                        for _, row in sample_df.iterrows():
                            transcripts.append({
                                'transcript': row.get('transcript', ''),
                                'metadata': {
                                    'customer_type': row.get('customer_type', ''),
                                    'city': row.get('city_name', ''),
                                    'is_repeat': row.get('is_ticket_repeat60d', '')
                                }
                            })
                        
                        result = aggregation_agent.analyze_multiple_transcripts(transcripts)
                        print_aggregated_results(result)
                        
                        # Generate executive summary
                        summary = aggregation_agent.generate_executive_summary(result)
                        print("\n" + "=" * 80)
                        print("📋 EXECUTIVE SUMMARY")
                        print("=" * 80)
                        print(summary)
                        
                        save_results(result, f"keyword_{keyword}")
        
        elif choice == "7" or choice.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")


def main():
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="IndiaMART Insights Engine - Agent Pipeline"
    )
    parser.add_argument("--transcript", action="store_true", help="Analyze a single transcript")
    parser.add_argument("--customer", type=int, help="Analyze by customer ID")
    parser.add_argument("--city", type=str, help="Analyze by city name")
    parser.add_argument("--type", type=str, help="Analyze by customer type")
    parser.add_argument("--sample-size", type=int, default=30, help="Sample size for analysis")
    parser.add_argument("--input", default="Data Voice Hackathon_Master.xlsx", help="Input data file")
    
    args = parser.parse_args()
    
    print(f"🤖 Model: {NVIDIA_MODEL}")
    
    # Load data
    df = load_data(args.input)
    if df is None:
        return
    
    # Initialize agents
    insights_agent = InsightsAgent(verbose=True)
    aggregation_agent = AggregationAgent(verbose=True)
    
    if args.transcript:
        analyze_single_transcript_interactive(insights_agent)
    
    elif args.customer:
        analyze_by_customer(df, args.customer, aggregation_agent)
    
    elif args.city:
        analyze_by_city(df, args.city, aggregation_agent)
    
    elif args.type:
        analyze_by_customer_type(df, args.type, aggregation_agent, args.sample_size)
    
    else:
        # Interactive mode
        interactive_mode(df)


if __name__ == "__main__":
    main()

