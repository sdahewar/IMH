"""
IndiaMART Insights Engine - Main Entry Point
Complete end-to-end system for extracting insights from call transcripts

Usage:
    python main.py --mode sample --sample-size 10    # Test on 10 samples
    python main.py --mode full                        # Process all 5600+ calls
    python main.py --mode quick-insights              # Quick insights from existing summaries
    python main.py --mode demo                        # Run 5 use case demo
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import OUTPUT_DIR, CHECKPOINT_DIR
from src.utils.helpers import print_header

# Load environment variables
load_dotenv()


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           IndiaMART INSIGHTS ENGINE                              ║
║           Voice Call Analysis Pipeline                           ║
║           Using NVIDIA NIM (Nemotron-4-Mini-Hindi)               ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def validate_api_key() -> str:
    """Validate and return NVIDIA NIM API key"""
    from src.config import NVIDIA_API_KEY
    
    # Check for override via environment
    api_key = os.getenv("NVIDIA_API_KEY") or NVIDIA_API_KEY
    
    if not api_key:
        print("ERROR: NVIDIA API key not found!")
        print("\nTo set your API key:")
        print("  1. Update src/config.py with your key")
        print("  2. Or set environment variable: $env:NVIDIA_API_KEY='your_key'")
        sys.exit(1)
    
    print("✅ NVIDIA NIM API key configured")
    print(f"   Model: nvidia/nemotron-4-mini-hindi-4b-instruct")
    return api_key


def load_data(filepath: str = "Data Voice Hackathon_Master.xlsx") -> pd.DataFrame:
    """Load and validate the dataset"""
    print(f"\n📂 Loading data from {filepath}...")
    
    # Try multiple paths
    paths_to_try = [
        filepath,
        f"data/{filepath}",
        os.path.join(os.path.dirname(__file__), filepath)
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            df = pd.read_excel(path)
            print(f"✅ Loaded {len(df):,} records with {len(df.columns)} columns")
            return df
    
    print(f"❌ ERROR: File not found. Tried: {paths_to_try}")
    sys.exit(1)


def run_sample_classification(df: pd.DataFrame, api_key: str, sample_size: int = 10):
    """Run classification on a sample for testing - outputs merged CSV with all columns"""
    from src.classifiers import classify_sample
    
    print(f"\n🔬 Running sample classification on {sample_size} records...")
    print("   Using NVIDIA NIM: Nemotron-4-Mini-Hindi (optimized for Hinglish)")
    
    # Use the classify_sample function that returns merged DataFrame with verbose output
    merged_df = classify_sample(df, sample_size=sample_size, api_key=api_key, verbose=True)
    
    # Save merged results to CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = f"{OUTPUT_DIR}/classified_calls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ Results saved to {output_file}")
    print(f"   Total columns: {len(merged_df.columns)} (Original: {len(df.columns)} + AI Metrics)")
    
    # Show column summary
    original_cols = [c for c in merged_df.columns if not c.startswith('ai_')]
    ai_cols = [c for c in merged_df.columns if c.startswith('ai_')]
    
    print(f"\n📋 Original columns: {len(original_cols)}")
    print(f"🤖 AI-derived columns: {len(ai_cols)}")
    print(f"   AI columns: {', '.join(ai_cols)}")
    
    # Print sample results
    print("\n" + "=" * 70)
    print("SAMPLE CLASSIFICATION RESULTS")
    print("=" * 70)
    
    for idx, row in merged_df.iterrows():
        print(f"\n{'─' * 70}")
        print(f"📞 Call #{idx + 1} | ID: {row.get('click_to_call_id', 'N/A')}")
        print(f"{'─' * 70}")
        print(f"Customer Type: {row.get('customer_type', 'N/A')}")
        print(f"City: {row.get('city_name', 'N/A')}")
        print(f"Repeat Ticket: {row.get('is_ticket_repeat60d', 'N/A')}")
        print(f"Call Duration: {row.get('call_duration', 'N/A')}s")
        print(f"\n🏷️  Primary Category: {row.get('ai_primary_category', 'N/A')}")
        print(f"📝 Issue Summary: {row.get('ai_issue_summary', 'N/A')}")
        print(f"😟 Pain Points: {row.get('ai_customer_pain_points', 'N/A')}")
        print(f"😊 Sentiment: {row.get('ai_sentiment', 'N/A')}")
        print(f"📈 Sentiment Shift: {row.get('ai_sentiment_shift', 'N/A')}")
        print(f"⚠️  Churn Risk: {row.get('ai_churn_risk', 'N/A')}")
        print(f"🚨 Urgency: {row.get('ai_urgency', 'N/A')}")
        print(f"✅ Resolution: {row.get('ai_resolution_status', 'N/A')}")
        print(f"\n👨‍💼 Executive Performance:")
        print(f"   Empathy: {row.get('ai_exec_empathy_shown', 'N/A')}")
        print(f"   Solution Offered: {row.get('ai_exec_solution_offered', 'N/A')}")
        print(f"   Followed Process: {row.get('ai_exec_followed_process', 'N/A')}")
        print(f"\n💡 Actionable Insight: {row.get('ai_actionable_insight', 'N/A')}")
        print(f"🔄 Follow-up Required: {row.get('ai_requires_follow_up', 'N/A')}")
        if row.get('ai_requires_follow_up'):
            print(f"   Reason: {row.get('ai_follow_up_reason', 'N/A')}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    if 'ai_primary_category' in merged_df.columns:
        print("\n📊 Category Distribution:")
        for cat, count in merged_df['ai_primary_category'].value_counts().items():
            print(f"   • {cat}: {count} ({count/len(merged_df)*100:.1f}%)")
    
    if 'ai_churn_risk' in merged_df.columns:
        print("\n⚠️ Churn Risk Distribution:")
        for risk, count in merged_df['ai_churn_risk'].value_counts().items():
            print(f"   • {risk}: {count}")
    
    if 'ai_sentiment' in merged_df.columns:
        print("\n😊 Sentiment Distribution:")
        for sent, count in merged_df['ai_sentiment'].value_counts().items():
            print(f"   • {sent}: {count}")
    
    if 'ai_resolution_status' in merged_df.columns:
        print("\n✅ Resolution Status:")
        for status, count in merged_df['ai_resolution_status'].value_counts().items():
            print(f"   • {status}: {count}")
    
    return merged_df


def run_full_classification(df: pd.DataFrame, api_key: str):
    """Run classification on the full dataset - outputs merged CSV with all columns"""
    from src.classifiers import NvidiaClassifier, BatchProcessor
    from src.classifiers.nvidia_classifier import flatten_classification_results
    
    print(f"\n🚀 Starting FULL classification of {len(df):,} records...")
    print("🤖 Using NVIDIA NIM: Nemotron-4-Mini-Hindi (optimized for Hinglish)")
    print("⏱️  Estimated time: ~2-3 hours (with rate limiting)")
    print("💾 Checkpoints will be saved every batch")
    
    classifier = NvidiaClassifier(api_key=api_key)
    processor = BatchProcessor(classifier, checkpoint_dir=CHECKPOINT_DIR)
    
    results_df = processor.process_with_checkpoints(df, batch_size=10, resume=True)
    
    # Flatten nested structures
    results_df = flatten_classification_results(results_df)
    
    # Add 'ai_' prefix to classification columns
    classification_cols = [c for c in results_df.columns if c not in ['original_index', 'call_id']]
    rename_dict = {c: f'ai_{c}' for c in classification_cols}
    results_df = results_df.rename(columns=rename_dict)
    
    # Merge with original DataFrame
    merged_df = df.copy()
    for col in results_df.columns:
        if col.startswith('ai_'):
            merged_df[col] = None
            for _, row in results_df.iterrows():
                orig_idx = row.get('original_index', None)
                if orig_idx is not None and orig_idx in merged_df.index:
                    merged_df.at[orig_idx, col] = row[col]
    
    # Save merged results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = f"{OUTPUT_DIR}/classified_calls_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ Full results saved to {output_file}")
    print(f"   Total rows: {len(merged_df):,}")
    print(f"   Total columns: {len(merged_df.columns)} (Original: {len(df.columns)} + AI Metrics)")
    
    # Print summary
    ai_cols = [c for c in merged_df.columns if c.startswith('ai_')]
    print(f"\n🤖 AI-derived columns ({len(ai_cols)}):")
    for col in ai_cols:
        print(f"   • {col}")
    
    return merged_df


def run_quick_insights(df: pd.DataFrame):
    """Run quick insights extraction from existing summaries"""
    from src.aggregators import quick_insights_from_summary
    import json
    
    print("\n⚡ Running quick insights extraction from existing summaries...")
    
    insights = quick_insights_from_summary(df)
    
    print("\n" + "=" * 70)
    print("QUICK INSIGHTS FROM EXISTING CALL SUMMARIES")
    print("=" * 70)
    
    print(f"\n📊 Total Calls Analyzed: {insights['total_calls']:,}")
    
    print(f"\n😊 Sentiment Distribution:")
    total = sum(insights['sentiment_distribution'].values())
    for sentiment, count in sorted(insights['sentiment_distribution'].items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {sentiment:15} {bar} {count:,} ({pct:.1f}%)")
    
    print(f"\n🚨 Alert Calls: {insights['alert_calls']} ({insights['alert_calls']/insights['total_calls']*100:.2f}%)")
    
    print(f"\n🏷️  Top 20 Key Topics from Calls:")
    for i, (topic, count) in enumerate(list(insights['key_topics'].items())[:20], 1):
        print(f"  {i:2}. {topic}: {count}")
    
    # Save insights
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = f"{OUTPUT_DIR}/quick_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(insights, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Quick insights saved to {output_file}")
    
    return insights


def run_demo():
    """Run the 5 use case demo"""
    from scripts.run_demo import run_demo as demo
    demo()


def main():
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="IndiaMART Insights Engine - Voice Call Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode sample --sample-size 10
  python main.py --mode full
  python main.py --mode quick-insights
  python main.py --mode demo
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["sample", "full", "quick-insights", "demo"],
        default="sample",
        help="Pipeline mode"
    )
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=10,
        help="Number of samples for sample mode"
    )
    parser.add_argument(
        "--input", 
        default="Data Voice Hackathon_Master.xlsx",
        help="Input data file"
    )
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    if args.mode == "demo":
        run_demo()
    
    elif args.mode == "quick-insights":
        df = load_data(args.input)
        run_quick_insights(df)
    
    else:
        # Sample and full modes need API key
        api_key = validate_api_key()
        df = load_data(args.input)
        
        if args.mode == "sample":
            run_sample_classification(df, api_key, args.sample_size)
        elif args.mode == "full":
            run_full_classification(df, api_key)
    
    print("\n" + "=" * 70)
    print("✅ Pipeline completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

