"""
Audio Pipeline - Complete Audio-to-Insights Pipeline
IndiaMART Insights Engine

Pipeline: Audio → Whisper STT → Transcript → NVIDIA NIM → Insights

Usage:
    # Single file
    python audio_pipeline.py --audio call.mp3
    
    # Folder of audio files
    python audio_pipeline.py --folder ./recordings
    
    # With metadata
    python audio_pipeline.py --audio call.mp3 --city Mumbai --type CATALOG
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           IndiaMART Insights Engine - Audio Pipeline             ║
║                Audio → Transcript → Insights                      ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def process_single_audio(args):
    """Process a single audio file"""
    from src.stt import STTManager
    
    print(f"\n🎤 Processing: {args.audio}")
    
    # Build metadata
    metadata = {}
    if args.city:
        metadata['city'] = args.city
    if args.customer_type:
        metadata['customer_type'] = args.customer_type
    if args.call_direction:
        metadata['call_direction'] = args.call_direction
    
    # Initialize pipeline
    manager = STTManager(
        stt_mode=args.stt_mode,
        openai_api_key=args.openai_key or os.getenv("OPENAI_API_KEY"),
        verbose=True
    )
    
    # Process
    result = manager.process_audio(args.audio, metadata)
    
    # Save result
    output_file = args.output or f"output/analysis_{Path(args.audio).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print summary
    if result.get('status') == 'success':
        insights = result.get('insights', {})
        print(f"\n📊 Analysis Summary:")
        print(f"   Category: {insights.get('primary_category', 'N/A')}")
        print(f"   Undertone: {insights.get('seller_undertone', 'N/A')}")
        print(f"   Churn Risk: {insights.get('churn_risk_assessment', {}).get('risk_level', 'N/A')}")
        
        if insights.get('opportunities', {}).get('upsell_opportunity'):
            print(f"   💰 UPSELL OPPORTUNITY DETECTED!")
    
    return result


def process_folder(args):
    """Process all audio files in a folder"""
    from src.stt import STTManager
    
    folder = Path(args.folder)
    print(f"\n📁 Processing folder: {folder}")
    
    # Initialize pipeline
    manager = STTManager(
        stt_mode=args.stt_mode,
        openai_api_key=args.openai_key or os.getenv("OPENAI_API_KEY"),
        verbose=True
    )
    
    # Output file
    output_file = args.output or f"output/batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Process
    results = manager.process_folder(
        folder_path=str(folder),
        output_file=output_file
    )
    
    # Print summary
    success = sum(1 for r in results if r.get('status') == 'success')
    print(f"\n📊 Batch Summary:")
    print(f"   Total files: {len(results)}")
    print(f"   Successful: {success}")
    print(f"   Failed: {len(results) - success}")
    
    # Category distribution
    categories = {}
    for r in results:
        if r.get('status') == 'success':
            cat = r.get('insights', {}).get('primary_category', 'UNKNOWN')
            categories[cat] = categories.get(cat, 0) + 1
    
    if categories:
        print(f"\n   Category Distribution:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"   • {cat}: {count}")
    
    return results


def interactive_mode():
    """Interactive mode for testing"""
    from src.stt import STTManager
    
    print("\n🎮 Interactive Mode")
    print("=" * 50)
    
    # Get OpenAI key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter OpenAI API key: ").strip()
    
    # Initialize
    manager = STTManager(
        stt_mode="api",
        openai_api_key=api_key,
        verbose=True
    )
    
    while True:
        print("\n" + "=" * 50)
        audio_path = input("Enter audio file path (or 'quit'): ").strip()
        
        if audio_path.lower() in ['quit', 'exit', 'q']:
            break
        
        if not Path(audio_path).exists():
            print(f"❌ File not found: {audio_path}")
            continue
        
        # Get optional metadata
        city = input("City (optional, press Enter to skip): ").strip() or None
        customer_type = input("Customer type (optional): ").strip() or None
        
        metadata = {}
        if city:
            metadata['city'] = city
        if customer_type:
            metadata['customer_type'] = customer_type
        
        # Process
        result = manager.process_audio(audio_path, metadata)
        
        # Show results
        if result.get('status') == 'success':
            print("\n📝 Transcript:")
            print("-" * 40)
            print(result['transcript']['text'][:500] + "..." if len(result['transcript']['text']) > 500 else result['transcript']['text'])
            
            print("\n📊 Insights:")
            print("-" * 40)
            insights = result.get('insights', {})
            print(f"Category: {insights.get('primary_category')}")
            print(f"Undertone: {insights.get('seller_undertone')}")
            print(f"Churn Risk: {insights.get('churn_risk_assessment', {}).get('risk_level')}")
            
            if insights.get('top_5_talking_points'):
                print("\nTalking Points:")
                for i, point in enumerate(insights['top_5_talking_points'][:5], 1):
                    print(f"  {i}. {point}")
    
    print("\n👋 Goodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="IndiaMART Audio Pipeline - Audio to Insights"
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        choices=['single', 'batch', 'interactive'],
        default='interactive',
        help='Processing mode'
    )
    
    # Audio input
    parser.add_argument(
        '--audio',
        type=str,
        help='Path to single audio file'
    )
    
    parser.add_argument(
        '--folder',
        type=str,
        help='Path to folder with audio files'
    )
    
    # STT settings
    parser.add_argument(
        '--stt-mode',
        choices=['api', 'local'],
        default='api',
        help='Whisper mode: api (OpenAI) or local'
    )
    
    parser.add_argument(
        '--openai-key',
        type=str,
        help='OpenAI API key (or set OPENAI_API_KEY env var)'
    )
    
    # Metadata
    parser.add_argument('--city', type=str, help='City name')
    parser.add_argument('--customer-type', type=str, help='Customer type (CATALOG/STAR)')
    parser.add_argument('--call-direction', type=str, help='Call direction')
    
    # Output
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path'
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Determine mode
    if args.audio:
        process_single_audio(args)
    elif args.folder:
        process_folder(args)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()

