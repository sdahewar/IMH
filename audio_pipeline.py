"""
Audio Pipeline - Complete Audio-to-Insights Pipeline
IndiaMART Insights Engine

Pipeline: Audio → Vosk STT → Transcript → NVIDIA NIM → Insights

Usage:
    # Single file
    python audio_pipeline.py --audio audio/testaud.mp3
    
    # Folder of audio files
    python audio_pipeline.py --folder ./recordings
    
    # Interactive mode
    python audio_pipeline.py
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         IndiaMART Insights Engine - Audio Pipeline               ║
║              Audio → Vosk STT → NVIDIA NIM → Insights            ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def process_single_audio(args):
    """Process a single audio file"""
    from src.stt import STTManager
    
    print(f"\n🎤 Processing: {args.audio}")
    
    metadata = {}
    if args.city:
        metadata['city'] = args.city
    if args.customer_type:
        metadata['customer_type'] = args.customer_type
    
    manager = STTManager(
        model_path=args.model_path,
        verbose=True
    )
    
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
    
    return result


def process_folder(args):
    """Process all audio files in a folder"""
    from src.stt import STTManager
    
    print(f"\n📁 Processing folder: {args.folder}")
    
    manager = STTManager(
        model_path=args.model_path,
        verbose=True
    )
    
    output_file = args.output or f"output/batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    results = manager.process_folder(
        folder_path=args.folder,
        output_file=output_file
    )
    
    return results


def interactive_mode(args):
    """Interactive mode"""
    from src.stt import STTManager
    
    print("\n🎮 Interactive Mode")
    print("="*50)
    
    manager = STTManager(
        model_path=args.model_path,
        verbose=True
    )
    
    while True:
        print("\n" + "="*50)
        audio_path = input("Enter audio file path (or 'quit'): ").strip()
        
        if audio_path.lower() in ['quit', 'exit', 'q']:
            break
        
        if not Path(audio_path).exists():
            print(f"❌ File not found: {audio_path}")
            continue
        
        city = input("City (optional): ").strip() or None
        customer_type = input("Customer type (optional): ").strip() or None
        
        metadata = {}
        if city:
            metadata['city'] = city
        if customer_type:
            metadata['customer_type'] = customer_type
        
        result = manager.process_audio(audio_path, metadata)
        
        if result.get('status') == 'success':
            print("\n📝 Transcript:")
            print("-"*40)
            transcript = result.get('transcript', '')
            print(transcript[:1000] + "..." if len(transcript) > 1000 else transcript)
            
            print("\n📊 Insights:")
            print("-"*40)
            insights = result.get('insights', {})
            print(f"Category: {insights.get('primary_category')}")
            print(f"Undertone: {insights.get('seller_undertone')}")
            print(f"Churn Risk: {insights.get('churn_risk_assessment', {}).get('risk_level')}")
    
    print("\n👋 Goodbye!")


def main():
    parser = argparse.ArgumentParser(description="IndiaMART Audio Pipeline")
    
    parser.add_argument('--audio', type=str, help='Path to single audio file')
    parser.add_argument('--folder', type=str, help='Path to folder with audio files')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--city', type=str, help='City name')
    parser.add_argument('--customer-type', type=str, help='Customer type')
    parser.add_argument(
        '--model-path',
        type=str,
        default='vosk-model-small-hi-0.22',
        help='Path to Vosk model'
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.audio:
        process_single_audio(args)
    elif args.folder:
        process_folder(args)
    else:
        interactive_mode(args)


if __name__ == "__main__":
    main()

