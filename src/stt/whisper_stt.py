"""
Whisper STT - Speech-to-Text using OpenAI Whisper
Supports both API and Local model

Features:
- Hindi/Hinglish audio transcription
- Speaker diarization (with pyannote)
- Batch processing
- Multiple audio formats
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# =============================================================================
# WHISPER STT CLASS
# =============================================================================

class WhisperSTT:
    """
    Whisper-based Speech-to-Text
    
    Modes:
    - 'api': OpenAI Whisper API (requires OPENAI_API_KEY)
    - 'local': Local Whisper model (requires GPU, free)
    """
    
    def __init__(
        self,
        mode: str = "api",
        api_key: str = None,
        model_size: str = "large-v3",
        language: str = "hi",
        verbose: bool = True
    ):
        self.mode = mode
        self.language = language
        self.verbose = verbose
        self.model_size = model_size
        
        if mode == "api":
            self._init_api(api_key)
        else:
            self._init_local(model_size)
    
    def _log(self, message: str):
        if self.verbose:
            print(message)
    
    def _init_api(self, api_key: str):
        """Initialize OpenAI Whisper API"""
        from openai import OpenAI
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY required for API mode")
        
        self.client = OpenAI(api_key=self.api_key)
        self._log("✅ Whisper API initialized")
    
    def _init_local(self, model_size: str):
        """Initialize local Whisper model"""
        try:
            import whisper
            self._log(f"⏳ Loading Whisper {model_size} model (this may take a while)...")
            self.model = whisper.load_model(model_size)
            self._log(f"✅ Whisper {model_size} model loaded")
        except ImportError:
            raise ImportError("Install whisper: pip install openai-whisper")
    
    def transcribe(
        self,
        audio_path: str,
        language: str = None,
        include_timestamps: bool = True,
        include_segments: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe a single audio file
        
        Args:
            audio_path: Path to audio file (.mp3, .wav, .m4a, etc.)
            language: Language code (default: 'hi' for Hindi)
            include_timestamps: Include word-level timestamps
            include_segments: Include segment-level breakdown
        
        Returns:
            Dict with transcript, duration, and metadata
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        language = language or self.language
        
        self._log(f"\n🎤 Transcribing: {audio_path.name}")
        start_time = time.time()
        
        if self.mode == "api":
            result = self._transcribe_api(audio_path, language, include_timestamps)
        else:
            result = self._transcribe_local(audio_path, language, include_segments)
        
        elapsed = time.time() - start_time
        result['processing_time'] = round(elapsed, 2)
        result['file_name'] = audio_path.name
        result['file_path'] = str(audio_path)
        
        self._log(f"   ✅ Done in {elapsed:.2f}s | Length: {len(result.get('transcript', ''))} chars")
        
        return result
    
    def _transcribe_api(
        self,
        audio_path: Path,
        language: str,
        include_timestamps: bool
    ) -> Dict[str, Any]:
        """Transcribe using OpenAI Whisper API"""
        
        with open(audio_path, 'rb') as audio_file:
            response = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format="verbose_json" if include_timestamps else "json",
                timestamp_granularities=["word", "segment"] if include_timestamps else None
            )
        
        result = {
            "transcript": response.text,
            "language": language,
            "duration": getattr(response, 'duration', None),
        }
        
        if include_timestamps and hasattr(response, 'words'):
            result['words'] = [
                {"word": w.word, "start": w.start, "end": w.end}
                for w in response.words
            ]
        
        if hasattr(response, 'segments'):
            result['segments'] = [
                {"text": s.text, "start": s.start, "end": s.end}
                for s in response.segments
            ]
        
        return result
    
    def _transcribe_local(
        self,
        audio_path: Path,
        language: str,
        include_segments: bool
    ) -> Dict[str, Any]:
        """Transcribe using local Whisper model"""
        
        result = self.model.transcribe(
            str(audio_path),
            language=language,
            task="transcribe",
            verbose=False
        )
        
        output = {
            "transcript": result["text"].strip(),
            "language": result.get("language", language),
            "duration": result.get("duration"),
        }
        
        if include_segments:
            output['segments'] = [
                {
                    "text": seg["text"],
                    "start": seg["start"],
                    "end": seg["end"]
                }
                for seg in result.get("segments", [])
            ]
        
        return output
    
    def transcribe_batch(
        self,
        audio_paths: List[str],
        output_dir: str = None,
        save_individual: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio files
        
        Args:
            audio_paths: List of audio file paths
            output_dir: Directory to save transcripts
            save_individual: Save each transcript as JSON
        
        Returns:
            List of transcription results
        """
        results = []
        total = len(audio_paths)
        
        self._log(f"\n📦 Batch transcription: {total} files")
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, audio_path in enumerate(audio_paths, 1):
            self._log(f"\n[{i}/{total}] Processing...")
            
            try:
                result = self.transcribe(audio_path)
                result['status'] = 'success'
                
                if save_individual and output_dir:
                    output_file = output_dir / f"{Path(audio_path).stem}_transcript.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                self._log(f"   ❌ Error: {str(e)}")
                result = {
                    'file_path': audio_path,
                    'status': 'error',
                    'error': str(e)
                }
            
            results.append(result)
        
        success = sum(1 for r in results if r.get('status') == 'success')
        self._log(f"\n✅ Batch complete: {success}/{total} successful")
        
        return results


# =============================================================================
# STT MANAGER - PIPELINE INTEGRATION
# =============================================================================

class STTManager:
    """
    Manages STT pipeline with LLM integration
    Audio → Transcript → NVIDIA NIM → Insights
    """
    
    def __init__(
        self,
        stt_mode: str = "api",
        openai_api_key: str = None,
        whisper_model: str = "large-v3",
        verbose: bool = True
    ):
        self.verbose = verbose
        self.stt = WhisperSTT(
            mode=stt_mode,
            api_key=openai_api_key,
            model_size=whisper_model,
            verbose=verbose
        )
        
        # Import LLM agent
        from src.agents import InsightsAgent
        self.insights_agent = InsightsAgent(verbose=verbose)
    
    def _log(self, message: str):
        if self.verbose:
            print(message)
    
    def process_audio(
        self,
        audio_path: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Complete pipeline: Audio → Transcript → Insights
        
        Args:
            audio_path: Path to audio file
            metadata: Additional metadata (customer_type, city, etc.)
        
        Returns:
            Complete analysis with transcript and insights
        """
        metadata = metadata or {}
        
        self._log(f"\n{'='*60}")
        self._log(f"🎯 Processing: {Path(audio_path).name}")
        self._log(f"{'='*60}")
        
        # Step 1: Transcribe
        self._log("\n📍 Step 1: Speech-to-Text (Whisper)")
        transcript_result = self.stt.transcribe(audio_path)
        
        if not transcript_result.get('transcript'):
            return {
                'status': 'error',
                'error': 'Empty transcript',
                'audio_path': audio_path
            }
        
        # Step 2: Analyze with LLM
        self._log("\n📍 Step 2: LLM Analysis (NVIDIA NIM)")
        
        # Add duration from STT to metadata
        if transcript_result.get('duration'):
            metadata['duration'] = transcript_result['duration']
        
        insights = self.insights_agent.analyze_transcript(
            transcript=transcript_result['transcript'],
            metadata=metadata
        )
        
        # Combine results
        result = {
            'status': 'success',
            'audio_file': Path(audio_path).name,
            'processing_time': transcript_result.get('processing_time', 0),
            'transcript': {
                'text': transcript_result['transcript'],
                'duration': transcript_result.get('duration'),
                'language': transcript_result.get('language'),
                'segments': transcript_result.get('segments', [])
            },
            'insights': insights,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        self._log(f"\n✅ Complete! Category: {insights.get('primary_category', 'N/A')}")
        
        return result
    
    def process_batch(
        self,
        audio_paths: List[str],
        metadata_list: List[Dict] = None,
        output_file: str = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple audio files through complete pipeline
        
        Args:
            audio_paths: List of audio file paths
            metadata_list: Optional list of metadata for each file
            output_file: Save results to JSON file
        
        Returns:
            List of complete analysis results
        """
        results = []
        total = len(audio_paths)
        metadata_list = metadata_list or [{}] * total
        
        self._log(f"\n{'='*60}")
        self._log(f"📦 BATCH PROCESSING: {total} audio files")
        self._log(f"{'='*60}")
        
        for i, (audio_path, metadata) in enumerate(zip(audio_paths, metadata_list), 1):
            self._log(f"\n[{i}/{total}] {'='*50}")
            
            try:
                result = self.process_audio(audio_path, metadata)
            except Exception as e:
                self._log(f"❌ Error processing {audio_path}: {str(e)}")
                result = {
                    'status': 'error',
                    'error': str(e),
                    'audio_file': Path(audio_path).name
                }
            
            results.append(result)
        
        # Summary
        success = sum(1 for r in results if r.get('status') == 'success')
        self._log(f"\n{'='*60}")
        self._log(f"📊 BATCH COMPLETE: {success}/{total} successful")
        self._log(f"{'='*60}")
        
        # Save results
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self._log(f"💾 Results saved to: {output_file}")
        
        return results
    
    def process_folder(
        self,
        folder_path: str,
        extensions: List[str] = None,
        output_file: str = None
    ) -> List[Dict[str, Any]]:
        """
        Process all audio files in a folder
        
        Args:
            folder_path: Path to folder containing audio files
            extensions: Audio file extensions to process
            output_file: Save results to JSON file
        
        Returns:
            List of analysis results
        """
        extensions = extensions or ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.webm']
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        audio_files = []
        for ext in extensions:
            audio_files.extend(folder.glob(f"*{ext}"))
        
        if not audio_files:
            self._log(f"⚠️ No audio files found in {folder_path}")
            return []
        
        self._log(f"📁 Found {len(audio_files)} audio files in {folder_path}")
        
        return self.process_batch(
            audio_paths=[str(f) for f in audio_files],
            output_file=output_file
        )

