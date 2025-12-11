"""
Vosk STT - Speech-to-Text using Vosk (Offline, Free)
Hindi model: vosk-model-small-hi-0.22

Features:
- Offline transcription (no API needed)
- Hindi language support
- Fast processing
- Word-level timestamps
"""

import os
import wave
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from vosk import Model, KaldiRecognizer

# =============================================================================
# VOSK STT CLASS
# =============================================================================

class VoskSTT:
    """
    Vosk-based Speech-to-Text for Hindi
    
    Model: vosk-model-hi-0.22 (larger, more accurate)
    """
    
    def __init__(
        self,
        model_path: str = "vosk-model-hi-0.22",
        verbose: bool = True
    ):
        self.model_path = model_path
        self.verbose = verbose
        self.model = None
        self.ffmpeg_path = None
        
        self._init_ffmpeg()
        self._load_model()
    
    def _log(self, message: str):
        if self.verbose:
            print(message)
    
    def _init_ffmpeg(self):
        """Find ffmpeg for audio conversion"""
        try:
            import imageio_ffmpeg
            self.ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            self._log(f"✅ ffmpeg found")
        except:
            self.ffmpeg_path = "ffmpeg"
            self._log("⚠️ Using system ffmpeg")
    
    def _load_model(self):
        """Load Vosk model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Vosk model not found at: {self.model_path}")
        
        self._log(f"⏳ Loading Vosk model: {self.model_path}")
        self.model = Model(self.model_path)
        self._log(f"✅ Vosk model loaded")
    
    def _convert_to_wav(self, audio_path: str) -> str:
        """Convert audio to WAV format (16kHz mono)"""
        audio_path = Path(audio_path)
        wav_path = audio_path.with_suffix('.wav')
        
        if wav_path.exists():
            return str(wav_path)
        
        self._log(f"   Converting to WAV...")
        
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", str(audio_path),
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            str(wav_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr}")
        
        return str(wav_path)
    
    def transcribe(
        self,
        audio_path: str,
        include_words: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file
        
        Args:
            audio_path: Path to audio file (.mp3, .wav, etc.)
            include_words: Include word-level timestamps
        
        Returns:
            Dict with transcript, duration, and metadata
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        self._log(f"\n🎤 Transcribing: {audio_path.name}")
        
        # Convert to WAV if needed
        if audio_path.suffix.lower() != '.wav':
            wav_path = self._convert_to_wav(str(audio_path))
        else:
            wav_path = str(audio_path)
        
        # Open WAV file
        wf = wave.open(wav_path, "rb")
        duration = wf.getnframes() / wf.getframerate()
        
        self._log(f"   Duration: {duration:.1f} seconds")
        
        # Create recognizer
        rec = KaldiRecognizer(self.model, wf.getframerate())
        rec.SetWords(include_words)
        
        # Process audio
        transcript_parts = []
        words_list = []
        
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                transcript_parts.append(result.get("text", ""))
                if include_words and "result" in result:
                    words_list.extend(result["result"])
        
        # Get final result
        final_result = json.loads(rec.FinalResult())
        transcript_parts.append(final_result.get("text", ""))
        if include_words and "result" in final_result:
            words_list.extend(final_result["result"])
        
        wf.close()
        
        transcript = " ".join(transcript_parts).strip()
        
        self._log(f"   ✅ Transcription complete ({len(transcript)} chars)")
        
        result = {
            "transcript": transcript,
            "duration": round(duration, 2),
            "file_name": audio_path.name,
            "language": "hi"
        }
        
        if include_words:
            result["words"] = words_list
        
        return result
    
    def transcribe_batch(
        self,
        audio_paths: List[str],
        output_dir: str = None
    ) -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio files
        """
        results = []
        total = len(audio_paths)
        
        self._log(f"\n📦 Batch transcription: {total} files")
        
        for i, audio_path in enumerate(audio_paths, 1):
            self._log(f"\n[{i}/{total}]")
            try:
                result = self.transcribe(audio_path)
                result['status'] = 'success'
            except Exception as e:
                self._log(f"   ❌ Error: {str(e)}")
                result = {
                    'file_name': Path(audio_path).name,
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
    Audio → Vosk STT → Transcript → NVIDIA NIM → Insights
    """
    
    def __init__(
        self,
        model_path: str = "vosk-model-hi-0.22",
        verbose: bool = True
    ):
        self.verbose = verbose
        self.stt = VoskSTT(model_path=model_path, verbose=verbose)
        
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
        """
        metadata = metadata or {}
        
        self._log(f"\n{'='*60}")
        self._log(f"🎯 Processing: {Path(audio_path).name}")
        self._log(f"{'='*60}")
        
        # Step 1: Transcribe with Vosk
        self._log("\n📍 Step 1: Speech-to-Text (Vosk)")
        transcript_result = self.stt.transcribe(audio_path)
        
        if not transcript_result.get('transcript'):
            return {
                'status': 'error',
                'error': 'Empty transcript',
                'audio_path': audio_path
            }
        
        # Step 2: Analyze with LLM
        self._log("\n📍 Step 2: LLM Analysis (NVIDIA NIM)")
        
        metadata['duration'] = transcript_result.get('duration')
        
        insights = self.insights_agent.analyze_transcript(
            transcript=transcript_result['transcript'],
            metadata=metadata
        )
        
        result = {
            'status': 'success',
            'audio_file': Path(audio_path).name,
            'transcript': transcript_result['transcript'],
            'duration': transcript_result.get('duration'),
            'insights': insights,
            'metadata': metadata
        }
        
        self._log(f"\n✅ Complete! Category: {insights.get('primary_category', 'N/A')}")
        
        return result
    
    def process_folder(
        self,
        folder_path: str,
        extensions: List[str] = None,
        output_file: str = None
    ) -> List[Dict[str, Any]]:
        """
        Process all audio files in a folder
        """
        extensions = extensions or ['.mp3', '.wav', '.m4a', '.ogg']
        folder = Path(folder_path)
        
        audio_files = []
        for ext in extensions:
            audio_files.extend(folder.glob(f"*{ext}"))
        
        if not audio_files:
            self._log(f"⚠️ No audio files found")
            return []
        
        results = []
        for audio_path in audio_files:
            result = self.process_audio(str(audio_path))
            results.append(result)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self._log(f"💾 Saved to: {output_file}")
        
        return results

