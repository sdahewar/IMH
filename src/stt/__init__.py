"""
Speech-to-Text (STT) Module
Whisper-based transcription for Hindi/Hinglish audio
"""

from .whisper_stt import WhisperSTT, STTManager

__all__ = ['WhisperSTT', 'STTManager']

