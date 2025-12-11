"""
Speech-to-Text (STT) Module
Vosk-based transcription for Hindi audio
"""

from .vosk_stt import VoskSTT, STTManager

__all__ = ['VoskSTT', 'STTManager']

