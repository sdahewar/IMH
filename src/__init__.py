"""
IndiaMART Insights Engine
Voice Call Analytics Pipeline

Modules:
- stt: Speech-to-Text (Whisper)
- agents: LLM-based analysis agents
- classifiers: Transcript classification
- aggregators: Insights aggregation
- utils: Helper utilities
"""

__version__ = "2.0.0"
__author__ = "Data Voice Hackathon Team"

# Import main components for easy access
from .agents import InsightsAgent, AggregationAgent
from .classifiers import NvidiaClassifier
from .stt import WhisperSTT, STTManager
