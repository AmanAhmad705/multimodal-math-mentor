"""
Audio Parser: Speech-to-text for audio problems
"""

import logging
from pathlib import Path
from typing import Tuple
import whisper

from src.config import settings

logger = logging.getLogger(__name__)
def normalize_spoken_math(text: str) -> str:
    """
    Normalize common spoken math phrases into symbolic form.
    This reduces ambiguity before LLM parsing.
    """
    replacements = {
        "is equal to": "=",
        "equals": "=",
        "equal to": "=",
        "plus": "+",
        "minus": "-",
        "times": "*",
        "multiplied by": "*",
        "into": "*",
        "divided by": "/",
        "over": "/",
        "square": "^2",
        "squared": "^2",
    }

    normalized = text.lower()
    for phrase, symbol in replacements.items():
        normalized = normalized.replace(phrase, symbol)

    return normalized.strip()


class AudioParser:
    """Handles audio input and ASR"""
    
    def __init__(self):
        self.model = whisper.load_model("base")
        self.confidence_threshold = settings.asr_confidence_threshold
    
    def parse_audio(self, audio_path: str) -> Tuple[str, float, bool]:
        """
        Convert audio to text using Whisper
        
        Args:
            audio_path: Path to audio file (MP3, WAV, M4A)
            
        Returns:
            (transcript, confidence_score, needs_hitl)
        """
        try:
            logger.info(f"Parsing audio: {audio_path}")
            
            # Validate file
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Transcribe
            result = self.model.transcribe(audio_path, language="en")
            
            raw_transcript = result.get("text", "").strip()
            normalized_transcript = normalize_spoken_math(raw_transcript)

            
            if not normalized_transcript:
                logger.warning("No speech detected in audio")
                return "", 0.0, True
            
            # Calculate confidence from segment-level probabilities
            segments = result.get("segments", [])
            if segments:
                # Whisper provides 'no_speech_prob' for each segment
                # Confidence = 1 - average no_speech_prob
                no_speech_probs = [seg.get("no_speech_prob", 0.0) for seg in segments]
                avg_no_speech_prob = sum(no_speech_probs) / len(no_speech_probs)
                confidence = 1.0 - avg_no_speech_prob
            else:
                # Fallback: assume moderate confidence if no metadata
                confidence = 0.85
            
            # HITL trigger if confidence is low
            needs_hitl = confidence < self.confidence_threshold
            
            logger.info(
                f"ASR raw: '{raw_transcript[:50]}...' | "
                f"Normalized: '{normalized_transcript[:50]}...' | "
                f"Confidence: {confidence:.2%} | "
                f"Needs HITL: {needs_hitl}"
            )

            
            return normalized_transcript, confidence, needs_hitl
            
        except Exception as e:
            logger.error(f"Audio parsing failed: {e}")
            return "", 0.0, True
    
    def validate_audio(self, audio_path: str) -> bool:
        """Validate audio format and size"""
        try:
            path = Path(audio_path)
            
            if not path.exists():
                return False
            
            if path.suffix.lower() not in ['.mp3', '.wav', '.m4a', '.flac', '.ogg']:
                return False
            
            if path.stat().st_size > 100 * 1024 * 1024:  # 100MB limit
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            return False
