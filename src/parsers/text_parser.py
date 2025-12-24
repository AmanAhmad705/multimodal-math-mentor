"""
Text Parser: Cleanup for typed text input
"""

import logging
import re
from typing import Tuple

logger = logging.getLogger(__name__)

class TextParser:
    """Handles text input cleanup and normalization"""
    
    COMMON_REPLACEMENTS = {
        "sqaure": "square",
        "sqrt": "√",
        "raised to": "^",
        "power of": "^",
        "pi": "π",
        "degree": "°",
        "theta": "θ",
        "alpha": "α",
        "beta": "β",
        "gamma": "γ",
        "infinity": "∞",
        "perpendicular": "⊥",
        "parallel": "∥",
    }
    
    @staticmethod
    def parse_text(raw_text: str) -> Tuple[str, float]:
        """
        Clean and normalize typed text
        
        Args:
            raw_text: Raw typed input
            
        Returns:
            (cleaned_text, confidence_score)
        """
        try:
            logger.info(f"Parsing text input: {raw_text[:50]}...")
            
            cleaned = raw_text.strip()
            
            # Fix common OCR/typos
            for typo, correction in TextParser.COMMON_REPLACEMENTS.items():
                cleaned = re.sub(
                    rf"\b{typo}\b",
                    correction,
                    cleaned,
                    flags=re.IGNORECASE
                )
            
            # Remove extra spaces
            cleaned = re.sub(r"\s+", " ", cleaned)
            
            # Normalize operators
            cleaned = cleaned.replace("  +  ", " + ")
            cleaned = cleaned.replace("  -  ", " - ")
            cleaned = cleaned.replace("  *  ", " × ")
            cleaned = cleaned.replace("  /  ", " ÷ ")
            
            confidence = 0.95  # Typed text is usually clear
            
            logger.info(f"Text cleaned. Confidence: {confidence}")
            return cleaned, confidence
            
        except Exception as e:
            logger.error(f"Text parsing failed: {e}")
            return raw_text, 0.7
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect if text contains mathematical notation"""
        math_indicators = [
            "√", "^", "×", "÷", "∞", "°", "π", "θ", "∑", "∫",
            "±", "≤", "≥", "≠", "∝", "≈",
        ]
        
        count = sum(1 for ind in math_indicators if ind in text)
        return "mixed" if count > 0 else "english"
    
    @staticmethod
    def validate_math_expression(text: str) -> bool:
        """Basic validation of math expression"""
        # Check for balanced parentheses/brackets
        open_parens = text.count("(") + text.count("[") + text.count("{")
        close_parens = text.count(")") + text.count("]") + text.count("}")
        
        if open_parens != close_parens:
            logger.warning("Unbalanced parentheses detected")
            return False
        
        return True
