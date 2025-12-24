"""
Image Parser: OCR for math problem images
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import easyocr

from src.config import settings

logger = logging.getLogger(__name__)

class ImageParser:
    """Handles image input and OCR"""
    
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.confidence_threshold = settings.ocr_confidence_threshold
    
    def parse_image(self, image_path: str) -> Tuple[str, float, bool]:
        """
        Extract text from image using OCR
        
        Args:
            image_path: Path to image file (JPG, PNG)
            
        Returns:
            (extracted_text, confidence_score, needs_hitl)
        """
        try:
            logger.info(f"Parsing image: {image_path}")
            
            # Validate file
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Read image
            image = Image.open(image_path)
            
            # OCR
            results = self.reader.readtext(image_path, detail=1)
            
            if not results:
                logger.warning("No text detected in image")
                return "", 0.0, True
            
            # Extract text and confidence
            extracted_lines = []
            confidences = []
            
            for detection in results:
                text = detection[1]
                confidence = detection[2]
                extracted_lines.append(text)
                confidences.append(confidence)
            
            extracted_text = "\n".join(extracted_lines)
            avg_confidence = np.mean(confidences)
            
            # HITL trigger
            needs_hitl = avg_confidence < self.confidence_threshold
            
            logger.info(
                f"OCR extracted {len(extracted_lines)} lines. "
                f"Confidence: {avg_confidence:.2%}. "
                f"Needs HITL: {needs_hitl}"
            )
            
            return extracted_text, avg_confidence, needs_hitl
            
        except Exception as e:
            logger.error(f"Image parsing failed: {e}")
            return "", 0.0, True
    
    def validate_image(self, image_path: str) -> bool:
        """Validate image format and size"""
        try:
            path = Path(image_path)
            
            if not path.exists():
                return False
            
            if path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                return False
            
            if path.stat().st_size > 50 * 1024 * 1024:  # 50MB limit
                return False
            
            Image.open(image_path).verify()
            return True
            
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False
