"""
Application Configuration & Constants
Production-grade settings management
"""

import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from typing import Literal, ClassVar, List

load_dotenv()

class Settings(BaseSettings):
    """Central configuration using Pydantic"""
    
    # Environment
    environment: Literal["development", "production"] = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # API Keys
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent
    memory_db_path: str = os.getenv("MEMORY_DB_PATH", str(base_dir / "data" / "math_mentor.db"))
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", str(base_dir / "data" / "faiss_index"))
    knowledge_base_path: str = os.getenv("KNOWLEDGE_BASE_PATH", str(base_dir / "data" / "knowledge_base"))
    
    # OCR
    ocr_engine: Literal["paddleocr", "tesseract"] = os.getenv("OCR_ENGINE", "paddleocr")
    ocr_confidence_threshold: float = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.75"))
    
    # ASR
    asr_confidence_threshold: float = float(os.getenv("ASR_CONFIDENCE_THRESHOLD", "0.80"))
    
    # RAG
    rag_chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "500"))
    rag_chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))
    
    # HITL
    enable_hitl: bool = os.getenv("ENABLE_HITL", "true").lower() == "true"
    auto_reject_confidence_threshold: float = float(os.getenv("AUTO_REJECT_CONFIDENCE_THRESHOLD", "0.5"))
    
    # Math Scope
    MATH_TOPICS: ClassVar[List[str]] = [
    "algebra",
    "probability",
    "calculus",
    "linear_algebra",
    "trigonometry",
    "coordinate_geometry",
    ]

    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()

# Agent Configuration
AGENT_CONFIG = {
    "parser": {
        "model": settings.groq_model,
        "temperature": 0.3,
        "max_tokens": 500,
    },
    "router": {
        "model": settings.groq_model,
        "temperature": 0.2,
        "max_tokens": 200,
    },
    "solver": {
        "model": settings.groq_model,
        "temperature": 0.4,
        "max_tokens": 2000,
    },
    "verifier": {
        "model": settings.groq_model,
        "temperature": 0.2,
        "max_tokens": 800,
    },
    "explainer": {
        "model": settings.groq_model,
        "temperature": 0.5,
        "max_tokens": 2000,
    },
}

# Math Problem Template
PROBLEM_SCHEMA = {
    "problem_text": str,
    "topic": str,
    "difficulty": Literal["easy", "medium", "hard"],
    "variables": list,
    "constraints": list,
    "required_formulas": list,
    "needs_clarification": bool,
    "clarification_questions": list,
}
