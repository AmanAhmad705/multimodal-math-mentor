"""
Memory Layer Schema: Database tables for HITL and learning
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT,
    status TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS raw_inputs (
    input_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    input_type TEXT,  -- 'image', 'audio', 'text'
    raw_content TEXT,  -- file path or text
    confidence REAL,
    ocr_transcript TEXT,  -- for image
    asr_transcript TEXT,  -- for audio
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS parsed_problems (
    problem_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    input_id TEXT NOT NULL,
    problem_text TEXT,
    topic TEXT,
    difficulty TEXT,
    variables TEXT,  -- JSON
    constraints TEXT,  -- JSON
    needs_clarification BOOLEAN,
    clarification_questions TEXT,  -- JSON
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id),
    FOREIGN KEY(input_id) REFERENCES raw_inputs(input_id)
);

CREATE TABLE IF NOT EXISTS rag_retrievals (
    retrieval_id TEXT PRIMARY KEY,
    problem_id TEXT NOT NULL,
    query TEXT,
    retrieved_docs TEXT,  -- JSON with doc IDs and content
    top_k INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(problem_id) REFERENCES parsed_problems(problem_id)
);

CREATE TABLE IF NOT EXISTS solutions (
    solution_id TEXT PRIMARY KEY,
    problem_id TEXT NOT NULL,
    problem_type TEXT,
    solving_strategy TEXT,
    solution_steps TEXT,  -- JSON
    final_answer TEXT,
    key_formulas TEXT,  -- JSON
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(problem_id) REFERENCES parsed_problems(problem_id)
);

CREATE TABLE IF NOT EXISTS verifications (
    verification_id TEXT PRIMARY KEY,
    solution_id TEXT NOT NULL,
    is_correct BOOLEAN,
    confidence REAL,
    issues_found TEXT,  -- JSON
    edge_cases_checked TEXT,  -- JSON
    units_verified BOOLEAN,
    requires_hitl BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(solution_id) REFERENCES solutions(solution_id)
);

CREATE TABLE IF NOT EXISTS hitl_requests (
    hitl_id TEXT PRIMARY KEY,
    problem_id TEXT,
    solution_id TEXT,
    verification_id TEXT,
    hitl_trigger_reason TEXT,  -- 'low_confidence', 'ambiguity', 'low_ocr', 'low_asr'
    human_feedback TEXT,
    human_correction TEXT,
    human_decision TEXT,  -- 'approved', 'rejected', 'corrected'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    FOREIGN KEY(problem_id) REFERENCES parsed_problems(problem_id),
    FOREIGN KEY(solution_id) REFERENCES solutions(solution_id),
    FOREIGN KEY(verification_id) REFERENCES verifications(verification_id)
);

CREATE TABLE IF NOT EXISTS explanations (
    explanation_id TEXT PRIMARY KEY,
    solution_id TEXT NOT NULL,
    conceptual_overview TEXT,
    step_by_step TEXT,  -- JSON
    key_insights TEXT,  -- JSON
    common_mistakes TEXT,  -- JSON
    related_concepts TEXT,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(solution_id) REFERENCES solutions(solution_id)
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id TEXT PRIMARY KEY,
    solution_id TEXT NOT NULL,
    user_rating TEXT,  -- 'correct', 'incorrect', 'partial', 'needs_revision'
    user_comment TEXT,
    helpful_rating INT,  -- 1-5
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(solution_id) REFERENCES solutions(solution_id)
);

CREATE INDEX IF NOT EXISTS idx_session_id ON parsed_problems(session_id);
CREATE INDEX IF NOT EXISTS idx_problem_topic ON parsed_problems(topic);
CREATE INDEX IF NOT EXISTS idx_solution_confidence ON solutions(confidence);
CREATE INDEX IF NOT EXISTS idx_verification_status ON verifications(is_correct);
CREATE INDEX IF NOT EXISTS idx_hitl_trigger ON hitl_requests(hitl_trigger_reason);
"""

def init_database(db_path: str = None):
    """Initialize SQLite database with schema"""
    if db_path is None:
        db_path = settings.memory_db_path
    
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        for statement in CREATE_TABLES_SQL.split(';'):
            if statement.strip():
                cursor.execute(statement)
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {db_path}")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
