"""
Memory Store: Persistent storage for learning and HITL
"""

import sqlite3
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.config import settings
from src.memory.schema import init_database

logger = logging.getLogger(__name__)

class MemoryStore:
    """Manages persistent memory for learning and HITL"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.memory_db_path
        init_database(self.db_path)
    
    def _get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    # Session Management
    def create_session(
        self,
        session_id: Optional[str] = None,
        user_id: str = "default",
    ) -> str:
        """Create new session (single source of truth)"""

        session_id = session_id or str(uuid.uuid4())

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (session_id, user_id) VALUES (?, ?)",
            (session_id, user_id)
        )
        conn.commit()
        conn.close()

        logger.info(f"Created session: {session_id}")
        return session_id

    
    # Raw Input Logging
    def log_raw_input(
        self,
        session_id: str,
        input_type: str,
        raw_content: str,
        confidence: float = 0.0,
        ocr_transcript: str = None,
        asr_transcript: str = None,
    ) -> str:
        """Log raw input (image, audio, text)"""
        input_id = str(uuid.uuid4())
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO raw_inputs 
            (input_id, session_id, input_type, raw_content, confidence, ocr_transcript, asr_transcript)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (input_id, session_id, input_type, raw_content, confidence, ocr_transcript, asr_transcript)
        )
        conn.commit()
        conn.close()
        return input_id
    
    # Parsed Problem Logging
    def log_parsed_problem(
        self,
        session_id: str,
        input_id: str,
        problem_text: str,
        topic: str,
        difficulty: str,
        variables: List[str],
        constraints: List[str],
        needs_clarification: bool,
        clarification_questions: List[str] = None,
        confidence: float = 0.0,
    ) -> str:
        """Log parsed problem"""
        problem_id = str(uuid.uuid4())
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO parsed_problems
            (problem_id, session_id, input_id, problem_text, topic, difficulty, variables, 
             constraints, needs_clarification, clarification_questions, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                problem_id, session_id, input_id, problem_text, topic, difficulty,
                json.dumps(variables), json.dumps(constraints), needs_clarification,
                json.dumps(clarification_questions or []), confidence
            )
        )
        conn.commit()
        conn.close()
        return problem_id
    
    # Solution Logging
    def log_solution(
        self,
        problem_id: str,
        problem_type: str,
        solving_strategy: str,
        solution_steps: List[str],
        final_answer: str,
        key_formulas: List[str],
        confidence: float,
    ) -> str:
        """Log solution"""
        solution_id = str(uuid.uuid4())
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO solutions
            (solution_id, problem_id, problem_type, solving_strategy, solution_steps, 
             final_answer, key_formulas, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                solution_id, problem_id, problem_type, solving_strategy,
                json.dumps(solution_steps), final_answer, json.dumps(key_formulas), confidence
            )
        )
        conn.commit()
        conn.close()
        return solution_id
    
    # Verification Logging
    def log_verification(
        self,
        solution_id: str,
        is_correct: bool,
        confidence: float,
        issues_found: List[str],
        edge_cases_checked: List[str],
        units_verified: bool,
        requires_hitl: bool,
    ) -> str:
        """Log verification result"""
        verification_id = str(uuid.uuid4())
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO verifications
            (verification_id, solution_id, is_correct, confidence, issues_found, 
             edge_cases_checked, units_verified, requires_hitl)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                verification_id, solution_id, is_correct, confidence,
                json.dumps(issues_found), json.dumps(edge_cases_checked),
                units_verified, requires_hitl
            )
        )
        conn.commit()
        conn.close()
        return verification_id
    
    # HITL Logging
    def create_hitl_request(
        self,
        problem_id: str,
        solution_id: str,
        verification_id: str,
        trigger_reason: str,
    ) -> str:
        """Create HITL request"""
        hitl_id = str(uuid.uuid4())
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO hitl_requests
            (hitl_id, problem_id, solution_id, verification_id, hitl_trigger_reason)
            VALUES (?, ?, ?, ?, ?)""",
            (hitl_id, problem_id, solution_id, verification_id, trigger_reason)
        )
        conn.commit()
        conn.close()
        logger.info(f"Created HITL request: {hitl_id}, Reason: {trigger_reason}")
        return hitl_id
    
    def update_hitl_request(
        self,
        hitl_id: str,
        human_feedback: str,
        human_decision: str,
        human_correction: str = None,
    ):
        """Update HITL request with human feedback"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """UPDATE hitl_requests
            SET human_feedback = ?, human_decision = ?, human_correction = ?, updated_at = CURRENT_TIMESTAMP
            WHERE hitl_id = ?""",
            (human_feedback, human_decision, human_correction, hitl_id)
        )
        conn.commit()
        conn.close()
        logger.info(f"Updated HITL request: {hitl_id}, Decision: {human_decision}")
    
    # Feedback Logging
    def log_feedback(
        self,
        solution_id: str,
        user_rating: str,
        user_comment: str = "",
        helpful_rating: int = 3,
    ) -> str:
        """Log user feedback on solution"""
        feedback_id = str(uuid.uuid4())
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO feedback
            (feedback_id, solution_id, user_rating, user_comment, helpful_rating)
            VALUES (?, ?, ?, ?, ?)""",
            (feedback_id, solution_id, user_rating, user_comment, helpful_rating)
        )
        conn.commit()
        conn.close()
        return feedback_id
    
    # Query for Similar Problems
    def find_similar_problems(self, topic: str, difficulty: str, limit: int = 5) -> List[Dict]:
        """Find similar solved problems for pattern reuse"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT pp.problem_text, pp.topic, s.final_answer, f.user_rating
            FROM parsed_problems pp
            JOIN solutions s ON pp.problem_id = s.problem_id
            LEFT JOIN feedback f ON s.solution_id = f.solution_id
            WHERE pp.topic = ? AND pp.difficulty = ?
            ORDER BY s.confidence DESC
            LIMIT ?""",
            (topic, difficulty, limit)
        )
        results = cursor.fetchall()
        conn.close()
        
        similar_problems = [
            {
                "problem_text": r[0],
                "topic": r[1],
                "answer": r[2],
                "feedback": r[3],
            }
            for r in results
        ]
        
        logger.info(f"Found {len(similar_problems)} similar problems")
        return similar_problems
    
    # Query Statistics
    def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT COUNT(*) FROM parsed_problems WHERE session_id = ?",
            (session_id,)
        )
        problem_count = cursor.fetchone()[0]
        
        cursor.execute(
            "SELECT COUNT(*) FROM solutions WHERE problem_id IN (SELECT problem_id FROM parsed_problems WHERE session_id = ?)",
            (session_id,)
        )
        solution_count = cursor.fetchone()[0]
        
        cursor.execute(
            "SELECT COUNT(*) FROM hitl_requests WHERE problem_id IN (SELECT problem_id FROM parsed_problems WHERE session_id = ?)",
            (session_id,)
        )
        hitl_count = cursor.fetchone()[0]
        
        cursor.execute(
            "SELECT AVG(confidence) FROM solutions WHERE problem_id IN (SELECT problem_id FROM parsed_problems WHERE session_id = ?)",
            (session_id,)
        )
        avg_confidence = cursor.fetchone()[0] or 0.0
        
        conn.close()
        
        return {
            "session_id": session_id,
            "problems_solved": problem_count,
            "solutions_generated": solution_count,
            "hitl_triggers": hitl_count,
            "average_confidence": round(avg_confidence, 2),
        }
