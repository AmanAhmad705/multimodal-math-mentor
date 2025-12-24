"""
Parser Agent
-------------
Converts raw user input into a structured math problem.

Design goals:
- ZERO prompt-formatting crashes
- STRICT JSON-only LLM output
- Pydantic validation
- Hard fallback on any failure
- LangChain v0.2+ compatible
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from src.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# Internal Parsed Representation (used across pipeline)
# ============================================================================
@dataclass
class ParsedProblem:
    raw_input: str
    problem_text: str
    topic: str
    difficulty: str
    variables: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    required_formulas: List[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_questions: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    parsed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ============================================================================
# Strict Schema Expected from LLM
# ============================================================================
class ProblemStructure(BaseModel):
    problem_text: str
    topic: str = Field(
        ...,
        description="One of: algebra, probability, calculus, trigonometry, linear_algebra, coordinate_geometry",
    )
    difficulty: str = Field(..., description="easy | medium | hard")
    variables: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    required_formulas: List[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_questions: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.9)


# ============================================================================
# Parser Agent
# ============================================================================
class ParserAgent:
    def __init__(self) -> None:
        self.llm = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.groq_model,
            temperature=0.1,
            max_tokens=600,
        )

        # IMPORTANT:
        # All literal `{}` are escaped as `{{}}`
        # Only `{raw_input}` is a real variable
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a math problem parser. "
                    "You MUST output ONLY valid JSON. "
                    "NO explanations. NO markdown. NO prose.",
                ),
                (
                    "human",
                    """
Convert the following input into structured JSON.

Input:
{raw_input}

JSON schema (STRICT):
{{
  "problem_text": string,
  "topic": one of ["algebra","probability","calculus","trigonometry","linear_algebra","coordinate_geometry"],
  "difficulty": one of ["easy","medium","hard"],
  "variables": [string],
  "constraints": [string],
  "required_formulas": [string],
  "needs_clarification": boolean,
  "clarification_questions": [string],
  "confidence_score": number (0 to 1)
}}

Return ONLY JSON. Do NOT include any extra text.
""",
                ),
            ]
        )

    # ----------------------------------------------------------------------
    # Safe JSON extraction (handles hallucinated text)
    # ----------------------------------------------------------------------
    def _safe_json_load(self, text: str) -> dict:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end == -1:
                raise ValueError("No JSON object found")
            return json.loads(text[start:end])
        except Exception as exc:
            raise ValueError(f"Invalid JSON returned by model: {exc}") from exc

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------
    def parse(self, raw_input: str) -> ParsedProblem:
        logger.info("Parsing raw input: %s", raw_input)

        try:
            messages = self.prompt.format_messages(raw_input=raw_input)
            response = self.llm.invoke(messages)

            data = self._safe_json_load(response.content)
            validated = ProblemStructure(**data)
            # ------------------------------------------------------------------
            # Guardrail: reject incomplete or non-mathematical problems
            # ------------------------------------------------------------------
            problem_text_lower = validated.problem_text.lower()

            has_digit = any(char.isdigit() for char in problem_text_lower)
            has_operator = any(op in problem_text_lower for op in ["=", "+", "-", "*", "/", "^"])

            if not has_digit or not has_operator:
                return ParsedProblem(
                    raw_input=raw_input,
                    problem_text=validated.problem_text,
                    topic=validated.topic,
                    difficulty=validated.difficulty,
                    variables=validated.variables,
                    constraints=validated.constraints,
                    required_formulas=validated.required_formulas,
                    needs_clarification=True,
                    clarification_questions=[
                        "Please provide the complete equation or mathematical expression to solve."
                    ],
                    confidence_score=0.3,
                )


            return ParsedProblem(
                raw_input=raw_input,
                problem_text=validated.problem_text,
                topic=validated.topic,
                difficulty=validated.difficulty,
                variables=validated.variables,
                constraints=validated.constraints,
                required_formulas=validated.required_formulas,
                needs_clarification=validated.needs_clarification,
                clarification_questions=validated.clarification_questions,
                confidence_score=validated.confidence_score,
            )

        except (ValidationError, ValueError, Exception) as exc:
            logger.error("Parser failed, activating fallback: %s", exc)

            # ------------------------------------------------------------------
            # HARD FALLBACK — GUARANTEED NO CRASH
            # ------------------------------------------------------------------
            return ParsedProblem(
                raw_input=raw_input,
                problem_text=raw_input.strip(),
                topic="unknown",
                difficulty="unknown",
                variables=[],
                constraints=[],
                required_formulas=[],
                needs_clarification=True,
                clarification_questions=[
                    "The problem statement is incomplete or unclear. Please provide the full equation or mathematical expression."
                ],
                confidence_score=0.2,
            )

