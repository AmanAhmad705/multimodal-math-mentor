"""
Verifier Agent: Checks solution correctness and triggers HITL if uncertain
"""

import logging
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.config import settings, AGENT_CONFIG
from src.agents.parser_agent import ParsedProblem
from src.agents.solver_agent import Solution
RISK_KEYWORDS = [
    "assume",
    "assumption",
    "approximately",
    "approximation",
    "numerical",
    "iteration",
    "fixed point",
    "no closed form",
    "cannot be solved",
    "lambert",
    "transcendental",
]


logger = logging.getLogger(__name__)

class VerificationResult(BaseModel):
    """Verification output"""
    is_correct: bool = Field(..., description="Solution is mathematically correct")
    confidence_score: float = Field(..., description="Verification confidence 0-1")
    issues_found: list[str] = Field(default_factory=list, description="Errors or concerns")
    edge_cases_checked: list[str] = Field(default_factory=list, description="Edge cases verified")
    units_verified: bool = Field(default=True)
    requires_hitl: bool = Field(default=False, description="Needs human review")
    reasoning: str = Field(..., description="Verification reasoning")

class VerifierAgent:
    """Verifies correctness of solutions"""
    
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.groq_model,
            temperature=AGENT_CONFIG["verifier"]["temperature"],
            max_tokens=AGENT_CONFIG["verifier"]["max_tokens"],
        )
        self.parser = JsonOutputParser(pydantic_object=VerificationResult)
        self._setup_prompt()
    
    def _setup_prompt(self):
        self.prompt = ChatPromptTemplate.from_template("""
    You are a math verification agent.

    Your job:
    - Verify whether the final answer is mathematically correct.
    - Treat mathematically equivalent answers as correct (e.g., 3/6 = 1/2).
    - Do NOT penalize formatting or wording differences.
    - Only flag issues if there is a real mathematical error.

    Problem:
    {problem_text}

    Constraints:
    {constraints}

    Solution Steps:
    {solution_steps}

    Final Answer:
    {final_answer}

    Instructions:
    - If the math is correct, set is_correct = true.
    - Set requires_hitl = true ONLY if you are genuinely uncertain.
    - Confidence should reflect certainty in correctness, not presentation quality.

    {format_instructions}
    """)

    
    def verify(
        self,
        parsed_problem: ParsedProblem,
        solution: Solution,
    ) -> VerificationResult:
        try:
            logger.info("Verifying solution...")

            constraints_str = ", ".join(parsed_problem.constraints) if parsed_problem.constraints else "None"
            steps_str = "\n".join(solution.solution_steps)

            chain = self.prompt | self.llm | self.parser

            raw = chain.invoke({
                "problem_text": parsed_problem.problem_text,
                "constraints": constraints_str,
                "solution_steps": steps_str,
                "final_answer": solution.final_answer,
                "format_instructions": self.parser.get_format_instructions(),
            })

            # 🔒 ALWAYS return Pydantic model
            result = VerificationResult.model_validate(raw)
            # --------------------------------------------------
            # Generic risk-based HITL detection (production-grade)
            # --------------------------------------------------
            reasoning_lower = result.reasoning.lower()
            steps_text = " ".join(solution.solution_steps).lower()

            risk_detected = any(
                keyword in reasoning_lower or keyword in steps_text
                for keyword in RISK_KEYWORDS
            )

            if risk_detected:
                result.requires_hitl = True
                result.confidence_score = min(result.confidence_score, 0.7)
                result.issues_found.append(
                    "Solution involves assumptions, approximations, or non-verifiable reasoning"
                )

            
            # HITL only if incorrect OR very low confidence
            if settings.enable_hitl:
                if result.requires_hitl:
                    pass  # already decided by risk rules
                elif not result.is_correct:
                    result.requires_hitl = True
                elif result.confidence_score < 0.8:
                    result.requires_hitl = True




            logger.info(
                f"Verification complete. Correct: {result.is_correct}, "
                f"Confidence: {result.confidence_score}"
            )
            return result

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return VerificationResult(
                is_correct=False,
                confidence_score=0.0,
                issues_found=["Verification failed due to system error"],
                edge_cases_checked=[],
                units_verified=False,
                requires_hitl=True,
                reasoning=f"Verification error: {e}",
            )

