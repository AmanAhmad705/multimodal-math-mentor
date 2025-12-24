"""
Router Agent: Classifies problem type and routes workflow
"""

import logging
from typing import Literal
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.config import settings, AGENT_CONFIG
from src.agents.parser_agent import ParsedProblem

logger = logging.getLogger(__name__)

class RoutingDecision(BaseModel):
    """Router output schema"""
    problem_type: Literal["analytical", "geometric", "statistical", "computational"]
    solving_strategy: Literal[
        "direct_formula",
        "step_by_step",
        "graphical",
        "numerical"
    ]
    requires_rag: bool = Field(default=True, description="Whether to use knowledge base")
    requires_tools: bool = Field(default=False, description="Whether to use computation tools")
    priority_topics: list[str] = Field(default_factory=list, description="Key concepts to focus on")
    estimated_complexity: int = Field(default=2, description="1-5, for step planning")

class RouterAgent:
    """Routes problem to appropriate solving strategy"""
    
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.groq_model,
            temperature=AGENT_CONFIG["router"]["temperature"],
            max_tokens=AGENT_CONFIG["router"]["max_tokens"],
        )
        self.parser = JsonOutputParser(pydantic_object=RoutingDecision)
        self._setup_prompt()
    
    def _setup_prompt(self):
        self.prompt = ChatPromptTemplate.from_template("""
    You are a routing agent for a math problem.

    You MUST return ONLY valid JSON.
    Do NOT include explanations, markdown, or text outside JSON.

    JSON schema (follow exactly):

    {format_instructions}

    Rules:
    - Linear equations (e.g., 3x + 5 = 14):
    - problem_type = "analytical"
    - solving_strategy = "direct_formula"
    - requires_rag = false
    - requires_tools = false
    - estimated_complexity = 1

    - Quadratic equations:
    - solving_strategy = "direct_formula"
    - estimated_complexity = 1 or 2
    - requires_rag = false

    - Formula recall or theory explanation → requires_rag = true
    - Heavy computation → requires_tools = true

    Problem:
    {problem_text}
    """)

    
    def route(self, parsed_problem: ParsedProblem) -> RoutingDecision:
        try:
            logger.info(f"Routing problem: {parsed_problem.topic}")

            chain = self.prompt | self.llm | self.parser

            raw = chain.invoke({
                "problem_text": parsed_problem.problem_text,
                "topic": parsed_problem.topic,
                "difficulty": parsed_problem.difficulty,
                "format_instructions": self.parser.get_format_instructions(),
            })

            # 🔒 ALWAYS return Pydantic object
            result = RoutingDecision.model_validate(raw)

            logger.info(
                f"Routing decision: {result.problem_type}, "
                f"Strategy: {result.solving_strategy}"
            )
            return result

        except Exception as e:
            logger.error(f"Routing failed: {e}")
            return RoutingDecision(
                problem_type="analytical",
                solving_strategy="direct_formula",
                requires_rag=False,
                requires_tools=False,
                priority_topics=[],
                estimated_complexity=1,
            )

