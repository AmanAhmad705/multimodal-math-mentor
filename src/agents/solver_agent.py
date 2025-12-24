"""
Solver Agent: Solves problem using RAG, memory, and reasoning
Reuses solution patterns from memory for consistency
"""

import logging
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.config import settings, AGENT_CONFIG
from src.agents.parser_agent import ParsedProblem
from src.agents.router_agent import RoutingDecision

logger = logging.getLogger(__name__)

class Solution(BaseModel):
    """Solution representation"""
    solution_steps: list[str] = Field(..., description="Step-by-step solution")
    final_answer: str = Field(..., description="Final numerical or symbolic answer")
    key_formulas_used: list[str] = Field(default_factory=list)
    reasoning: str = Field(..., description="Overall reasoning approach")
    confidence_score: float = Field(default=0.8, description="Solution confidence")

class SolverAgent:
    """Solves math problems using RAG, memory patterns, and LLM reasoning"""
    
    def __init__(self, retriever=None, memory_store=None):
        self.llm = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.groq_model,
            temperature=AGENT_CONFIG["solver"]["temperature"],
            max_tokens=AGENT_CONFIG["solver"]["max_tokens"],
        )
        self.retriever = retriever
        self.memory_store = memory_store
        self._setup_prompt()
    
    def _setup_prompt(self):
        """Create solving prompt with memory injection"""
        self.prompt = ChatPromptTemplate.from_template("""You are an expert math tutor solving JEE-style problems.

Problem:
{problem_text}

Constraints: {constraints}
Variables: {variables}

Retrieved Formulas & Concepts:
{retrieved_context}

Similar Previously Solved Problems (for pattern reference):
{similar_problems_context}

Solving Strategy: {strategy}

Instructions:
1. Show clear, numbered solution steps
2. Explain each step with reasoning
3. Use retrieved formulas if applicable
4. Reference similar solved problems if pattern matches
5. State the final answer clearly
6. Be precise and avoid approximations unless justified
7. IMPORTANT: If retrieval indicated failure or hallucination risk, state this clearly

Provide solution in this format:
STEP 1: [description]
STEP 2: [description]
...
FINAL ANSWER: [answer]

Key Formulas Used: [comma-separated list]
Confidence: [0-1]""")
    
    def solve(
        self,
        parsed_problem: ParsedProblem,
        routing_decision: RoutingDecision,
        retrieved_context: str = "",
        rag_retrieval_failed: bool = False,
        memory_store=None,
    ) -> Solution:
        """
        Solve the problem using RAG, memory patterns, and reasoning
        
        Args:
            parsed_problem: Parsed problem from ParserAgent
            routing_decision: Routing strategy from RouterAgent
            retrieved_context: RAG context (formulas, examples)
            rag_retrieval_failed: Flag if RAG lookup had issues
            memory_store: Memory store for pattern lookup
            
        Returns:
            Solution with steps and answer
        """
        try:
            logger.info(f"Solving problem with strategy: {routing_decision.solving_strategy}")
            
            constraints_str = ", ".join(parsed_problem.constraints) if parsed_problem.constraints else "None"
            variables_str = ", ".join(parsed_problem.variables) if parsed_problem.variables else "x"
            
            # Retrieve similar problems from memory for pattern reuse
            similar_problems_context = self._get_similar_problems_context(
                parsed_problem, memory_store
            )
            
            # Handle RAG retrieval failure
            if rag_retrieval_failed:
                retrieved_context = (
                    "⚠️ Knowledge base retrieval had issues. "
                    "Use general mathematical knowledge and verify all steps carefully."
                )
            
            prompt_value = self.prompt.format_prompt(
                problem_text=parsed_problem.problem_text,
                constraints=constraints_str,
                variables=variables_str,
                retrieved_context=retrieved_context or "No specific formulas retrieved.",
                similar_problems_context=similar_problems_context,
                strategy=routing_decision.solving_strategy,
            )
            
            response = self.llm.invoke(prompt_value.to_messages())
            solution_text = response.content
            
            # Parse response
            solution = self._parse_solution(solution_text)
            logger.info(f"Solution generated with {len(solution.solution_steps)} steps")
            return solution
            
        except Exception as e:
            logger.error(f"Solving failed: {e}")
            return Solution(
                solution_steps=["Could not generate solution"],
                final_answer="Unable to solve",
                reasoning=str(e),
                confidence_score=0.0,
            )
    
    def _get_similar_problems_context(
        self,
        parsed_problem: ParsedProblem,
        memory_store,
    ) -> str:
        if not memory_store:
            logger.info("Memory store not available in SolverAgent")
            return "No similar problems in memory yet."

        try:
            similar_problems = memory_store.find_similar_problems(
                topic=parsed_problem.topic,
                difficulty=parsed_problem.difficulty,
                limit=3,
            )

            if not similar_problems:
                logger.info("No similar problems found in memory")
                return "No similar problems in memory."

            # ✅ SAFE: variable exists here
            logger.info(
                f"Retrieved {len(similar_problems)} similar problems from memory"
            )

            context_parts = []
            for i, prob in enumerate(similar_problems, 1):
                context_parts.append(
                    f"[Memory {i}] Similar Problem\n"
                    f"Problem: {prob['problem_text']}\n"
                    f"Solution Pattern: {prob['answer']}\n"
                    f"User Feedback: {prob.get('feedback', 'not rated')}\n"
                )

            return "\n---\n".join(context_parts)

        except Exception as e:
            logger.error(f"Memory lookup failed: {e}")
            return "Memory lookup failed, proceeding without pattern reference."

    
    def _parse_solution(self, solution_text: str) -> Solution:
        """Extract structured solution from LLM response"""
        lines = solution_text.split("\n")
        steps = []
        final_answer = ""
        formulas = []
        confidence = 0.8
        
        for line in lines:
            line = line.strip()
            if line.startswith("STEP"):
                steps.append(line)
            elif line.startswith("FINAL ANSWER"):
                final_answer = line.replace("FINAL ANSWER:", "").strip()
            elif line.startswith("Key Formulas"):
                formulas_str = line.replace("Key Formulas Used:", "").strip()
                formulas = [f.strip() for f in formulas_str.split(",") if f.strip()]
            elif line.startswith("Confidence"):
                try:
                    conf_str = line.split(":")[1].strip()
                    confidence = float(conf_str)
                except (ValueError, IndexError):
                    pass
        
        return Solution(
            solution_steps=steps if steps else [solution_text],
            final_answer=final_answer if final_answer else "See steps above",
            key_formulas_used=formulas,
            reasoning=solution_text,
            confidence_score=min(confidence, 0.95),
        )
