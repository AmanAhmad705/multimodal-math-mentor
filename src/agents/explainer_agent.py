"""
Explainer Agent: Generates student-friendly step-by-step explanations
"""

import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.config import settings, AGENT_CONFIG
from src.agents.parser_agent import ParsedProblem
from src.agents.solver_agent import Solution

logger = logging.getLogger(__name__)

class Explanation(BaseModel):
    """Student-friendly explanation"""
    title: str = Field(..., description="Problem title")
    conceptual_overview: str = Field(..., description="What is this problem about?")
    step_by_step: list[str] = Field(..., description="Detailed steps with reasoning")
    key_insights: list[str] = Field(default_factory=list, description="Important takeaways")
    common_mistakes: list[str] = Field(default_factory=list, description="Pitfalls to avoid")
    related_concepts: list[str] = Field(default_factory=list, description="Broader context")
    final_answer_explained: str = Field(..., description="Why the answer is correct")

class ExplainerAgent:
    """Generates educational explanations"""
    
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.groq_model,
            temperature=AGENT_CONFIG["explainer"]["temperature"],
            max_tokens=AGENT_CONFIG["explainer"]["max_tokens"],
        )
        self._setup_prompt()
    
    def _setup_prompt(self):
        """Create explanation prompt"""
        self.prompt = ChatPromptTemplate.from_template("""You are an expert math tutor explaining problems to students.

Problem: {problem_text}
Topic: {topic}
Difficulty: {difficulty}

Verified Solution:
{solution_steps}

Final Answer: {final_answer}

Create a student-friendly explanation:

1. CONCEPTUAL OVERVIEW: What is the core idea? (2-3 sentences)

2. STEP-BY-STEP BREAKDOWN: Explain each step clearly, including WHY we do it.

3. KEY INSIGHTS: What should the student remember? (3-4 bullet points)

4. COMMON MISTAKES: What errors do students often make? (2-3 examples)

5. RELATED CONCEPTS: How does this connect to other topics?

6. FINAL ANSWER: Why is "{final_answer}" correct?

Keep language simple and encouraging. Use analogies where helpful.""")
    
    def explain(
        self,
        parsed_problem: ParsedProblem,
        solution: Solution,
    ) -> Explanation:
        """
        Generate student-friendly explanation
        
        Args:
            parsed_problem: Original problem
            solution: Verified solution
            
        Returns:
            Explanation with multiple facets
        """
        try:
            logger.info("Generating explanation...")
            
            steps_str = "\n".join(solution.solution_steps)
            
            prompt_value = self.prompt.format_prompt(
                problem_text=parsed_problem.problem_text,
                topic=parsed_problem.topic,
                difficulty=parsed_problem.difficulty,
                solution_steps=steps_str,
                final_answer=solution.final_answer,
            )
            
            response = self.llm.invoke(prompt_value.to_messages())
            explanation_text = response.content
            
            # Parse explanation
            explanation = self._parse_explanation(explanation_text, parsed_problem, solution)
            logger.info("Explanation generated")
            return explanation
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return Explanation(
                title=parsed_problem.problem_text[:50],
                conceptual_overview="See detailed solution above.",
                step_by_step=solution.solution_steps,
                key_insights=["Focus on the key formulas used."],
                final_answer_explained=solution.final_answer,
            )
    
    def _parse_explanation(
        self,
        explanation_text: str,
        parsed_problem: ParsedProblem,
        solution: Solution,
    ) -> Explanation:
        """Parse explanation text into structured form"""
        sections = {
            "conceptual_overview": "",
            "step_by_step": [],
            "key_insights": [],
            "common_mistakes": [],
            "related_concepts": [],
            "final_answer_explained": "",
        }
        
        current_section = None
        lines = explanation_text.split("\n")
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if "CONCEPTUAL OVERVIEW" in line.upper():
                current_section = "conceptual_overview"
            elif "STEP" in line.upper() and "BREAKDOWN" in line.upper():
                current_section = "step_by_step"
            elif "KEY INSIGHTS" in line.upper():
                current_section = "key_insights"
            elif "COMMON MISTAKES" in line.upper():
                current_section = "common_mistakes"
            elif "RELATED CONCEPTS" in line.upper():
                current_section = "related_concepts"
            elif "FINAL ANSWER" in line.upper():
                current_section = "final_answer_explained"
            elif current_section and line and not line.endswith(":"):
                if current_section in ["step_by_step", "key_insights", "common_mistakes", "related_concepts"]:
                    sections[current_section].append(line.lstrip("-•*").strip())
                else:
                    sections[current_section] += line + " "
        
        return Explanation(
            title=parsed_problem.problem_text[:60],
            conceptual_overview=sections["conceptual_overview"].strip(),
            step_by_step=sections["step_by_step"] or solution.solution_steps,
            key_insights=sections["key_insights"],
            common_mistakes=sections["common_mistakes"],
            related_concepts=sections["related_concepts"],
            final_answer_explained=sections["final_answer_explained"].strip() or solution.final_answer,
        )
