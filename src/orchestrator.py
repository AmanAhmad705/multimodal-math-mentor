"""
Orchestrator: Coordinates all agents and components (HARDENED)
"""

import logging
from typing import Dict, Optional

from src.config import settings
from src.agents.parser_agent import ParserAgent, ParsedProblem
from src.agents.router_agent import RouterAgent, RoutingDecision
from src.agents.solver_agent import SolverAgent, Solution
from src.agents.verifier_agent import VerifierAgent, VerificationResult
from src.agents.explainer_agent import ExplainerAgent
from src.rag.retriever import RAGRetriever
from src.memory.store import MemoryStore
from src.parsers.image_parser import ImageParser
from src.parsers.audio_parser import AudioParser
from src.parsers.text_parser import TextParser

logger = logging.getLogger(__name__)


# -----------------------------
# Execution Trace (UI/debug)
# -----------------------------
class ExecutionTrace:
    def __init__(self):
        self.steps = []
        self.metadata = {}

    def add_step(
        self,
        agent: str,
        action: str,
        result: str,
        status: str = "success",
    ):
        self.steps.append(
            {
                "agent": agent,
                "action": action,
                "result": result,
                "status": status,
            }
        )

    def to_dict(self) -> Dict:
        return {
            "steps": self.steps,
            "total_steps": len(self.steps),
            "metadata": self.metadata,
        }


# -----------------------------
# Main Orchestrator
# -----------------------------
class MathMentorOrchestrator:
    def __init__(self, session_id: Optional[str] = None):
        
        # RAG
        self.rag_retriever = RAGRetriever()

        # Memory
        self.memory_store = MemoryStore()
        self.session_id = self.memory_store.create_session(session_id)

        # Agents
        self.parser_agent = ParserAgent()
        self.router_agent = RouterAgent()
        self.solver_agent = SolverAgent(
            retriever=self.rag_retriever,
            memory_store=self.memory_store
        )
        self.verifier_agent = VerifierAgent()
        self.explainer_agent = ExplainerAgent()


        # Input parsers
        self.image_parser = ImageParser()
        self.audio_parser = AudioParser()
        self.text_parser = TextParser()

        self.trace = ExecutionTrace()

        logger.info(f"Orchestrator initialized for session: {self.session_id}")

    # --------------------------------------------------
    # Input processing
    # --------------------------------------------------
    def process_text(self, text_input: str) -> Dict:
        self.trace.add_step("TextParser", "Normalize", "Cleaning text...")
        cleaned_text, confidence = self.text_parser.parse_text(text_input)

        input_id = self.memory_store.log_raw_input(
            self.session_id,
            "text",
            text_input,
            confidence,
        )

        self.trace.add_step(
            "TextParser",
            "Normalize",
            f"Cleaned text (confidence={confidence:.2%})",
        )

        return {
            "raw_text": cleaned_text,
            "confidence": confidence,
            "needs_hitl": False,
            "input_id": input_id,
        }
    def process_image(self, image_path: str) -> Dict:
        """
        Extract text from image and log input.
        """
        self.trace.add_step("ImageParser", "OCR", "Extracting text from image...")

        # parse_image returns tuple: (text, confidence, needs_hitl)
        result = self.image_parser.parse_image(image_path)
        text, confidence, needs_hitl = result  # unpack tuple

        input_id = self.memory_store.log_raw_input(
            self.session_id,
            "image",
            text,
            confidence,
        )

        self.trace.add_step(
            "ImageParser",
            "OCR",
            f"Extracted text (confidence={confidence:.2%})",
        )

        return {
            "raw_text": text,
            "confidence": confidence,
            "needs_hitl": needs_hitl,
            "input_id": input_id,
        }




    def process_audio(self, audio_path: str) -> Dict:
        """
        Transcribe audio and log input.
        """
        self.trace.add_step("AudioParser", "Transcribe", "Transcribing audio...")

        # parse_audio returns tuple: (text, confidence, needs_hitl)
        result = self.audio_parser.parse_audio(audio_path)
        text, confidence, needs_hitl = result  # unpack tuple

        input_id = self.memory_store.log_raw_input(
            self.session_id,
            "audio",
            text,
            confidence,
        )

        self.trace.add_step(
            "AudioParser",
            "Transcribe",
            f"Transcribed text (confidence={confidence:.2%})",
        )

        return {
            "raw_text": text,
            "confidence": confidence,
            "needs_hitl": needs_hitl,
            "input_id": input_id,
        }


    # --------------------------------------------------
    # Main solve pipeline
    # --------------------------------------------------
    def solve_problem(
        self,
        raw_text: str,
        input_id: str,
        handle_hitl_callback=None,
    ) -> Dict:
        try:
            # -----------------------------
            # 1. PARSE
            # -----------------------------
            self.trace.add_step("ParserAgent", "Parse", "Parsing problem...")
            parsed_problem: ParsedProblem = self.parser_agent.parse(raw_text)

            self.trace.add_step(
                "ParserAgent",
                "Parse",
                f"Topic={parsed_problem.topic}, confidence={parsed_problem.confidence_score:.2%}",
                status="warning" if parsed_problem.needs_clarification else "success",
            )

            problem_id = self.memory_store.log_parsed_problem(
                self.session_id,
                input_id,
                parsed_problem.problem_text,
                parsed_problem.topic,
                parsed_problem.difficulty,
                parsed_problem.variables,
                parsed_problem.constraints,
                parsed_problem.needs_clarification,
                parsed_problem.clarification_questions,
                parsed_problem.confidence_score,
            )


            # HITL — PARSE AMBIGUITY
            # -----------------------------
            if parsed_problem.needs_clarification:
                self.trace.add_step(
                    "HITL",
                    "Clarification",
                    "Problem needs clarification",
                    "warning",
                )

                hitl_id = self.memory_store.create_hitl_request(
                    problem_id,
                    None,
                    None,
                    "ambiguity",
                )

                return {
                    "status": "hitl_pending",
                    "hitl_id": hitl_id,
                    "clarification_questions": parsed_problem.clarification_questions,
                    "trace": self.trace.to_dict(),
                }



                    

            # -----------------------------
            # 2. ROUTE
            # -----------------------------
            self.trace.add_step("RouterAgent", "Route", "Routing problem...")
            routing: RoutingDecision = self.router_agent.route(parsed_problem)

            self.trace.add_step(
                "RouterAgent",
                "Route",
                f"Strategy={routing.solving_strategy}, complexity={routing.estimated_complexity}/5",
            )

            # -----------------------------
            # 3. RAG
            # -----------------------------
            self.trace.add_step("RAG", "Retrieve", "Retrieving knowledge...")

            # Require RAG only for non-trivial problems
            requires_rag = (
                routing.solving_strategy not in ["direct_formula", "simple_manipulation"]
                and routing.estimated_complexity >= 2
            )

            retrieved_docs, context_str, rag_needs_hitl = self.rag_retriever.retrieve(
                parsed_problem.problem_text,
                top_k=settings.rag_top_k,
                requires_rag=requires_rag,
            )

            if rag_needs_hitl and settings.enable_hitl:
                self.trace.add_step(
                    "HITL",
                    "RAG Retrieval",
                    "No reliable documents retrieved. Human review required.",
                    "warning",
                )
                return {
                    "status": "hitl_pending",
                    "stage": "rag_retrieval",
                    "message": context_str,
                    "trace": self.trace.to_dict(),
                }

            self.trace.add_step(
                "RAG",
                "Retrieve",
                f"Retrieved {len(retrieved_docs)} documents",
            )


            # -----------------------------
            # 4. SOLVE
            # -----------------------------
            self.trace.add_step("SolverAgent", "Solve", "Solving problem...")
            solution: Solution = self.solver_agent.solve(
            parsed_problem=parsed_problem,
            routing_decision=routing,
            retrieved_context=context_str,
            rag_retrieval_failed=False,
            memory_store=self.memory_store,
        )




            self.trace.add_step(
                "SolverAgent",
                "Solve",
                f"Solved (confidence={solution.confidence_score:.2%})",
                status="warning" if solution.confidence_score < 0.7 else "success",
            )

            solution_id = self.memory_store.log_solution(
                problem_id,
                routing.problem_type,
                routing.solving_strategy,
                solution.solution_steps,
                solution.final_answer,
                solution.key_formulas_used,
                solution.confidence_score,
            )

            # -----------------------------
            # 5. VERIFY
            # -----------------------------
            self.trace.add_step("VerifierAgent", "Verify", "Verifying solution...")
            verification: VerificationResult = self.verifier_agent.verify(
                parsed_problem,
                solution,
            )

            self.trace.add_step(
                "VerifierAgent",
                "Verify",
                f"Correct={verification.is_correct}, confidence={verification.confidence_score:.2%}",
                status="warning" if verification.requires_hitl else "success",
            )

            verification_id = self.memory_store.log_verification(
                solution_id,
                verification.is_correct,
                verification.confidence_score,
                verification.issues_found,
                verification.edge_cases_checked,
                verification.units_verified,
                verification.requires_hitl,
            )

            # HITL (verification)
            if (
                    verification.requires_hitl
                    and settings.enable_hitl
                    and (not verification.is_correct or verification.confidence_score < 0.6)
                ):
                self.trace.add_step(
                    "HITL",
                    "Verification",
                    "Low verification confidence",
                    "warning",
                )

                if handle_hitl_callback:
                    hitl_id = self.memory_store.create_hitl_request(
                        problem_id,
                        solution_id,
                        verification_id,
                        "low_confidence",
                    )
                    result = handle_hitl_callback(
                        hitl_id,
                        {
                            "solution_steps": solution.solution_steps,
                            "final_answer": solution.final_answer,
                            "confidence": solution.confidence_score,
                            "reasoning": getattr(solution, "reasoning", ""),
                        }
                    )


                    if not result or "decision" not in result:
                        return {
                            "status": "hitl_pending",
                            "trace": self.trace.to_dict(),
                        }

                    if result["decision"] == "rejected":
                        self.memory_store.update_hitl_request(
                            hitl_id,
                            result.get("feedback", ""),
                            "rejected",
                        )
                        return {
                            "status": "rejected_at_verification",
                            "trace": self.trace.to_dict(),
                        }

                    self.memory_store.update_hitl_request(
                        hitl_id,
                        result.get("feedback", ""),
                        "approved",
                    )

            # -----------------------------
            # 6. EXPLAIN
            # -----------------------------
            self.trace.add_step(
                "ExplainerAgent",
                "Explain",
                "Generating explanation...",
            )
            explanation = self.explainer_agent.explain(
                parsed_problem,
                solution,
            )

            self.trace.add_step(
                "ExplainerAgent",
                "Explain",
                "Explanation generated",
            )

            # -----------------------------
            # SUCCESS
            # -----------------------------
            retrieved_context = [
                {
                    "title": doc.get("title", f"Document {i+1}"),  # fallback if title missing
                    "content": doc.get("content", ""),
                    "relevance": doc.get("relevance", 0)
                }
                for i, doc in enumerate(retrieved_docs)
            ]

            return {
                "status": "success",
                "parsed_problem": {
                    "text": parsed_problem.problem_text,
                    "topic": parsed_problem.topic,
                    "difficulty": parsed_problem.difficulty,
                    "variables": parsed_problem.variables,
                    "constraints": parsed_problem.constraints,
                },
                "routing": {
                    "problem_type": routing.problem_type,
                    "strategy": routing.solving_strategy,
                    "complexity": routing.estimated_complexity,
                },
                "solution": {
                    "steps": solution.solution_steps,
                    "final_answer": solution.final_answer,
                    "confidence": solution.confidence_score,
                    "formulas_used": getattr(solution, "key_formulas_used", []),  # safe access
                },
                "verification": {
                    "is_correct": verification.is_correct,
                    "confidence": verification.confidence_score,
                    "issues": verification.issues_found,
                },
                "explanation": {
                    "overview": explanation.conceptual_overview,
                    "steps": explanation.step_by_step,
                    "insights": explanation.key_insights,
                    "mistakes": explanation.common_mistakes,
                    "related": explanation.related_concepts,
                },
                "retrieved_context": retrieved_context,
                "trace": self.trace.to_dict(),
                "session_id": self.session_id,
                "problem_id": problem_id,
                "solution_id": solution_id,
            }


        except Exception as e:
            logger.error("Solving pipeline failed", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "trace": self.trace.to_dict(),
            }
