"""
Knowledge Base: Math formulas, concepts, and solution templates
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
from langchain.schema import Document

from src.config import settings

logger = logging.getLogger(__name__)

class MathKnowledgeBase:
    """Manages curated math knowledge base"""
    
    KB_DATA = [
        {
            "id": "algebra_001",
            "title": "Quadratic Equation Formula",
            "content": "For ax² + bx + c = 0, the solutions are x = (-b ± √(b² - 4ac)) / 2a. The discriminant Δ = b² - 4ac determines the nature of roots. If Δ > 0: two distinct real roots. If Δ = 0: one repeated real root. If Δ < 0: two complex conjugate roots.",
            "topic": "algebra",
            "category": "formulas",
            "difficulty": "medium",
            "prerequisites": ["linear_equations", "factorization"],
        },
        {
            "id": "algebra_002",
            "title": "Vieta's Formulas",
            "content": "For polynomial ax² + bx + c = 0 with roots α and β: sum of roots = α + β = -b/a, product of roots = αβ = c/a. These relations hold for any quadratic regardless of root nature.",
            "topic": "algebra",
            "category": "formulas",
            "difficulty": "medium",
            "prerequisites": ["quadratic_equations"],
        },
        {
            "id": "algebra_003",
            "title": "Arithmetic Progression (AP)",
            "content": "AP series: a, a+d, a+2d, ..., a+(n-1)d. General term: aₙ = a + (n-1)d. Sum of n terms: Sₙ = n/2 * (2a + (n-1)d) = n/2 * (first + last). Constraint: common difference d is constant.",
            "topic": "algebra",
            "category": "formulas",
            "difficulty": "easy",
            "prerequisites": ["sequences"],
        },
        {
            "id": "algebra_004",
            "title": "Geometric Progression (GP)",
            "content": "GP series: a, ar, ar², ..., ar^(n-1). General term: aₙ = ar^(n-1). Sum of n terms: Sₙ = a(1-r^n)/(1-r) if r ≠ 1. For |r| < 1 (infinite GP): S∞ = a/(1-r). Common ratio r ≠ 1.",
            "topic": "algebra",
            "category": "formulas",
            "difficulty": "medium",
            "prerequisites": ["sequences"],
        },
        {
            "id": "trigonometry_001",
            "title": "Basic Trigonometric Identities",
            "content": "sin²θ + cos²θ = 1 (fundamental identity). 1 + tan²θ = sec²θ. 1 + cot²θ = csc²θ. sin(A±B) = sinA cosB ± cosA sinB. cos(A±B) = cosA cosB ∓ sinA sinB. tan(A±B) = (tanA ± tanB)/(1 ∓ tanA tanB).",
            "topic": "trigonometry",
            "category": "formulas",
            "difficulty": "medium",
            "prerequisites": ["trigonometric_ratios"],
        },
        {
            "id": "calculus_001",
            "title": "Derivatives - Power Rule",
            "content": "If f(x) = x^n, then f'(x) = n * x^(n-1). This is the power rule for differentiation. Example: d/dx(x³) = 3x². Applies for all real n. Chain rule: d/dx(f(g(x))) = f'(g(x)) * g'(x).",
            "topic": "calculus",
            "category": "formulas",
            "difficulty": "medium",
            "prerequisites": ["limits"],
        },
        {
            "id": "calculus_002",
            "title": "Derivatives - Product and Quotient Rule",
            "content": "Product rule: d/dx(uv) = u*dv/dx + v*du/dx. Quotient rule: d/dx(u/v) = (v*du/dx - u*dv/dx) / v². Used for composite functions.",
            "topic": "calculus",
            "category": "formulas",
            "difficulty": "medium",
            "prerequisites": ["power_rule"],
        },
        {
            "id": "calculus_003",
            "title": "Indefinite Integration - Basic Rules",
            "content": "∫x^n dx = x^(n+1)/(n+1) + C for n ≠ -1. ∫1/x dx = ln|x| + C. ∫eˣ dx = eˣ + C. ∫sinx dx = -cosx + C. ∫cosx dx = sinx + C. Always add constant C.",
            "topic": "calculus",
            "category": "formulas",
            "difficulty": "medium",
            "prerequisites": ["derivatives"],
        },
        {
            "id": "probability_001",
            "title": "Probability Basics",
            "content": "P(A) = (Number of favorable outcomes) / (Total number of outcomes). Range: 0 ≤ P(A) ≤ 1. Complementary event: P(A') = 1 - P(A). Addition rule: P(A∪B) = P(A) + P(B) - P(A∩B). Multiplication rule: P(A∩B) = P(A) * P(B|A).",
            "topic": "probability",
            "category": "formulas",
            "difficulty": "easy",
            "prerequisites": ["counting"],
        },
        {
            "id": "probability_002",
            "title": "Permutations and Combinations",
            "content": "Permutations (order matters): P(n,r) = n! / (n-r)!. Combinations (order doesn't matter): C(n,r) = n! / (r!(n-r)!). Key: nPr = r! * nCr. Constraint: r ≤ n. Used in probability counting.",
            "topic": "probability",
            "category": "formulas",
            "difficulty": "medium",
            "prerequisites": ["factorials"],
        },
        {
            "id": "linear_algebra_001",
            "title": "Determinant of 2x2 Matrix",
            "content": "For matrix [[a,b],[c,d]], determinant = ad - bc. Non-zero determinant means matrix is invertible. Used in Cramer's rule for solving linear systems.",
            "topic": "linear_algebra",
            "category": "formulas",
            "difficulty": "easy",
            "prerequisites": ["matrices"],
        },
        {
            "id": "linear_algebra_002",
            "title": "Matrix Inversion",
            "content": "For 2x2 matrix A = [[a,b],[c,d]], inverse A⁻¹ = (1/(ad-bc)) * [[d,-b],[-c,a]]. Condition: det(A) ≠ 0. For larger matrices, use Gauss-Jordan elimination or adjugate method.",
            "topic": "linear_algebra",
            "category": "formulas",
            "difficulty": "medium",
            "prerequisites": ["determinants"],
        },
        {
            "id": "coordinate_geometry_001",
            "title": "Distance and Midpoint Formula",
            "content": "Distance between P(x₁,y₁) and Q(x₂,y₂): d = √((x₂-x₁)² + (y₂-y₁)²). Midpoint M = ((x₁+x₂)/2, (y₁+y₂)/2). Slope m = (y₂-y₁)/(x₂-x₁).",
            "topic": "coordinate_geometry",
            "category": "formulas",
            "difficulty": "easy",
            "prerequisites": ["coordinate_system"],
        },
        {
            "id": "coordinate_geometry_002",
            "title": "Equation of Circle",
            "content": "Standard form: (x-h)² + (y-k)² = r², where center = (h,k), radius = r. General form: x² + y² + 2gx + 2fy + c = 0, where center = (-g,-f), radius = √(g²+f²-c).",
            "topic": "coordinate_geometry",
            "category": "formulas",
            "difficulty": "medium",
            "prerequisites": ["circle_basics"],
        },
        {
            "id": "concept_001",
            "title": "Common OCR/Parsing Errors",
            "content": "Typical OCR mistakes: '0' confused with 'O', '1' with 'I', 'sqrt' misspelled as 'sqaure' or 'sqrt'. Audio transcription: 'squared' for '²', 'pi' for 'π'. Always validate parsed equations against original input.",
            "topic": "general",
            "category": "pitfalls",
            "difficulty": "easy",
            "prerequisites": [],
        },
        {
            "id": "concept_002",
            "title": "Domain and Range Constraints",
            "content": "For √x: domain x ≥ 0. For 1/x: domain x ≠ 0. For log(x): domain x > 0. For tan(x): undefined at x = π/2 + nπ. Always verify constraints in solution.",
            "topic": "general",
            "category": "constraints",
            "difficulty": "medium",
            "prerequisites": ["functions"],
        },
    ]
    
    @staticmethod
    def load_knowledge_base() -> List[Document]:
        """
        Load knowledge base as Document objects
        
        Returns:
            List of Document objects for vector store
        """
        documents = []
        
        for item in MathKnowledgeBase.KB_DATA:
            doc = Document(
                page_content=f"{item['title']}\n{item['content']}",
                metadata={
                    "id": item["id"],
                    "title": item["title"],
                    "topic": item["topic"],
                    "category": item["category"],
                    "difficulty": item["difficulty"],
                    "prerequisites": item["prerequisites"],
                }
            )
            documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} knowledge base documents")
        return documents
    
    @staticmethod
    def save_to_json(output_path: str = None):
        """Save knowledge base to JSON file"""
        if output_path is None:
            output_path = settings.knowledge_base_path + "/math_formulas.json"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(MathKnowledgeBase.KB_DATA, f, indent=2)
        
        logger.info(f"Knowledge base saved to {output_path}")
