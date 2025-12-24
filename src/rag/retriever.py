"""
RAG Retriever: Orchestrates knowledge base retrieval
Enforces no-hallucination policy
"""

import logging
from typing import List, Dict, Tuple
from src.rag.vector_store import VectorStore
from src.rag.knowledge_base import MathKnowledgeBase
from src.config import settings
import streamlit as st

logger = logging.getLogger(__name__)

class RAGRetriever:
    """Retrieves relevant context from knowledge base"""

    def __init__(self):
        if "vector_store" not in st.session_state:
            vector_store = VectorStore()

            if not vector_store.load():
                logger.info("No existing vector store found. Building new one...")
                documents = MathKnowledgeBase.load_knowledge_base()
                vector_store.initialize_from_knowledge_base(documents)
            else:
                logger.info("Loaded existing vector store from disk")

            st.session_state.vector_store = vector_store

        self.vector_store = st.session_state.vector_store

    
    def _initialize(self):
        """Initialize vector store"""
        if not self.vector_store.load():
            logger.info("Building new vector store...")
            documents = MathKnowledgeBase.load_knowledge_base()
            self.vector_store.initialize_from_knowledge_base(documents)
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        requires_rag: bool = True,
    ) -> Tuple[List[Dict], str, bool]:
        """
        Retrieve relevant documents from knowledge base
        
        Args:
            query: Problem statement or concept
            top_k: Number of documents to retrieve
            requires_rag: If True and no docs found → trigger HITL
            
        Returns:
            (List of relevant documents, formatted context string, needs_hitl_for_rag)
        """
        if top_k is None:
            top_k = settings.rag_top_k
        
        try:
            logger.info(f"Retrieving documents for: {query[:60]}...")
            
            results = self.vector_store.search(query, top_k=top_k)
            
            # CRITICAL: If requires_rag=true and no docs retrieved → trigger HITL
            if requires_rag and not results:
                logger.warning(
                    f"RAG retrieval failed: No documents met similarity threshold "
                    f"({settings.similarity_threshold:.2f}). requires_rag=True. "
                    f"Triggering HITL to prevent hallucination."
                )
                return [], (
                    f"⚠️ RETRIEVAL FAILED: Knowledge base lookup returned no relevant documents. "
                    f"Similarity threshold: {settings.similarity_threshold:.2f}. "
                    f"Cannot proceed without verified sources. "
                    f"Human review required."
                ), True
            
            if not results:
                logger.warning("No relevant documents found, but requires_rag=False")
                return [], (
                    "Note: No specific formulas retrieved from knowledge base. "
                    "Using general mathematical knowledge."
                ), False
            
            # Format context for LLM
            context_parts = []
            for i, doc in enumerate(results, 1):
                context_parts.append(
                    f"[Document {i}] {doc['metadata'].get('title', 'Unknown')}\n"
                    f"Content: {doc['content']}\n"
                    f"Topic: {doc['metadata'].get('topic', 'general')}\n"
                    f"Difficulty: {doc['metadata'].get('difficulty', 'N/A')}\n"
                    f"Relevance Score: {doc['relevance_score']:.2%}\n"
                )
            
            context_str = "\n---\n".join(context_parts)
            
            logger.info(f"Retrieved {len(results)} relevant documents")
            return results, context_str, False
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return [], f"Error retrieving context: {str(e)}", requires_rag
    
    def add_learned_document(self, problem_text: str, solution: str, feedback: str):
        """
        Add solved problem to knowledge base for future learning
        
        Args:
            problem_text: Original problem
            solution: Verified solution
            feedback: User feedback (correct/incorrect/needs_revision)
        """
        try:
            from langchain.schema import Document
            
            new_doc = Document(
                page_content=f"Learned Solution: {problem_text}\nSolution: {solution}",
                metadata={
                    "source": "learned_problem",
                    "feedback": feedback,
                    "problem_summary": problem_text[:50],
                }
            )
            
            self.vector_store.add_documents([new_doc])
            logger.info(f"Added learned document to knowledge base (feedback: {feedback})")
            
        except Exception as e:
            logger.error(f"Failed to add learned document: {e}")
