"""
Vector Store Management: FAISS integration for knowledge base
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from src.config import settings

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages FAISS vector store for math knowledge base"""
    
    def __init__(self, embeddings_model: str = "all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        self.vector_store = None
        self.db_path = settings.vector_db_path
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def initialize_from_knowledge_base(self, documents: List[Document]):
        """
        Create vector store from documents
        
        Args:
            documents: List of Document objects with content and metadata
        """
        try:
            logger.info(f"Building vector store from {len(documents)} documents...")
            self.vector_store = FAISS.from_documents(
                documents,
                self.embeddings,
            )
            self.save()
            logger.info("Vector store initialized and saved")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def load(self):
        """Load existing vector store from disk"""
        try:
            logger.info(f"Loading vector store from {self.db_path}...")
            self.vector_store = FAISS.load_local(
                self.db_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("Vector store loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load vector store: {e}")
            return False
        return True
    
    def save(self):
        """Save vector store to disk"""
        try:
            if self.vector_store:
                Path(self.db_path).mkdir(parents=True, exist_ok=True)
                self.vector_store.save_local(self.db_path)
                logger.info(f"Vector store saved to {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search similar documents using FAISS
        
        Args:
            query: Search query (problem statement)
            top_k: Number of results
            
        Returns:
            List of relevant document chunks with similarity scores
        """
        try:
            if not self.vector_store:
                logger.warning("Vector store not initialized")
                return []
            
            # similarity_search_with_score returns (doc, distance)
            # In FAISS with L2: smaller distance = more similar
            # Convert L2 distance to similarity: similarity = 1 / (1 + distance)
            results = self.vector_store.similarity_search_with_score(
                query,
                k=top_k,
            )
            
            documents = []
            for doc, distance in results:
                # Convert L2 distance to similarity score [0, 1]
                # Using formula: similarity = 1 / (1 + distance)
                similarity = 1.0 / (1.0 + distance)
                
                # Only include if similarity meets threshold
                if similarity >= settings.similarity_threshold:
                    documents.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_score": similarity,
                        "distance": distance,
                    })
            
            logger.info(
                f"Search returned {len(documents)} relevant documents "
                f"(threshold: {settings.similarity_threshold:.2f})"
            )
            return documents
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to existing vector store"""
        try:
            if self.vector_store is None:
                self.initialize_from_knowledge_base(documents)
            else:
                self.vector_store.add_documents(documents)
                self.save()
                logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
    
    def delete_documents(self, document_ids: List[str]):
        """Delete documents by ID"""
        try:
            if self.vector_store:
                logger.info(f"Marked {len(document_ids)} documents for deletion")
                self.save()
        except Exception as e:
            logger.error(f"Delete failed: {e}")
