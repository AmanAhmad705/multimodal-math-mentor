"""
RAG (Retrieval-Augmented Generation) utilities for Math Mentor
Handles vector store operations, embeddings, retrieval, and synchronization
"""

import logging
import json
import hashlib
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ==================== Data Models ====================

@dataclass
class Document:
    """Represents a document in the knowledge base"""
    doc_id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict = None
    created_at: datetime = None
    
    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class RetrievalResult:
    """Result of a retrieval query"""
    doc_id: str
    content: str
    similarity_score: float
    metadata: Dict = None


# ==================== Embedding Manager ====================

class EmbeddingManager:
    """Manages embeddings for documents and queries"""
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize embedding manager
        
        Args:
            embedding_model_name: HuggingFace model identifier
        """
        self.model_name = embedding_model_name
        self.model_version = self._extract_version(embedding_model_name)
        self.model = None
        self._load_model()
    
    def _extract_version(self, model_name: str) -> str:
        """Extract version from model name"""
        return hashlib.md5(model_name.encode()).hexdigest()[:8]
    
    def _load_model(self):
        """Load embedding model (lazy-loaded)"""
        try:
            # Try to import sentence_transformers
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not available, using mock embeddings")
            self.model = None
    
    def embed_text(self, text: str) -> List[float]:
        """
        Embed text using the model
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if self.model:
            try:
                embedding = self.model.encode(text, convert_to_numpy=True)
                return embedding.tolist()
            except Exception as e:
                logger.error(f"Failed to embed text: {e}")
                return self._mock_embedding(text)
        else:
            return self._mock_embedding(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if self.model:
            try:
                embeddings = self.model.encode(texts, convert_to_numpy=True)
                return embeddings.tolist()
            except Exception as e:
                logger.error(f"Failed to embed texts: {e}")
                return [self._mock_embedding(t) for t in texts]
        else:
            return [self._mock_embedding(t) for t in texts]
    
    def _mock_embedding(self, text: str) -> List[float]:
        """
        Generate mock embedding for testing/fallback
        (based on text hash)
        """
        np.random.seed(int(hashlib.md5(text.encode()).hexdigest(), 16) % 2**32)
        return np.random.randn(384).tolist()  # MPNET dimension


# ==================== Vector Store ====================

class VectorStore:
    """In-memory vector store with similarity search"""
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize vector store
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.documents: Dict[str, Document] = {}
        self.vectors: Dict[str, np.ndarray] = {}
        self.index = None
    
    def add_document(self, doc: Document) -> bool:
        """
        Add document to vector store
        
        Args:
            doc: Document to add
            
        Returns:
            True if added successfully
        """
        try:
            if doc.embedding is None:
                logger.error(f"Document {doc.doc_id} has no embedding")
                return False
            
            # Store document
            self.documents[doc.doc_id] = doc
            
            # Store vector
            self.vectors[doc.doc_id] = np.array(doc.embedding)
            
            logger.debug(f"Added document {doc.doc_id} to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return False
    
    def add_documents(self, docs: List[Document]) -> int:
        """
        Add multiple documents to vector store
        
        Args:
            docs: List of documents to add
            
        Returns:
            Number of documents added successfully
        """
        count = 0
        for doc in docs:
            if self.add_document(doc):
                count += 1
        
        return count
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.5,
    ) -> List[RetrievalResult]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of retrieval results
        """
        if not self.vectors:
            return []
        
        try:
            query_vec = np.array(query_embedding)
            
            # Compute similarities
            similarities = {}
            for doc_id, doc_vec in self.vectors.items():
                # Cosine similarity
                similarity = np.dot(query_vec, doc_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(doc_vec) + 1e-8
                )
                
                if similarity >= similarity_threshold:
                    similarities[doc_id] = similarity
            
            # Sort by similarity
            sorted_docs = sorted(
                similarities.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
            # Build results
            results = []
            for doc_id, similarity in sorted_docs:
                doc = self.documents[doc_id]
                results.append(
                    RetrievalResult(
                        doc_id=doc.doc_id,
                        content=doc.content,
                        similarity_score=float(similarity),
                        metadata=doc.metadata,
                    )
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove document from vector store"""
        try:
            if doc_id in self.documents:
                del self.documents[doc_id]
                del self.vectors[doc_id]
                logger.debug(f"Removed document {doc_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove document: {e}")
            return False
    
    def clear(self):
        """Clear all documents from vector store"""
        self.documents.clear()
        self.vectors.clear()
        logger.info("Vector store cleared")
    
    def size(self) -> int:
        """Get number of documents in store"""
        return len(self.documents)


# ==================== Knowledge Base Manager ====================

class KnowledgeBaseManager:
    """Manages knowledge base documents and synchronization"""
    
    def __init__(self, memory_store, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialize knowledge base manager
        
        Args:
            memory_store: Persistent memory store
            vector_store: In-memory vector store
            embedding_manager: Embedding model
        """
        self.memory_store = memory_store
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
    
    def load_from_file(self, json_path: str) -> int:
        """
        Load knowledge base from JSON file
        
        Args:
            json_path: Path to knowledge base JSON
            
        Returns:
            Number of documents loaded
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            documents = data.get("documents", [])
            loaded = 0
            
            for doc_data in documents:
                doc_id = doc_data.get("id")
                content = doc_data.get("content")
                metadata = doc_data.get("metadata", {})
                
                if not doc_id or not content:
                    continue
                
                # Embed content
                embedding = self.embedding_manager.embed_text(content)
                
                # Create document
                doc = Document(
                    doc_id=doc_id,
                    content=content,
                    embedding=embedding,
                    metadata=metadata,
                    created_at=datetime.utcnow(),
                )
                
                # Add to vector store
                if self.vector_store.add_document(doc):
                    loaded += 1
            
            logger.info(f"Loaded {loaded} documents from {json_path}")
            return loaded
            
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            return 0
    
    def add_learned_document(
        self,
        problem_id: str,
        solution: str,
        metadata: Dict = None,
    ) -> bool:
        """
        Add a learned solution to knowledge base
        
        Args:
            problem_id: Problem identifier
            solution: Solution text
            metadata: Associated metadata
            
        Returns:
            True if added successfully
        """
        try:
            doc_id = f"learned_{problem_id}"
            
            # Embed solution
            embedding = self.embedding_manager.embed_text(solution)
            
            # Create document
            doc = Document(
                doc_id=doc_id,
                content=solution,
                embedding=embedding,
                metadata=metadata or {},
                created_at=datetime.utcnow(),
            )
            
            # Add to vector store
            if self.vector_store.add_document(doc):
                # Also persist to DB
                self.memory_store.log_learned_document(
                    doc_id, problem_id, solution, metadata
                )
                logger.info(f"Added learned document: {doc_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to add learned document: {e}")
            return False
    
    def reconcile_vector_store(self) -> Dict:
        """
        Reconcile in-memory vector store with persistent DB
        Handles model version changes and rebuilds
        
        Returns:
            Reconciliation report
        """
        logger.info("Starting vector store reconciliation...")
        
        report = {
            "documents_in_db": 0,
            "documents_in_store": 0,
            "added": 0,
            "removed": 0,
            "model_changed": False,
        }
        
        try:
            # Check if embedding model has changed
            current_version = self.embedding_manager.model_version
            recorded_version = self.memory_store.get_embedding_model_version(
                "vector_embedding_model"
            )
            
            if recorded_version and recorded_version != current_version:
                logger.warning(
                    f"Embedding model changed: {recorded_version} → {current_version}"
                )
                report["model_changed"] = True
                
                # Clear vector store and rebuild
                self.vector_store.clear()
                
                # Rebuild from DB
                learned_docs = self.memory_store.get_learned_documents()
                report["documents_in_db"] = len(learned_docs)
                
                for doc in learned_docs:
                    # Re-embed with new model
                    embedding = self.embedding_manager.embed_text(doc["content"])
                    
                    doc_obj = Document(
                        doc_id=doc["doc_id"],
                        content=doc["content"],
                        embedding=embedding,
                        metadata=doc.get("metadata", {}),
                    )
                    
                    if self.vector_store.add_document(doc_obj):
                        report["added"] += 1
            
            else:
                # Normal reconciliation (no model change)
                
                # Get DB documents
                db_docs = self.memory_store.get_learned_documents()
                report["documents_in_db"] = len(db_docs)
                
                # Get store documents
                store_doc_ids = set(self.vector_store.documents.keys())
                db_doc_ids = {doc["doc_id"] for doc in db_docs}
                report["documents_in_store"] = len(store_doc_ids)
                
                # Add missing DB docs
                for db_doc in db_docs:
                    if db_doc["doc_id"] not in store_doc_ids:
                        embedding = self.embedding_manager.embed_text(
                            db_doc["content"]
                        )
                        
                        doc_obj = Document(
                            doc_id=db_doc["doc_id"],
                            content=db_doc["content"],
                            embedding=embedding,
                            metadata=db_doc.get("metadata", {}),
                        )
                        
                        if self.vector_store.add_document(doc_obj):
                            report["added"] += 1
                
                # Remove orphan store docs
                for store_doc_id in store_doc_ids:
                    if store_doc_id not in db_doc_ids:
                        if self.vector_store.remove_document(store_doc_id):
                            report["removed"] += 1
            
            # Update model version
            self.memory_store.set_embedding_model_version(
                "vector_embedding_model", current_version
            )
            
            logger.info(f"Reconciliation complete: {report}")
            return report
            
        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            report["error"] = str(e)
            return report


# ==================== RAG Retriever ====================

class RAGRetriever:
    """Main RAG retriever interface"""
    
    def __init__(
        self,
        memory_store,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
        kb_manager: KnowledgeBaseManager,
    ):
        """
        Initialize RAG retriever
        
        Args:
            memory_store: Persistent memory
            vector_store: Vector store for similarity search
            embedding_manager: Embedding model
            kb_manager: Knowledge base manager
        """
        self.memory_store = memory_store
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.kb_manager = kb_manager
        self.last_reconcile = datetime.utcnow()
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query text
            top_k: Number of results
            similarity_threshold: Minimum similarity
            
        Returns:
            List of retrieval results
        """
        try:
            # Reconcile periodically (not on every retrieve)
            now = datetime.utcnow()
            if (now - self.last_reconcile).total_seconds() > 3600:  # 1 hour
                self.kb_manager.reconcile_vector_store()
                self.last_reconcile = now
            
            # Embed query
            query_embedding = self.embedding_manager.embed_text(query)
            
            # Search
            results = self.vector_store.search(
                query_embedding,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
            
            logger.debug(
                f"Retrieved {len(results)} documents for query "
                f"(threshold: {similarity_threshold})"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def learn_solution(
        self,
        problem_id: str,
        solution: str,
        metadata: Dict = None,
    ) -> bool:
        """
        Learn from a successfully solved problem
        
        Args:
            problem_id: Problem identifier
            solution: Solution text
            metadata: Associated metadata
            
        Returns:
            True if learned successfully
        """
        return self.kb_manager.add_learned_document(
            problem_id, solution, metadata
        )


# ==================== Utility Functions ====================

def compute_document_hash(content: str) -> str:
    """Compute hash of document content"""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def chunk_document(content: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split document into overlapping chunks
    
    Args:
        content: Document content
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of chunks
    """
    chunks = []
    start = 0
    
    while start < len(content):
        end = min(start + chunk_size, len(content))
        chunks.append(content[start:end])
        
        # Move start with overlap
        start = end - overlap
    
    return chunks


def filter_results_by_metadata(
    results: List[RetrievalResult],
    metadata_filter: Dict,
) -> List[RetrievalResult]:
    """
    Filter retrieval results by metadata
    
    Args:
        results: Retrieval results
        metadata_filter: Filter criteria
        
    Returns:
        Filtered results
    """
    filtered = []
    
    for result in results:
        if result.metadata is None:
            continue
        
        match = True
        for key, value in metadata_filter.items():
            if result.metadata.get(key) != value:
                match = False
                break
        
        if match:
            filtered.append(result)
    
    return filtered
