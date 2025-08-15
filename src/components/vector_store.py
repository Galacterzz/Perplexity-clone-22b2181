"""
Vector store implementation using FAISS for similarity search.
Updated for faiss-cpu v1.9.0 (August 2025).
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import faiss
import pickle
import os
from ..utils.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """FAISS-based vector store for document similarity search."""
    
    def __init__(self, dimension: int = None):
        """Initialize FAISS vector store."""
        self.dimension = dimension or config.embedding_dimension
        self.index = None
        self.documents = []  # Store original documents
        self.metadata = []   # Store document metadata
        self.is_trained = False
        
        # Initialize index
        self._initialize_index()
        
        logger.info(f"Initialized FAISS vector store with dimension {self.dimension}")
    
    def _initialize_index(self):
        """Initialize FAISS index."""
        # Use IndexFlatL2 for exact search (good for small to medium datasets)
        # For larger datasets, consider IndexIVFFlat or IndexHNSWFlat
        self.index = faiss.IndexFlatL2(self.dimension)
        self.is_trained = True
        logger.info("Initialized FAISS IndexFlatL2")
    
    def add_documents(
        self, 
        texts: List[str], 
        embeddings: List[List[float]], 
        metadata: List[Dict[str, Any]]
    ):
        """Add documents with their embeddings to the vector store."""
        if len(texts) != len(embeddings) or len(texts) != len(metadata):
            raise ValueError("Texts, embeddings, and metadata lists must have the same length")
        
        if not embeddings:
            logger.warning("No embeddings provided to add")
            return
        
        # Convert embeddings to numpy array with proper dtype
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Verify dimension
        if embeddings_array.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings_array.shape[1]} doesn't match expected {self.dimension}")
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Store documents and metadata
        self.documents.extend(texts)
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(texts)} documents to vector store. Total documents: {len(self.documents)}")
    
    def search(
        self, 
        query_embedding: List[float], 
        k: int = 5, 
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using embedding similarity."""
        if not self.documents:
            logger.warning("No documents in vector store")
            return []
        
        # Convert to numpy array with proper dtype
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Verify dimension
        if query_vector.shape[1] != self.dimension:
            raise ValueError(f"Query embedding dimension {query_vector.shape[1]} doesn't match expected {self.dimension}")
        
        # Search in FAISS index
        distances, indices = self.index.search(query_vector, min(k, len(self.documents)))
        
        # Process results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            # FAISS returns -1 for invalid indices
            if idx == -1:
                continue
            
            # Apply similarity threshold (for L2 distance, lower is better)
            if distance > similarity_threshold and similarity_threshold > 0:
                continue
            
            result = {
                "text": self.documents[idx],
                "metadata": self.metadata[idx].copy(),
                "similarity_score": float(distance),
                "rank": i + 1
            }
            results.append(result)
        
        logger.info(f"Found {len(results)} similar documents")
        return results
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents in the vector store."""
        return [
            {
                "text": text,
                "metadata": meta.copy()
            }
            for text, meta in zip(self.documents, self.metadata)
        ]
    
    def clear(self):
        """Clear all documents from the vector store."""
        self._initialize_index()
        self.documents.clear()
        self.metadata.clear()
        logger.info("Cleared vector store")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "total_documents": len(self.documents),
            "dimension": self.dimension,
            "index_type": type(self.index).__name__,
            "is_trained": self.is_trained,
            "total_vectors": self.index.ntotal if self.index else 0,
            "faiss_version": faiss.__version__ if hasattr(faiss, '__version__') else "unknown"
        }
    
    def save(self, filepath: str):
        """Save vector store to disk."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save documents and metadata
            data = {
                "documents": self.documents,
                "metadata": self.metadata,
                "dimension": self.dimension
            }
            
            with open(f"{filepath}.pkl", "wb") as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved vector store to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
    
    def load(self, filepath: str):
        """Load vector store from disk."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load documents and metadata
            with open(f"{filepath}.pkl", "rb") as f:
                data = pickle.load(f)
            
            self.documents = data["documents"]
            self.metadata = data["metadata"]
            self.dimension = data["dimension"]
            self.is_trained = True
            
            logger.info(f"Loaded vector store from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            # Initialize new index on error
            self._initialize_index()
    
    def similarity_search_with_filter(
        self, 
        query_embedding: List[float], 
        filter_criteria: Dict[str, Any], 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search with metadata filtering."""
        # Get all similar documents first
        all_results = self.search(query_embedding, k=len(self.documents))
        
        # Filter based on criteria
        filtered_results = []
        for result in all_results:
            metadata = result["metadata"]
            
            # Check if all filter criteria are met
            matches = True
            for key, value in filter_criteria.items():
                if key not in metadata or metadata[key] != value:
                    matches = False
                    break
            
            if matches:
                filtered_results.append(result)
                if len(filtered_results) >= k:
                    break
        
        logger.info(f"Filtered to {len(filtered_results)} results from {len(all_results)} candidates")
        return filtered_results
