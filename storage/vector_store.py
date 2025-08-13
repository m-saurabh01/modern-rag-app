"""
Abstract base class for vector stores.
Provides common interface for ChromaDB, Qdrant, and other vector databases.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import uuid


@dataclass
class VectorDocument:
    """Document with vector embedding and metadata."""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class SearchResult:
    """Search result with relevance score."""
    document: VectorDocument
    score: float
    distance: float


@dataclass
class CollectionInfo:
    """Vector collection information."""
    name: str
    count: int
    dimension: int
    metadata: Dict[str, Any]


class VectorStore(ABC):
    """
    Abstract base class for vector database implementations.
    
    Provides a common interface for different vector databases while allowing
    implementation-specific optimizations.
    """
    
    def __init__(self, collection_name: str):
        """
        Initialize vector store with collection name.
        
        Args:
            collection_name: Name of the collection/index to use
        """
        self.collection_name = collection_name
    
    @abstractmethod
    async def create_collection(
        self, 
        dimension: int, 
        distance_metric: str = "cosine",
        **kwargs
    ) -> bool:
        """
        Create a new collection/index.
        
        Args:
            dimension: Vector dimension
            distance_metric: Distance metric (cosine, euclidean, dot)
            **kwargs: Implementation-specific parameters
            
        Returns:
            True if created successfully, False if already exists
        """
        pass
    
    @abstractmethod
    async def collection_exists(self) -> bool:
        """Check if collection exists."""
        pass
    
    @abstractmethod
    async def get_collection_info(self) -> Optional[CollectionInfo]:
        """Get collection information."""
        pass
    
    @abstractmethod
    async def add_documents(
        self, 
        documents: List[VectorDocument],
        batch_size: int = 100
    ) -> bool:
        """
        Add documents to the collection.
        
        Args:
            documents: List of documents to add
            batch_size: Batch size for bulk operations
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            metadata_filter: Metadata filtering conditions
            
        Returns:
            List of search results ordered by relevance
        """
        pass
    
    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[VectorDocument]:
        """Get document by ID."""
        pass
    
    @abstractmethod
    async def update_document(self, document: VectorDocument) -> bool:
        """Update existing document."""
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete document by ID."""
        pass
    
    @abstractmethod
    async def delete_collection(self) -> bool:
        """Delete the entire collection."""
        pass
    
    @abstractmethod
    async def count_documents(self) -> int:
        """Get total number of documents in collection."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check database health and connectivity."""
        pass
    
    async def bulk_search(
        self,
        query_embeddings: List[List[float]],
        limit: int = 10,
        score_threshold: float = 0.0
    ) -> List[List[SearchResult]]:
        """
        Perform bulk search operations.
        
        Default implementation uses individual searches.
        Implementations can override for better performance.
        
        Args:
            query_embeddings: List of query vectors
            limit: Maximum results per query
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results for each query
        """
        results = []
        for query_embedding in query_embeddings:
            query_results = await self.search(
                query_embedding=query_embedding,
                limit=limit,
                score_threshold=score_threshold
            )
            results.append(query_results)
        return results
