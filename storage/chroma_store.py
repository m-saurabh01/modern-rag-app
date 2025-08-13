"""
ChromaDB implementation of the VectorStore interface.
Provides embedded vector database functionality with persistent storage.
"""

import asyncio
from typing import List, Dict, Any, Optional, Union
import uuid
from pathlib import Path
import json
import time

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from config import settings, get_logger, log_performance
from core.exceptions import VectorStoreError, VectorIndexError, VectorSearchError
from .vector_store import VectorStore, VectorDocument, SearchResult, CollectionInfo


class ChromaVectorStore(VectorStore):
    """
    ChromaDB implementation of vector store.
    
    Features:
    - Embedded database with persistent storage
    - Efficient similarity search with multiple distance metrics
    - Metadata filtering and hybrid search capabilities
    - Automatic collection management
    - Health monitoring and error recovery
    """
    
    def __init__(self, collection_name: Optional[str] = None):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the collection. Defaults to settings value.
        """
        collection_name = collection_name or settings.vector_db.chromadb_collection_name
        super().__init__(collection_name)
        
        self.logger = get_logger(__name__)
        self._client: Optional[chromadb.Client] = None
        self._collection: Optional[chromadb.Collection] = None
        
        # Connection settings
        self.persist_directory = Path(settings.vector_db.chromadb_persist_directory)
        self.host = settings.vector_db.chromadb_host
        self.port = settings.vector_db.chromadb_port
        
        # Performance tracking
        self._operation_stats = {
            'total_documents_added': 0,
            'total_searches': 0,
            'average_search_time': 0.0,
            'last_operation_time': 0.0
        }
        
        self.logger.info(
            "ChromaVectorStore initialized",
            collection_name=self.collection_name,
            persist_directory=str(self.persist_directory)
        )
    
    async def _get_client(self) -> chromadb.Client:
        """
        Get or create ChromaDB client.
        
        Returns:
            ChromaDB client instance
            
        Raises:
            VectorStoreError: If client creation fails
        """
        if self._client is None:
            try:
                # Ensure persist directory exists
                self.persist_directory.mkdir(parents=True, exist_ok=True)
                
                # Create client with persistent storage
                self._client = chromadb.PersistentClient(
                    path=str(self.persist_directory),
                    settings=ChromaSettings(
                        allow_reset=True,
                        anonymized_telemetry=False
                    )
                )
                
                self.logger.info(
                    "ChromaDB client created",
                    persist_directory=str(self.persist_directory)
                )
                
            except Exception as e:
                raise VectorStoreError(
                    f"Failed to create ChromaDB client: {str(e)}",
                    store_type="chromadb",
                    cause=e
                )
        
        return self._client
    
    async def _get_collection(self) -> chromadb.Collection:
        """
        Get or create collection.
        
        Returns:
            ChromaDB collection instance
            
        Raises:
            VectorStoreError: If collection access fails
        """
        if self._collection is None:
            client = await self._get_client()
            
            try:
                # Try to get existing collection
                self._collection = client.get_collection(name=self.collection_name)
                self.logger.debug(
                    "Retrieved existing collection",
                    collection_name=self.collection_name
                )
                
            except Exception:
                # Collection doesn't exist, will be created when needed
                self.logger.debug(
                    "Collection not found, will create when needed",
                    collection_name=self.collection_name
                )
        
        return self._collection
    
    @log_performance("create_collection")
    async def create_collection(
        self, 
        dimension: int, 
        distance_metric: str = "cosine",
        **kwargs
    ) -> bool:
        """
        Create a new collection.
        
        Args:
            dimension: Vector dimension (informational for ChromaDB)
            distance_metric: Distance metric (cosine, l2, ip)
            **kwargs: Additional ChromaDB parameters
            
        Returns:
            True if created, False if already exists
            
        Raises:
            VectorStoreError: If creation fails
        """
        client = await self._get_client()
        
        try:
            # Map distance metrics
            metric_mapping = {
                "cosine": "cosine",
                "euclidean": "l2", 
                "dot": "ip"
            }
            chroma_metric = metric_mapping.get(distance_metric, "cosine")
            
            # Create collection
            self._collection = client.create_collection(
                name=self.collection_name,
                metadata={
                    "dimension": dimension,
                    "distance_metric": distance_metric,
                    "created_at": time.time(),
                    **kwargs
                }
            )
            
            self.logger.info(
                "Collection created successfully",
                collection_name=self.collection_name,
                dimension=dimension,
                distance_metric=distance_metric
            )
            
            return True
            
        except Exception as e:
            if "already exists" in str(e).lower():
                self.logger.debug(
                    "Collection already exists",
                    collection_name=self.collection_name
                )
                return False
            else:
                raise VectorStoreError(
                    f"Failed to create collection '{self.collection_name}': {str(e)}",
                    store_type="chromadb",
                    cause=e
                )
    
    async def collection_exists(self) -> bool:
        """Check if collection exists."""
        try:
            client = await self._get_client()
            collections = client.list_collections()
            return any(col.name == self.collection_name for col in collections)
        except Exception as e:
            self.logger.error(
                "Failed to check collection existence",
                collection_name=self.collection_name,
                error=str(e)
            )
            return False
    
    async def get_collection_info(self) -> Optional[CollectionInfo]:
        """Get collection information."""
        try:
            if not await self.collection_exists():
                return None
            
            collection = await self._get_collection()
            count = collection.count()
            metadata = collection.metadata or {}
            
            return CollectionInfo(
                name=self.collection_name,
                count=count,
                dimension=metadata.get("dimension", 0),
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to get collection info",
                collection_name=self.collection_name,
                error=str(e)
            )
            return None
    
    @log_performance("add_documents")
    async def add_documents(
        self, 
        documents: List[VectorDocument],
        batch_size: int = 100
    ) -> bool:
        """
        Add documents to collection.
        
        Args:
            documents: Documents to add
            batch_size: Batch size for bulk operations
            
        Returns:
            True if successful
            
        Raises:
            VectorIndexError: If indexing fails
        """
        if not documents:
            return True
        
        try:
            # Ensure collection exists
            if not await self.collection_exists():
                # Create collection with dimension from first document
                dimension = len(documents[0].embedding)
                await self.create_collection(dimension=dimension)
            
            collection = await self._get_collection()
            
            # Process in batches
            total_added = 0
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Prepare batch data
                ids = [doc.id for doc in batch]
                embeddings = [doc.embedding for doc in batch]
                metadatas = []
                documents_text = []
                
                for doc in batch:
                    # Add content as document text
                    documents_text.append(doc.content)
                    
                    # Prepare metadata (ChromaDB requires string values)
                    metadata = {}
                    for key, value in doc.metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            metadata[key] = value
                        else:
                            metadata[key] = json.dumps(value)
                    
                    metadatas.append(metadata)
                
                # Add to collection
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents_text
                )
                
                total_added += len(batch)
                
                self.logger.debug(
                    "Added document batch",
                    batch_size=len(batch),
                    total_added=total_added,
                    total_documents=len(documents)
                )
            
            # Update statistics
            self._operation_stats['total_documents_added'] += total_added
            
            self.logger.info(
                "Documents added successfully",
                total_documents=total_added,
                collection_name=self.collection_name
            )
            
            return True
            
        except Exception as e:
            raise VectorIndexError(
                f"Failed to add {len(documents)} documents: {str(e)}",
                operation="add_documents",
                vector_count=len(documents),
                cause=e
            )
    
    @log_performance("vector_search")
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
            List of search results
            
        Raises:
            VectorSearchError: If search fails
        """
        try:
            start_time = time.time()
            collection = await self._get_collection()
            
            # Perform search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=metadata_filter,
                include=["metadatas", "documents", "distances"]
            )
            
            # Update performance stats
            search_time = time.time() - start_time
            self._operation_stats['total_searches'] += 1
            self._operation_stats['last_operation_time'] = search_time
            
            if self._operation_stats['total_searches'] > 0:
                self._operation_stats['average_search_time'] = (
                    (self._operation_stats['average_search_time'] * 
                     (self._operation_stats['total_searches'] - 1) + search_time) /
                    self._operation_stats['total_searches']
                )
            
            # Convert to SearchResult objects
            search_results = []
            
            if results['ids'] and results['ids'][0]:  # Check if we have results
                for i, doc_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i]
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    score = 1 - distance if distance <= 1 else 0
                    
                    # Skip if below threshold
                    if score < score_threshold:
                        continue
                    
                    # Parse metadata
                    metadata = results['metadatas'][0][i] or {}
                    parsed_metadata = {}
                    for key, value in metadata.items():
                        try:
                            # Try to parse JSON values
                            parsed_metadata[key] = json.loads(value) if isinstance(value, str) and value.startswith(('{', '[')) else value
                        except:
                            parsed_metadata[key] = value
                    
                    # Create VectorDocument
                    document = VectorDocument(
                        id=doc_id,
                        content=results['documents'][0][i] or "",
                        embedding=query_embedding,  # We don't store embeddings in the result
                        metadata=parsed_metadata
                    )
                    
                    search_results.append(SearchResult(
                        document=document,
                        score=score,
                        distance=distance
                    ))
            
            self.logger.debug(
                "Search completed",
                query_results=len(search_results),
                search_time_ms=round(search_time * 1000, 2),
                score_threshold=score_threshold
            )
            
            return search_results
            
        except Exception as e:
            raise VectorSearchError(
                f"Search failed: {str(e)}",
                query=str(query_embedding[:5]) + "..." if len(query_embedding) > 5 else str(query_embedding),
                collection_name=self.collection_name,
                cause=e
            )
    
    async def get_document(self, document_id: str) -> Optional[VectorDocument]:
        """Get document by ID."""
        try:
            collection = await self._get_collection()
            
            results = collection.get(
                ids=[document_id],
                include=["metadatas", "documents", "embeddings"]
            )
            
            if not results['ids'] or not results['ids'][0]:
                return None
            
            # Parse metadata
            metadata = results['metadatas'][0] or {}
            parsed_metadata = {}
            for key, value in metadata.items():
                try:
                    parsed_metadata[key] = json.loads(value) if isinstance(value, str) and value.startswith(('{', '[')) else value
                except:
                    parsed_metadata[key] = value
            
            return VectorDocument(
                id=document_id,
                content=results['documents'][0] or "",
                embedding=results['embeddings'][0] or [],
                metadata=parsed_metadata
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to get document",
                document_id=document_id,
                error=str(e)
            )
            return None
    
    async def update_document(self, document: VectorDocument) -> bool:
        """Update existing document."""
        try:
            collection = await self._get_collection()
            
            # Prepare metadata
            metadata = {}
            for key, value in document.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                else:
                    metadata[key] = json.dumps(value)
            
            # Update document
            collection.update(
                ids=[document.id],
                embeddings=[document.embedding],
                metadatas=[metadata],
                documents=[document.content]
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to update document",
                document_id=document.id,
                error=str(e)
            )
            return False
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete document by ID."""
        try:
            collection = await self._get_collection()
            collection.delete(ids=[document_id])
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to delete document",
                document_id=document_id,
                error=str(e)
            )
            return False
    
    async def delete_collection(self) -> bool:
        """Delete the entire collection."""
        try:
            client = await self._get_client()
            client.delete_collection(name=self.collection_name)
            self._collection = None
            
            self.logger.info(
                "Collection deleted successfully",
                collection_name=self.collection_name
            )
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to delete collection",
                collection_name=self.collection_name,
                error=str(e)
            )
            return False
    
    async def count_documents(self) -> int:
        """Get total number of documents."""
        try:
            collection = await self._get_collection()
            return collection.count()
        except Exception:
            return 0
    
    async def health_check(self) -> Dict[str, Any]:
        """Check database health and connectivity."""
        health = {
            "status": "unknown",
            "collection_exists": False,
            "document_count": 0,
            "last_operation_time": self._operation_stats.get('last_operation_time', 0),
            "total_searches": self._operation_stats.get('total_searches', 0),
            "error": None
        }
        
        try:
            # Test basic connectivity
            client = await self._get_client()
            
            # Check if collection exists and get info
            if await self.collection_exists():
                health["collection_exists"] = True
                info = await self.get_collection_info()
                if info:
                    health["document_count"] = info.count
            
            health["status"] = "healthy"
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
            
            self.logger.error(
                "Health check failed",
                error=str(e),
                collection_name=self.collection_name
            )
        
        return health
