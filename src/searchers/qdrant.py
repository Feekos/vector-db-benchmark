"""Поисковый адаптер для Qdrant"""
import os
import sys
import time
import numpy as np
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.searchers.base import BaseSearcher, SearchResult
from src.utils import logger

try:
    from qdrant_client import QdrantClient, models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


class QdrantSearcher(BaseSearcher):
    """Поисковый адаптер для Qdrant"""
    
    def __init__(self, config: Dict[str, Any], db_config: Dict[str, Any]):
        super().__init__(config, db_config)
        self.collection_name = db_config.get('collection_name', 'ru_rag_test')
        self.client = None
        self.ef_search = self.search_params.get('ef_search', 200)
        
    def connect(self) -> bool:
        if not QDRANT_AVAILABLE:
            return False
        try:
            self.client = QdrantClient(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 6333),
                grpc=self.db_config.get('grpc', True)
            )
            # Проверка существования коллекции
            collections = [c.name for c in self.client.get_collections().collections]
            self.connected = self.collection_name in collections
            return self.connected
        except Exception as e:
            logger.error(f"Qdrant connect error: {e}")
            return False
    
    def disconnect(self) -> None:
        if self.client:
            self.client = None
        self.connected = False
    
    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 10,
        timeout: Optional[float] = None
    ) -> SearchResult:
        start_time = time.time()
        
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=["context_id"],
                params=models.SearchParams(
                    hnsw_ef=self.ef_search,
                    exact=False
                ),
                timeout=timeout
            )
            
            latency = (time.time() - start_time) * 1000
            
            return SearchResult(
                query_idx=-1,
                retrieved_ids=[str(hit.payload.get("context_id")) for hit in results],
                scores=[float(hit.score) for hit in results],
                latency_ms=latency
            )
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error(f"Qdrant search error: {e}")
            return SearchResult(
                query_idx=-1,
                retrieved_ids=[],
                scores=[],
                latency_ms=latency
            )
    
    def warmup(self, queries: int = 10) -> None:
        if not self.connected:
            return
        emb_dim = self.config['benchmark']['embedding']['dim']
        for _ in range(queries):
            dummy = np.random.randn(emb_dim).tolist()
            self.search(dummy, top_k=5)
        logger.info(f"Qdrant: выполнено {queries} warmup запросов")