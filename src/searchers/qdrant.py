"""Поисковый адаптер для Qdrant"""
import httpx
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
        self.base_url = f"http://{self.db_config.get('host', 'localhost')}:{self.db_config.get('port', 6333)}"
        self.ef_search = self.search_params.get('ef_search', 200)
        
    def connect(self) -> bool:
        if not QDRANT_AVAILABLE:
            return False
        try:
            self.client = QdrantClient(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 6333),
                prefer_grpc=False,  # Используем HTTP для загрузки
                check_compatibility=False
            )
            # Проверка существования коллекции
            collections = [c.name for c in self.client.get_collections().collections]
            self.connected = self.collection_name in collections
            if self.connected:
                logger.info(f"Qdrant connected, collection {self.collection_name} exists")
            else:
                logger.error(f"Qdrant connected, but collection {self.collection_name} not found. Available: {collections}")
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
            query_vector = list(query_embedding)
            payload = {
                "vector": query_vector,
                "limit": top_k,
                "with_payload": ["context_id"],
                "params": {
                    "hnsw_ef": self.ef_search,
                    "exact": False
                }
            }
            request_timeout = timeout if timeout is not None else 30.0
            response = httpx.post(
                f"{self.base_url}/collections/{self.collection_name}/points/search",
                json=payload,
                timeout=request_timeout
            )
            response.raise_for_status()
            result = response.json().get("result", [])
            latency = (time.time() - start_time) * 1000

            retrieved_ids = []
            scores = []
            for hit in result:
                payload_data = hit.get("payload") or {}
                context_id = payload_data.get("context_id")
                if context_id is None:
                    context_id = hit.get("id")
                retrieved_ids.append(str(context_id))
                scores.append(float(hit.get("score", 0.0)))

            return SearchResult(
                query_idx=-1,
                retrieved_ids=retrieved_ids,
                scores=scores,
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