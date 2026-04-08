"""Поисковый адаптер для Milvus"""
import os
import sys
import time
import numpy as np
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.searchers.base import BaseSearcher, SearchResult
from src.utils import logger

try:
    from pymilvus import connections, Collection, utility
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False


class MilvusSearcher(BaseSearcher):
    """Поисковый адаптер для Milvus"""
    
    def __init__(self, config: Dict[str, Any], db_config: Dict[str, Any]):
        super().__init__(config, db_config)
        self.collection_name = db_config.get('collection_name', 'ru_rag_test')
        self.collection = None
        self.ef = self.search_params.get('ef', 200)
        
    def connect(self) -> bool:
        if not MILVUS_AVAILABLE:
            logger.error("Milvus недоступен: пакет pymilvus не установлен")
            return False
        try:
            connections.connect(
                "default",
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 19530)
            )
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.collection.load()
                self.connected = True
                return True
            return False
        except Exception as e:
            logger.error(f"Milvus connect error: {e}")
            return False
    
    def disconnect(self) -> None:
        if self.collection:
            self.collection.release()
        try:
            connections.disconnect("default")
        except:
            pass
        self.connected = False
    
    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 10,
        timeout: Optional[float] = None
    ) -> SearchResult:
        start_time = time.time()
        
        try:
            search_params = {
                "metric_type": self.db_config.get('index', {}).get('metric_type', 'COSINE'),
                "params": {"ef": self.ef}
            }
            
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["context_id"],
                timeout=timeout
            )
            
            latency = (time.time() - start_time) * 1000
            
            hits = results[0]
            return SearchResult(
                query_idx=-1,
                retrieved_ids=[str(hit.entity.get("context_id")) for hit in hits],
                scores=[float(hit.score) for hit in hits],
                latency_ms=latency
            )
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error(f"Milvus search error: {e}")
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
        logger.info(f"Milvus: выполнено {queries} warmup запросов")