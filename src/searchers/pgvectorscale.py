"""Поисковый адаптер для pgvectorscale"""
import os
import sys
import time
import random
from typing import Any, Dict, List, Optional
import numpy as np

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.searchers.base import BaseSearcher, SearchResult
from src.utils import logger

try:
    import psycopg
    from pgvector import Vector
    from pgvector.psycopg import register_vector
    PG_AVAILABLE = True
except ImportError:
    PG_AVAILABLE = False


class PGVectorScaleSearcher(BaseSearcher):
    """Поисковый адаптер для pgvectorscale"""
    
    def __init__(self, config: Dict[str, Any], db_config: Dict[str, Any]):
        super().__init__(config, db_config)
        self.conn = None
        self.table_name = db_config.get('table_name', 'wiki_chunks')
        self.ef_search = self.search_params.get('ef_search', 200)
        
    def connect(self) -> bool:
        if not PG_AVAILABLE:
            return False
        try:
            self.conn = psycopg.connect(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 5432),
                dbname=self.db_config.get('database', 'rag_benchmark'),
                user=self.db_config.get('user', 'postgres'),
                password=self.db_config.get('password', ''),
                autocommit=True
            )
            register_vector(self.conn)
            self.connected = True
            return True
        except Exception as e:
            logger.error(f"PGVectorScale connect error: {e}")
            return False
    
    def disconnect(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None
            self.connected = False
    
    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 10,
        timeout: Optional[float] = None
    ) -> SearchResult:
        start_time = time.time()
        
        try:
            with self.conn.cursor() as cur:
                if self.ef_search:
                    cur.execute(f"SET hnsw.ef_search = {int(self.ef_search)}")

                query_vector = Vector(query_embedding)

                # Поиск с использованием оператора <=> (cosine distance)
                cur.execute(
                    f"""
                    SELECT context_id, embedding <=> %s AS distance
                    FROM {self.table_name}
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (query_vector, query_vector, top_k)
                )
                results = cur.fetchall()
            
            latency = (time.time() - start_time) * 1000
            
            return SearchResult(
                query_idx=-1,  # заполняется в benchmark.py
                retrieved_ids=[str(r[0]) for r in results],
                scores=[float(1 - r[1]) for r in results],  # конвертируем distance в similarity
                latency_ms=latency
            )
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error(f"Search error: {e}")
            return SearchResult(
                query_idx=-1,
                retrieved_ids=[],
                scores=[],
                latency_ms=latency
            )
    
    def warmup(self, queries: int = 10) -> None:
        """Прогрев: случайные вектора"""
        if not self.connected:
            return
        emb_dim = self.config['benchmark']['embedding']['dim']
        for _ in range(queries):
            dummy = np.random.randn(emb_dim).tolist()
            self.search(dummy, top_k=5)
        logger.info(f"PGVectorScale: выполнено {queries} warmup запросов")