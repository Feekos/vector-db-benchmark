"""Загрузчик для pgvectorscale (PostgreSQL extension)"""
import os
import sys
from typing import Any, Dict, List, Optional
import logging
import json

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.loaders.base import BaseLoader
from src.utils import logger

try:
    import psycopg
    from pgvector.psycopg import register_vector
    PG_AVAILABLE = True
except ImportError:
    PG_AVAILABLE = False
    logger.warning("pgvector/psycopg не установлен. pgvectorscale недоступен.")


class PGVectorScaleLoader(BaseLoader):
    """Загрузчик данных в pgvectorscale"""
    
    def __init__(self, config: Dict[str, Any], db_config: Dict[str, Any]):
        super().__init__(config, db_config)
        self.conn = None
        self.table_name = db_config.get('table_name', 'wiki_chunks')
        self.index_params = db_config.get('index', {}).get('params', {})
        
    def connect(self) -> bool:
        """Подключение к PostgreSQL"""
        if not PG_AVAILABLE:
            logger.error("pgvector не установлен")
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
            logger.info(f"Подключено к PostgreSQL: {self.db_config.get('host')}")
            return True
        except Exception as e:
            logger.error(f"Ошибка подключения к PostgreSQL: {e}")
            return False
    
    def disconnect(self) -> None:
        """Закрытие соединения"""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.connected = False
            logger.info("Отключено от PostgreSQL")
    
    def create_schema(self) -> bool:
        """Создание таблицы и индекса pgvectorscale"""
        if not self.conn:
            return False
            
        try:
            with self.conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                
                # Создание таблицы
                emb_dim = self.config['benchmark']['embedding']['dim']
                cur.execute(f"""
                    DROP TABLE IF EXISTS {self.table_name} CASCADE
                """)
                cur.execute(f"""
                    CREATE TABLE {self.table_name} (
                        id SERIAL PRIMARY KEY,
                        context_id TEXT UNIQUE,
                        content TEXT,
                        file_id TEXT,
                        embedding vector({emb_dim})
                    )
                """)
                
                # Создание индекса для ускорения поиска
                index_type = self.index_params.get('type', 'ivfflat')
                if index_type == 'hnsw':
                    cur.execute(f"""
                        CREATE INDEX idx_{self.table_name}_embedding 
                        ON {self.table_name} 
                        USING hnsw (embedding vector_cosine_ops)
                        WITH (m = {self.index_params.get('m', 16)}, ef_construction = {self.index_params.get('ef_construction', 64)})
                    """)
                else:
                    cur.execute(f"""
                        CREATE INDEX idx_{self.table_name}_embedding 
                        ON {self.table_name} 
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100)
                    """)
                
                self.conn.commit()
                self.collection_ready = True
                logger.info(f"Таблица и индекс созданы: {self.table_name}")
                return True
                
        except Exception as e:
            logger.error(f"Ошибка создания схемы: {e}")
            if self.conn:
                self.conn.rollback()
            return False
    
    def load_data(
        self, 
        df: Any,
        batch_size: int = 500,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Загрузка данных в pgvectorscale"""
        if not self.conn or not self.collection_ready:
            raise RuntimeError("Сначала подключитесь и создайте схему")
        
        stats = {
            "inserted": 0,
            "skipped": 0,
            "errors": 0,
            "batches": 0
        }
        
        # Получение имён колонок из конфига
        cols = self.config['benchmark']['datasets'][0]
        id_col = cols.get('id_column') or df.index.name or 'index'
        
        total = len(df)
        
        for i in range(0, total, batch_size):
            batch = df.iloc[i:i+batch_size]
            stats["batches"] += 1
            
            try:
                with self.conn.cursor() as cur:
                    for idx, row in batch.iterrows():
                        context_id = str(idx) if id_col == 'index' else str(row.get(id_col, idx))
                        embedding = row.get('embedding')
                        
                        # Преобразуем embedding если это список
                        if isinstance(embedding, list):
                            embedding = embedding
                        elif hasattr(embedding, 'tolist'):
                            embedding = embedding.tolist()
                        
                        cur.execute(
                            f"""
                            INSERT INTO {self.table_name} 
                            (context_id, content, file_id, embedding)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (context_id) DO NOTHING
                            """,
                            (
                                context_id,
                                str(row.get(cols['context_column'], '')),
                                str(row.get(cols.get('file_column', ''), '')),
                                embedding
                            )
                        )
                    
                    self.conn.commit()
                    stats["inserted"] += len(batch)
                    
            except Exception as e:
                logger.error(f"Ошибка при вставке батча {i}: {e}")
                stats["errors"] += 1
                if self.conn:
                    self.conn.rollback()
            
            # Прогресс
            if progress_callback:
                progress_callback(i + len(batch), total)
            elif (i // batch_size) % 10 == 0:
                logger.info(f"Загружено {min(i+batch_size, total)}/{total}")
        
        # Оптимизация после загрузки
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"VACUUM ANALYZE {self.table_name}")
            logger.info("VACUUM ANALYZE выполнен")
        except Exception as e:
            logger.warning(f"Не удалось выполнить VACUUM: {e}")
        
        return stats
    
    def count_records(self) -> int:
        """Количество записей в таблице"""
        if not self.conn:
            return 0
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                return cur.fetchone()[0]
        except:
            return 0
    
    def clear_collection(self) -> bool:
        """Очистка таблицы"""
        if not self.conn:
            return False
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {self.table_name} RESTART IDENTITY")
            self.conn.commit()
            logger.info(f"Таблица {self.table_name} очищена")
            return True
        except Exception as e:
            logger.error(f"Ошибка очистки: {e}")
            return False