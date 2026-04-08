"""Загрузчик для Milvus"""
import os
import sys
from typing import Any, Dict, List, Optional
import logging
import numpy as np

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.loaders.base import BaseLoader
from src.utils import logger, Timer

try:
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logger.warning("pymilvus не установлен. Milvus недоступен.")


class MilvusLoader(BaseLoader):
    """Загрузчик данных в Milvus"""
    
    def __init__(self, config: Dict[str, Any], db_config: Dict[str, Any]):
        super().__init__(config, db_config)
        self.collection_name = db_config.get('collection_name', 'ru_rag_test')
        self.index_params = db_config.get('index', {})
        self.collection = None
        
    def connect(self) -> bool:
        """Подключение к Milvus"""
        if not MILVUS_AVAILABLE:
            logger.error("pymilvus не установлен")
            return False
            
        try:
            connections.connect(
                "default",
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 19530)
            )
            self.connected = True
            logger.info(f"✓ Подключено к Milvus: {self.db_config.get('host')}")
            return True
        except Exception as e:
            logger.error(f"Ошибка подключения к Milvus: {e}")
            return False
    
    def disconnect(self) -> None:
        """Закрытие соединения"""
        if self.connected:
            try:
                connections.disconnect("default")
            except:
                pass
            self.connected = False
            logger.info("Отключено от Milvus")
    
    def create_schema(self) -> bool:
        """Создание коллекции и индекса в Milvus"""
        if not self.connected:
            return False
            
        try:
            # Удаление существующей коллекции для чистого теста
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"Удалена существующая коллекция: {self.collection_name}")
            
            # Схема коллекции
            emb_dim = self.config['benchmark']['embedding']['dim']
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="context_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=emb_dim)
            ]
            schema = CollectionSchema(fields, "RAG benchmark collection")
            
            # Создание коллекции
            self.collection = Collection(self.collection_name, schema)
            
            # Создание индекса
            with Timer("Создание индекса Milvus") as timer:
                index_params = {
                    "metric_type": self.index_params.get('metric_type', 'COSINE'),
                    "index_type": self.index_params.get('type', 'HNSW'),
                    "params": self.index_params.get('params', {"M": 16, "efConstruction": 200})
                }
                self.collection.create_index("embedding", index_params)
            
            self.index_build_time_seconds = timer.elapsed
            
            # Загрузка в память для поиска
            self.collection.load()
            
            self.collection_ready = True
            logger.info(f"Коллекция создана: {self.collection_name} (индекс: {timer})")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка создания схемы Milvus: {e}")
            return False
    
    def load_data(
        self, 
        df: Any,
        batch_size: int = 500,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Загрузка данных в Milvus"""
        if not self.collection or not self.collection_ready:
            raise RuntimeError("Сначала подключитесь и создайте коллекцию")
        
        stats = {
            "inserted": 0,
            "errors": 0,
            "batches": 0
        }
        
        cols = self.config['benchmark']['datasets'][0]
        total = len(df)
        
        for i in range(0, total, batch_size):
            batch = df.iloc[i:i+batch_size]
            stats["batches"] += 1
            
            try:
                # Подготовка эмбеддингов
                embeddings = []
                for emb in batch['embedding'].values:
                    if isinstance(emb, list):
                        embeddings.append(emb)
                    else:
                        embeddings.append(emb.tolist() if hasattr(emb, 'tolist') else list(emb))
                
                # Получаем файлы если они есть
                file_col = cols.get('file_column', '')
                file_ids = []
                if file_col and file_col in batch.columns:
                    file_ids = batch[file_col].fillna('').astype(str).tolist()
                else:
                    file_ids = [''] * len(batch)
                
                # Подготовка данных для вставки
                entities = [
                    [str(idx) for idx in batch.index],  # context_id
                    batch[cols['context_column']].fillna('').astype(str).tolist(),  # content
                    file_ids,  # file_id
                    embeddings  # embeddings as list of lists
                ]
                
                # Вставка
                self.collection.insert(entities)
                stats["inserted"] += len(batch)
                
                # Коммит каждые 10 батчей для экономии времени
                if stats["batches"] % 10 == 0:
                    self.collection.flush()
                    
            except Exception as e:
                logger.error(f"Ошибка при вставке батча {i}: {e}")
                stats["errors"] += 1
            
            # Прогресс
            if progress_callback:
                progress_callback(i + len(batch), total)
            elif (i // batch_size) % 10 == 0:
                logger.info(f"Загружено {min(i+batch_size, total)}/{total}")
        
        # Финальный флеш и сбор индекса
        try:
            self.collection.flush()
            # Пересоздание индекса после загрузки всех данных
            with Timer("Пересоздание индекса Milvus") as timer:
                index_params = {
                    "metric_type": self.index_params.get('metric_type', 'COSINE'),
                    "index_type": self.index_params.get('type', 'HNSW'),
                    "params": self.index_params.get('params', {})
                }
                self.collection.create_index("embedding", index_params)
            
            self.index_build_time_seconds += timer.elapsed
            self.collection.load()
            logger.info(f"Индекс пересоздан и загружен в память: {timer}")
        except Exception as e:
            logger.warning(f"Не удалось оптимизировать индекс: {e}")
        
        return stats
    
    def count_records(self) -> int:
        """Количество записей в коллекции"""
        if not self.collection:
            return 0
        try:
            return self.collection.num_entities
        except:
            return 0
    
    def clear_collection(self) -> bool:
        """Удаление коллекции"""
        if not self.connected:
            return False
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
            logger.info(f"Коллекция {self.collection_name} удалена")
            self.collection = None
            self.collection_ready = False
            return True
        except Exception as e:
            logger.error(f"Ошибка удаления коллекции: {e}")
            return False