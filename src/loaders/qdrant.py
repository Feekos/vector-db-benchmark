"""Загрузчик для Qdrant"""
import os
import sys
from typing import Any, Dict, List, Optional
import logging
import numpy as np

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.loaders.base import BaseLoader
from src.utils import logger, Timer

try:
    from qdrant_client import QdrantClient, models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("qdrant-client не установлен. Qdrant недоступен.")


class QdrantLoader(BaseLoader):
    """Загрузчик данных в Qdrant"""
    
    def __init__(self, config: Dict[str, Any], db_config: Dict[str, Any]):
        super().__init__(config, db_config)
        self.collection_name = db_config.get('collection_name', 'ru_rag_test')
        self.index_params = db_config.get('index', {})
        self.client = None
        
    def connect(self) -> bool:
        """Подключение к Qdrant"""
        if not QDRANT_AVAILABLE:
            logger.error("qdrant-client не установлен")
            return False
            
        try:
            self.client = QdrantClient(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 6333),
                prefer_grpc=False,  # Используем HTTP для загрузки
                check_compatibility=False
            )
            # Проверка соединения
            self.client.get_collections()
            self.connected = True
            logger.info(f"Подключено к Qdrant: {self.db_config.get('host')}")
            return True
        except Exception as e:
            logger.error(f"Ошибка подключения к Qdrant: {e}")
            return False
    
    def disconnect(self) -> None:
        """Закрытие соединения"""
        if self.client:
            self.client = None
            self.connected = False
            logger.info("Отключено от Qdrant")
    
    def create_schema(self) -> bool:
        """Создание коллекции и настройка индекса в Qdrant"""
        if not self.client:
            return False
            
        try:
            emb_dim = self.config['benchmark']['embedding']['dim']
            index_cfg = self.index_params
            
            # Настройка квантизации для экономии памяти
            quant_config = None
            quant_type = index_cfg.get('quantization', 'none')
            if quant_type == 'binary':
                quant_config = models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(
                        always_ram=index_cfg.get('always_ram', True)
                    )
                )
            elif quant_type == 'product':
                quant_config = models.ProductQuantization(
                    product=models.ProductQuantizationConfig(
                        compression=index_cfg.get('compression', 8),
                        always_ram=index_cfg.get('always_ram', True)
                    )
                )
            
            # Создание/пересоздание коллекции
            with Timer("Создание коллекции и индекса Qdrant") as timer:
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=emb_dim,
                        distance=models.Distance.COSINE,
                        hnsw_config=models.HnswConfigDiff(
                            m=index_cfg.get('params', {}).get('m', 16),
                            ef_construct=index_cfg.get('params', {}).get('ef_construct', 200)
                        ),
                        quantization_config=quant_config,
                        on_disk=index_cfg.get('on_disk', False)
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        default_segment_number=4,
                        memmap_threshold=20000 if index_cfg.get('on_disk') else None
                    )
                )
            
            self.index_build_time_seconds = timer.elapsed
            
            self.collection_ready = True
            logger.info(f"Коллекция создана: {self.collection_name} (индекс: {timer})")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка создания схемы Qdrant: {e}")
            return False
    
    def load_data(
        self, 
        df: Any,
        batch_size: int = 500,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Загрузка данных в Qdrant"""
        if not self.client or not self.collection_ready:
            raise RuntimeError("Сначала подключитесь и создайте коллекцию")
        
        stats = {
            "inserted": 0,
            "errors": 0,
            "batches": 0
        }
        
        cols = self.config['benchmark']['datasets'][0]
        emb_dim = self.config['benchmark']['embedding']['dim']
        total = len(df)
        
        for i in range(0, total, batch_size):
            batch = df.iloc[i:i+batch_size]
            stats["batches"] += 1
            
            try:
                # Подготовка точек для вставки
                points = []
                batch_errors = 0
                
                for idx, row in batch.iterrows():
                    # Преобразуем embedding в список
                    embedding = row['embedding']
                    if isinstance(embedding, list):
                        vector = embedding
                    elif hasattr(embedding, 'tolist'):
                        vector = embedding.tolist()
                    else:
                        vector = list(embedding) if embedding is not None else []
                    
                    # Проверяем что вектор имеет правильный размер
                    if len(vector) != emb_dim:
                        batch_errors += 1
                        continue
                    
                    point_id = int(idx) if isinstance(idx, (int, np.integer)) else hash(str(idx)) % (2**63)
                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector=vector,
                            payload={
                                "context_id": str(idx),
                                "content": str(row.get(cols['context_column'], '')),
                                "file_id": str(row.get(cols.get('file_column', ''), '')),
                                "answer": str(row.get(cols.get('answer_column', ''), ''))
                            }
                        )
                    )
                
                stats["errors"] += batch_errors
                
                if points:
                    logger.info(f"Qdrant batch {i}: trying to insert {len(points)} points (skipped {batch_errors})")
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                        wait=True
                    )
                    stats["inserted"] += len(points)
                else:
                    logger.warning(f"Qdrant batch {i}: no valid points to insert (all {len(batch)} skipped)")
                
            except Exception as e:
                logger.error(f"Ошибка при вставке батча {i}: {e}")
                stats["errors"] += 1
            
            # Прогресс
            if progress_callback:
                progress_callback(i + len(batch), total)
            elif (i // batch_size) % 10 == 0:
                logger.info(f"Загружено {min(i+batch_size, total)}/{total}")
        
        # Оптимизация коллекции после загрузки
        try:
            self.client.update_collection(
                collection_name=self.collection_name,
                optimizer_config=models.OptimizersConfigDiff(
                    indexing_threshold=20000
                )
            )
            logger.info("Коллекция оптимизирована")
        except Exception as e:
            logger.warning(f"Не удалось оптимизировать коллекцию: {e}")
        
        return stats
    
    def count_records(self) -> int:
        """Количество записей в коллекции"""
        if not self.client:
            return 0
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except:
            return 0
    
    def clear_collection(self) -> bool:
        """Удаление коллекции"""
        if not self.client:
            return False
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Коллекция {self.collection_name} удалена")
            self.collection_ready = False
            return True
        except Exception as e:
            logger.error(f"Ошибка удаления коллекции: {e}")
            return False