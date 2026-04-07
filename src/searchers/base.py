"""Базовый класс для поисковых адаптеров"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Результат одного поискового запроса"""
    query_idx: int
    retrieved_ids: List[str]
    scores: List[float]
    latency_ms: float
    metadata: Optional[Dict] = None


class BaseSearcher(ABC):
    """Абстрактный базовый класс для поисковых адаптеров"""
    
    def __init__(self, config: Dict[str, Any], db_config: Dict[str, Any]):
        self.config = config
        self.db_config = db_config
        self.connected = False
        self.search_params = db_config.get('index', {}).get('params', {})
        
    @abstractmethod
    def connect(self) -> bool:
        """Установка соединения с БД"""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Закрытие соединения"""
        pass
    
    @abstractmethod
    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 10,
        timeout: Optional[float] = None
    ) -> SearchResult:
        """
        Выполнение поискового запроса
        
        Args:
            query_embedding: вектор запроса
            top_k: количество результатов
            timeout: таймаут в секундах
            
        Returns:
            SearchResult с ID, скорингом и метриками
        """
        pass
    
    @abstractmethod
    def warmup(self, queries: int = 10) -> None:
        """Прогрев кэшей БД тестовыми запросами"""
        pass
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()