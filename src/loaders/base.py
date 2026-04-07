"""Базовый класс для загрузчиков данных"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """Абстрактный базовый класс для загрузчиков в векторные БД"""
    
    def __init__(self, config: Dict[str, Any], db_config: Dict[str, Any]):
        """
        Инициализация загрузчика
        
        Args:
            config: общая конфигурация бенчмарка
            db_config: специфичная конфигурация для данной БД
        """
        self.config = config
        self.db_config = db_config
        self.connected = False
        self.collection_ready = False
        
    @abstractmethod
    def connect(self) -> bool:
        """Установка соединения с БД. Возвращает True при успехе."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Закрытие соединения с БД"""
        pass
    
    @abstractmethod
    def create_schema(self) -> bool:
        """Создание коллекции/таблицы и индекса. Возвращает True при успехе."""
        pass
    
    @abstractmethod
    def load_data(
        self, 
        df: Any,  # pandas DataFrame
        batch_size: int = 500,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Загрузка данных в БД
        
        Args:
            df: DataFrame с данными и эмбеддингами
            batch_size: размер батча для вставки
            progress_callback: функция для отображения прогресса
            
        Returns:
            Dict со статистикой загрузки
        """
        pass
    
    @abstractmethod
    def count_records(self) -> int:
        """Возвращает количество записей в коллекции"""
        pass
    
    @abstractmethod
    def clear_collection(self) -> bool:
        """Очистка коллекции/таблицы. Возвращает True при успехе."""
        pass
    
    def __enter__(self):
        """Контекстный менеджер: подключение"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер: отключение"""
        self.disconnect()
    
    def check_ready(self) -> bool:
        """Проверка готовности БД к загрузке"""
        if not self.connected:
            self.connect()
        
        if not self.collection_ready:
            self.collection_ready = self.create_schema()
        
        return self.connected and self.collection_ready
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики по коллекции"""
        return {
            "connected": self.connected,
            "collection_ready": self.collection_ready,
            "record_count": self.count_records() if self.connected else 0
        }