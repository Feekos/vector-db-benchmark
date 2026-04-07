"""Модуль поисковых адаптеров для векторных БД"""
from .base import BaseSearcher
from .pgvectorscale import PGVectorScaleSearcher
from .milvus import MilvusSearcher
from .qdrant import QdrantSearcher

__all__ = ['BaseSearcher', 'PGVectorScaleSearcher', 'MilvusSearcher', 'QdrantSearcher']