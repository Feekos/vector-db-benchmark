"""Модуль загрузчиков данных в векторные БД"""
from .base import BaseLoader
from .pgvectorscale import PGVectorScaleLoader
from .milvus import MilvusLoader
from .qdrant import QdrantLoader

__all__ = ['BaseLoader', 'PGVectorScaleLoader', 'MilvusLoader', 'QdrantLoader']