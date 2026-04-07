"""Расчёт метрик качества поиска"""
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchMetrics:
    """Метрики одного поискового запроса"""
    query_idx: int
    recall_at_1: Optional[float] = None
    recall_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None
    mrr: float = 0.0
    latency_ms: float = 0.0
    retrieved_ids: List[str] = None
    
    def __post_init__(self):
        if self.retrieved_ids is None:
            self.retrieved_ids = []


@dataclass
class AggregateMetrics:
    """Агрегированные метрики по всем запросам"""
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    qps: float = 0.0
    total_queries: int = 0
    successful_queries: int = 0
    
    def to_dict(self) -> Dict:
        return {k: round(v, 4) if isinstance(v, float) else v 
                for k, v in asdict(self).items()}


def calculate_recall(retrieved_ids: List[str], ground_truth_id: str, k: int) -> float:
    """
    Расчёт Recall@K
    Возвращает 1.0 если ground_truth_id есть в первых k результатах, иначе 0.0
    """
    if not retrieved_ids or not ground_truth_id:
        return 0.0
    return 1.0 if ground_truth_id in retrieved_ids[:k] else 0.0


def calculate_mrr(retrieved_ids: List[str], ground_truth_id: str) -> float:
    """
    Mean Reciprocal Rank для одного запроса
    Возвращает 1/rank если найден, иначе 0.0
    """
    if not ground_truth_id or ground_truth_id not in retrieved_ids:
        return 0.0
    rank = retrieved_ids.index(ground_truth_id) + 1
    return 1.0 / rank


def calculate_percentile(values: List[float], percentile: float) -> float:
    """Расчёт перцентиля из списка значений"""
    if not values:
        return 0.0
    return float(np.percentile(values, percentile))


def aggregate_metrics(
    metrics_list: List[SearchMetrics], 
    total_time_seconds: float,
    top_ks: List[int] = [1, 5, 10]
) -> AggregateMetrics:
    """Агрегация метрик по всем запросам"""
    if not metrics_list:
        return AggregateMetrics()
    
    # Извлечение значений
    recalls = {k: [] for k in top_ks}
    mrrs = []
    latencies = []
    successful = 0
    
    for m in metrics_list:
        if m.retrieved_ids:  # успешный запрос
            successful += 1
            for k in top_ks:
                recalls[k].append(getattr(m, f'recall_at_{k}', 0.0) or 0.0)
            mrrs.append(m.mrr)
            latencies.append(m.latency_ms)
    
    total = len(metrics_list)
    
    return AggregateMetrics(
        recall_at_1=np.mean(recalls.get(1, [0])) if recalls.get(1) else 0.0,
        recall_at_5=np.mean(recalls.get(5, [0])) if recalls.get(5) else 0.0,
        recall_at_10=np.mean(recalls.get(10, [0])) if recalls.get(10) else 0.0,
        mrr=np.mean(mrrs) if mrrs else 0.0,
        latency_p50=calculate_percentile(latencies, 50),
        latency_p95=calculate_percentile(latencies, 95),
        latency_p99=calculate_percentile(latencies, 99),
        qps=successful / total_time_seconds if total_time_seconds > 0 else 0.0,
        total_queries=total,
        successful_queries=successful
    )


def evaluate_search_results(
    retrieved_results: List[List[str]],  # list of retrieved_ids for each query
    ground_truth_ids: List[str],  # correct id for each query
    latencies_ms: List[float],
    top_ks: List[int] = [1, 5, 10]
) -> tuple[List[SearchMetrics], AggregateMetrics]:
    """
    Полная оценка результатов поиска
    
    Returns:
        Tuple of (list of per-query metrics, aggregated metrics)
    """
    if len(retrieved_results) != len(ground_truth_ids):
        raise ValueError("Несоответствие количества результатов и ground truth")
    
    if len(retrieved_results) != len(latencies_ms):
        raise ValueError("Несоответствие количества результатов и замеров времени")
    
    per_query_metrics = []
    
    for i, (retrieved, gt_id, latency) in enumerate(
        zip(retrieved_results, ground_truth_ids, latencies_ms)
    ):
        metrics = SearchMetrics(
            query_idx=i,
            retrieved_ids=retrieved,
            latency_ms=latency
        )
        
        # Расчёт recall для каждого K
        for k in top_ks:
            setattr(metrics, f'recall_at_{k}', calculate_recall(retrieved, gt_id, k))
        
        metrics.mrr = calculate_mrr(retrieved, gt_id)
        per_query_metrics.append(metrics)
    
    # Агрегация
    total_time = sum(latencies_ms) / 1000  # конвертируем в секунды
    aggregated = aggregate_metrics(per_query_metrics, total_time, top_ks)
    
    return per_query_metrics, aggregated