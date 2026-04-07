"""
Vector DB Benchmark Package
Пакет для сравнительного тестирования векторных БД: pgvectorscale, Milvus, Qdrant
"""
__version__ = "1.0.0"

def __getattr__(name):
    if name == "load_config" or name == "logger" or name == "Timer":
        from .utils import load_config, logger, Timer
        return locals()[name]
    if name == "evaluate_search_results" or name == "AggregateMetrics":
        from .metrics import evaluate_search_results, AggregateMetrics
        return locals()[name]
    if name == "BenchmarkRunner":
        from .benchmark import BenchmarkRunner
        return BenchmarkRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "__version__",
    "load_config",
    "logger",
    "Timer",
    "evaluate_search_results",
    "AggregateMetrics",
    "BenchmarkRunner"
]