"""Основной модуль запуска бенчмарка"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, ensure_dirs, save_results, get_timestamp, logger, Timer
from src.metrics import evaluate_search_results, AggregateMetrics
from src.searchers.pgvectorscale import PGVectorScaleSearcher
from src.searchers.milvus import MilvusSearcher
from src.searchers.qdrant import QdrantSearcher


class BenchmarkRunner:
    """Основной класс для запуска бенчмарка"""
    
    SEARCHERS = {
        "pgvectorscale": PGVectorScaleSearcher,
        "milvus": MilvusSearcher,
        "qdrant": QdrantSearcher
    }
    
    def __init__(self, config: Dict):
        self.config = config
        self.results = {}
        
    def run_single_db(
        self, 
        db_name: str,
        question_embeddings: np.ndarray,
        ground_truth_ids: List[str],
        output_dir: str
    ) -> Dict:
        """Запуск бенчмарка для одной БД"""
        db_config = self.config['benchmark']['databases'].get(db_name)
        if not db_config or not db_config.get('enabled', True):
            logger.info(f"⏭ {db_name}: отключено в конфигурации")
            return {}
        
        logger.info(f"\n🎯 Запуск бенчмарка: {db_name}")
        
        searcher_class = self.SEARCHERS.get(db_name)
        if not searcher_class:
            logger.error(f"Неизвестный searcher: {db_name}")
            return {}
        
        search_config = self.config['benchmark']['search']
        top_k = search_config.get('top_k', 10)
        warmup = search_config.get('warmup_queries', 10)
        timeout = search_config.get('timeout_seconds', 300)
        
        all_retrieved = []
        all_latencies = []
        
        with searcher_class(self.config, db_config) as searcher:
            if not searcher.connected:
                logger.error(f"{db_name}: не удалось подключиться")
                return {}
            
            # Прогрев
            logger.info("🔥 Прогрев кэшей...")
            searcher.warmup(warmup)
            
            # Основной тест
            logger.info(f"🔍 Выполнение {len(question_embeddings)} поисковых запросов...")
            
            with Timer("Поиск") as timer:
                for i, query_emb in enumerate(tqdm(question_embeddings, desc=f"{db_name} поиск")):
                    # Для отладки: ограничить количество запросов
                    # if i >= 100: break
                    
                    result = searcher.search(
                        query_embedding=query_emb.tolist(),
                        top_k=top_k,
                        timeout=timeout
                    )
                    result.query_idx = i
                    
                    all_retrieved.append(result.retrieved_ids)
                    all_latencies.append(result.latency_ms)
            
            # Оценка метрик
            logger.info("📊 Расчёт метрик...")
            per_query, aggregated = evaluate_search_results(
                retrieved_results=all_retrieved,
                ground_truth_ids=ground_truth_ids,
                latencies_ms=all_latencies,
                top_ks=[1, 5, 10]
            )
            
            # Формирование отчёта
            report = {
                "database": db_name,
                "index_config": db_config.get('index', {}),
                "metrics": aggregated.to_dict(),
                "total_time_seconds": timer.elapsed,
                "config_snapshot": {
                    "top_k": top_k,
                    "embedding_dim": self.config['benchmark']['embedding']['dim'],
                    "dataset_size": len(question_embeddings)
                }
            }
            
            # Сохранение сырых результатов если нужно
            if self.config['benchmark']['output'].get('save_raw', True):
                raw_path = os.path.join(output_dir, f"{db_name}_raw.json")
                save_results({
                    "per_query_metrics": [
                        {"query_idx": m.query_idx, "recall@10": m.recall_at_10, "mrr": m.mrr, "latency_ms": m.latency_ms}
                        for m in per_query[:100]  # только первые 100 для экономии места
                    ],
                    "all_latencies_ms": all_latencies
                }, output_dir, f"{db_name}_raw.json")
            
            logger.info(f"✅ {db_name}: завершено за {timer}")
            return report
    
    def run_all(self, dataset_path: str, output_dir: str) -> Dict:
        """Запуск бенчмарка для всех включённых БД"""
        ensure_dirs(output_dir)
        
        # Загрузка данных
        logger.info(f"📦 Загрузка данных: {dataset_path}")
        df = pd.read_pickle(dataset_path)
        question_emb = np.load(dataset_path.replace('dataset_processed.pkl', 'question_embeddings.npy'))

        expected_dim = self.config['benchmark']['embedding']['dim']
        if question_emb.ndim != 2 or question_emb.shape[1] != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: config={expected_dim}, loaded={question_emb.shape if question_emb.ndim == 2 else question_emb.shape}"
            )
        if len(df) != len(question_emb):
            raise ValueError(
                f"Dataset size mismatch: data rows={len(df)}, embeddings={len(question_emb)}"
            )
        
        # Ground truth IDs (используем индекс строки как ID контекста)
        ground_truth_ids = [str(idx) for idx in df.index]
        
        # Запуск по каждой БД
        all_reports = []
        for db_name in self.SEARCHERS.keys():
            report = self.run_single_db(db_name, question_emb, ground_truth_ids, output_dir)
            if report:
                all_reports.append(report)
        
        # Сводный отчёт
        if all_reports:
            summary_df = pd.DataFrame([r['metrics'] | {'database': r['database']} for r in all_reports])
            summary_path = save_results(
                {"reports": all_reports, "summary": summary_df.to_dict('records')},
                output_dir,
                "summary.json"
            )
            
            # CSV для удобного просмотра
            summary_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
            
            # Markdown отчёт
            self._generate_markdown_report(all_reports, output_dir)
            
            logger.info(f"📈 Сводный отчёт: {os.path.join(output_dir, 'metrics.csv')}")
        
        return {"reports": all_reports}
    
    def _generate_markdown_report(self, reports: List[Dict], output_dir: str):
        """Генерация MD отчёта"""
        lines = ["# 📊 Отчёт бенчмарка векторных БД", ""]
        lines.append(f"**Дата запуска:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Таблица метрик
        lines.append("## 🎯 Сравнительные метрики")
        lines.append("")
        lines.append("| Система | Recall@1 | Recall@5 | Recall@10 | MRR | p50 (ms) | p95 (ms) | p99 (ms) | QPS |")
        lines.append("|---------|----------|----------|-----------|-----|----------|----------|----------|-----|")
        
        for r in reports:
            m = r['metrics']
            lines.append(
                f"| {r['database']} | "
                f"{m['recall_at_1']:.3f} | {m['recall_at_5']:.3f} | {m['recall_at_10']:.3f} | "
                f"{m['mrr']:.3f} | {m['latency_p50']:.2f} | {m['latency_p95']:.2f} | "
                f"{m['latency_p99']:.2f} | {m['qps']:.1f} |"
            )
        
        lines.append("")
        lines.append("## ⚙️ Конфигурация")
        lines.append("")
        for r in reports:
            lines.append(f"### {r['database']}")
            lines.append(f"- Индекс: `{r['index_config'].get('type', 'N/A')}`")
            params = r['index_config'].get('params', {})
            if params:
                lines.append(f"- Параметры: `{params}`")
            lines.append("")
        
        # Рекомендации
        lines.append("## 💡 Рекомендации на основе прогонов")
        lines.append("")
        if reports:
            best_recall = max(reports, key=lambda x: x['metrics']['recall_at_10'])
            best_latency = min(reports, key=lambda x: x['metrics']['latency_p99'])
            best_qps = max(reports, key=lambda x: x['metrics']['qps'])
            
            lines.append(f"- 🎯 Лучший Recall@10: **{best_recall['database']}** ({best_recall['metrics']['recall_at_10']:.3f})")
            lines.append(f"- ⚡ Лучшая p99 latency: **{best_latency['database']}** ({best_latency['metrics']['latency_p99']:.2f} ms)")
            lines.append(f"- 🚀 Лучший QPS: **{best_qps['database']}** ({best_qps['metrics']['qps']:.1f})")
        
        with open(os.path.join(output_dir, "report.md"), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description="Запуск бенчмарка векторных БД")
    parser.add_argument("--config", default="config.yaml", help="Путь к config.yaml")
    parser.add_argument("--data", default="data/processed/dataset_processed.pkl", help="Путь к обработанным данным (pickle)")
    parser.add_argument("--output", default=None, help="Директория для результатов (авто если не указан)")
    parser.add_argument("--sample", type=int, default=None, help="Ограничить количество тестовых запросов")
    
    args = parser.parse_args()
    
    # Загрузка конфигурации
    config = load_config(args.config)
    
    # Определение путей
    dataset_path = args.data
    output_dir = args.output or os.path.join(
        config['benchmark']['output']['results_dir'],
        f"benchmark_{get_timestamp()}"
    )
    
    # Запуск бенчмарка
    runner = BenchmarkRunner(config)
    results = runner.run_all(dataset_path, output_dir)
    
    # Финальный вывод
    print("\n" + "="*60)
    print("✅ БЕНЧМАРК ЗАВЕРШЕН")
    print(f"📁 Результаты: {output_dir}")
    print(f"📄 Отчёт: {output_dir}/report.md")
    print("="*60)
    
    # Вывод сводной таблицы
    if results.get('reports'):
        print("\n📊 Краткие результаты:")
        for r in results['reports']:
            m = r['metrics']
            print(f"  {r['database']:15} | R@10: {m['recall_at_10']:.3f} | p99: {m['latency_p99']:6.1f}ms | QPS: {m['qps']:6.1f}")


if __name__ == "__main__":
    main()