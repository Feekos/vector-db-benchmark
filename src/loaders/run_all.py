"""Скрипт для загрузки данных во все включённые БД"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import load_config, load_dataset, logger, save_results
from src.loaders.pgvectorscale import PGVectorScaleLoader
from src.loaders.milvus import MilvusLoader
from src.loaders.qdrant import QdrantLoader


def run_loader(loader_class, config, db_config, df, batch_size, db_name):
    """Запуск загрузчика с обработкой ошибок"""
    logger.info(f"\n🔄 {db_name}: Начало загрузки...")
    
    try:
        with loader_class(config, db_config) as loader:
            if not loader.check_ready():
                logger.error(f"{db_name}: Не удалось подготовить схему")
                return False, 0.0
            
            # Загрузка данных
            def progress(current, total):
                pct = current / total * 100
                logger.info(f"{db_name}: {current}/{total} ({pct:.1f}%)")
            
            stats = loader.load_data(df, batch_size=batch_size, progress_callback=progress)
            
            # Отчёт
            count = loader.count_records()
            loader_stats = loader.get_stats()
            logger.info(f"✅ {db_name}: Загружено {stats['inserted']} записей, всего в БД: {count}, время индексации: {loader_stats['index_build_time_seconds']:.2f} сек")
            return True, loader_stats['index_build_time_seconds']
            
    except Exception as e:
        logger.error(f"❌ {db_name}: Ошибка загрузки: {e}")
        return False, 0.0


def main():
    parser = argparse.ArgumentParser(description="Загрузка данных во все БД")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", default="data/processed/dataset_processed.pkl")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--db", choices=["pgvectorscale", "milvus", "qdrant", "all"], default="all")
    
    args = parser.parse_args()
    
    # Загрузка конфигурации и данных
    config = load_config(args.config)
    df = load_dataset(args.data)
    batch_size = args.batch_size
    
    # Маппинг загрузчиков
    loaders = {
        "pgvectorscale": (PGVectorScaleLoader, config['benchmark']['databases']['pgvectorscale']),
        "milvus": (MilvusLoader, config['benchmark']['databases']['milvus']),
        "qdrant": (QdrantLoader, config['benchmark']['databases']['qdrant'])
    }
    
    # Определение какие БД запускать
    targets = [args.db] if args.db != "all" else [k for k, v in config['benchmark']['databases'].items() if v.get('enabled', True)]
    
    results = {}
    index_build_times = {}
    for db_name in targets:
        if db_name not in loaders:
            logger.warning(f"Неизвестная БД: {db_name}")
            continue
        
        loader_class, db_config = loaders[db_name]
        if not db_config.get('enabled', True):
            logger.info(f"⏭ {db_name}: отключено в конфигурации")
            continue
            
        success, build_time = run_loader(loader_class, config, db_config, df, batch_size, db_name)
        results[db_name] = success
        if success:
            index_build_times[db_name] = build_time
    
    # Сохранение времени индексации для бенчмарка
    if index_build_times:
        save_results(index_build_times, "data/processed", "index_build_times.json")
        logger.info("💾 Время индексации сохранено в data/processed/index_build_times.json")
    
    # Итоговый отчёт
    print("\n" + "="*50)
    print("📊 Итоги загрузки:")
    for db_name, success in results.items():
        status = "✅ Успешно" if success else "❌ Ошибка"
        build_time = f" ({index_build_times.get(db_name, 0):.2f} сек индекса)" if success else ""
        print(f"  {db_name}: {status}{build_time}")
    print("="*50)
    
    # Exit code
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()