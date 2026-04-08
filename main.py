#!/usr/bin/env python3
"""
Единая точка входа для запуска полного пайплайна бенчмарка
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_cmd(cmd: str, desc: str) -> bool:
    """Запуск внешней команды с выводом в реальном времени"""
    print(f"\n🔄 {desc}...")
    try:
        process = subprocess.run(
            cmd, shell=True, check=True, 
            stdout=sys.stdout, stderr=sys.stderr
        )
        print(f"✅ {desc} завершено")
        return True
    except subprocess.CalledProcessError:
        print(f"\n❌ Ошибка на шаге: {desc}")
        print("💡 Проверьте логи выше и убедитесь, что БД запущены: docker compose up -d")
        return False

def check_docker():
    """Проверка, что контейнеры запущены"""
    try:
        out = subprocess.check_output(
            "docker compose ps --format '{{.Status}}'", 
            shell=True, text=True
        )
        if "healthy" in out.lower() or "up" in out.lower():
            return True
        print("⚠️ Контейнеры не в статусе Up/Healthy.")
        return False
    except Exception:
        print("⚠️ Не удалось проверить статус Docker. Убедитесь, что выполнили: docker compose up -d")
        return False

def main():
    parser = argparse.ArgumentParser(description="🚀 Полный пайплайн бенчмарка векторных БД")
    parser.add_argument("--skip-embed", action="store_true", help="Пропустить генерацию эмбеддингов")
    parser.add_argument("--skip-load", action="store_true", help="Пропустить загрузку данных в БД")
    parser.add_argument("--skip-bench", action="store_true", help="Пропустить запуск бенчмарка")
    parser.add_argument("--sample", type=int, help="Ограничить число тестовых запросов (для быстрой проверки)")
    parser.add_argument("--config", default="config.yaml", help="Путь к конфигу")
    args = parser.parse_args()

    print("="*60)
    print("🚀 VECTOR DB BENCHMARK PIPELINE")
    print("="*60)

    # Проверка БД
    if not check_docker():
        sys.exit(1)
    print("✅ БД работают и готовы к подключению")

    # Генерация эмбеддингов
    if not args.skip_embed:
        if not os.path.exists("data/processed/question_embeddings.npy"):
            if not run_cmd(f"python -m src.embed --config {args.config}", "Генерация эмбеддингов"):
                sys.exit(1)
        else:
            print("\nЭмбеддинги уже сгенерированы (data/processed/)")
    else:
        print("\nПропуск генерации эмбеддингов")

    # Загрузка данных в БД
    if not args.skip_load:
        if not run_cmd(f"python -m src.loaders.run_all --config {args.config}", "Загрузка данных в векторные БД"):
            sys.exit(1)
    else:
        print("⏭ Пропуск загрузки данных")

    # Запуск бенчмарка
    if not args.skip_bench:
        bench_cmd = f"python -m src.benchmark --config {args.config}"
        if args.sample:
            bench_cmd += f" --sample {args.sample}"
        
        if not run_cmd(bench_cmd, "Поисковый бенчмарк"):
            sys.exit(1)
    else:
        print("⏭ Пропуск бенчмарка")

    # Итоговые результаты
    print("\n" + "="*60)
    print("🎉 БЕНЧМАРК ЗАВЕРШЁН!")
    print("📁 Папка с результатами: data/results/benchmark_*/")
    print("📄 Текстовый отчёт: data/results/benchmark_*/report.md")
    print("📊 CSV с метриками: data/results/benchmark_*/metrics.csv")
    print("="*60)

if __name__ == "__main__":
    main()