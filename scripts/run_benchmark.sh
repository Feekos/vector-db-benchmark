#!/bin/bash
# Полный пайплайн запуска бенчмарка

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "🚀 Vector DB Benchmark Pipeline"
echo "================================"
echo "Дата: $(date)"
echo ""

# Загрузка .env
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# 1. Проверка и запуск БД
echo "📦 Шаг 1: Проверка БД..."
if ! docker compose ps | grep -q "healthy\|Up"; then
    echo "  ⚠️ Контейнеры не запущены, запускаем..."
    bash "$SCRIPT_DIR/setup_dbs.sh"
else
    echo "  ✅ Контейнеры уже работают"
fi

# 2. Генерация эмбеддингов (если нужно)
echo ""
echo "🔤 Шаг 2: Генерация эмбеддингов..."
if [ ! -f "data/processed/question_embeddings.npy" ]; then
    echo "  🔄 Генерация..."
    python -m src.embed --config config.yaml
else
    echo "  ✅ Эмбеддинги уже сгенерированы"
    echo "     Проверьте актуальность: data/processed/"
fi

# 3. Загрузка данных
echo ""
echo "📥 Шаг 3: Загрузка данных в БД..."
python -m src.loaders.run_all --config config.yaml

# 4. Запуск бенчмарка
echo ""
echo "🎯 Шаг 4: Запуск бенчмарка..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="data/results/benchmark_${TIMESTAMP}"

python -m src.benchmark \
    --config config.yaml \
    --data data/processed \
    --output "$OUTPUT_DIR" \
    "${@:1}"  # передача дополнительных аргументов

# 5. Открытие отчёта
echo ""
echo "📊 Результаты готовы: $OUTPUT_DIR"
if [ -f "$OUTPUT_DIR/report.md" ]; then
    echo ""
    echo "📄 Краткий отчёт:"
    echo "----------------------------------------"
    head -30 "$OUTPUT_DIR/report.md"
    echo "..."
    echo "----------------------------------------"
    echo ""
    echo "💡 Полный отчёт: $OUTPUT_DIR/report.md"
fi

echo ""
echo "✅ Пайплайн завершён!"