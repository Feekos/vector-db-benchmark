#!/bin/bash
# Очистка результатов и данных

set -e

echo "🧹 Очистка проекта векторного бенчмарка"
echo "======================================="
echo ""

# Парсинг аргументов
CLEAN_RESULTS=false
CLEAN_PROCESSED=false
CLEAN_DOCKER=false
CLEAN_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --results) CLEAN_RESULTS=true; shift ;;
        --processed) CLEAN_PROCESSED=true; shift ;;
        --docker) CLEAN_DOCKER=true; shift ;;
        --all) CLEAN_ALL=true; shift ;;
        *) echo "Неизвестный аргумент: $1"; exit 1 ;;
    esac
done

# Если нет аргументов - показать справку
if [[ "$CLEAN_RESULTS$CLEAN_PROCESSED$CLEAN_DOCKER$CLEAN_ALL" == "falsefalsefalsefalse" ]]; then
    echo "Использование: $0 [опции]"
    echo ""
    echo "Опции:"
    echo "  --results    Удалить результаты бенчмарков (data/results/)"
    echo "  --processed  Удалить обработанные данные (data/processed/)"
    echo "  --docker     Остановить и удалить контейнеры + volumes"
    echo "  --all        Удалить всё вышеперечисленное"
    echo ""
    echo "Примеры:"
    echo "  $0 --results              # только результаты"
    echo "  $0 --all                  # полная очистка"
    exit 0
fi

# Если --all, включить всё
if $CLEAN_ALL; then
    CLEAN_RESULTS=true
    CLEAN_PROCESSED=true
    CLEAN_DOCKER=true
fi

# Очистка результатов
if $CLEAN_RESULTS; then
    echo "🗑️  Удаление результатов..."
    rm -rf data/results/*
    echo "   ✅ data/results/ очищена"
fi

# Очистка обработанных данных
if $CLEAN_PROCESSED; then
    echo "🗑️  Удаление обработанных данных..."
    rm -rf data/processed/*
    echo "   ✅ data/processed/ очищена"
    echo "   ⚠️  При следующем запуске эмбеддинги будут сгенерированы заново"
fi

# Очистка Docker
if $CLEAN_DOCKER; then
    echo "🗑️  Остановка и удаление контейнеров..."
    docker compose down -v 2>/dev/null || true
    echo "   ✅ Контейнеры остановлены, volumes удалены"
    
    # Очистка образов (опционально, закомментировано по умолчанию)
    # echo "🗑️  Удаление образов (опционально)..."
    # docker rmi timescale/timescaledb-ha:pg16 milvusdb/milvus:v2.3.2 qdrant/qdrant:v1.7.3 2>/dev/null || true
fi

echo ""
echo "✅ Очистка завершена!"