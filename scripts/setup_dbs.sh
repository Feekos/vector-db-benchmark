#!/bin/bash
# Подготовка и проверка векторных БД

set -e

echo "🔧 Настройка векторных БД для бенчмарка..."

# Проверка Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен"
    exit 1
fi

# Загрузка переменных окружения
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Запуск контейнеров
echo "🐳 Запуск контейнеров..."
docker compose up -d

# Ожидание готовности
echo "⏳ Ожидание готовности сервисов..."

# PostgreSQL
echo "  → PostgreSQL..."
for i in {1..30}; do
    if docker exec pgvectorscale_test pg_isready -U ${POSTGRES_USER:-postgres} &>/dev/null; then
        echo "    ✅ PostgreSQL готов"
        break
    fi
    sleep 2
done

# Milvus
echo "  → Milvus..."
for i in {1..60}; do
    if curl -s "http://localhost:${MILVUS_UI_PORT:-9091}/healthz" &>/dev/null; then
        echo "    ✅ Milvus готов"
        break
    fi
    sleep 3
done

# Qdrant
echo "  → Qdrant..."
for i in {1..30}; do
    if curl -s "http://localhost:${QDRANT_PORT:-6333}/healthz" &>/dev/null; then
        echo "    ✅ Qdrant готов"
        break
    fi
    sleep 2
done

echo ""
echo "✅ Все сервисы запущены!"
echo ""
echo "📋 Статус контейнеров:"
docker compose ps

echo ""
echo "💡 Далее выполните:"
echo "   1. python -m src.embed --config config.yaml"
echo "   2. python -m src.loaders.run_all --config config.yaml"
echo "   3. python -m src.benchmark --config config.yaml"