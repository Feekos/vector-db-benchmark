# 🚀 Vector DB Benchmark

Сравнительное тестирование векторных баз данных: **pgvectorscale**, **Milvus**, **Qdrant**

## 📋 Требования

- Docker + Docker Compose v2+
- Python 3.10+
- 8+ GB RAM (рекомендуется 16 GB)
- GPU опционально (для ускорения эмбеддингов)

## 🚀 Быстрый старт

```bash
# 1. Клонирование и настройка
git clone <repo>
cd vector-db-benchmark
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Подготовка датасета
mkdir -p data/raw
# Скачайте ru_rag_test_dataset.pkl в data/raw/

# 3. Запуск БД
docker compose up -d
# Подождите 3-5 минут пока все сервисы станут healthy

# 4. Запуск пайплайна для бенчмарка от эмбеддинга до поиска по базам данных с генерацией результатов
python main.py --config config.yaml

# 5. Просмотр результатов
cat data/results/test_run/report.md