-- Инициализация pgvectorscale
-- Этот скрипт выполняется при первом запуске контейнера PostgreSQL

-- Попытка включить vectorscale, если он установлен
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_available_extensions WHERE name = 'vectorscale'
    ) THEN
        CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
        RAISE NOTICE 'pgvectorscale подключен';
    ELSE
        RAISE NOTICE 'pgvectorscale не найден, используем только pgvector';
        CREATE EXTENSION IF NOT EXISTS vector CASCADE;
    END IF;
END $$;

-- Создание пользователя
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'benchmark_user') THEN
        CREATE ROLE benchmark_user WITH LOGIN PASSWORD 'benchmark_pass';
    END IF;
END $$;

GRANT ALL ON DATABASE rag_benchmark TO benchmark_user;
GRANT ALL ON SCHEMA public TO benchmark_user;