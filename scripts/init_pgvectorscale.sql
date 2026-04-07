-- Инициализация pgvectorscale
-- Этот скрипт выполняется при первом запуске контейнера PostgreSQL

-- Создание расширения vectorscale
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;

-- Создание пользователя для бенчмарка (опционально)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'benchmark_user') THEN
        CREATE ROLE benchmark_user WITH LOGIN PASSWORD 'benchmark_pass';
    END IF;
END $$;

-- Предоставление прав
GRANT ALL ON DATABASE rag_benchmark TO benchmark_user;
GRANT ALL ON SCHEMA public TO benchmark_user;

SELECT 'pgvectorscale initialized' AS status;