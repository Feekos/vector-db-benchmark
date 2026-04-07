"""Вспомогательные функции для бенчмарка"""
import os
import yaml
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Загрузка конфигурации из YAML файла"""
    load_dotenv()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Подстановка переменных окружения в конфиг
    config = _substitute_env_vars(config)
    
    return config


def _substitute_env_vars(obj: Any) -> Any:
    """Рекурсивная замена ${VAR} на значения из окружения"""
    if isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
        var_name = obj[2:-1]
        return os.getenv(var_name, obj)
    return obj


def ensure_dirs(*paths: str) -> None:
    """Создание директорий если не существуют"""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_results(data: Dict[str, Any], output_dir: str, filename: str) -> str:
    """Сохранение результатов в различных форматах"""
    ensure_dirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    
    ext = Path(filename).suffix.lower()
    
    if ext == '.json':
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    elif ext == '.yaml' or ext == '.yml':
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True)
    elif ext == '.csv':
        import pandas as pd
        pd.DataFrame([data] if isinstance(data, dict) else data).to_csv(
            filepath, index=False, encoding='utf-8'
        )
    else:
        # По умолчанию JSON
        with open(filepath + '.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        filepath += '.json'
    
    logger.info(f"Результаты сохранены: {filepath}")
    return filepath


def format_duration(seconds: float) -> str:
    """Форматирование времени в человекочитаемый вид"""
    if seconds < 60:
        return f"{seconds:.2f} сек"
    elif seconds < 3600:
        return f"{seconds/60:.2f} мин"
    else:
        return f"{seconds/3600:.2f} час"


class Timer:
    """Контекстный менеджер для замера времени"""
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
    
    @property
    def elapsed(self) -> float:
        """Возвращает прошедшее время в секундах"""
        if self.start_time is None:
            return 0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def __str__(self):
        return f"{self.name}: {format_duration(self.elapsed)}"


def get_timestamp() -> str:
    """Возвращает текущую метку времени для имён файлов"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_dataset(path: str) -> Any:
    """Загрузка датасета из pickle файла"""
    import pandas as pd
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Датасет не найден: {path}")
    
    logger.info(f"Загрузка датасета: {path}")
    return pd.read_pickle(path)