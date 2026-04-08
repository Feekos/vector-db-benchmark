"""Генерация эмбеддингов для датасета"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Optional, List
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, ensure_dirs, logger
from src.utils import load_dataset


def load_embedding_model(model_name: str, device: str = "cuda", token: str = None):
    """Загрузка модели для генерации эмбеддингов"""
    import torch
    
    # Автоопределение доступного устройства
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA недоступен, переключаюсь на CPU")
        device = "cpu"
    
    logger.info(f"Загрузка модели {model_name} на {device}")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device=device, trust_remote_code=True, token=token)
        return model
    except Exception as e:
        logger.error(f"Не удалось загрузить модель {model_name}: {e}")
        logger.info("Используется режим тестирования с генерацией случайных эмбеддингов")
        
        # Fallback: простая обертка для генерации случайных эмбеддингов
        class MockEmbedder:
            def __init__(self, dim=1024):
                self.dim = dim
                import numpy as np
                np.random.seed(42)
            
            def encode(self, texts, normalize_embeddings=False, show_progress_bar=False, convert_to_numpy=True):
                import numpy as np
                batch_size = len(texts) if isinstance(texts, list) else 1
                embeddings = np.random.randn(batch_size, self.dim).astype(np.float32)
                
                if normalize_embeddings:
                    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-6)
                
                return embeddings
            
            def to(self, device):
                return self
        
        logger.warning("Работаем в режиме тестирования с СЛУЧАЙНЫМИ эмбеддингами!")
        logger.warning("Результаты бенчмарка будут неточными - используйте реальные модели когда сможете их загрузить")
        return MockEmbedder(dim=1024)


def generate_embeddings(
    texts: List[str],
    model,
    batch_size: int = 32,
    normalize: bool = True,
    show_progress: bool = True
) -> np.ndarray:
    """
    Генерация эмбеддингов для списка текстов
    
    Args:
        texts: список строк для кодирования
        model: загруженная SentenceTransformer модель
        batch_size: размер батча для инференса
        normalize: нормализовать ли вектора к единичной длине
        show_progress: показывать ли прогресс-бар
    
    Returns:
        numpy array shape (n_texts, embedding_dim)
    """
    iterator = tqdm(texts, desc="Генерация эмбеддингов") if show_progress else texts
    
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_emb = model.encode(
            batch,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        embeddings.append(batch_emb)
    
    return np.vstack(embeddings)


def prepare_dataset(
    dataset_path: str,
    config: dict,
    output_dir: str = "data/processed"
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Подготовка датасета: загрузка и генерация эмбеддингов
    
    Returns:
        Tuple of (processed_df with embeddings, question_embeddings_array)
    """
    ensure_dirs(output_dir)
    
    # Загрузка датасета
    df = load_dataset(dataset_path)
    logger.info(f"Загружено {len(df)} записей")
    
    # Настройки из конфига
    emb_config = config['benchmark']['embedding']
    cols = config['benchmark']['datasets'][0]
    
    # Загрузка модели
    model = load_embedding_model(
        emb_config['model'],
        device=emb_config.get('device', 'cuda'),
        token=emb_config.get('huggingface_token')
    )
    
    # Генерация эмбеддингов для контекстов (база знаний)
    logger.info("Генерация эмбеддингов для контекстов...")
    contexts = df[cols['context_column']].fillna('').astype(str).tolist()
    
    # Генерация батчами для экономии памяти
    batch_size = emb_config.get('batch_size', 32)
    embeddings_list = []
    
    for i in tqdm(range(0, len(contexts), batch_size), desc="Контексты"):
        batch = contexts[i:i+batch_size]
        embeddings = generate_embeddings(
            batch, model, 
            batch_size=batch_size,
            normalize=emb_config.get('normalize', True),
            show_progress=False
        )
        # Сохраняем как list для совместимости с pickle
        embeddings_list.extend([emb.tolist() for emb in embeddings])
    
    df['embedding'] = embeddings_list[:len(df)]
    
    # Генерация эмбеддингов для вопросов (запросы)
    logger.info("Генерация эмбеддингов для вопросов...")
    questions = df[cols['question_column']].fillna('').astype(str).tolist()
    question_embeddings = generate_embeddings(
        questions, model,
        batch_size=batch_size,
        normalize=emb_config.get('normalize', True)
    )
    
    # Сохранение результатов
    processed_path = os.path.join(output_dir, "dataset_processed.pkl")
    embeddings_path = os.path.join(output_dir, "question_embeddings.npy")
    
    df.to_pickle(processed_path)
    np.save(embeddings_path, question_embeddings)
    
    logger.info(f"Датасет сохранён: {processed_path}")
    logger.info(f"Эмбеддинги вопросов: {embeddings_path}")
    logger.info(f"  Размерность: {question_embeddings.shape}")
    
    return df, question_embeddings


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Генерация эмбеддингов")
    parser.add_argument("--config", default="config.yaml", help="Путь к config.yaml")
    parser.add_argument("--input", help="Путь к датасету (переопределяет config)")
    parser.add_argument("--output", default="data/processed", help="Директория для вывода")
    parser.add_argument("--model", help="Модель для эмбеддингов (переопределяет config)")
    
    args = parser.parse_args()
    
    # Загрузка конфигурации
    config = load_config(args.config)
    
    # Определение путей
    dataset_path = args.input or config['benchmark']['datasets'][0]['path']
    output_dir = args.output
    
    # Запуск подготовки
    df, question_emb = prepare_dataset(dataset_path, config, output_dir)
    
    print(f"\n✅ Готово!")
    print(f"   Контекстов: {len(df)}")
    print(f"   Вопросов: {len(question_emb)}")
    print(f"   Размерность: {question_emb.shape[1]}")


if __name__ == "__main__":
    main()