# Music Topology Analysis Project

Проект для анализа топологии музыки с использованием методов TDA (Topological Data Analysis) и MIR (Music Information Retrieval).

## Структура проекта

```
3rdCourseWork/
├── notebooks/          # Jupyter ноутбуки для анализа и визуализации
│   ├── topology_visualization.ipynb
│   ├── topology_tda_pipeline.ipynb
│   ├── topology_mir_visualization.ipynb
│   └── topology_mir_umap_visualization.ipynb
│
├── src/               # Python модули
│   ├── __init__.py
│   ├── topology_methods.py    # CQT-хрома эмбеддинги тактов
│   ├── mir_bar_features.py    # MIR-баровые признаки
│   └── chaos_methods.py       # Методы анализа хаоса
│
├── scripts/           # Утилитарные скрипты (опционально)
│   └── spotify_scraper.py    # Скрипт для скачивания музыки с Spotify
│
├── data/              # Данные (аудиофайлы)
│   ├── top50musicSpotify/
│   └── 1000musicSpotify/
│
├── venv/              # Виртуальное окружение (не в git)
├── setup.py           # Установка пакета
├── requirements_scraper.txt  # Зависимости для скрейпера
└── .env               # Переменные окружения (не в git)
```

## Установка

1. Создайте виртуальное окружение:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

2. Установите зависимости:
```bash
pip install -r requirements_scraper.txt  # Для скрейпера
pip install -e .  # Установка проекта в режиме разработки
```

Или установите основные зависимости вручную:
```bash
pip install numpy scipy librosa soundfile matplotlib scikit-learn umap-learn hdbscan ripser persim plotly pandas tqdm antropy ordpy
```

## Использование

### Быстрый старт

1. **Активируйте виртуальное окружение:**
```bash
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

2. **Установите зависимости (если еще не установлены):**
```bash
pip install -e .  # Установка проекта в режиме разработки
```

3. **Запустите Jupyter:**
```bash
jupyter notebook notebooks/
```

### Ноутбуки

Все ноутбуки находятся в папке `notebooks/`. Они автоматически настраивают пути для импорта модулей из `src/`.

**Важно:** Запускайте Jupyter из корня проекта, чтобы пути к данным работали корректно.

```bash
# Из корня проекта (рекомендуется)
jupyter notebook notebooks/
```

Подробные инструкции см. в:
- [USAGE.md](USAGE.md) - общие инструкции
- [DIRECT_USAGE.md](DIRECT_USAGE.md) - использование ноутбуков напрямую (без терминала)

### Импорт модулей

В ноутбуках и скриптах модули импортируются так:

```python
from src.topology_methods import cqt_chroma_bar_embeddings
from src.mir_bar_features import mir_bar_embeddings
from src.chaos_methods import ...
```

### Пути к данным

Пути к данным настроены относительно корня проекта. В ноутбуках используется `project_root` для надежной работы из любой директории:

```python
base_genre_dir = str(project_root / "data/top50musicSpotify")
```

## Модули

### `topology_methods.py`
- `cqt_chroma_bar_embeddings()` - CQT-хрома эмбеддинги тактов

### `mir_bar_features.py`
- `mir_bar_embeddings()` - MIR-баровые признаки (MFCC, спектральные признаки, хрома)

### `chaos_methods.py`
- Методы анализа хаоса для аудио

## Примечания

- Виртуальное окружение `venv/` не должно попадать в git (уже в .gitignore)
- Файл `.env` с API ключами не должен попадать в git
- Большие данные в `data/` могут быть исключены из git (раскомментируйте в .gitignore)

