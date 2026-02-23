# Инструкция по использованию проекта

## Быстрый старт

### 1. Активация виртуального окружения

```bash
# Перейдите в корень проекта
cd /Users/mverzhbitskiy/Documents/GitHub/3rdCourseWork

# Активируйте venv
source venv/bin/activate
```

После активации в начале строки терминала появится `(venv)`.

### 2. Установка зависимостей

Если зависимости еще не установлены:

```bash
# Установка проекта в режиме разработки (рекомендуется)
pip install -e .

# Или установка зависимостей вручную
pip install numpy scipy librosa soundfile matplotlib scikit-learn umap-learn hdbscan ripser persim plotly pandas tqdm antropy ordpy
```

### 3. Запуск Jupyter ноутбуков

#### Вариант 1: Запуск из корня проекта (рекомендуется)

```bash
# Убедитесь, что venv активирован
source venv/bin/activate

# Установите jupyter, если еще не установлен
pip install jupyter ipykernel

# Запустите Jupyter из корня проекта
jupyter notebook notebooks/
```

#### Вариант 2: Запуск из папки notebooks

```bash
cd notebooks
jupyter notebook
```

### 4. Использование модулей в ноутбуках

Все ноутбуки уже настроены для автоматического импорта модулей. В первой ячейке каждого ноутбука есть код:

```python
import sys
from pathlib import Path

# Автоматическое определение корня проекта
project_root = Path(__file__).parent.parent if '__file__' in globals() else Path.cwd().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))
```

После этого модули импортируются так:

```python
from src.topology_methods import cqt_chroma_bar_embeddings
from src.mir_bar_features import mir_bar_embeddings
from src.chaos_methods import ...
```

### 5. Пути к данным

Пути к данным автоматически настраиваются относительно корня проекта:

```python
# В ноутбуках уже настроено:
base_genre_dir = str(project_root / "data/top50musicSpotify")
```

## Работа с отдельными скриптами

### Использование spotify_scraper.py

```bash
# Активируйте venv
source venv/bin/activate

# Установите зависимости для скрейпера (если еще не установлены)
pip install -r requirements_scraper.txt

# Запустите скрипт
python spotify_scraper.py
```

## Создание нового ноутбука

Если создаете новый ноутбук в папке `notebooks/`, добавьте в первую ячейку:

```python
import os
import sys
from pathlib import Path

# Определение корня проекта
if 'notebooks' in os.getcwd():
    project_root = Path(os.getcwd()).parent
else:
    project_root = Path(os.getcwd())

# Добавление src/ в путь
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Теперь можно импортировать модули
from src.topology_methods import cqt_chroma_bar_embeddings
from src.mir_bar_features import mir_bar_embeddings
```

## Создание нового Python скрипта

Если создаете новый скрипт в корне проекта или в отдельной папке:

```python
import sys
from pathlib import Path

# Добавление src/ в путь
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# Импорт модулей
from src.topology_methods import cqt_chroma_bar_embeddings
```

## Проверка установки

Проверьте, что все работает:

```bash
# Активируйте venv
source venv/bin/activate

# Проверьте импорт модулей
python -c "import sys; sys.path.insert(0, 'src'); from src.topology_methods import cqt_chroma_bar_embeddings; print('✓ Импорт работает!')"
```

## Решение проблем

### Проблема: ModuleNotFoundError

**Решение:**
1. Убедитесь, что venv активирован: `source venv/bin/activate`
2. Установите зависимости: `pip install -e .`
3. Проверьте, что вы находитесь в корне проекта

### Проблема: Пути к данным не работают

**Решение:**
- Убедитесь, что используете `project_root` для путей:
  ```python
  data_path = str(project_root / "data/top50musicSpotify")
  ```

### Проблема: Ноутбук не видит модули

**Решение:**
1. Убедитесь, что первая ячейка с настройкой путей выполнена
2. Перезапустите ядро ноутбука (Kernel → Restart)
3. Выполните все ячейки заново

## Структура импортов

```
notebooks/
  └── *.ipynb
      └── from src.topology_methods import ...
      └── from src.mir_bar_features import ...

src/
  ├── topology_methods.py
  ├── mir_bar_features.py  ──┐
  └── chaos_methods.py       │
                              │
      mir_bar_features.py ────┘ from .topology_methods import ...
```

## Полезные команды

```bash
# Деактивация venv
deactivate

# Проверка установленных пакетов
pip list

# Обновление пакетов
pip install --upgrade -e .

# Очистка кэша Python
find . -type d -name __pycache__ -exec rm -r {} +
```

