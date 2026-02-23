# Использование ноутбуков напрямую (без терминала)

## ✅ Да, можно использовать ноутбуки напрямую!

Ноутбуки настроены для работы напрямую через VS Code, JupyterLab или любой другой редактор, без необходимости запуска через терминал.

## Способы открытия ноутбуков

### 1. Через VS Code (рекомендуется)

1. Откройте VS Code в корне проекта:
   ```bash
   code /Users/mverzhbitskiy/Documents/GitHub/3rdCourseWork
   ```

2. Откройте любой ноутбук из папки `notebooks/`

3. **Важно:** Выберите правильное ядро (kernel):
   - Нажмите на кнопку выбора ядра в правом верхнем углу ноутбука
   - Выберите `Python 3.x.x ('venv': venv)` или создайте новое ядро:
     ```bash
     # В терминале VS Code (если нужно):
     source venv/bin/activate
     python -m ipykernel install --user --name=venv --display-name="Python (venv)"
     ```

4. Запустите первую ячейку - она автоматически настроит пути

### 2. Через JupyterLab

1. Откройте JupyterLab:
   ```bash
   # Один раз активируйте venv и запустите:
   source venv/bin/activate
   jupyter lab
   ```

2. Откройте ноутбук из папки `notebooks/`

3. Убедитесь, что выбран правильный kernel (из venv)

### 3. Через обычный Jupyter Notebook

1. Запустите Jupyter (один раз через терминал):
   ```bash
   source venv/bin/activate
   jupyter notebook
   ```

2. Откройте ноутбук из папки `notebooks/`

## Как это работает

В первой ячейке каждого ноутбука есть код, который:

1. **Автоматически определяет корень проекта** через путь к самому ноутбуку
2. **Добавляет `src/` в путь** для импорта модулей
3. **Настраивает пути к данным** относительно корня проекта

```python
# Этот код работает автоматически при запуске первой ячейки
import inspect
notebook_path = Path(inspect.getfile(inspect.currentframe())).resolve()
if 'notebooks' in str(notebook_path):
    project_root = notebook_path.parent.parent  # Поднимаемся на уровень выше
```

## Проверка работы

После запуска первой ячейки вы должны увидеть:

```
✓ Корень проекта: /Users/mverzhbitskiy/Documents/GitHub/3rdCourseWork
✓ Путь к src: /Users/mverzhbitskiy/Documents/GitHub/3rdCourseWork/src
```

Если видите эти сообщения - всё работает правильно!

## Импорт модулей

После выполнения первой ячейки можно использовать:

```python
from src.topology_methods import cqt_chroma_bar_embeddings
from src.mir_bar_features import mir_bar_embeddings
from src.chaos_methods import ...
```

## Важные моменты

### ✅ Что работает автоматически:
- Определение корня проекта
- Настройка путей для импорта
- Пути к данным (`data/top50musicSpotify`)

### ⚠️ Что нужно проверить:
- **Правильный kernel выбран** (должен быть из venv)
- **Зависимости установлены** в venv (`pip install -e .`)
- **Первая ячейка выполнена** перед использованием модулей

## Решение проблем

### Проблема: ModuleNotFoundError

**Решение:**
1. Проверьте, что выбран правильный kernel (из venv)
2. В VS Code: нажмите на kernel в правом верхнем углу → выберите venv
3. Перезапустите ядро (Kernel → Restart)

### Проблема: Пути не определяются

**Решение:**
1. Убедитесь, что ноутбук находится в папке `notebooks/`
2. Перезапустите ядро и выполните первую ячейку заново
3. Проверьте вывод - должны быть видны пути

### Проблема: Kernel не найден

**Решение:**
```bash
# Активируйте venv и установите ipykernel
source venv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=venv --display-name="Python (venv)"
```

Затем выберите этот kernel в ноутбуке.

## Преимущества прямого использования

✅ Не нужно запускать терминал каждый раз  
✅ Удобная работа в VS Code  
✅ Автоматическая настройка путей  
✅ Интеграция с Git в VS Code  

## Рекомендации

- **VS Code:** Лучший вариант для работы с ноутбуками
- **JupyterLab:** Хорошая альтернатива с расширенными возможностями
- **Обычный Jupyter:** Работает, но требует запуска через терминал

