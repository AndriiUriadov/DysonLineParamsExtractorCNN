# DysonLineParamsExtractorCNN

Система для виконання коду в Google Colab з виводом результатів в термінал Cursor.

## 🚀 Швидкий старт

### Метод 1: Копіювання коду в Colab (рекомендовано)

```bash
# Генерує код для Colab та копіює в буфер обміну
python3 update_colab_connection.py improved_gpu_check_notebook.py
```

### Метод 2: Швидкий запуск з меню

```bash
# Інтерактивне меню з вибором методу
python3 quick_colab_runner.py improved_gpu_check_notebook.py
```

### Метод 3: SSH виконання (експериментально)

```bash
# Виконання через SSH з'єднання
python3 simple_colab_runner.py improved_gpu_check_notebook.py
```

## 📋 Доступні скрипти

| Скрипт | Опис |
|--------|------|
| `update_colab_connection.py` | Генерує код для прямого виконання в Colab |
| `quick_colab_runner.py` | Інтерактивне меню з вибором методу |
| `simple_colab_runner.py` | SSH виконання (нестабільне) |
| `run_in_colab.py` | Розширений SSH виконавець |

## 🔧 Налаштування

### 1. Встановлення залежностей

```bash
pip install pyperclip
```

### 2. Налаштування Colab

1. Відкрийте [Google Colab](https://colab.research.google.com/)
2. Створіть новий notebook
3. Налаштуйте runtime:
   - Runtime → Change runtime type
   - Hardware accelerator: **GPU**
   - Runtime shape: **High-RAM**

## 📖 Використання

### Крок 1: Підготовка коду

```bash
# Генерує код для Colab
python3 update_colab_connection.py your_file.py
```

### Крок 2: Виконання в Colab

1. Код автоматично копіюється в буфер обміну
2. Відкрийте Colab (автоматично відкриється)
3. Вставте код в комірку (Cmd+V)
4. Запустіть комірку (Shift + Enter)
5. Результати з'являться в Colab

## 🎯 Приклади

### Перевірка GPU та RAM

```bash
python3 update_colab_connection.py improved_gpu_check_notebook.py
```

### Виконання будь-якого Python файлу

```bash
python3 update_colab_connection.py your_script.py
```

## ⚠️ Обмеження

- **SSH метод**: Нестабільний через ефемерні тунелі `colab-ssh`
- **Копіювання**: Потребує ручного вставлення в Colab
- **Результати**: Відображаються в Colab, не в терміналі Cursor

## 🔄 Робочий процес

1. **Розробка**: Пишіть код в Cursor
2. **Генерація**: Запустіть `update_colab_connection.py`
3. **Виконання**: Вставте код в Colab
4. **Результати**: Переглядайте в Colab

## 📝 Структура проекту

```
DysonLineParamsExtractorCNN/
├── improved_gpu_check_notebook.py    # Основний файл для перевірки GPU
├── update_colab_connection.py        # Генератор коду для Colab
├── quick_colab_runner.py             # Швидкий запуск з меню
├── simple_colab_runner.py            # SSH виконавець
├── run_in_colab.py                   # Розширений SSH виконавець
└── README.md                         # Документація
```

## 🛠️ Розробка

### Додавання нових файлів

1. Створіть Python файл в Cursor
2. Використовуйте `update_colab_connection.py` для генерації коду
3. Виконайте в Colab

### Налаштування середовища

```bash
# Віртуальне середовище (опціонально)
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# або
.venv\Scripts\activate     # Windows
```

## 📞 Підтримка

При проблемах:

1. Перевірте налаштування runtime в Colab
2. Спробуйте перезапустити notebook
3. Використовуйте метод копіювання замість SSH

## 🎉 Переваги

- ✅ Працює в Cursor IDE
- ✅ Використовує потужні ресурси Colab
- ✅ Простий у використанні
- ✅ Автоматична генерація коду
- ✅ Підтримка GPU та High-RAM