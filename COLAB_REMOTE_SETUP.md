# 🚀 Налаштування віддаленого виконання коду в Google Colab

## Опис

Цей набір скриптів дозволяє працювати з кодом локально в Cursor, але виконувати його в потужному середовищі Google Colab через SSH з'єднання.

## 📋 Покрокова інструкція

### **Крок 1: Налаштування Google Colab**

1. **Відкрийте Google Colab** в браузері: https://colab.research.google.com/
2. **Створіть новий ноутбук** або відкрийте існуючий
3. **Налаштуйте runtime**:
   - Перейдіть до **Runtime** → **Change runtime type**
   - У **Hardware accelerator** виберіть **GPU**
   - У **Runtime shape** виберіть **High-RAM** (рекомендовано)

### **Крок 2: Запуск SSH сервера в Colab**

Вставте наступний код в **першу комірку** Colab:

```python
# Встановлюємо colab-ssh
!pip install colab-ssh

# Імпортуємо необхідні модулі
from colab_ssh import launch_ssh_cloudflared
import random
import string

# Генеруємо випадковий пароль
password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
print(f"🔐 Пароль для SSH: {password}")

# Запускаємо SSH сервер
launch_ssh_cloudflared(password=password, verbose=True)
```

4. **Запустіть комірку** (Shift + Enter)
5. **Скопіюйте SSH команду** з виводу (виглядає як `ssh -p 12345 root@localhost`)
6. **Запам'ятайте пароль** з виводу

### **Крок 3: Підключення з Cursor**

1. **Відкрийте термінал в Cursor**
2. **Виконайте скопійовану SSH команду**
3. **Введіть пароль** коли запитає

### **Крок 4: Тестування з'єднання**

```bash
# Тестуємо з'єднання
python3 run_in_colab.py --test
```

### **Крок 5: Налаштування середовища**

```bash
# Налаштовуємо середовище в Colab
python3 run_in_colab.py your_file.py --setup
```

### **Крок 6: Виконання коду**

```bash
# Виконуємо Python файл
python3 run_in_colab.py improved_gpu_check_notebook.py

# Виконуємо Jupyter notebook
python3 run_in_colab.py DysonianLineCNN_multihead_30K.ipynb
```

## 📁 Створені файли

- `run_in_colab.py` - Основний скрипт для віддаленого виконання
- `colab_remote_setup.py` - Допоміжний скрипт налаштування
- `remote_colab_executor.py` - Альтернативний виконавець

## 🔧 Використання

### **Базове використання:**
```bash
# Виконати Python файл
python3 run_in_colab.py your_script.py

# Виконати notebook
python3 run_in_colab.py your_notebook.ipynb

# Налаштувати середовище
python3 run_in_colab.py your_script.py --setup

# Тестувати з'єднання
python3 run_in_colab.py --test
```

### **Розширені опції:**
```bash
# Вказати хост та порт
python3 run_in_colab.py your_script.py --host localhost --port 12345

# Виконати з налаштуванням середовища
python3 run_in_colab.py your_script.py --setup
```

## 💡 Переваги

### **🚀 Продуктивність:**
- Доступ до потужних GPU (Tesla T4, V100, A100)
- Велика RAM (до 100GB в Pro+)
- Швидке навчання моделей

### **🛠️ Зручність:**
- Робота в звичному редакторі Cursor
- Автоматична синхронізація файлів
- Не потрібно копіювати код вручну

### **🔒 Безпека:**
- SSH з'єднання з шифруванням
- Тимчасові файли автоматично видаляються
- Ізольоване середовище виконання

## ⚠️ Важливі зауваження

1. **SSH з'єднання** потрібно встановлювати кожного разу при перезапуску Colab
2. **Пароль генерується** автоматично для кожного сеансу
3. **Таймаут сеансу** - Colab має обмеження часу виконання
4. **Версії бібліотек** можуть відрізнятися від локальних

## 🔧 Розв'язання проблем

### **Проблема: SSH з'єднання не працює**
```bash
# Перевірте, чи запущений SSH сервер в Colab
# Перевірте правильність команди та пароля
python3 run_in_colab.py --test
```

### **Проблема: Файл не знайдено**
```bash
# Перевірте шлях до файлу
ls -la your_file.py
```

### **Проблема: Помилки імпорту**
```bash
# Налаштуйте середовище
python3 run_in_colab.py your_file.py --setup
```

### **Проблема: Таймаут виконання**
- Збільшіть таймаут в коді
- Розділіть великий код на менші частини
- Використовуйте Colab Pro для довшого часу виконання

## 📊 Приклад використання

```bash
# 1. Тестуємо з'єднання
python3 run_in_colab.py --test

# 2. Налаштовуємо середовище
python3 run_in_colab.py improved_gpu_check_notebook.py --setup

# 3. Виконуємо покращений код перевірки GPU
python3 run_in_colab.py improved_gpu_check_notebook.py

# 4. Виконуємо основний notebook
python3 run_in_colab.py DysonianLineCNN_multihead_30K.ipynb
```

## 🎯 Результат

Тепер ви можете:
- ✅ Працювати з кодом локально в Cursor
- ✅ Виконувати код в потужному середовищі Colab
- ✅ Використовувати GPU та велику RAM
- ✅ Не копіювати код вручну
- ✅ Зберігати всю історію змін в Git

**Насолоджуйтесь потужним середовищем Colab з зручністю локальної розробки!** 🚀 