#!/usr/bin/env python3
"""
Скрипт для запуску improved_gpu_check_notebook.py в Google Colab
Використовує SSH з'єднання для віддаленого виконання
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def print_header():
    """Виводить заголовок скрипта"""
    print("=" * 60)
    print("🚀 Запуск GPU Check в Google Colab")
    print("=" * 60)
    print()

def check_ssh_connection():
    """Перевіряє SSH з'єднання з Colab"""
    try:
        # Перевіряємо чи можемо підключитися до localhost
        result = subprocess.run(['ssh', '-o', 'ConnectTimeout=5', 'root@localhost', 'echo "SSH connection test"'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def setup_colab_ssh():
    """Налаштовує SSH з'єднання з Colab"""
    print("🔧 Налаштування SSH з'єднання з Colab...")
    print()
    print("📋 ІНСТРУКЦІЯ:")
    print("1. Відкрийте Google Colab: https://colab.research.google.com/")
    print("2. Створіть новий notebook")
    print("3. Вставте наступний код в першу комірку:")
    print()
    print("```python")
    print("# Встановлюємо colab-ssh")
    print("!pip install colab-ssh")
    print()
    print("# Імпортуємо необхідні модулі")
    print("from colab_ssh import launch_ssh_cloudflared")
    print("import random")
    print("import string")
    print()
    print("# Генеруємо випадковий пароль")
    print("password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))")
    print("print(f\"🔐 Пароль для SSH: {password}\")")
    print()
    print("# Запускаємо SSH сервер")
    print("launch_ssh_cloudflared(password=password, verbose=True)")
    print("```")
    print()
    print("4. Запустіть комірку (Shift + Enter)")
    print("5. Скопіюйте SSH команду з виводу")
    print("6. Поверніться сюди і натисніть Enter для продовження")
    print()
    input("Натисніть Enter коли SSH буде налаштовано...")

def run_gpu_check_remotely(ssh_command, password):
    """Запускає GPU check в Colab через SSH"""
    print(f"🚀 Запуск improved_gpu_check_notebook.py в Colab...")
    print()
    
    # Підготовка команди для копіювання файлу
    copy_command = f"scp improved_gpu_check_notebook.py root@localhost:/tmp/"
    
    # Підготовка команди для виконання
    execute_command = f"ssh root@localhost 'cd /tmp && python3 improved_gpu_check_notebook.py'"
    
    try:
        # Копіюємо файл
        print("📁 Копіювання файлу в Colab...")
        copy_result = subprocess.run(copy_command, shell=True, capture_output=True, text=True)
        
        if copy_result.returncode != 0:
            print(f"❌ Помилка копіювання файлу: {copy_result.stderr}")
            return False
        
        print("✅ Файл скопійовано")
        
        # Виконуємо файл
        print("⚡ Виконання GPU check...")
        print("-" * 50)
        
        execute_result = subprocess.run(execute_command, shell=True, text=True)
        
        if execute_result.returncode == 0:
            print("-" * 50)
            print("✅ GPU check успішно виконано в Colab!")
        else:
            print(f"❌ Помилка виконання: {execute_result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Помилка: {str(e)}")
        return False
    
    return True

def run_gpu_check_locally():
    """Запускає GPU check локально"""
    print("🔍 Запуск GPU check локально...")
    print("-" * 50)
    
    try:
        result = subprocess.run([sys.executable, 'improved_gpu_check_notebook.py'], 
                              text=True, capture_output=False)
        
        if result.returncode == 0:
            print("-" * 50)
            print("✅ GPU check успішно виконано локально!")
        else:
            print("❌ Помилка виконання GPU check")
            
    except Exception as e:
        print(f"❌ Помилка: {str(e)}")

def main():
    """Головна функція"""
    print_header()
    
    # Перевіряємо чи існує файл
    if not os.path.exists('improved_gpu_check_notebook.py'):
        print("❌ Файл improved_gpu_check_notebook.py не знайдено!")
        print("Переконайтеся, що файл знаходиться в поточній директорії.")
        return
    
    print("📁 Файл improved_gpu_check_notebook.py знайдено")
    print()
    
    # Питаємо користувача про спосіб виконання
    print("🔧 Виберіть спосіб виконання:")
    print("1. 🚀 Запустити в Google Colab (рекомендовано для GPU)")
    print("2. 💻 Запустити локально")
    print("3. 🔍 Тільки перевірити SSH з'єднання")
    print()
    
    choice = input("Введіть номер (1-3): ").strip()
    
    if choice == "1":
        # Перевіряємо SSH з'єднання
        if not check_ssh_connection():
            print("❌ SSH з'єднання не знайдено")
            print("Потрібно налаштувати SSH з'єднання з Colab")
            setup_colab_ssh()
            
            # Повторна перевірка
            if not check_ssh_connection():
                print("❌ SSH з'єднання все ще не працює")
                print("Перевірте налаштування та спробуйте ще раз")
                return
        
        print("✅ SSH з'єднання працює")
        
        # Запускаємо в Colab
        ssh_command = "ssh root@localhost"
        password = input("Введіть пароль SSH (якщо потрібно): ").strip()
        
        run_gpu_check_remotely(ssh_command, password)
        
    elif choice == "2":
        # Запускаємо локально
        run_gpu_check_locally()
        
    elif choice == "3":
        # Тільки перевірка SSH
        if check_ssh_connection():
            print("✅ SSH з'єднання працює")
        else:
            print("❌ SSH з'єднання не працює")
            print("Потрібно налаштувати SSH з'єднання з Colab")
            setup_colab_ssh()
    
    else:
        print("❌ Невірний вибір")

if __name__ == "__main__":
    main() 