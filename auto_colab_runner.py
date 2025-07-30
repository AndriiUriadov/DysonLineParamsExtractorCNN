#!/usr/bin/env python3
"""
Автоматичний скрипт для запуску GPU check в Google Colab
Автоматично налаштовує SSH з'єднання та виконує код
"""

import subprocess
import sys
import os
import time
import webbrowser
from datetime import datetime

def print_header():
    """Виводить заголовок скрипта"""
    print("=" * 60)
    print("🚀 Автоматичний запуск GPU Check в Google Colab")
    print("=" * 60)
    print()

def open_colab():
    """Відкриває Google Colab в браузері"""
    print("🌐 Відкриваємо Google Colab...")
    webbrowser.open("https://colab.research.google.com/")
    print("✅ Google Colab відкрито в браузері")
    print()

def generate_colab_code():
    """Генерує код для Colab"""
    colab_code = '''# Встановлюємо colab-ssh
!pip install colab-ssh

# Імпортуємо необхідні модулі
from colab_ssh import launch_ssh_cloudflared
import random
import string

# Генеруємо випадковий пароль
password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
print(f"🔐 Пароль для SSH: {password}")

# Запускаємо SSH сервер
launch_ssh_cloudflared(password=password, verbose=True)'''
    
    return colab_code

def show_instructions():
    """Показує інструкції для налаштування"""
    print("📋 ІНСТРУКЦІЯ ДЛЯ НАЛАШТУВАННЯ SSH:")
    print("=" * 50)
    print()
    print("1. 🌐 Google Colab вже відкрито в браузері")
    print("2. 📝 Створіть новий notebook")
    print("3. 📋 Скопіюйте наступний код в першу комірку:")
    print()
    print("-" * 50)
    print(generate_colab_code())
    print("-" * 50)
    print()
    print("4. ⚡ Запустіть комірку (Shift + Enter)")
    print("5. 📋 Скопіюйте SSH команду з виводу (виглядає як ssh -p 12345 root@localhost)")
    print("6. 🔄 Поверніться сюди і натисніть Enter")
    print()
    input("Натисніть Enter коли SSH буде налаштовано...")

def test_ssh_connection():
    """Перевіряє SSH з'єднання"""
    print("🔍 Перевіряємо SSH з'єднання...")
    
    try:
        # Перевіряємо різні порти
        for port in [22, 12345, 12346, 12347, 12348, 12349]:
            try:
                result = subprocess.run([
                    'ssh', '-p', str(port), '-o', 'ConnectTimeout=5', 
                    'root@localhost', 'echo "SSH test"'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    print(f"✅ SSH з'єднання працює на порту {port}")
                    return True, port
            except:
                continue
        
        print("❌ SSH з'єднання не знайдено")
        return False, None
        
    except Exception as e:
        print(f"❌ Помилка перевірки SSH: {e}")
        return False, None

def run_gpu_check_remotely(port):
    """Запускає GPU check в Colab"""
    print(f"🚀 Запуск improved_gpu_check_notebook.py в Colab...")
    print()
    
    try:
        # Копіюємо файл
        print("📁 Копіювання файлу в Colab...")
        copy_result = subprocess.run([
            'scp', '-P', str(port), 'improved_gpu_check_notebook.py', 
            'root@localhost:/tmp/'
        ], capture_output=True, text=True)
        
        if copy_result.returncode != 0:
            print(f"❌ Помилка копіювання: {copy_result.stderr}")
            return False
        
        print("✅ Файл скопійовано")
        
        # Виконуємо файл
        print("⚡ Виконання GPU check...")
        print("-" * 50)
        
        execute_result = subprocess.run([
            'ssh', '-p', str(port), 'root@localhost',
            'cd /tmp && python3 improved_gpu_check_notebook.py'
        ], text=True)
        
        if execute_result.returncode == 0:
            print("-" * 50)
            print("✅ GPU check успішно виконано в Colab!")
            return True
        else:
            print("❌ Помилка виконання GPU check")
            return False
            
    except Exception as e:
        print(f"❌ Помилка: {str(e)}")
        return False

def run_gpu_check_locally():
    """Запускає GPU check локально"""
    print("🔍 Запуск GPU check локально...")
    print("-" * 50)
    
    try:
        result = subprocess.run([sys.executable, 'improved_gpu_check_notebook.py'], 
                              text=True)
        
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
    print("3. 🔧 Налаштувати SSH з'єднання")
    print()
    
    choice = input("Введіть номер (1-3): ").strip()
    
    if choice == "1":
        # Перевіряємо SSH з'єднання
        ssh_works, port = test_ssh_connection()
        
        if not ssh_works:
            print("❌ SSH з'єднання не знайдено")
            print("Потрібно налаштувати SSH з'єднання з Colab")
            
            # Відкриваємо Colab
            open_colab()
            
            # Показуємо інструкції
            show_instructions()
            
            # Повторна перевірка
            ssh_works, port = test_ssh_connection()
            
            if not ssh_works:
                print("❌ SSH з'єднання все ще не працює")
                print("Перевірте налаштування та спробуйте ще раз")
                return
        
        print("✅ SSH з'єднання працює")
        
        # Запускаємо в Colab
        success = run_gpu_check_remotely(port)
        
        if success:
            print("\n🎉 GPU check успішно виконано в Google Colab!")
            print("Тепер ви можете переглянути результати в Colab")
        
    elif choice == "2":
        # Запускаємо локально
        run_gpu_check_locally()
        
    elif choice == "3":
        # Тільки налаштування SSH
        open_colab()
        show_instructions()
        
        ssh_works, port = test_ssh_connection()
        if ssh_works:
            print("✅ SSH з'єднання успішно налаштовано!")
        else:
            print("❌ SSH з'єднання не працює")
    
    else:
        print("❌ Невірний вибір")

if __name__ == "__main__":
    main() 