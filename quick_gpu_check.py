#!/usr/bin/env python3
"""
Швидкий запуск GPU check з інструкціями для Colab
"""

import subprocess
import sys
import os

def run_gpu_check():
    """Запускає GPU check"""
    print("🔍 Запуск GPU check...")
    print("-" * 50)
    
    try:
        result = subprocess.run([sys.executable, 'improved_gpu_check_notebook.py'], 
                              text=True)
        
        if result.returncode == 0:
            print("-" * 50)
            print("✅ GPU check успішно виконано!")
        else:
            print("❌ Помилка виконання GPU check")
            
    except Exception as e:
        print(f"❌ Помилка: {str(e)}")

def show_colab_instructions():
    """Показує інструкції для Colab"""
    print("\n" + "=" * 60)
    print("🚀 ІНСТРУКЦІЯ ДЛЯ ЗАПУСКУ В GOOGLE COLAB")
    print("=" * 60)
    print()
    print("Для запуску в Google Colab (рекомендовано для GPU):")
    print()
    print("1. 🌐 Відкрийте: https://colab.research.google.com/")
    print("2. 📝 Створіть новий notebook")
    print("3. 📋 Вставте код в першу комірку:")
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
    print("4. ⚡ Запустіть комірку (Shift + Enter)")
    print("5. 📋 Скопіюйте SSH команду з виводу")
    print("6. 🔄 Виконайте команду в терміналі")
    print("7. 🚀 Запустіть: python3 run_in_colab.py improved_gpu_check_notebook.py")
    print()

def main():
    """Головна функція"""
    print("🚀 Швидкий запуск GPU Check")
    print("=" * 40)
    
    # Перевіряємо чи існує файл
    if not os.path.exists('improved_gpu_check_notebook.py'):
        print("❌ Файл improved_gpu_check_notebook.py не знайдено!")
        return
    
    # Запускаємо GPU check
    run_gpu_check()
    
    # Показуємо інструкції для Colab
    show_colab_instructions()

if __name__ == "__main__":
    main() 