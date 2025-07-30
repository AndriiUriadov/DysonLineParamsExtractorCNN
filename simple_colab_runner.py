#!/usr/bin/env python3
"""
Спрощений скрипт для виконання файлів в Google Colab
З автоматичним налаштуванням SSH з'єднання
"""

import subprocess
import sys
import os
import time
import webbrowser

def open_colab_and_setup():
    """Відкриває Colab та показує інструкції"""
    print("🚀 Відкриваю Google Colab...")
    webbrowser.open("https://colab.research.google.com/")
    
    print("\n📋 ІНСТРУКЦІЯ ДЛЯ НАЛАШТУВАННЯ:")
    print("=" * 50)
    print("1. Створіть новий notebook")
    print("2. Налаштуйте runtime:")
    print("   - Runtime → Change runtime type")
    print("   - Hardware accelerator: GPU")
    print("   - Runtime shape: High-RAM")
    print("3. Вставте код в комірку:")
    print()
    print("```python")
    print("!pip install colab-ssh")
    print("from colab_ssh import launch_ssh_cloudflared")
    print("import random, string")
    print("password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))")
    print("print(f'🔐 Пароль: {password}')")
    print("launch_ssh_cloudflared(password=password, verbose=True)")
    print("```")
    print()
    print("4. Запустіть комірку (Shift + Enter)")
    print("5. Скопіюйте SSH хост та пароль")
    print("6. Поверніться в термінал і натисніть Enter")
    print("=" * 50)
    
    input("\nНатисніть Enter коли налаштуєте SSH в Colab...")

def test_ssh_connection(host, password):
    """Тестує SSH з'єднання"""
    print(f"🔍 Тестування SSH з'єднання до {host}...")
    
    try:
        result = subprocess.run([
            'ssh', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=no',
            f'root@{host}', 'echo "SSH connection test"'
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print("✅ SSH з'єднання працює!")
            return True
        else:
            print(f"❌ SSH з'єднання не працює: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Таймаут SSH з'єднання")
        return False
    except Exception as e:
        print(f"❌ Помилка SSH з'єднання: {e}")
        return False

def copy_and_run_file(host, file_path):
    """Копіює та виконує файл в Colab"""
    if not os.path.exists(file_path):
        print(f"❌ Файл не знайдено: {file_path}")
        return False
    
    filename = os.path.basename(file_path)
    print(f"📁 Копіювання файлу в Colab: {filename}")
    
    try:
        # Копіюємо файл
        result = subprocess.run([
            'scp', '-o', 'StrictHostKeyChecking=no', file_path,
            f'root@{host}:/tmp/'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"❌ Помилка копіювання: {result.stderr}")
            return False
        
        print("✅ Файл успішно скопійовано в Colab")
        
        # Виконуємо файл
        print(f"🚀 Виконання {filename} в Colab...")
        print("=" * 60)
        
        result = subprocess.run([
            'ssh', '-o', 'StrictHostKeyChecking=no', f'root@{host}',
            f'cd /tmp && python3 {filename}'
        ], text=True, timeout=300)  # 5 хвилин таймаут
        
        print("=" * 60)
        
        if result.returncode == 0:
            print("✅ Файл успішно виконано в Colab!")
        else:
            print(f"❌ Помилка виконання (код: {result.returncode})")
        
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Таймаут виконання (5 хвилин)")
        return False
    except Exception as e:
        print(f"❌ Помилка: {e}")
        return False

def main():
    """Головна функція"""
    if len(sys.argv) < 2:
        print("📋 Використання:")
        print("   python3 simple_colab_runner.py <файл.py>")
        print()
        print("📁 Приклади:")
        print("   python3 simple_colab_runner.py improved_gpu_check_notebook.py")
        print("   python3 simple_colab_runner.py colab_gpu_check.py")
        return
    
    file_path = sys.argv[1]
    
    print("🚀 Запуск файлу в Google Colab")
    print("=" * 50)
    
    # Перевіряємо файл
    if not os.path.exists(file_path):
        print(f"❌ Файл не знайдено: {file_path}")
        return
    
    print(f"📁 Файл знайдено: {file_path}")
    print()
    
    # Відкриваємо Colab та налаштовуємо SSH
    open_colab_and_setup()
    
    # Отримуємо SSH дані
    print("\n🔐 Введіть SSH дані з Colab:")
    host = input("SSH хост: ").strip()
    password = input("SSH пароль: ").strip()
    
    if not host or not password:
        print("❌ Не введено хост або пароль")
        return
    
    # Тестуємо з'єднання
    if not test_ssh_connection(host, password):
        print("\n❌ SSH з'єднання не працює.")
        print("💡 Спробуйте:")
        print("   1. Перезапустіть комірку в Colab")
        print("   2. Отримайте новий SSH хост та пароль")
        print("   3. Запустіть скрипт знову")
        return
    
    # Виконуємо файл
    copy_and_run_file(host, file_path)

if __name__ == "__main__":
    main() 