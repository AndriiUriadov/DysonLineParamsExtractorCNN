#!/usr/bin/env python3
"""
Скрипт для налаштування SSH з'єднання з Google Colab
"""

import subprocess
import sys
import os
import time
import webbrowser

def print_header():
    """Виводить заголовок"""
    print("=" * 60)
    print("🔧 Налаштування SSH з'єднання з Google Colab")
    print("=" * 60)
    print()

def check_ssh_keys():
    """Перевіряє наявність SSH ключів"""
    print("🔍 Перевірка SSH ключів...")
    
    ssh_dir = os.path.expanduser("~/.ssh")
    if not os.path.exists(ssh_dir):
        print("❌ Директорія ~/.ssh не знайдена")
        return False
    
    # Перевіряємо наявність приватного ключа
    private_key = os.path.join(ssh_dir, "id_ed25519")
    if os.path.exists(private_key):
        print("✅ Приватний ключ знайдено: id_ed25519")
        return True
    else:
        print("❌ Приватний ключ не знайдено")
        return False

def generate_ssh_key():
    """Генерує SSH ключ"""
    print("🔑 Генерування SSH ключа...")
    
    try:
        result = subprocess.run([
            'ssh-keygen', '-t', 'ed25519', '-f', '~/.ssh/id_ed25519', '-N', ''
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ SSH ключ успішно згенеровано")
            return True
        else:
            print(f"❌ Помилка генерації ключа: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Помилка: {e}")
        return False

def open_colab():
    """Відкриває Google Colab"""
    print("🌐 Відкриваємо Google Colab...")
    webbrowser.open("https://colab.research.google.com/")
    print("✅ Google Colab відкрито в браузері")

def show_colab_instructions():
    """Показує інструкції для Colab"""
    print("\n📋 ІНСТРУКЦІЯ ДЛЯ COLAB:")
    print("=" * 50)
    print()
    print("1. 🌐 Google Colab вже відкрито в браузері")
    print("2. 📝 Створіть новий notebook")
    print("3. ⚙️  Налаштуйте runtime:")
    print("   - Runtime → Change runtime type")
    print("   - Hardware accelerator: GPU")
    print("   - Runtime shape: High-RAM")
    print("4. 📋 Вставте код в першу комірку:")
    print()
    
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
    
    print("-" * 50)
    print(colab_code)
    print("-" * 50)
    print()
    print("5. ⚡ Запустіть комірку (Shift + Enter)")
    print("6. 📋 Скопіюйте SSH команду з виводу")
    print("7. 🔄 Поверніться сюди і натисніть Enter")
    print()

def test_ssh_connection(host, password):
    """Тестує SSH з'єднання"""
    print(f"🔍 Тестування SSH з'єднання до {host}...")
    
    try:
        # Тестуємо з'єднання з таймаутом
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

def run_gpu_check_ssh(host, password):
    """Запускає GPU check через SSH"""
    print(f"🚀 Запуск GPU check через SSH...")
    print()
    
    # Копіюємо файл
    print("📁 Копіювання файлу...")
    copy_cmd = f"scp -o StrictHostKeyChecking=no improved_gpu_check_notebook.py root@{host}:/tmp/"
    
    try:
        copy_result = subprocess.run(copy_cmd, shell=True, capture_output=True, text=True)
        
        if copy_result.returncode != 0:
            print(f"❌ Помилка копіювання: {copy_result.stderr}")
            return False
        
        print("✅ Файл скопійовано")
        
        # Виконуємо GPU check
        print("⚡ Виконання GPU check...")
        print("-" * 50)
        
        execute_cmd = f"ssh -o StrictHostKeyChecking=no root@{host} 'cd /tmp && python3 improved_gpu_check_notebook.py'"
        execute_result = subprocess.run(execute_cmd, shell=True, text=True)
        
        if execute_result.returncode == 0:
            print("-" * 50)
            print("✅ GPU check успішно виконано!")
            return True
        else:
            print("❌ Помилка виконання GPU check")
            return False
            
    except Exception as e:
        print(f"❌ Помилка: {e}")
        return False

def main():
    """Головна функція"""
    print_header()
    
    # Перевіряємо SSH ключі
    if not check_ssh_keys():
        print("🔑 Потрібно згенерувати SSH ключ...")
        if not generate_ssh_key():
            print("❌ Не вдалося згенерувати SSH ключ")
            return
    
    # Відкриваємо Colab
    open_colab()
    
    # Показуємо інструкції
    show_colab_instructions()
    
    # Очікуємо введення користувача
    input("Натисніть Enter коли SSH буде налаштовано в Colab...")
    
    # Отримуємо дані для SSH
    print("\n🔧 Введіть дані SSH з Colab:")
    host = input("SSH хост (наприклад, partition-been-indoor-barrier.trycloudflare.com): ").strip()
    password = input("SSH пароль: ").strip()
    
    if not host or not password:
        print("❌ Не введено хост або пароль")
        return
    
    # Тестуємо з'єднання
    if test_ssh_connection(host, password):
        print("\n🎉 SSH з'єднання налаштовано!")
        
        # Питаємо чи запустити GPU check
        choice = input("\n🚀 Запустити GPU check через SSH? (y/n): ").strip().lower()
        
        if choice in ['y', 'yes', 'так', 'да']:
            if not os.path.exists('improved_gpu_check_notebook.py'):
                print("❌ Файл improved_gpu_check_notebook.py не знайдено")
                return
            
            success = run_gpu_check_ssh(host, password)
            if success:
                print("\n🎉 GPU check успішно виконано в Colab!")
            else:
                print("\n❌ Не вдалося виконати GPU check")
        else:
            print("\n📝 Ви можете запустити GPU check пізніше командою:")
            print(f"   scp improved_gpu_check_notebook.py root@{host}:/tmp/")
            print(f"   ssh root@{host} 'cd /tmp && python3 improved_gpu_check_notebook.py'")
    else:
        print("\n❌ SSH з'єднання не працює")
        print("Перевірте налаштування в Colab та спробуйте ще раз")

if __name__ == "__main__":
    main() 