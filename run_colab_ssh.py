#!/usr/bin/env python3
"""
Скрипт для запуску GPU check через SSH з'єднання з Colab
"""

import subprocess
import sys
import os

def run_gpu_check_via_ssh():
    """Запускає GPU check через SSH з'єднання з Colab"""
    print("🚀 Запуск GPU check через SSH з'єднання з Colab")
    print("=" * 60)
    print()
    
    # SSH команда з Colab
    ssh_host = "partition-been-indoor-barrier.trycloudflare.com"
    ssh_password = "N5509oaF2k5J"
    
    print(f"🔗 Підключення до: {ssh_host}")
    print(f"🔐 Пароль: {ssh_password}")
    print()
    
    # Створюємо команду для копіювання файлу
    copy_command = f"scp improved_gpu_check_notebook.py root@{ssh_host}:/tmp/"
    
    # Створюємо команду для виконання
    execute_command = f"ssh root@{ssh_host} 'cd /tmp && python3 improved_gpu_check_notebook.py'"
    
    try:
        # Копіюємо файл
        print("📁 Копіювання файлу в Colab...")
        copy_result = subprocess.run(copy_command, shell=True, capture_output=True, text=True)
        
        if copy_result.returncode != 0:
            print(f"❌ Помилка копіювання: {copy_result.stderr}")
            print("💡 Спробуйте підключитися вручну:")
            print(f"   ssh {ssh_host}")
            print(f"   Пароль: {ssh_password}")
            return False
        
        print("✅ Файл скопійовано")
        
        # Виконуємо файл
        print("⚡ Виконання GPU check в Colab...")
        print("-" * 50)
        
        execute_result = subprocess.run(execute_command, shell=True, text=True)
        
        if execute_result.returncode == 0:
            print("-" * 50)
            print("✅ GPU check успішно виконано в Colab!")
            return True
        else:
            print("❌ Помилка виконання GPU check")
            return False
            
    except Exception as e:
        print(f"❌ Помилка: {str(e)}")
        print("💡 Спробуйте підключитися вручну:")
        print(f"   ssh {ssh_host}")
        print(f"   Пароль: {ssh_password}")
        return False

def show_manual_instructions():
    """Показує інструкції для ручного підключення"""
    print("\n" + "=" * 60)
    print("🔧 РУЧНЕ ПІДКЛЮЧЕННЯ ДО COLAB")
    print("=" * 60)
    print()
    print("Якщо автоматичне підключення не працює:")
    print()
    print("1. 🔗 Підключіться до Colab:")
    print("   ssh partition-been-indoor-barrier.trycloudflare.com")
    print("   Пароль: N5509oaF2k5J")
    print()
    print("2. 📁 Скопіюйте файл:")
    print("   scp improved_gpu_check_notebook.py root@partition-been-indoor-barrier.trycloudflare.com:/tmp/")
    print()
    print("3. ⚡ Виконайте GPU check:")
    print("   cd /tmp && python3 improved_gpu_check_notebook.py")
    print()
    print("4. 📊 Перевірте GPU:")
    print("   nvidia-smi")
    print()
    print("5. 💾 Перевірте RAM:")
    print("   free -h")
    print()

def main():
    """Головна функція"""
    print("🚀 Запуск GPU Check в Google Colab")
    print("=" * 50)
    
    # Перевіряємо чи існує файл
    if not os.path.exists('improved_gpu_check_notebook.py'):
        print("❌ Файл improved_gpu_check_notebook.py не знайдено!")
        return
    
    print("📁 Файл improved_gpu_check_notebook.py знайдено")
    print()
    
    # Питаємо користувача
    print("🔧 Виберіть спосіб:")
    print("1. 🚀 Автоматичний запуск через SSH")
    print("2. 🔧 Показати інструкції для ручного підключення")
    print("3. 💻 Запустити локально")
    print()
    
    choice = input("Введіть номер (1-3): ").strip()
    
    if choice == "1":
        success = run_gpu_check_via_ssh()
        if not success:
            show_manual_instructions()
    
    elif choice == "2":
        show_manual_instructions()
    
    elif choice == "3":
        print("🔍 Запуск GPU check локально...")
        subprocess.run([sys.executable, 'improved_gpu_check_notebook.py'])
    
    else:
        print("❌ Невірний вибір")

if __name__ == "__main__":
    main() 