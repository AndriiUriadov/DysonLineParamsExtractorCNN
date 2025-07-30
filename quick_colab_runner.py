#!/usr/bin/env python3
"""
Швидкий скрипт для запуску файлів в Google Colab
З вибором методу виконання
"""

import os
import sys
import subprocess
import webbrowser

def show_menu():
    """Показує меню вибору"""
    print("🚀 ШВИДКИЙ ЗАПУСК В GOOGLE COLAB")
    print("=" * 50)
    print("Виберіть метод виконання:")
    print("1. 📋 Копіювати код в Colab (рекомендовано)")
    print("2. 🔌 SSH з'єднання (експериментально)")
    print("3. 📖 Показати інструкції")
    print("4. ❌ Вихід")
    print("=" * 50)

def copy_to_colab(file_path):
    """Копіює код в Colab"""
    print("📋 Копіювання коду в Colab...")
    try:
        result = subprocess.run([
            "python3", "update_colab_connection.py", file_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Код готовий для вставки в Colab!")
            print("💡 Відкрийте Colab і вставте код (Cmd+V)")
        else:
            print("❌ Помилка копіювання")
            
    except Exception as e:
        print(f"❌ Помилка: {e}")

def ssh_execution(file_path):
    """Виконує через SSH"""
    print("🔌 SSH виконання...")
    try:
        result = subprocess.run([
            "python3", "simple_colab_runner.py", file_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ SSH виконання завершено!")
        else:
            print("❌ Помилка SSH виконання")
            
    except Exception as e:
        print(f"❌ Помилка: {e}")

def show_instructions():
    """Показує інструкції"""
    print("\n📖 ІНСТРУКЦІЇ ДЛЯ РОБОТИ З COLAB")
    print("=" * 50)
    print("1. Відкрийте Google Colab: https://colab.research.google.com/")
    print("2. Створіть новий notebook")
    print("3. Налаштуйте runtime:")
    print("   - Runtime → Change runtime type")
    print("   - Hardware accelerator: GPU")
    print("   - Runtime shape: High-RAM")
    print("4. Вставте код в комірку")
    print("5. Запустіть комірку (Shift + Enter)")
    print("6. Результати з'являться в Colab")
    print("=" * 50)
    
    # Відкриваємо Colab
    webbrowser.open("https://colab.research.google.com/")

def main():
    """Головна функція"""
    if len(sys.argv) < 2:
        print("📋 Використання:")
        print("   python3 quick_colab_runner.py <файл.py>")
        print()
        print("📁 Приклади:")
        print("   python3 quick_colab_runner.py improved_gpu_check_notebook.py")
        return
    
    file_path = sys.argv[1]
    
    # Перевіряємо файл
    if not os.path.exists(file_path):
        print(f"❌ Файл не знайдено: {file_path}")
        return
    
    print(f"📁 Файл: {file_path}")
    print()
    
    while True:
        show_menu()
        choice = input("Виберіть опцію (1-4): ").strip()
        
        if choice == "1":
            copy_to_colab(file_path)
            break
        elif choice == "2":
            ssh_execution(file_path)
            break
        elif choice == "3":
            show_instructions()
            break
        elif choice == "4":
            print("👋 До побачення!")
            break
        else:
            print("❌ Невірний вибір. Спробуйте ще раз.")

if __name__ == "__main__":
    main() 