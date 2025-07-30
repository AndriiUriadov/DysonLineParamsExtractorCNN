#!/usr/bin/env python3
"""
Скрипт для генерації коду для прямого виконання в Google Colab
Без використання SSH - просто копіюємо код в Colab
"""

import os
import sys
import webbrowser
import pyperclip  # pip install pyperclip

def read_file_content(file_path):
    """Читає вміст файлу"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Помилка читання файлу: {e}")
        return None

def generate_colab_code(file_content, filename):
    """Генерує код для виконання в Colab"""
    colab_code = f'''# =============================================================================
# КОД ДЛЯ ВИКОНАННЯ В GOOGLE COLAB
# Файл: {filename}
# =============================================================================

# Встановлюємо необхідні бібліотеки
!pip install psutil

# =============================================================================
# ОСНОВНИЙ КОД З ФАЙЛУ {filename}
# =============================================================================

{file_content}

# =============================================================================
# КІНЕЦЬ КОДУ
# =============================================================================
'''
    return colab_code

def main():
    """Головна функція"""
    if len(sys.argv) < 2:
        print("📋 Використання:")
        print("   python3 update_colab_connection.py <файл.py>")
        print()
        print("📁 Приклади:")
        print("   python3 update_colab_connection.py improved_gpu_check_notebook.py")
        return
    
    file_path = sys.argv[1]
    
    # Перевіряємо файл
    if not os.path.exists(file_path):
        print(f"❌ Файл не знайдено: {file_path}")
        return
    
    print("🚀 Генерація коду для Google Colab")
    print("=" * 50)
    print(f"📁 Файл: {file_path}")
    
    # Читаємо вміст файлу
    file_content = read_file_content(file_path)
    if not file_content:
        return
    
    # Генеруємо код для Colab
    filename = os.path.basename(file_path)
    colab_code = generate_colab_code(file_content, filename)
    
    # Копіюємо в буфер обміну
    try:
        pyperclip.copy(colab_code)
        print("✅ Код скопійовано в буфер обміну!")
    except ImportError:
        print("⚠️  Встановіть pyperclip: pip install pyperclip")
        print("📋 Код для копіювання:")
        print("=" * 60)
        print(colab_code)
        print("=" * 60)
        return
    
    # Відкриваємо Colab
    print("🚀 Відкриваю Google Colab...")
    webbrowser.open("https://colab.research.google.com/")
    
    print("\n📋 ІНСТРУКЦІЯ:")
    print("=" * 50)
    print("1. Створіть новий notebook")
    print("2. Налаштуйте runtime:")
    print("   - Runtime → Change runtime type")
    print("   - Hardware accelerator: GPU")
    print("   - Runtime shape: High-RAM")
    print("3. Вставте код в комірку (Cmd+V)")
    print("4. Запустіть комірку (Shift + Enter)")
    print("5. Результати з'являться в Colab")
    print("=" * 50)
    
    print("\n💡 Код вже скопійовано в буфер обміну!")
    print("   Просто вставте його в Colab (Cmd+V)")

if __name__ == "__main__":
    main() 