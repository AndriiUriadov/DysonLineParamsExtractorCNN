#!/usr/bin/env python3
"""
Прямий виконавець для Google Colab
Автоматично створює notebook і виконує код
"""

import os
import sys
import webbrowser
import pyperclip
import subprocess
import time

class DirectColabRunner:
    """Прямий виконавець для Colab"""
    
    def __init__(self):
        self.notebook_url = None
        
    def create_notebook_with_code(self, file_path):
        """Створює notebook з кодом"""
        print("📝 Створення notebook в Colab...")
        
        # Читаємо файл
        with open(file_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
        
        # Генеруємо повний код для Colab
        filename = os.path.basename(file_path)
        full_code = f"""# =============================================================================
# АВТОМАТИЧНО ЗГЕНЕРОВАНО З ФАЙЛУ: {filename}
# =============================================================================

# Встановлюємо необхідні бібліотеки
!pip install psutil

# =============================================================================
# ОСНОВНИЙ КОД
# =============================================================================

{code_content}

# =============================================================================
# КІНЕЦЬ КОДУ
# =============================================================================

print("\\n✅ Код успішно виконано в Colab!")
print("📊 Результати вище")
"""
        
        # Копіюємо код в буфер обміну
        try:
            pyperclip.copy(full_code)
            print("✅ Код скопійовано в буфер обміну!")
        except ImportError:
            print("⚠️  Встановіть pyperclip: pip install pyperclip")
            print("📋 Код для копіювання:")
            print("=" * 60)
            print(full_code)
            print("=" * 60)
        
        # Відкриваємо Colab
        print("🚀 Відкриваю Google Colab...")
        webbrowser.open("https://colab.research.google.com/")
        
        return True
    
    def setup_runtime_instructions(self):
        """Показує інструкції для налаштування runtime"""
        print("\n📋 ІНСТРУКЦІЯ ДЛЯ НАЛАШТУВАННЯ:")
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
    
    def run(self, file_path):
        """Основний метод запуску"""
        print("🚀 ПРЯМИЙ ЗАПУСК В GOOGLE COLAB")
        print("=" * 50)
        print(f"📁 Файл: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"❌ Файл не знайдено: {file_path}")
            return False
        
        # Створюємо notebook з кодом
        if self.create_notebook_with_code(file_path):
            self.setup_runtime_instructions()
            print("\n💡 Код готовий для вставки в Colab!")
            return True
        else:
            print("❌ Помилка створення notebook")
            return False

def main():
    """Головна функція"""
    if len(sys.argv) < 2:
        print("📋 Використання:")
        print("   python3 direct_colab_runner.py <файл.py>")
        print()
        print("📁 Приклади:")
        print("   python3 direct_colab_runner.py colab_gpu_check.py")
        print("   python3 direct_colab_runner.py improved_gpu_check_notebook.py")
        return
    
    file_path = sys.argv[1]
    runner = DirectColabRunner()
    runner.run(file_path)

if __name__ == "__main__":
    main() 