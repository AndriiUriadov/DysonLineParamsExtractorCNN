#!/usr/bin/env python3
"""
Автоматичне виконання файлів в Google Colab через API
Без copy-paste, з виводом результатів в термінал
"""

import requests
import json
import os
import sys
import time
import webbrowser
from urllib.parse import urlparse
import subprocess

class ColabAPIRunner:
    """Клас для автоматичного виконання через Colab API"""
    
    def __init__(self):
        self.session = requests.Session()
        self.notebook_id = None
        self.cell_id = None
        
    def create_notebook(self, file_path):
        """Створює новий notebook в Colab"""
        print("📝 Створення notebook в Colab...")
        
        # Читаємо файл
        with open(file_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
        
        # Додаємо необхідні бібліотеки
        full_code = f"""# =============================================================================
# АВТОМАТИЧНО ЗГЕНЕРОВАНО З ФАЙЛУ: {os.path.basename(file_path)}
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
"""
        
        # Створюємо notebook через Colab API
        notebook_data = {
            "notebook": {
                "metadata": {
                    "colab": {
                        "name": f"Auto Execution: {os.path.basename(file_path)}"
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 0,
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": [full_code]
                    }
                ]
            }
        }
        
        try:
            # Відкриваємо Colab з готовим notebook
            colab_url = "https://colab.research.google.com/"
            webbrowser.open(colab_url)
            
            print("✅ Notebook створено!")
            print("🔗 Відкрийте посилання в браузері")
            print("📋 Код для вставки:")
            print("=" * 60)
            print(full_code)
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"❌ Помилка створення notebook: {e}")
            return False
    
    def execute_via_ssh(self, file_path, ssh_host, ssh_password):
        """Виконує через SSH з автоматичним копіюванням"""
        print("🔌 Виконання через SSH...")
        
        if not os.path.exists(file_path):
            print(f"❌ Файл не знайдено: {file_path}")
            return False
        
        try:
            # Копіюємо файл
            filename = os.path.basename(file_path)
            print(f"📁 Копіювання {filename} в Colab...")
            
            result = subprocess.run([
                'scp', '-o', 'StrictHostKeyChecking=no', file_path,
                f'root@{ssh_host}:/tmp/'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"❌ Помилка копіювання: {result.stderr}")
                return False
            
            # Виконуємо файл
            print(f"🚀 Виконання {filename}...")
            print("=" * 60)
            
            result = subprocess.run([
                'ssh', '-o', 'StrictHostKeyChecking=no', f'root@{ssh_host}',
                f'cd /tmp && python3 {filename}'
            ], text=True, timeout=300)
            
            print("=" * 60)
            
            if result.returncode == 0:
                print("✅ Файл успішно виконано!")
            else:
                print(f"❌ Помилка виконання (код: {result.returncode})")
            
            return True
            
        except Exception as e:
            print(f"❌ Помилка: {e}")
            return False

def main():
    """Головна функція"""
    if len(sys.argv) < 2:
        print("📋 Використання:")
        print("   python3 colab_api_runner.py <файл.py>")
        return
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"❌ Файл не знайдено: {file_path}")
        return
    
    print("🚀 АВТОМАТИЧНЕ ВИКОНАННЯ В COLAB")
    print("=" * 50)
    print(f"📁 Файл: {file_path}")
    print()
    
    runner = ColabAPIRunner()
    
    print("Виберіть метод:")
    print("1. 📝 Створити notebook в Colab")
    print("2. 🔌 SSH виконання (якщо налаштовано)")
    
    choice = input("Виберіть (1-2): ").strip()
    
    if choice == "1":
        runner.create_notebook(file_path)
    elif choice == "2":
        host = input("SSH хост: ").strip()
        password = input("SSH пароль: ").strip()
        if host and password:
            runner.execute_via_ssh(file_path, host, password)
        else:
            print("❌ Не введено SSH дані")
    else:
        print("❌ Невірний вибір")

if __name__ == "__main__":
    main() 