#!/usr/bin/env python3
"""
Скрипт для виконання файлів в Google Colab через SSH
З виводом результатів в термінал Cursor
"""

import subprocess
import sys
import os
import time
from datetime import datetime

class ColabExecutor:
    """Клас для виконання файлів в Colab"""
    
    def __init__(self, ssh_host=None, ssh_password=None):
        self.ssh_host = ssh_host
        self.ssh_password = ssh_password
        self.connection_active = False
        
    def setup_ssh_connection(self):
        """Налаштовує SSH з'єднання"""
        print("🔧 Налаштування SSH з'єднання з Colab...")
        print()
        
        if not self.ssh_host:
            print("📋 ІНСТРУКЦІЯ:")
            print("1. Відкрийте Google Colab: https://colab.research.google.com/")
            print("2. Створіть новий notebook")
            print("3. Налаштуйте runtime (GPU + High-RAM)")
            print("4. Вставте код в комірку:")
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
            print("5. Запустіть комірку (Shift + Enter)")
            print("6. Скопіюйте SSH хост та пароль")
            print()
            
            self.ssh_host = input("SSH хост (наприклад, partition-been-indoor-barrier.trycloudflare.com): ").strip()
            self.ssh_password = input("SSH пароль: ").strip()
            
            if not self.ssh_host or not self.ssh_password:
                print("❌ Не введено хост або пароль")
                return False
        
        # Тестуємо з'єднання
        print(f"🔍 Тестування SSH з'єднання до {self.ssh_host}...")
        
        try:
            result = subprocess.run([
                'ssh', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=no',
                f'root@{self.ssh_host}', 'echo "SSH connection test"'
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                print("✅ SSH з'єднання працює!")
                self.connection_active = True
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
    
    def copy_file_to_colab(self, file_path):
        """Копіює файл в Colab"""
        if not self.connection_active:
            print("❌ SSH з'єднання не активне")
            return False
            
        if not os.path.exists(file_path):
            print(f"❌ Файл не знайдено: {file_path}")
            return False
            
        print(f"📁 Копіювання файлу в Colab: {file_path}")
        
        try:
            result = subprocess.run([
                'scp', '-o', 'StrictHostKeyChecking=no', file_path,
                f'root@{self.ssh_host}:/tmp/'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("✅ Файл успішно скопійовано в Colab")
                return True
            else:
                print(f"❌ Помилка копіювання: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Помилка: {e}")
            return False
    
    def execute_file_in_colab(self, file_path):
        """Виконує файл в Colab з виводом в термінал"""
        if not self.connection_active:
            print("❌ SSH з'єднання не активне")
            return False
            
        # Копіюємо файл
        if not self.copy_file_to_colab(file_path):
            return False
        
        # Виконуємо файл
        filename = os.path.basename(file_path)
        print(f"🚀 Виконання {filename} в Colab...")
        print("=" * 60)
        
        try:
            result = subprocess.run([
                'ssh', '-o', 'StrictHostKeyChecking=no', f'root@{self.ssh_host}',
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
    
    def run_file(self, file_path):
        """Основний метод для запуску файлу"""
        print("🚀 Запуск файлу в Google Colab")
        print("=" * 50)
        
        # Перевіряємо файл
        if not os.path.exists(file_path):
            print(f"❌ Файл не знайдено: {file_path}")
            return False
        
        print(f"📁 Файл знайдено: {file_path}")
        print()
        
        # Налаштовуємо SSH
        if not self.setup_ssh_connection():
            return False
        
        # Виконуємо файл
        return self.execute_file_in_colab(file_path)

def main():
    """Головна функція"""
    if len(sys.argv) < 2:
        print("📋 Використання:")
        print("   python3 run_in_colab.py <файл.py>")
        print()
        print("📁 Приклади:")
        print("   python3 run_in_colab.py improved_gpu_check_notebook.py")
        print("   python3 run_in_colab.py colab_gpu_check.py")
        return
    
    file_path = sys.argv[1]
    executor = ColabExecutor()
    executor.run_file(file_path)

if __name__ == "__main__":
    main() 