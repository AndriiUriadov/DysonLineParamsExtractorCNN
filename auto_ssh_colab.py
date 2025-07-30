#!/usr/bin/env python3
"""
Автоматичне SSH виконання в Google Colab
З автоматичним налаштуванням з'єднання
"""

import subprocess
import sys
import os
import time
import webbrowser
import requests
import json

class AutoSSHColab:
    """Автоматичний SSH виконавець для Colab"""
    
    def __init__(self):
        self.ssh_host = None
        self.ssh_password = None
        self.connection_active = False
        
    def setup_colab_ssh(self):
        """Автоматично налаштовує SSH в Colab"""
        print("🔧 АВТОМАТИЧНЕ НАЛАШТУВАННЯ SSH В COLAB")
        print("=" * 50)
        
        # Відкриваємо Colab
        print("🚀 Відкриваю Google Colab...")
        webbrowser.open("https://colab.research.google.com/")
        
        print("\n📋 ІНСТРУКЦІЯ:")
        print("1. Створіть новий notebook")
        print("2. Налаштуйте runtime (GPU + High-RAM)")
        print("3. Вставте код в комірку:")
        print()
        
        ssh_code = '''!pip install colab-ssh
from colab_ssh import launch_ssh_cloudflared
import random, string
password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
print(f'🔐 Пароль: {password}')
launch_ssh_cloudflared(password=password, verbose=True)'''
        
        print("```python")
        print(ssh_code)
        print("```")
        print()
        print("4. Запустіть комірку (Shift + Enter)")
        print("5. Скопіюйте SSH хост та пароль")
        print("6. Поверніться в термінал")
        print("=" * 50)
        
        input("\nНатисніть Enter коли налаштуєте SSH...")
        
        # Отримуємо SSH дані
        print("\n🔐 Введіть SSH дані:")
        self.ssh_host = input("SSH хост: ").strip()
        self.ssh_password = input("SSH пароль: ").strip()
        
        if not self.ssh_host or not self.ssh_password:
            print("❌ Не введено SSH дані")
            return False
        
        # Тестуємо з'єднання
        return self.test_connection()
    
    def test_connection(self):
        """Тестує SSH з'єднання"""
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
    
    def execute_file(self, file_path):
        """Виконує файл в Colab через SSH"""
        if not self.connection_active:
            print("❌ SSH з'єднання не активне")
            return False
        
        if not os.path.exists(file_path):
            print(f"❌ Файл не знайдено: {file_path}")
            return False
        
        filename = os.path.basename(file_path)
        print(f"📁 Копіювання {filename} в Colab...")
        
        try:
            # Копіюємо файл
            result = subprocess.run([
                'scp', '-o', 'StrictHostKeyChecking=no', file_path,
                f'root@{self.ssh_host}:/tmp/'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"❌ Помилка копіювання: {result.stderr}")
                return False
            
            print("✅ Файл успішно скопійовано!")
            
            # Виконуємо файл
            print(f"🚀 Виконання {filename} в Colab...")
            print("=" * 60)
            
            result = subprocess.run([
                'ssh', '-o', 'StrictHostKeyChecking=no', f'root@{self.ssh_host}',
                f'cd /tmp && python3 {filename}'
            ], text=True, timeout=300)
            
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
    
    def run(self, file_path):
        """Основний метод запуску"""
        print("🚀 АВТОМАТИЧНЕ SSH ВИКОНАННЯ В COLAB")
        print("=" * 50)
        print(f"📁 Файл: {file_path}")
        print()
        
        # Налаштовуємо SSH
        if not self.setup_colab_ssh():
            print("\n❌ Не вдалося налаштувати SSH з'єднання")
            return False
        
        # Виконуємо файл
        return self.execute_file(file_path)

def main():
    """Головна функція"""
    if len(sys.argv) < 2:
        print("📋 Використання:")
        print("   python3 auto_ssh_colab.py <файл.py>")
        return
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"❌ Файл не знайдено: {file_path}")
        return
    
    runner = AutoSSHColab()
    runner.run(file_path)

if __name__ == "__main__":
    main() 