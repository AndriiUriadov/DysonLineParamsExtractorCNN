#!/usr/bin/env python3
"""
Скрипт для запуску GPU check через Cloudflare Tunnel
"""

import subprocess
import sys
import os
import requests
import json
from datetime import datetime

def print_header():
    """Виводить заголовок"""
    print("=" * 60)
    print("🚀 Запуск GPU Check через Cloudflare Tunnel")
    print("=" * 60)
    print()

def check_tunnel_status():
    """Перевіряє статус тунеля"""
    print("🔍 Перевірка статусу Cloudflare Tunnel...")
    
    try:
        # Перевіряємо локальний сервер
        local_response = requests.get("http://localhost:8080/api/status", timeout=5)
        if local_response.status_code == 200:
            print("✅ Локальний сервер працює")
        else:
            print("❌ Локальний сервер не працює")
            return False
        
        # Перевіряємо тунель
        tunnel_response = requests.get("https://dysonlinecnn.dysonline.org/api/status", timeout=10)
        if tunnel_response.status_code == 200:
            print("✅ Cloudflare Tunnel працює")
            return True
        else:
            print("❌ Cloudflare Tunnel не працює")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Помилка з'єднання: {e}")
        return False

def run_gpu_check_locally():
    """Запускає GPU check локально"""
    print("🔍 Запуск GPU check локально...")
    print("-" * 50)
    
    try:
        result = subprocess.run([sys.executable, 'improved_gpu_check_notebook.py'], 
                              text=True, capture_output=False)
        
        if result.returncode == 0:
            print("-" * 50)
            print("✅ GPU check успішно виконано локально!")
        else:
            print("❌ Помилка виконання GPU check")
            
    except Exception as e:
        print(f"❌ Помилка: {str(e)}")

def show_tunnel_info():
    """Показує інформацію про тунель"""
    print("\n📊 ІНФОРМАЦІЯ ПРО ТУНЕЛЬ:")
    print("=" * 40)
    print("🌐 Локальний доступ: http://localhost:8080")
    print("🌐 Через тунель: https://dysonlinecnn.dysonline.org")
    print("📊 API статус: https://dysonlinecnn.dysonline.org/api/status")
    print()
    print("🔧 Корисні команди:")
    print("   curl http://localhost:8080")
    print("   curl https://dysonlinecnn.dysonline.org")
    print("   curl https://dysonlinecnn.dysonline.org/api/status")

def main():
    """Головна функція"""
    print_header()
    
    # Перевіряємо чи існує файл
    if not os.path.exists('improved_gpu_check_notebook.py'):
        print("❌ Файл improved_gpu_check_notebook.py не знайдено!")
        return
    
    print("📁 Файл improved_gpu_check_notebook.py знайдено")
    print()
    
    # Перевіряємо статус тунеля
    if not check_tunnel_status():
        print("\n❌ Тунель не працює")
        print("Переконайтеся, що:")
        print("1. Тестовий сервер запущено: python3 test_tunnel_simple.py")
        print("2. Тунель запущено: cloudflared tunnel --config tunnel-config.yml run")
        return
    
    print("\n✅ Тунель працює!")
    
    # Показуємо інформацію про тунель
    show_tunnel_info()
    
    # Питаємо користувача
    print("\n🔧 Виберіть дію:")
    print("1. 🚀 Запустити GPU check локально")
    print("2. 🌐 Відкрити тунель в браузері")
    print("3. 📊 Перевірити статус тунеля")
    print()
    
    choice = input("Введіть номер (1-3): ").strip()
    
    if choice == "1":
        run_gpu_check_locally()
        
    elif choice == "2":
        print("🌐 Відкриваємо тунель в браузері...")
        import webbrowser
        webbrowser.open("https://dysonlinecnn.dysonline.org")
        
    elif choice == "3":
        print("📊 Перевірка статусу тунеля...")
        try:
            response = requests.get("https://dysonlinecnn.dysonline.org/api/status", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print("✅ Тунель працює:")
                print(f"   Статус: {data.get('status', 'unknown')}")
                print(f"   Сервер: {data.get('server', 'unknown')}")
                print(f"   Порт: {data.get('port', 'unknown')}")
                print(f"   Тунель: {data.get('tunnel', 'unknown')}")
                print(f"   Час: {data.get('timestamp', 'unknown')}")
            else:
                print(f"❌ Помилка: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ Помилка: {e}")
    
    else:
        print("❌ Невірний вибір")

if __name__ == "__main__":
    main() 