#!/usr/bin/env python3
"""
Скрипт для налаштування Cloudflare Tunnel
Допомагає налаштувати тунель для проекту DysonLineCNN
"""

import json
import os
import subprocess
import sys
from datetime import datetime

def print_header():
    """Виводить заголовок скрипта"""
    print("=" * 60)
    print("🚀 Cloudflare Tunnel Setup для DysonLineCNN")
    print("=" * 60)
    print()

def check_cloudflared():
    """Перевіряє чи встановлений cloudflared"""
    try:
        result = subprocess.run(['cloudflared', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ cloudflared встановлено")
            print(f"📦 Версія: {result.stdout.strip()}")
            return True
        else:
            print("❌ cloudflared не знайдено")
            return False
    except FileNotFoundError:
        print("❌ cloudflared не встановлено")
        return False

def install_cloudflared():
    """Встановлює cloudflared"""
    print("📦 Встановлення cloudflared...")
    
    if sys.platform == "darwin":  # macOS
        try:
            subprocess.run(['brew', 'install', 'cloudflare/cloudflare/cloudflared'], 
                         check=True)
            print("✅ cloudflared встановлено через Homebrew")
            return True
        except subprocess.CalledProcessError:
            print("❌ Помилка встановлення через Homebrew")
            return False
    else:
        print("⚠️  Встановіть cloudflared вручну:")
        print("   https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/")
        return False

def create_config_file():
    """Створює конфігураційний файл для тунеля"""
    config = {
        "tunnel": "66aaa8ad-3c2c-4d61-8a92-751613b5340b",
        "credentials-file": "./tunnel-credentials.json",
        "ingress": [
            {
                "hostname": "dysonlinecnn.your-domain.com",
                "service": "http://localhost:8080"
            },
            {
                "service": "http_status:404"
            }
        ]
    }
    
    with open('tunnel-config.yml', 'w') as f:
        f.write("tunnel: 66aaa8ad-3c2c-4d61-8a92-751613b5340b\n")
        f.write("credentials-file: ./tunnel-credentials.json\n")
        f.write("\n")
        f.write("ingress:\n")
        f.write("  - hostname: dysonlinecnn.your-domain.com\n")
        f.write("    service: http://localhost:8080\n")
        f.write("  - service: http_status:404\n")
    
    print("✅ Конфігураційний файл створено: tunnel-config.yml")

def show_next_steps():
    """Показує наступні кроки"""
    print("\n" + "=" * 60)
    print("📋 НАСТУПНІ КРОКИ:")
    print("=" * 60)
    
    steps = [
        "1. 🔧 Налаштуйте маршрутизацію в Cloudflare Dashboard:",
        "   - Перейдіть до Zero Trust → Access → Tunnels",
        "   - Знайдіть тунель 'DysonLineCNN-001'",
        "   - Додайте маршрут: dysonlinecnn.your-domain.com → http://localhost:8080",
        "",
        "2. 🌐 Замініть 'your-domain.com' на ваш домен",
        "",
        "3. 🚀 Запустіть тестовий сервер:",
        "   python3 test_tunnel_server.py",
        "",
        "4. 🔗 Тестуйте з'єднання:",
        "   - Локально: http://localhost:8080",
        "   - Через тунель: https://dysonlinecnn.your-domain.com",
        "",
        "5. 📊 Перевірте статус API:",
        "   https://dysonlinecnn.your-domain.com/api/status",
        "",
        "6. 🔒 Налаштуйте SSL сертифікат (автоматично через Cloudflare)",
        "",
        "7. 📱 Тестуйте з різних пристроїв та мереж"
    ]
    
    for step in steps:
        print(step)

def run_tunnel():
    """Запускає тунель"""
    print("\n🚀 Запуск Cloudflare Tunnel...")
    print("⚠️  Переконайтеся, що тестовий сервер запущено на порту 8080")
    
    try:
        subprocess.run(['cloudflared', 'tunnel', '--config', 'tunnel-config.yml', 'run'], 
                      check=True)
    except KeyboardInterrupt:
        print("\n🛑 Тунель зупинено")
    except subprocess.CalledProcessError as e:
        print(f"❌ Помилка запуску тунеля: {e}")

def main():
    """Головна функція"""
    print_header()
    
    # Перевіряємо cloudflared
    if not check_cloudflared():
        print("\n📦 Потрібно встановити cloudflared...")
        if not install_cloudflared():
            print("❌ Не вдалося встановити cloudflared")
            return
    
    print("\n✅ cloudflared готовий до використання")
    
    # Створюємо конфігурацію
    create_config_file()
    
    # Показуємо наступні кроки
    show_next_steps()
    
    # Питаємо чи запустити тунель
    print("\n" + "-" * 60)
    response = input("🚀 Запустити тунель зараз? (y/n): ").lower().strip()
    
    if response in ['y', 'yes', 'так', 'да']:
        run_tunnel()
    else:
        print("📝 Ви можете запустити тунель пізніше командою:")
        print("   cloudflared tunnel --config tunnel-config.yml run")

if __name__ == "__main__":
    main() 