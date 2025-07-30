#!/usr/bin/env python3
"""
Скрипт для створення credentials файлу з токена Cloudflare
"""

import json
import os

def create_credentials_file():
    """Створює credentials файл з токена"""
    print("🔧 Створення credentials файлу для Cloudflare Tunnel")
    print("=" * 60)
    print()
    
    print("📋 ІНСТРУКЦІЯ:")
    print("1. У Cloudflare Dashboard знайдіть розділ 'Install and run a connector'")
    print("2. Скопіюйте токен з команди (частина після 'eyJhIjoiMG...')")
    print("3. Вставте токен нижче")
    print()
    
    # Отримуємо токен від користувача
    token = input("Введіть токен з Cloudflare Dashboard: ").strip()
    
    if not token:
        print("❌ Токен не введено")
        return False
    
    # Створюємо credentials файл
    credentials_data = {
        "AccountTag": "your_account_tag",
        "TunnelID": "66aaa8ad-3c2c-4d61-8a92-751613b5340b",
        "TunnelSecret": token
    }
    
    try:
        with open('tunnel-credentials.json', 'w') as f:
            json.dump(credentials_data, f, indent=2)
        
        print("✅ Файл tunnel-credentials.json створено!")
        print(f"📁 Розмір файлу: {os.path.getsize('tunnel-credentials.json')} байт")
        
        # Показуємо вміст (без токена)
        print("\n📄 Структура файлу:")
        safe_data = credentials_data.copy()
        safe_data['TunnelSecret'] = '***' + token[-4:] if len(token) > 4 else '***'
        print(json.dumps(safe_data, indent=2))
        
        return True
        
    except Exception as e:
        print(f"❌ Помилка створення файлу: {e}")
        return False

def test_tunnel_config():
    """Тестує конфігурацію тунеля"""
    print("\n🧪 Тестування конфігурації тунеля...")
    
    if not os.path.exists('tunnel-config.yml'):
        print("❌ Файл tunnel-config.yml не знайдено")
        return False
    
    if not os.path.exists('tunnel-credentials.json'):
        print("❌ Файл tunnel-credentials.json не знайдено")
        return False
    
    print("✅ Всі необхідні файли знайдено")
    
    # Показуємо наступні кроки
    print("\n📋 НАСТУПНІ КРОКИ:")
    print("1. Запустіть тунель: cloudflared tunnel --config tunnel-config.yml run")
    print("2. Налаштуйте маршрутизацію в Cloudflare Dashboard")
    print("3. Тестуйте з'єднання: https://dysonlinecnn.your-domain.com")
    
    return True

def main():
    """Головна функція"""
    print("🚀 Створення Cloudflare Tunnel Credentials")
    print("=" * 50)
    
    # Створюємо credentials файл
    if create_credentials_file():
        # Тестуємо конфігурацію
        test_tunnel_config()
        
        print("\n🎉 Credentials файл успішно створено!")
        print("Тепер ви можете запустити тунель.")
    else:
        print("\n❌ Не вдалося створити credentials файл")

if __name__ == "__main__":
    main() 