#!/usr/bin/env python3
"""
Виконання через Cloudflare Tunnel HTTP API
Використовує налаштований Cloudflare Tunnel
"""

import requests
import json
import os
import sys
import subprocess
import time

class CloudflareAPIRunner:
    """Виконавець через Cloudflare Tunnel"""
    
    def __init__(self):
        self.tunnel_url = None
        self.server_running = False
        
    def start_local_server(self):
        """Запускає локальний сервер"""
        print("🚀 Запуск локального сервера...")
        
        try:
            # Запускаємо сервер в фоновому режимі
            self.server_process = subprocess.Popen([
                'python3', 'test_tunnel_simple.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            time.sleep(2)  # Даємо час на запуск
            
            # Перевіряємо чи сервер запущений
            try:
                response = requests.get('http://localhost:8080/api/status', timeout=5)
                if response.status_code == 200:
                    print("✅ Локальний сервер запущений!")
                    self.server_running = True
                    return True
            except:
                pass
            
            print("❌ Не вдалося запустити локальний сервер")
            return False
            
        except Exception as e:
            print(f"❌ Помилка запуску сервера: {e}")
            return False
    
    def start_cloudflare_tunnel(self):
        """Запускає Cloudflare Tunnel"""
        print("🌐 Запуск Cloudflare Tunnel...")
        
        try:
            # Запускаємо tunnel в фоновому режимі
            self.tunnel_process = subprocess.Popen([
                'cloudflared', 'tunnel', '--config', 'tunnel-config.yml', 'run'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            time.sleep(3)  # Даємо час на підключення
            
            # Отримуємо URL з конфігурації
            try:
                with open('tunnel-config.yml', 'r') as f:
                    config = f.read()
                    if 'hostname:' in config:
                        for line in config.split('\n'):
                            if 'hostname:' in line:
                                self.tunnel_url = f"https://{line.split(':')[1].strip()}"
                                break
                
                if self.tunnel_url:
                    print(f"✅ Cloudflare Tunnel запущений: {self.tunnel_url}")
                    return True
                else:
                    print("❌ Не знайдено URL в конфігурації")
                    return False
                    
            except Exception as e:
                print(f"❌ Помилка читання конфігурації: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Помилка запуску tunnel: {e}")
            return False
    
    def execute_file_via_api(self, file_path):
        """Виконує файл через HTTP API"""
        if not self.tunnel_url:
            print("❌ Cloudflare Tunnel не налаштований")
            return False
        
        if not os.path.exists(file_path):
            print(f"❌ Файл не знайдено: {file_path}")
            return False
        
        # Читаємо файл
        with open(file_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
        
        # Відправляємо код на виконання
        print(f"📁 Відправка {os.path.basename(file_path)} на виконання...")
        
        try:
            response = requests.post(f"{self.tunnel_url}/api/execute", json={
                'code': code_content,
                'filename': os.path.basename(file_path)
            }, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Файл успішно виконано!")
                print("\n📊 РЕЗУЛЬТАТИ:")
                print("=" * 60)
                print(result.get('output', 'Немає виводу'))
                print("=" * 60)
                return True
            else:
                print(f"❌ Помилка виконання: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Помилка API запиту: {e}")
            return False
    
    def run(self, file_path):
        """Основний метод запуску"""
        print("🚀 ВИКОНАННЯ ЧЕРЕЗ CLOUDFLARE TUNNEL")
        print("=" * 50)
        print(f"📁 Файл: {file_path}")
        print()
        
        # Запускаємо локальний сервер
        if not self.start_local_server():
            print("❌ Не вдалося запустити локальний сервер")
            return False
        
        # Запускаємо Cloudflare Tunnel
        if not self.start_cloudflare_tunnel():
            print("❌ Не вдалося запустити Cloudflare Tunnel")
            return False
        
        # Виконуємо файл
        return self.execute_file_via_api(file_path)

def main():
    """Головна функція"""
    if len(sys.argv) < 2:
        print("📋 Використання:")
        print("   python3 cloudflare_api_runner.py <файл.py>")
        return
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"❌ Файл не знайдено: {file_path}")
        return
    
    runner = CloudflareAPIRunner()
    runner.run(file_path)

if __name__ == "__main__":
    main() 