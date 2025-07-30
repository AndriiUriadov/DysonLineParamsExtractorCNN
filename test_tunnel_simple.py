#!/usr/bin/env python3
"""
Простий тестовий сервер для Cloudflare Tunnel
Працює без credentials файлу
"""

import http.server
import socketserver
import json
import os
from datetime import datetime

class SimpleTunnelHandler(http.server.SimpleHTTPRequestHandler):
    """Простий обробник для тестування тунеля"""
    
    def do_GET(self):
        """Обробка GET запитів"""
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            
            html_content = f"""
            <!DOCTYPE html>
            <html lang="uk">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>DysonLineCNN - Простий тест</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        max-width: 600px;
                        margin: 50px auto;
                        padding: 20px;
                        background: #f5f5f5;
                    }}
                    .container {{
                        background: white;
                        padding: 30px;
                        border-radius: 10px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }}
                    .status {{
                        background: #d4edda;
                        color: #155724;
                        padding: 15px;
                        border-radius: 5px;
                        margin: 20px 0;
                    }}
                    .info {{
                        background: #d1ecf1;
                        color: #0c5460;
                        padding: 15px;
                        border-radius: 5px;
                        margin: 20px 0;
                    }}
                    .warning {{
                        background: #fff3cd;
                        color: #856404;
                        padding: 15px;
                        border-radius: 5px;
                        margin: 20px 0;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🚀 DysonLineCNN - Простий тест</h1>
                    
                    <div class="status">
                        <h2>✅ Сервер працює!</h2>
                        <p>Час: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                    
                    <div class="info">
                        <h3>📊 Інформація:</h3>
                        <ul>
                            <li>Сервер: Python HTTP Server</li>
                            <li>Порт: 8080</li>
                            <li>Тунель: DysonLineCNN-001</li>
                            <li>Статус: Очікує налаштування</li>
                        </ul>
                    </div>
                    
                    <div class="warning">
                        <h3>⚠️ Наступні кроки:</h3>
                        <ol>
                            <li>Завантажте credentials файл з Cloudflare Dashboard</li>
                            <li>Збережіть як <code>tunnel-credentials.json</code></li>
                            <li>Запустіть тунель: <code>cloudflared tunnel --config tunnel-config.yml run</code></li>
                            <li>Налаштуйте маршрутизацію в Cloudflare Dashboard</li>
                        </ol>
                    </div>
                    
                    <div class="info">
                        <h3>🔗 Тестування:</h3>
                        <ul>
                            <li>Локально: <a href="http://localhost:8080">http://localhost:8080</a></li>
                            <li>API статус: <a href="/api/status">/api/status</a></li>
                        </ul>
                    </div>
                </div>
            </body>
            </html>
            """
            
            self.wfile.write(html_content.encode('utf-8'))
            
        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            status_data = {
                "status": "running",
                "server": "Simple Python HTTP Server",
                "port": 8080,
                "tunnel": "DysonLineCNN-001",
                "timestamp": datetime.now().isoformat(),
                "message": "Очікує налаштування credentials"
            }
            
            self.wfile.write(json.dumps(status_data, indent=2, ensure_ascii=False).encode('utf-8'))
            
        else:
            super().do_GET()

def run_simple_server(port=8080):
    """Запуск простого тестового сервера"""
    with socketserver.TCPServer(("", port), SimpleTunnelHandler) as httpd:
        print(f"🚀 Простий тестовий сервер запущено на порту {port}")
        print(f"📱 Відкрийте в браузері: http://localhost:{port}")
        print(f"📊 API статус: http://localhost:{port}/api/status")
        print(f"⏹️  Для зупинки натисніть Ctrl+C")
        print("-" * 50)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Сервер зупинено")

if __name__ == "__main__":
    run_simple_server() 