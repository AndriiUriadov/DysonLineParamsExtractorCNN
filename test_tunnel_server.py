#!/usr/bin/env python3
"""
Тестовий сервер для перевірки Cloudflare Tunnel
Створено для тестування з'єднання через Cloudflare Tunnel
"""

import http.server
import socketserver
import json
import os
from datetime import datetime

class TunnelTestHandler(http.server.SimpleHTTPRequestHandler):
    """Обробник запитів для тестування тунеля"""
    
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
                <title>DysonLineCNN - Cloudflare Tunnel Test</title>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        min-height: 100vh;
                    }}
                    .container {{
                        background: rgba(255, 255, 255, 0.1);
                        padding: 30px;
                        border-radius: 15px;
                        backdrop-filter: blur(10px);
                        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                    }}
                    h1 {{
                        text-align: center;
                        margin-bottom: 30px;
                        font-size: 2.5em;
                        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                    }}
                    .status {{
                        background: rgba(76, 175, 80, 0.2);
                        padding: 15px;
                        border-radius: 10px;
                        margin: 20px 0;
                        border-left: 5px solid #4CAF50;
                    }}
                    .info {{
                        background: rgba(33, 150, 243, 0.2);
                        padding: 15px;
                        border-radius: 10px;
                        margin: 20px 0;
                        border-left: 5px solid #2196F3;
                    }}
                    .timestamp {{
                        text-align: center;
                        font-size: 0.9em;
                        opacity: 0.8;
                        margin-top: 30px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🚀 DysonLineCNN</h1>
                    
                    <div class="status">
                        <h2>✅ Cloudflare Tunnel Активний</h2>
                        <p>Тунель "DysonLineCNN-001" успішно працює!</p>
                    </div>
                    
                    <div class="info">
                        <h3>📊 Інформація про сервер:</h3>
                        <ul>
                            <li><strong>Сервер:</strong> Python HTTP Server</li>
                            <li><strong>Порт:</strong> 8080</li>
                            <li><strong>Тунель:</strong> DysonLineCNN-001</li>
                            <li><strong>Статус:</strong> HEALTHY</li>
                        </ul>
                    </div>
                    
                    <div class="info">
                        <h3>🔧 Наступні кроки:</h3>
                        <ol>
                            <li>Налаштуйте маршрутизацію в Cloudflare Dashboard</li>
                            <li>Додайте домен до тунеля</li>
                            <li>Тестуйте з'єднання з різних пристроїв</li>
                            <li>Налаштуйте SSL сертифікат</li>
                        </ol>
                    </div>
                    
                    <div class="timestamp">
                        <p>Останнє оновлення: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
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
                "status": "healthy",
                "tunnel": "DysonLineCNN-001",
                "timestamp": datetime.now().isoformat(),
                "server": "Python HTTP Server",
                "port": 8080
            }
            
            self.wfile.write(json.dumps(status_data, indent=2).encode('utf-8'))
            
        else:
            super().do_GET()

def run_server(port=8080):
    """Запуск тестового сервера"""
    with socketserver.TCPServer(("", port), TunnelTestHandler) as httpd:
        print(f"🚀 Тестовий сервер запущено на порту {port}")
        print(f"📱 Відкрийте в браузері: http://localhost:{port}")
        print(f"🌐 Або через Cloudflare Tunnel: https://your-domain.com")
        print(f"⏹️  Для зупинки натисніть Ctrl+C")
        print("-" * 50)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Сервер зупинено")

if __name__ == "__main__":
    run_server() 