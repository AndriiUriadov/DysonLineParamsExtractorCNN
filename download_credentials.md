# 📥 Завантаження Credentials для Cloudflare Tunnel

## 🔧 Крок 1: Завантаження credentials файлу

1. **Перейдіть до Cloudflare Dashboard**
   - Відкрийте: https://dash.cloudflare.com/
   - Перейдіть до **Zero Trust** → **Access** → **Tunnels**

2. **Знайдіть ваш тунель**
   - Знайдіть тунель "DysonLineCNN-001"
   - Натисніть на нього

3. **Завантажте credentials**
   - У розділі **Install and run a connector**
   - Натисніть **Download credentials file**
   - Збережіть файл як `tunnel-credentials.json` в корені проекту

## 🔧 Крок 2: Перевірка файлу

```bash
# Перевірте, чи файл створено
ls -la tunnel-credentials.json

# Перевірте вміст (не показуйте він на публіку!)
cat tunnel-credentials.json
```

## 🔧 Крок 3: Запуск тунеля

```bash
# Запустіть тунель
cloudflared tunnel --config tunnel-config.yml run
```

## 🔧 Альтернативний спосіб

Якщо у вас немає доступу до Cloudflare Dashboard, можете створити тунель заново:

```bash
# Створіть новий тунель
cloudflared tunnel create dysonlinecnn-tunnel

# Завантажте credentials
cloudflared tunnel token dysonlinecnn-tunnel

# Отримайте Tunnel ID
cloudflared tunnel list
```

## ⚠️ Важливо

- **НЕ публікуйте** `tunnel-credentials.json` файл
- Додайте його до `.gitignore`
- Зберігайте в безпечному місці

## 📝 Приклад .gitignore

```gitignore
# Cloudflare Tunnel credentials
tunnel-credentials.json
*.pem
*.key
```

Після завантаження credentials файлу, тунель буде готовий до використання! 🚀 