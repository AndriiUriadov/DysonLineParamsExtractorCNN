# =============================================================================
# ПЕРЕВІРКА СИСТЕМНИХ РЕСУРСІВ СЕРЕДОВИЩА ВИКОНАННЯ
# =============================================================================

# Імпортуємо модулі
import subprocess
import sys
import platform
import warnings

# Спробуємо імпортувати psutil, якщо не встановлено - встановимо
try:
    from psutil import virtual_memory
except ImportError:
    print("📦 Встановлюємо psutil...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    from psutil import virtual_memory

def check_gpu_availability():
    """
    Перевіряє доступність GPU та повертає детальну інформацію
    Returns:
        dict: Словник з інформацією про GPU
    """
    gpu_info = {
        'available': False,
        'name': 'Unknown',
        'memory_total': 0,
        'memory_free': 0,
        'memory_used': 0,
        'driver_version': 'Unknown',
        'cuda_version': 'Unknown',
        'temperature': 0,
        'utilization': 0,
        'error_message': None
    }
    
    try:
        # Виконуємо команду nvidia-smi для отримання інформації про GPU
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              text=True, 
                              timeout=15)
        
        if result.returncode == 0:
            output = result.stdout
            
            # Перевіряємо наявність помилок у виводі
            error_indicators = ['failed', 'error', 'not found', 'no devices', 'nvidia-smi has failed']
            has_error = any(indicator in output.lower() for indicator in error_indicators)
            
            if not has_error and 'nvidia' in output.lower():
                gpu_info['available'] = True
                
                # Парсимо інформацію про GPU
                lines = output.split('\n')
                for line in lines:
                    line_lower = line.lower()
                    
                    # Драйвер версія
                    if 'driver version' in line_lower:
                        gpu_info['driver_version'] = line.split(':')[-1].strip()
                    
                    # CUDA версія
                    elif 'cuda version' in line_lower:
                        gpu_info['cuda_version'] = line.split(':')[-1].strip()
                    
                    # Назва GPU
                    elif 'nvidia' in line_lower and any(gpu in line_lower for gpu in ['a100', 'v100', 't4', 'k80', 'p100']):
                        gpu_info['name'] = line.strip()
                    
                    # Пам'ять GPU
                    elif 'memory' in line_lower and 'mi' in line_lower:
                        try:
                            memory_parts = line.split('|')
                            if len(memory_parts) >= 3:
                                memory_info = memory_parts[2].strip()
                                if 'MiB' in memory_info:
                                    # Формат: "0MiB / 40960MiB"
                                    memory_values = memory_info.split('/')
                                    if len(memory_values) >= 2:
                                        used_mem = memory_values[0].replace('MiB', '').strip()
                                        total_mem = memory_values[1].replace('MiB', '').strip()
                                        gpu_info['memory_used'] = int(used_mem)
                                        gpu_info['memory_total'] = int(total_mem)
                                        gpu_info['memory_free'] = gpu_info['memory_total'] - gpu_info['memory_used']
                        except (ValueError, IndexError):
                            pass
                    
                    # Температура
                    elif 'temp' in line_lower and '°c' in line_lower:
                        try:
                            temp_part = line.split('°C')[0].split()[-1]
                            gpu_info['temperature'] = int(temp_part)
                        except (ValueError, IndexError):
                            pass
                    
                    # Використання GPU
                    elif 'gpu-util' in line_lower:
                        try:
                            util_part = line.split('%')[0].split()[-1]
                            gpu_info['utilization'] = int(util_part)
                        except (ValueError, IndexError):
                            pass
            else:
                gpu_info['error_message'] = 'GPU не знайдено або недоступно'
        else:
            gpu_info['error_message'] = f'Помилка виконання nvidia-smi: {result.stderr}'
            
    except subprocess.TimeoutExpired:
        gpu_info['error_message'] = 'Таймаут при виконанні nvidia-smi'
    except FileNotFoundError:
        gpu_info['error_message'] = 'nvidia-smi не знайдено (драйвери NVIDIA не встановлені)'
    except Exception as e:
        gpu_info['error_message'] = f'Помилка: {str(e)}'
    
    return gpu_info

def check_ram_availability():
    """
    Перевіряє доступність RAM та повертає детальну інформацію
    Returns:
        dict: Словник з інформацією про RAM
    """
    ram_info = {
        'total_gb': 0,
        'available_gb': 0,
        'used_gb': 0,
        'usage_percent': 0,
        'is_high_ram': False,
        'recommendation': ''
    }
    
    try:
        memory = virtual_memory()
        ram_info['total_gb'] = memory.total / 1e9
        ram_info['available_gb'] = memory.available / 1e9
        ram_info['used_gb'] = memory.used / 1e9
        ram_info['usage_percent'] = memory.percent
        ram_info['is_high_ram'] = ram_info['total_gb'] >= 20
        
        # Рекомендації
        if ram_info['usage_percent'] > 80:
            ram_info['recommendation'] = '⚠️  Високе використання RAM - рекомендується перезапустити runtime'
        elif ram_info['total_gb'] < 8:
            ram_info['recommendation'] = '⚠️  Мало RAM - може вплинути на продуктивність'
        else:
            ram_info['recommendation'] = '✅ RAM у нормальному стані'
        
    except Exception as e:
        ram_info['recommendation'] = f'❌ Помилка при отриманні інформації про RAM: {str(e)}'
    
    return ram_info

# =============================================================================
# ВИКОНАННЯ ПЕРЕВІРОК
# =============================================================================

print("🔍 ПЕРЕВІРКА СИСТЕМНИХ РЕСУРСІВ В COLAB")
print("=" * 50)

# Перевірка GPU
print("\n📊 ІНФОРМАЦІЯ ПРО GPU:")
gpu_status = check_gpu_availability()

if gpu_status['available']:
    print("✅ GPU доступний")
    print(f"   Драйвер: {gpu_status['driver_version']}")
    print(f"   CUDA: {gpu_status['cuda_version']}")
    if gpu_status['memory_total'] > 0:
        memory_usage = (gpu_status['memory_used'] / gpu_status['memory_total']) * 100
        print(f"   Пам'ять: {gpu_status['memory_used']} MiB / {gpu_status['memory_total']} MiB ({memory_usage:.1f}%)")
        print(f"   Вільна пам'ять: {gpu_status['memory_free']} MiB")
    if gpu_status['temperature'] > 0:
        print(f"   Температура: {gpu_status['temperature']}°C")
    if gpu_status['utilization'] > 0:
        print(f"   Використання: {gpu_status['utilization']}%")
else:
    print("❌ GPU недоступний")
    if gpu_status['error_message']:
        print(f"   Причина: {gpu_status['error_message']}")

# Перевірка RAM
print("\n💾 ІНФОРМАЦІЯ ПРО RAM:")
ram_status = check_ram_availability()

print(f"   Загальна пам'ять: {ram_status['total_gb']:.1f} GB")
print(f"   Доступна пам'ять: {ram_status['available_gb']:.1f} GB")
print(f"   Використана пам'ять: {ram_status['used_gb']:.1f} GB")
print(f"   Відсоток використання: {ram_status['usage_percent']:.1f}%")

if ram_status['is_high_ram']:
    print("✅ Використовується runtime з високою RAM")
else:
    print("⚠️  Використовується runtime з низькою RAM (< 20 GB)")

print(f"   {ram_status['recommendation']}")

# Додаткова перевірка для Colab
def check_colab_environment():
    """Перевіряє чи працюємо в Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

if check_colab_environment():
    print("🌐 Працюємо в Google Colab")
    print("💡 Для кращої продуктивності рекомендується:")
    print("   - Використовувати GPU runtime")
    print("   - Активувати високі ресурси RAM")
else:
    print("\n❌ Середовище Google Colab не знайдено.")

# Загальні рекомендації
print("\n📋 РЕЗУЛЬТАТ ПЕРЕВІРКИ СЕРЕДОВИЩА ВИКОНАННЯ:")
if gpu_status['available'] and ram_status['is_high_ram']:
    print("✅ Система готова для навчання глибоких нейронних мереж")
elif gpu_status['available'] and not ram_status['is_high_ram']:
    print("⚠️  GPU доступний, але мало RAM - може вплинути на продуктивність")
elif not gpu_status['available'] and ram_status['is_high_ram']:
    print("⚠️  Багато RAM, але немає GPU - навчання буде повільним")
else:
    print("❌ Недостатньо ресурсів для ефективного навчання")

print("\n" + "=" * 50) 