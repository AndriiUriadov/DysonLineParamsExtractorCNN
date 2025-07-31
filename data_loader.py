# =============================================================================
# ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА ДАНИХ
# =============================================================================

import gdown
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def download_data():
    """
    Завантажує дані з Google Drive
    Returns:
        tuple: (X, y) - вхідні та вихідні дані
    """
    print("📥 Завантаження даних...")
    
    # Google Drive file IDs
    file_id_X = '1kOeVd4d1PZfPhfoVIPKXSUScV0tfiRcD'
    file_id_y = '1LKHYyAnb3Ls1qKbxlXOvc6mUY_fMSiAk'

    # Створюємо прямі посилання
    url_X = f'https://drive.google.com/uc?id={file_id_X}'
    url_y = f'https://drive.google.com/uc?id={file_id_y}'

    # Завантажуємо файли
    gdown.download(url_X, 'X_dyson.npy', quiet=False)
    gdown.download(url_y, 'y_dyson.npy', quiet=False)

    # Завантажуємо у змінні
    X = np.load('X_dyson.npy')
    y = np.load('y_dyson.npy')

    print("✅ Дані завантажено!")
    print(f"   X.shape = {X.shape}")
    print(f"   y.shape = {y.shape}")
    
    return X, y

def preprocess_data(X, y):
    """
    Підготовка даних для навчання
    Args:
        X: вхідні дані
        y: вихідні дані
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, scaler_y, y_min, y_max)
    """
    print("🔧 Підготовка даних...")
    
    # Нормалізація вхідних даних X (середнє 0, стандартне відхилення 1)
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_std = np.std(X, axis=1, keepdims=True)
    X_normalized = (X - X_mean) / X_std

    # Нормалізація вихідних даних y (масштабування до [0,1])
    scaler_y = MinMaxScaler()
    y_normalized = scaler_y.fit_transform(y)

    # Збережемо також максимуми і мінімуми y для подальшого денормування
    y_min = scaler_y.data_min_
    y_max = scaler_y.data_max_

    print("   Мінімальні значення y:", y_min)
    print("   Максимальні значення y:", y_max)

    # Розділення на train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X_normalized, y_normalized, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print("✅ Дані підготовлено!")
    print(f"   Train set: {X_train.shape}, {y_train.shape}")
    print(f"   Validation set: {X_val.shape}, {y_val.shape}")
    print(f"   Test set: {X_test.shape}, {y_test.shape}")
    print(f"   Min y: {np.min(y_train, axis=0)}")  # має бути ~0
    print(f"   Max y: {np.max(y_train, axis=0)}")  # має бути ~1

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_y, y_min, y_max

def visualize_random_spectra(X_train, X_val, X_test, y_train, y_val, y_test, X_original=None):
    """
    Візуалізує 2 випадкових спектри для контролю процесів завантаження та нормалізації
    
    Args:
        X_train, X_val, X_test: нормалізовані вхідні дані
        y_train, y_val, y_test: нормалізовані вихідні дані
        X_original: оригінальні дані (опціонально)
    """
    import matplotlib.pyplot as plt
    import random
    
    print("📊 Візуалізація випадкових спектрів для контролю...")
    
    # Вибір випадкових індексів
    train_idx = random.randint(0, len(X_train) - 1)
    val_idx = random.randint(0, len(X_val) - 1)
    
    # Створюємо графік
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Спектр 1: Train set
    axes[0, 0].plot(X_train[train_idx], 'b-', linewidth=1, alpha=0.8)
    axes[0, 0].set_title(f'Train Spectrum #{train_idx}', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Frequency Channel')
    axes[0, 0].set_ylabel('Normalized Intensity')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(0.02, 0.98, f'Parameters: B0={y_train[train_idx, 0]:.3f}, dB={y_train[train_idx, 1]:.3f}\np={y_train[train_idx, 2]:.3f}, I={y_train[train_idx, 3]:.3f}', 
                     transform=axes[0, 0].transAxes, verticalalignment='top', 
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Спектр 2: Validation set
    axes[0, 1].plot(X_val[val_idx], 'r-', linewidth=1, alpha=0.8)
    axes[0, 1].set_title(f'Validation Spectrum #{val_idx}', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Frequency Channel')
    axes[0, 1].set_ylabel('Normalized Intensity')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].text(0.02, 0.98, f'Parameters: B0={y_val[val_idx, 0]:.3f}, dB={y_val[val_idx, 1]:.3f}\np={y_val[val_idx, 2]:.3f}, I={y_val[val_idx, 3]:.3f}', 
                     transform=axes[0, 1].transAxes, verticalalignment='top', 
                     bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Порівняння оригінального та нормалізованого (якщо є оригінальні дані)
    if X_original is not None:
        # Знаходимо відповідні індекси в оригінальних даних
        original_train_idx = train_idx  # Приблизно
        original_val_idx = len(X_original) - len(X_test) - len(X_val) + val_idx
        
        axes[1, 0].plot(X_original[original_train_idx], 'g-', linewidth=1, alpha=0.6, label='Original')
        axes[1, 0].plot(X_train[train_idx], 'b-', linewidth=1, alpha=0.8, label='Normalized')
        axes[1, 0].set_title('Train: Original vs Normalized', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Frequency Channel')
        axes[1, 0].set_ylabel('Intensity')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(X_original[original_val_idx], 'g-', linewidth=1, alpha=0.6, label='Original')
        axes[1, 1].plot(X_val[val_idx], 'r-', linewidth=1, alpha=0.8, label='Normalized')
        axes[1, 1].set_title('Validation: Original vs Normalized', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Frequency Channel')
        axes[1, 1].set_ylabel('Intensity')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Статистика нормалізації
        axes[1, 0].hist(X_train.flatten(), bins=50, alpha=0.7, color='blue', label='Train')
        axes[1, 0].hist(X_val.flatten(), bins=50, alpha=0.7, color='red', label='Validation')
        axes[1, 0].set_title('Distribution of Normalized Values', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Normalized Intensity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Статистика параметрів
        param_names = ['B0', 'dB', 'p', 'I']
        train_means = [np.mean(y_train[:, i]) for i in range(4)]
        val_means = [np.mean(y_val[:, i]) for i in range(4)]
        
        x_pos = np.arange(len(param_names))
        width = 0.35
        
        axes[1, 1].bar(x_pos - width/2, train_means, width, label='Train', alpha=0.7, color='blue')
        axes[1, 1].bar(x_pos + width/2, val_means, width, label='Validation', alpha=0.7, color='red')
        axes[1, 1].set_title('Mean Parameter Values', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Parameters')
        axes[1, 1].set_ylabel('Normalized Value')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(param_names)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Виводимо статистику
    print(f"📈 Статистика нормалізації:")
    print(f"   Train X - Mean: {np.mean(X_train):.6f}, Std: {np.std(X_train):.6f}")
    print(f"   Val X - Mean: {np.mean(X_val):.6f}, Std: {np.std(X_val):.6f}")
    print(f"   Train y - Min: {np.min(y_train, axis=0)}, Max: {np.max(y_train, axis=0)}")
    print(f"   Val y - Min: {np.min(y_val, axis=0)}, Max: {np.max(y_val, axis=0)}")
    print(f"✅ Візуалізація завершена!")

def create_data_dict(y_train, y_val, y_test):
    """
    Створює словники з даними для зручності
    Args:
        y_train, y_val, y_test: вихідні дані
    Returns:
        tuple: (y_train_dict, y_val_dict, y_test_dict)
    """
    y_train_dict = {
        'B0': y_train[:, 0],
        'dB': y_train[:, 1],
        'p':  y_train[:, 2],
        'I':  y_train[:, 3]
    }
    
    y_val_dict = {
        'B0': y_val[:, 0],
        'dB': y_val[:, 1],
        'p':  y_val[:, 2],
        'I':  y_val[:, 3]
    }
    
    y_test_dict = {
        'B0': y_test[:, 0],
        'dB': y_test[:, 1],
        'p':  y_test[:, 2],
        'I':  y_test[:, 3]
    }
    
    return y_train_dict, y_val_dict, y_test_dict

def load_and_preprocess():
    """
    Повний процес завантаження та підготовки даних
    Returns:
        tuple: всі необхідні дані для навчання
    """
    # Завантажуємо дані
    X, y = download_data()
    
    # Підготовляємо дані
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_y, y_min, y_max = preprocess_data(X, y)
    
    # Створюємо словники
    y_train_dict, y_val_dict, y_test_dict = create_data_dict(y_train, y_val, y_test)
    
    # Візуалізуємо випадкові спектри для контролю
    visualize_random_spectra(X_train, X_val, X_test, y_train, y_val, y_test, X_original=X)
    
    return (X_train, X_val, X_test, y_train, y_val, y_test, 
            y_train_dict, y_val_dict, y_test_dict, scaler_y, y_min, y_max) 