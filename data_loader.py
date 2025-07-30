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
    
    return (X_train, X_val, X_test, y_train, y_val, y_test, 
            y_train_dict, y_val_dict, y_test_dict, scaler_y, y_min, y_max) 