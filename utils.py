# =============================================================================
# ДОПОМІЖНІ ФУНКЦІЇ
# =============================================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

def denormalize_predictions(predictions, scaler_y):
    """
    Денормалізує передбачення моделі
    
    Args:
        predictions: нормалізовані передбачення
        scaler_y: scaler, який використовувався для нормалізації
    
    Returns:
        denormalized: денормалізовані передбачення
    """
    return scaler_y.inverse_transform(predictions)

def normalize_predictions(predictions, scaler_y):
    """
    Нормалізує передбачення моделі
    
    Args:
        predictions: денормалізовані передбачення
        scaler_y: scaler, який використовувався для нормалізації
    
    Returns:
        normalized: нормалізовані передбачення
    """
    return scaler_y.transform(predictions)

def calculate_detailed_metrics(predictions, targets, param_names=['B0', 'dB', 'p', 'I']):
    """
    Обчислює детальні метрики для кожного параметра
    
    Args:
        predictions: передбачення моделі
        targets: справжні значення
        param_names: назви параметрів
    
    Returns:
        dict: детальні метрики
    """
    metrics = {}
    
    # Загальні метрики
    metrics['overall'] = {
        'MSE': mean_squared_error(targets, predictions),
        'RMSE': np.sqrt(mean_squared_error(targets, predictions)),
        'MAE': mean_absolute_error(targets, predictions),
        'R2': r2_score(targets, predictions)
    }
    
    # Метрики для кожного параметра
    for i, param in enumerate(param_names):
        param_pred = predictions[:, i]
        param_target = targets[:, i]
        
        metrics[param] = {
            'MSE': mean_squared_error(param_target, param_pred),
            'RMSE': np.sqrt(mean_squared_error(param_target, param_pred)),
            'MAE': mean_absolute_error(param_target, param_pred),
            'R2': r2_score(param_target, param_pred),
            'Mean': np.mean(param_pred),
            'Std': np.std(param_pred),
            'Min': np.min(param_pred),
            'Max': np.max(param_pred)
        }
    
    return metrics

def plot_predictions_vs_targets(predictions, targets, param_names=['B0', 'dB', 'p', 'I'], 
                               figsize=(15, 10)):
    """
    Візуалізує передбачення проти справжніх значень
    
    Args:
        predictions: передбачення моделі
        targets: справжні значення
        param_names: назви параметрів
        figsize: розмір графіка
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, param in enumerate(param_names):
        ax = axes[i]
        
        # Scatter plot
        ax.scatter(targets[:, i], predictions[:, i], alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(targets[:, i].min(), predictions[:, i].min())
        max_val = max(targets[:, i].max(), predictions[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        # Calculate R²
        r2 = r2_score(targets[:, i], predictions[:, i])
        rmse = np.sqrt(mean_squared_error(targets[:, i], predictions[:, i]))
        
        ax.set_xlabel(f'True {param}')
        ax.set_ylabel(f'Predicted {param}')
        ax.set_title(f'{param}: R² = {r2:.4f}, RMSE = {rmse:.4f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_residuals(predictions, targets, param_names=['B0', 'dB', 'p', 'I'], 
                   figsize=(15, 10)):
    """
    Візуалізує залишки (residuals) для кожного параметра
    
    Args:
        predictions: передбачення моделі
        targets: справжні значення
        param_names: назви параметрів
        figsize: розмір графіка
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, param in enumerate(param_names):
        ax = axes[i]
        
        residuals = predictions[:, i] - targets[:, i]
        
        # Histogram of residuals
        ax.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        
        ax.set_xlabel(f'Residuals {param}')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Residuals Distribution: {param}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, 
                        figsize=(15, 10)):
    """
    Візуалізує криві навчання
    
    Args:
        train_losses: втрати на тренуванні
        val_losses: втрати на валідації
        train_metrics: метрики на тренуванні
        val_metrics: метрики на валідації
        figsize: розмір графіка
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Loss curves
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Val Loss', color='red')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # R² curves
    train_r2 = [m['R2'] for m in train_metrics]
    val_r2 = [m['R2'] for m in val_metrics]
    axes[0, 1].plot(train_r2, label='Train R²', color='blue')
    axes[0, 1].plot(val_r2, label='Val R²', color='red')
    axes[0, 1].set_title('R² Curves')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # RMSE curves
    train_rmse = [m['RMSE'] for m in train_metrics]
    val_rmse = [m['RMSE'] for m in val_metrics]
    axes[1, 0].plot(train_rmse, label='Train RMSE', color='blue')
    axes[1, 0].plot(val_rmse, label='Val RMSE', color='red')
    axes[1, 0].set_title('RMSE Curves')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Parameter-specific R²
    param_names = ['B0', 'dB', 'p', 'I']
    for i, param in enumerate(param_names):
        train_param_r2 = [m['param_metrics'][param]['R2'] for m in train_metrics]
        val_param_r2 = [m['param_metrics'][param]['R2'] for m in val_metrics]
        axes[1, 1].plot(train_param_r2, label=f'Train {param}', alpha=0.7)
        axes[1, 1].plot(val_param_r2, label=f'Val {param}', linestyle='--', alpha=0.7)
    
    axes[1, 1].set_title('Parameter-specific R²')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def save_scaler(scaler, filename):
    """
    Зберігає scaler у файл
    
    Args:
        scaler: scaler для збереження
        filename: назва файлу
    """
    with open(filename, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"💾 Scaler збережено: {filename}")

def load_scaler(filename):
    """
    Завантажує scaler з файлу
    
    Args:
        filename: назва файлу
    
    Returns:
        scaler: завантажений scaler
    """
    with open(filename, 'rb') as f:
        scaler = pickle.load(f)
    print(f"📂 Scaler завантажено: {filename}")
    return scaler

def create_prediction_report(predictions, targets, param_names=['B0', 'dB', 'p', 'I']):
    """
    Створює детальний звіт про передбачення
    
    Args:
        predictions: передбачення моделі
        targets: справжні значення
        param_names: назви параметрів
    
    Returns:
        dict: звіт з метриками
    """
    metrics = calculate_detailed_metrics(predictions, targets, param_names)
    
    print("📊 ДЕТАЛЬНИЙ ЗВІТ ПРО ПЕРЕДБАЧЕННЯ")
    print("=" * 50)
    
    # Загальні метрики
    print("\n🎯 ЗАГАЛЬНІ МЕТРИКИ:")
    overall = metrics['overall']
    print(f"   MSE: {overall['MSE']:.6f}")
    print(f"   RMSE: {overall['RMSE']:.6f}")
    print(f"   MAE: {overall['MAE']:.6f}")
    print(f"   R²: {overall['R2']:.4f}")
    
    # Метрики для кожного параметра
    print("\n📈 МЕТРИКИ ПО ПАРАМЕТРАХ:")
    for param in param_names:
        param_metrics = metrics[param]
        print(f"\n   {param}:")
        print(f"     R²: {param_metrics['R2']:.4f}")
        print(f"     RMSE: {param_metrics['RMSE']:.6f}")
        print(f"     MAE: {param_metrics['MAE']:.6f}")
        print(f"     MSE: {param_metrics['MSE']:.6f}")
        print(f"     Mean: {param_metrics['Mean']:.4f}")
        print(f"     Std: {param_metrics['Std']:.4f}")
        print(f"     Min: {param_metrics['Min']:.4f}")
        print(f"     Max: {param_metrics['Max']:.4f}")
    
    return metrics

def plot_correlation_matrix(predictions, targets, param_names=['B0', 'dB', 'p', 'I']):
    """
    Візуалізує матрицю кореляції між передбаченнями та справжніми значеннями
    
    Args:
        predictions: передбачення моделі
        targets: справжні значення
        param_names: назви параметрів
    """
    # Створюємо DataFrame
    data = {}
    for i, param in enumerate(param_names):
        data[f'True_{param}'] = targets[:, i]
        data[f'Pred_{param}'] = predictions[:, i]
    
    df = pd.DataFrame(data)
    
    # Обчислюємо кореляційну матрицю
    corr_matrix = df.corr()
    
    # Візуалізуємо
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix: Predictions vs True Values')
    plt.tight_layout()
    plt.show()

def save_training_results(history, filename):
    """
    Зберігає результати навчання
    
    Args:
        history: історія навчання
        filename: назва файлу
    """
    with open(filename, 'wb') as f:
        pickle.dump(history, f)
    print(f"💾 Результати навчання збережено: {filename}")

def load_training_results(filename):
    """
    Завантажує результати навчання
    
    Args:
        filename: назва файлу
    
    Returns:
        history: історія навчання
    """
    with open(filename, 'rb') as f:
        history = pickle.load(f)
    print(f"📂 Результати навчання завантажено: {filename}")
    return history

def print_model_summary(model):
    """
    Виводить короткий опис моделі
    
    Args:
        model: PyTorch модель
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("🧠 ОПИС МОДЕЛІ:")
    print(f"   Загальна кількість параметрів: {total_params:,}")
    print(f"   Навчаємі параметри: {trainable_params:,}")
    print(f"   Розмір моделі: {total_params * 4 / 1024 / 1024:.2f} MB")

def check_device_availability():
    """
    Перевіряє доступність GPU
    
    Returns:
        str: доступний пристрій
    """
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ GPU доступний: {torch.cuda.get_device_name()}")
    else:
        device = 'cpu'
        print("⚠️  GPU недоступний, використовується CPU")
    
    return device 