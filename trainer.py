# =============================================================================
# НАВЧАННЯ МОДЕЛІ
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import time
import os

class DysonianLineTrainer:
    """
    Клас для навчання моделі DysonianLineCNN
    """
    
    def __init__(self, model, device='cuda', learning_rate=0.001, weight_decay=1e-5):
        """
        Ініціалізація тренера
        
        Args:
            model: модель для навчання
            device: пристрій для обчислень
            learning_rate: швидкість навчання
            weight_decay: регуляризація
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Критерій втрат
        self.criterion = nn.MSELoss()
        
        # Оптимізатор
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Scheduler для зменшення learning rate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10, 
            verbose=True
        )
        
        # Історія навчання
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        print(f"🎯 Тренер ініціалізовано:")
        print(f"   Пристрій: {device}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Weight decay: {weight_decay}")
    
    def create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size=16):
        """
        Створює DataLoader для тренування, валідації та тестування
        
        Args:
            X_train, X_val, X_test: вхідні дані
            y_train, y_val, y_test: вихідні дані
            batch_size: розмір батчу (зменшено до 16 для економії пам'яті)
        
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        print("📦 Створення DataLoader...")
        
        # Конвертуємо в PyTorch тензори (на CPU для економії пам'яті)
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val)
        X_test_tensor = torch.FloatTensor(X_test)
        
        y_train_tensor = torch.FloatTensor(y_train)
        y_val_tensor = torch.FloatTensor(y_val)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Створюємо датасети
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Створюємо DataLoader з меншим batch_size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        print(f"✅ DataLoader створено:")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        print(f"   Batch size: {batch_size} (оптимізовано для пам'яті)")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader):
        """
        Навчання на одній епосі
        
        Args:
            train_loader: DataLoader для тренування
        
        Returns:
            tuple: (середня втрата, метрики)
        """
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Переносимо дані на GPU
            data = data.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Прямий прохід
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Зворотний прохід
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Зберігаємо для метрик (на CPU для економії пам'яті)
            all_predictions.append(output.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())
            
            # Показуємо прогрес кожні 100 batch'ів
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f"     [BATCH] Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.6f}")
            
            # Очищаємо кеш GPU
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        # Обчислюємо метрики
        avg_loss = total_loss / len(train_loader)
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        metrics = self.calculate_metrics(predictions, targets)
        
        return avg_loss, metrics
    
    def validate_epoch(self, val_loader):
        """
        Валідація на одній епосі
        
        Args:
            val_loader: DataLoader для валідації
        
        Returns:
            tuple: (середня втрата, метрики)
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                # Переносимо дані на GPU
                data = data.to(self.device)
                target = target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        metrics = self.calculate_metrics(predictions, targets)
        
        return avg_loss, metrics
    
    def calculate_metrics(self, predictions, targets):
        """
        Обчислює метрики якості
        
        Args:
            predictions: передбачення моделі
            targets: справжні значення
        
        Returns:
            dict: словник з метриками
        """
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)
        
        # Обчислюємо метрики для кожного параметра
        param_metrics = {}
        param_names = ['B0', 'dB', 'p', 'I']
        
        for i, param in enumerate(param_names):
            param_mse = mean_squared_error(targets[:, i], predictions[:, i])
            param_rmse = np.sqrt(param_mse)
            param_r2 = r2_score(targets[:, i], predictions[:, i])
            
            param_metrics[param] = {
                'MSE': param_mse,
                'RMSE': param_rmse,
                'R2': param_r2
            }
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'param_metrics': param_metrics
        }
    
    def train(self, train_loader, val_loader, num_epochs=100, early_stopping_patience=20):
        """
        Повний процес навчання
        
        Args:
            train_loader: DataLoader для тренування
            val_loader: DataLoader для валідації
            num_epochs: кількість епох
            early_stopping_patience: терпіння для early stopping
        
        Returns:
            dict: історія навчання
        """
        print(f"🚀 Початок навчання на {num_epochs} епох...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Навчання
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Валідація
            val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # Оновлюємо scheduler
            self.scheduler.step(val_loss)
            
            # Зберігаємо історію
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)
            
            # Перевіряємо best loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Зберігаємо найкращу модель
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
            
            # Виводимо прогрес
            epoch_time = time.time() - epoch_start
            print(f"[EPOCH] Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s):")
            print(f"   [LOSS] Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"   [R2] Train R²: {train_metrics['R2']:.4f}, Val R²: {val_metrics['R2']:.4f}")
            print(f"   [LR] Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"   [TIME] Загальний час: {(time.time() - start_time)/60:.1f} хв")
            print("-" * 50)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"⏹️  Early stopping на епосі {epoch+1}")
                break
        
        total_time = time.time() - start_time
        print(f"✅ Навчання завершено за {total_time/60:.1f} хвилин")
        print(f"   Найкраща val loss: {best_val_loss:.6f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_val_loss': best_val_loss
        }
    
    def evaluate(self, test_loader):
        """
        Оцінка моделі на тестових даних
        
        Args:
            test_loader: DataLoader для тестування
        
        Returns:
            dict: метрики на тестових даних
        """
        print("📊 Оцінка моделі на тестових даних...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                # Переносимо дані на GPU
                data = data.to(self.device)
                target = target.to(self.device)
                
                output = self.model(data)
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        metrics = self.calculate_metrics(predictions, targets)
        
        print("✅ Результати на тестових даних:")
        print(f"   MSE: {metrics['MSE']:.6f}")
        print(f"   RMSE: {metrics['RMSE']:.6f}")
        print(f"   R²: {metrics['R2']:.4f}")
        
        # Виводимо метрики для кожного параметра
        for param, param_metrics in metrics['param_metrics'].items():
            print(f"   {param}: R²={param_metrics['R2']:.4f}, RMSE={param_metrics['RMSE']:.6f}")
        
        return metrics
    
    def save_model(self, filename):
        """
        Зберігає модель
        
        Args:
            filename: назва файлу
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }, filename)
        print(f"💾 Модель збережено: {filename}")
    
    def load_model(self, filename):
        """
        Завантажує модель
        
        Args:
            filename: назва файлу
        """
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_metrics = checkpoint['train_metrics']
        self.val_metrics = checkpoint['val_metrics']
        print(f"📂 Модель завантажено: {filename}")
    
    def plot_training_history(self):
        """
        Візуалізує історію навчання
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Loss History')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # R²
        train_r2 = [m['R2'] for m in self.train_metrics]
        val_r2 = [m['R2'] for m in self.val_metrics]
        axes[0, 1].plot(train_r2, label='Train R²')
        axes[0, 1].plot(val_r2, label='Val R²')
        axes[0, 1].set_title('R² History')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # RMSE
        train_rmse = [m['RMSE'] for m in self.train_metrics]
        val_rmse = [m['RMSE'] for m in self.val_metrics]
        axes[1, 0].plot(train_rmse, label='Train RMSE')
        axes[1, 0].plot(val_rmse, label='Val RMSE')
        axes[1, 0].set_title('RMSE History')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate
        lr_history = []
        for i in range(len(self.train_losses)):
            lr_history.append(self.optimizer.param_groups[0]['lr'])
        axes[1, 1].plot(lr_history)
        axes[1, 1].set_title('Learning Rate History')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_loss_history(self, figsize=(12, 8)):
        """
        Побудова графіка втрат, аналогічний до Keras
        
        Args:
            figsize: розмір графіка
        """
        plt.figure(figsize=figsize)
        
        # Побудова ліній втрат
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', linewidth=2, label='Train Loss', alpha=0.8)
        plt.plot(epochs, self.val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
        
        # Налаштування графіка
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.title('Зміна Loss під час навчання', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Покращення візуалізації
        plt.tight_layout()
        plt.show()
        
        # Виводимо статистику
        print(f"📊 Статистика втрат:")
        print(f"   Початковий Train Loss: {self.train_losses[0]:.6f}")
        print(f"   Фінальний Train Loss: {self.train_losses[-1]:.6f}")
        print(f"   Покращення Train Loss: {((self.train_losses[0] - self.train_losses[-1]) / self.train_losses[0] * 100):.1f}%")
        print(f"   Початковий Val Loss: {self.val_losses[0]:.6f}")
        print(f"   Фінальний Val Loss: {self.val_losses[-1]:.6f}")
        print(f"   Покращення Val Loss: {((self.val_losses[0] - self.val_losses[-1]) / self.val_losses[0] * 100):.1f}%")
        print(f"   Найкращий Val Loss: {min(self.val_losses):.6f} (епока {self.val_losses.index(min(self.val_losses)) + 1})")
    
    def plot_r2_history(self, figsize=(12, 8)):
        """
        Побудова графіка R², аналогічний до Keras
        
        Args:
            figsize: розмір графіка
        """
        plt.figure(figsize=figsize)
        
        # Отримуємо R² значення
        train_r2 = [m['R2'] for m in self.train_metrics]
        val_r2 = [m['R2'] for m in self.val_metrics]
        
        # Побудова ліній R²
        epochs = range(1, len(train_r2) + 1)
        plt.plot(epochs, train_r2, 'g-', linewidth=2, label='Train R²', alpha=0.8)
        plt.plot(epochs, val_r2, 'm-', linewidth=2, label='Validation R²', alpha=0.8)
        
        # Налаштування графіка
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('R² Score', fontsize=12)
        plt.title('Зміна R² під час навчання', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Покращення візуалізації
        plt.tight_layout()
        plt.show()
        
        # Виводимо статистику
        print(f"📊 Статистика R²:")
        print(f"   Початковий Train R²: {train_r2[0]:.4f}")
        print(f"   Фінальний Train R²: {train_r2[-1]:.4f}")
        print(f"   Покращення Train R²: {((train_r2[-1] - train_r2[0]) / abs(train_r2[0]) * 100):.1f}%")
        print(f"   Початковий Val R²: {val_r2[0]:.4f}")
        print(f"   Фінальний Val R²: {val_r2[-1]:.4f}")
        print(f"   Покращення Val R²: {((val_r2[-1] - val_r2[0]) / abs(val_r2[0]) * 100):.1f}%")
        print(f"   Найкращий Val R²: {max(val_r2):.4f} (епока {val_r2.index(max(val_r2)) + 1})")

def create_trainer(model, learning_rate=0.001, weight_decay=1e-5, device='cuda'):
    """
    Створює тренер для моделі
    
    Args:
        model: модель для навчання
        learning_rate: швидкість навчання
        weight_decay: регуляризація
        device: пристрій для обчислень
    
    Returns:
        trainer: ініціалізований тренер
    """
    return DysonianLineTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    ) 