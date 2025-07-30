# =============================================================================
# ОПТИМІЗОВАНИЙ ТРЕНЕР ДЛЯ ШВИДШОГО НАВЧАННЯ
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

class OptimizedDysonianLineTrainer:
    """
    Оптимізований клас для швидкого навчання моделі DysonianLineCNN
    """
    
    def __init__(self, model, device='cuda', learning_rate=0.002, weight_decay=1e-4):
        """
        Ініціалізація оптимізованого тренера
        
        Args:
            model: модель для навчання
            device: пристрій для обчислень
            learning_rate: збільшена швидкість навчання
            weight_decay: збільшена регуляризація
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Оптимізація PyTorch для швидкості
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Критерій втрат
        self.criterion = nn.MSELoss()
        
        # Оптимізатор з більшою швидкістю навчання
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Більш агресивний scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.7,  # Менш агресивне зменшення
            patience=5,   # Менше терпіння
            verbose=True,
            min_lr=1e-6
        )
        
        # Історія навчання
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        print(f"🚀 Оптимізований тренер ініціалізовано:")
        print(f"   Пристрій: {device}")
        print(f"   Learning rate: {learning_rate} (збільшено)")
        print(f"   Weight decay: {weight_decay} (збільшено)")
        print(f"   Оптимізатор: AdamW")
        print(f"   CUDNN benchmark: увімкнено")
    
    def create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
        """
        Створює оптимізовані DataLoader для швидшого навчання
        
        Args:
            X_train, X_val, X_test: вхідні дані
            y_train, y_val, y_test: вихідні дані
            batch_size: збільшений розмір батчу для швидкості
        
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        print("📦 Створення оптимізованих DataLoader...")
        
        # Конвертуємо в PyTorch тензори
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
        
        # Оптимізовані DataLoader з більшим batch_size та num_workers
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=True,
            num_workers=2,  # Паралельне завантаження
            persistent_workers=True  # Зберігаємо workers між епохами
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            pin_memory=True,
            num_workers=2,
            persistent_workers=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            pin_memory=True,
            num_workers=2,
            persistent_workers=True
        )
        
        print(f"✅ Оптимізовані DataLoader створено:")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        print(f"   Batch size: {batch_size} (збільшено для швидкості)")
        print(f"   Workers: 2 (паралельне завантаження)")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader):
        """
        Оптимізоване навчання на одній епосі
        
        Args:
            train_loader: DataLoader для тренування
        
        Returns:
            tuple: (середня втрата, метрики)
        """
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        # Використовуємо torch.cuda.amp для mixed precision
        scaler = torch.cuda.amp.GradScaler()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Переносимо дані на GPU
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)  # Швидше очищення
            
            # Mixed precision training
            with torch.cuda.amp.autocast():
                output = self.model(data)
                loss = self.criterion(output, target)
            
            # Зворотний прохід з mixed precision
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            # Зберігаємо для метрик
            all_predictions.append(output.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())
            
            # Показуємо прогрес кожні 50 batch'ів (менше для швидкості)
            if batch_idx % 50 == 0 and batch_idx > 0:
                print(f"     [BATCH] Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.6f}")
            
            # Очищаємо кеш GPU рідше
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()
        
        # Обчислюємо метрики
        avg_loss = total_loss / len(train_loader)
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        metrics = self.calculate_metrics(predictions, targets)
        
        return avg_loss, metrics
    
    def validate_epoch(self, val_loader):
        """
        Оптимізована валідація на одній епосі
        
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
                # Переносимо дані на GPU з non_blocking
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                # Mixed precision inference
                with torch.cuda.amp.autocast():
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
    
    def train(self, train_loader, val_loader, num_epochs=100, early_stopping_patience=15):
        """
        Оптимізований процес навчання
        
        Args:
            train_loader: DataLoader для тренування
            val_loader: DataLoader для валідації
            num_epochs: кількість епох
            early_stopping_patience: терпіння для early stopping
        
        Returns:
            dict: історія навчання
        """
        print(f"🚀 Початок оптимізованого навчання на {num_epochs} епох...")
        
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
                self.save_model('best_model_optimized.pth')
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
        print(f"✅ Оптимізоване навчання завершено за {total_time/60:.1f} хвилин")
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
        print("📊 Оцінка оптимізованої моделі на тестових даних...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        metrics = self.calculate_metrics(predictions, targets)
        
        print("✅ Результати оптимізованої моделі на тестових даних:")
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
        print(f"💾 Оптимізована модель збережена: {filename}")
    
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
        print(f"📂 Оптимізована модель завантажена: {filename}")
    
    def plot_training_history(self):
        """
        Візуалізує історію навчання
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Loss History (Optimized)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # R²
        train_r2 = [m['R2'] for m in self.train_metrics]
        val_r2 = [m['R2'] for m in self.val_metrics]
        axes[0, 1].plot(train_r2, label='Train R²')
        axes[0, 1].plot(val_r2, label='Val R²')
        axes[0, 1].set_title('R² History (Optimized)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # RMSE
        train_rmse = [m['RMSE'] for m in self.train_metrics]
        val_rmse = [m['RMSE'] for m in self.val_metrics]
        axes[1, 0].plot(train_rmse, label='Train RMSE')
        axes[1, 0].plot(val_rmse, label='Val RMSE')
        axes[1, 0].set_title('RMSE History (Optimized)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate
        lr_history = []
        for i in range(len(self.train_losses)):
            lr_history.append(self.optimizer.param_groups[0]['lr'])
        axes[1, 1].plot(lr_history)
        axes[1, 1].set_title('Learning Rate History (Optimized)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

def create_optimized_trainer(model, learning_rate=0.002, weight_decay=1e-4, device='cuda'):
    """
    Створює оптимізований тренер для моделі
    
    Args:
        model: модель для навчання
        learning_rate: збільшена швидкість навчання
        weight_decay: збільшена регуляризація
        device: пристрій для обчислень
    
    Returns:
        trainer: ініціалізований оптимізований тренер
    """
    return OptimizedDysonianLineTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    ) 