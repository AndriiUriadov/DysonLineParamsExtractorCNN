# =============================================================================
# АРХІТЕКТУРА НЕЙРОННОЇ МЕРЕЖІ
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DysonianLineCNN(nn.Module):
    """
    CNN модель для передбачення параметрів лінії Дайсона
    
    Архітектура:
    - Вхідний шар: 4096 параметрів
    - Згорткові шари з багатоголовою увагою
    - Повнозв'язні шари
    - Вихідний шар: 4 параметри (B0, dB, p, I)
    """
    
    def __init__(self, input_size=4096, hidden_sizes=[512, 256, 128], num_heads=8, dropout=0.2):
        """
        Ініціалізація моделі
        
        Args:
            input_size: розмір вхідних даних
            hidden_sizes: список розмірів прихованих шарів
            num_heads: кількість голів у multi-head attention
            dropout: коефіцієнт dropout
        """
        super(DysonianLineCNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Перший шар - згортковий
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        
        # Multi-head attention шар
        self.attention = nn.MultiheadAttention(
            embed_dim=64, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Згорткові шари
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        in_channels = 64
        for hidden_size in hidden_sizes:
            self.conv_layers.append(nn.Conv1d(in_channels, hidden_size, kernel_size=5, padding=2))
            self.bn_layers.append(nn.BatchNorm1d(hidden_size))
            in_channels = hidden_size
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Повнозв'язні шари
        self.fc_layers = nn.ModuleList()
        fc_sizes = [hidden_sizes[-1]] + [256, 128, 64]
        
        for i in range(len(fc_sizes) - 1):
            self.fc_layers.append(nn.Linear(fc_sizes[i], fc_sizes[i + 1]))
        
        # Вихідний шар
        self.output_layer = nn.Linear(fc_sizes[-1], 4)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Прямий прохід через мережу
        
        Args:
            x: вхідні дані shape (batch_size, input_size)
        
        Returns:
            output: передбачення параметрів shape (batch_size, 4)
        """
        batch_size = x.size(0)
        
        # Reshape для згорткових шарів: (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        
        # Перший згортковий шар
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout_layer(x)
        
        # Multi-head attention
        # Reshape для attention: (batch_size, seq_len, features)
        x_att = x.transpose(1, 2)  # (batch_size, input_size, 64)
        att_output, _ = self.attention(x_att, x_att, x_att)
        x = att_output.transpose(1, 2)  # Повертаємо до (batch_size, 64, input_size)
        
        # Згорткові шари
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = F.relu(bn(conv(x)))
            x = self.dropout_layer(x)
        
        # Global Average Pooling
        x = self.global_pool(x)  # (batch_size, hidden_sizes[-1], 1)
        x = x.squeeze(-1)  # (batch_size, hidden_sizes[-1])
        
        # Повнозв'язні шари
        for fc in self.fc_layers:
            x = F.relu(fc(x))
            x = self.dropout_layer(x)
        
        # Вихідний шар
        output = self.output_layer(x)
        
        return output
    
    def get_model_summary(self):
        """
        Повертає короткий опис моделі
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        summary = f"""
        🧠 АРХІТЕКТУРА МОДЕЛІ DysonLineCNN:
        
        📊 Параметри:
        - Вхідний розмір: {self.input_size}
        - Приховані шари: {self.hidden_sizes}
        - Кількість голів attention: {self.num_heads}
        - Dropout: {self.dropout}
        
        📈 Розміри:
        - Загальна кількість параметрів: {total_params:,}
        - Навчаємі параметри: {trainable_params:,}
        
        🏗️ Структура:
        1. Згортковий шар (1 → 64 канали)
        2. Multi-head attention (8 голів)
        3. Згорткові шари: {self.hidden_sizes}
        4. Global Average Pooling
        5. Повнозв'язні шари: {self.hidden_sizes[-1]} → 256 → 128 → 64
        6. Вихідний шар: 64 → 4 параметри
        """
        
        return summary

def create_model(input_size=4096, hidden_sizes=[512, 256, 128], num_heads=8, dropout=0.2, device=None):
    """
    Створює та ініціалізує модель
    
    Args:
        input_size: розмір вхідних даних
        hidden_sizes: список розмірів прихованих шарів
        num_heads: кількість голів у multi-head attention
        dropout: коефіцієнт dropout
        device: пристрій для обчислень ('cuda' або 'cpu'), якщо None - автоматично визначається
    
    Returns:
        model: ініціалізована модель
    """
    print("🏗️ Створення моделі...")
    
    # Автоматично визначаємо пристрій, якщо не вказано
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🎯 Автоматично визначено пристрій: {device}")
    
    # Створюємо модель
    model = DysonianLineCNN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_heads=num_heads,
        dropout=dropout
    )
    
    # Переміщуємо на пристрій
    model = model.to(device)
    
    # Виводимо інформацію про модель
    print(model.get_model_summary())
    
    return model

def count_parameters(model):
    """
    Підраховує кількість параметрів моделі
    
    Args:
        model: PyTorch модель
    
    Returns:
        tuple: (загальна кількість, навчаємі параметри)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

def test_model(model, input_size=4096, batch_size=4):
    """
    Тестує модель на випадкових даних
    
    Args:
        model: модель для тестування
        input_size: розмір вхідних даних
        batch_size: розмір батчу
    
    Returns:
        bool: True якщо тест пройшов успішно
    """
    print("🧪 Тестування моделі...")
    
    try:
        # Створюємо випадкові дані
        x = torch.randn(batch_size, input_size)
        
        # Переміщуємо на той же пристрій, що й модель
        device = next(model.parameters()).device
        x = x.to(device)
        
        # Прямий прохід
        with torch.no_grad():
            output = model(x)
        
        print(f"✅ Тест пройшов успішно!")
        print(f"   Вхідний розмір: {x.shape}")
        print(f"   Вихідний розмір: {output.shape}")
        print(f"   Пристрій: {device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Помилка при тестуванні: {str(e)}")
        return False 