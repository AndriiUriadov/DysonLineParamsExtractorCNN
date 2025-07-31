# =============================================================================
# ВІЗУАЛІЗАЦІЯ МОДЕЛІ DysonLineCNN
# =============================================================================

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def visualize_model_architecture(model, device=None, save_path='model_architecture.png'):
    """
    Візуалізує архітектуру моделі
    
    Args:
        model: DysonianLineCNN модель
        device: пристрій (GPU/CPU) - опціонально
        save_path: шлях для збереження зображення
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Створюємо схему архітектури
    layers = [
        ('Input', 4096, 'lightblue'),
        ('Conv1D', 64, 'lightgreen'),
        ('Attention', 64, 'orange'),
        ('Conv1D', 512, 'lightcoral'),
        ('Conv1D', 256, 'lightcoral'),
        ('Conv1D', 128, 'lightcoral'),
        ('Global Pool', 128, 'yellow'),
        ('FC', 256, 'lightpink'),
        ('FC', 128, 'lightpink'),
        ('FC', 64, 'lightpink'),
        ('Output', 4, 'red')
    ]
    
    y_pos = np.arange(len(layers))
    colors = [layer[2] for layer in layers]
    sizes = [layer[1] for layer in layers]
    names = [layer[0] for layer in layers]
    
    bars = ax.barh(y_pos, sizes, color=colors, alpha=0.7)
    
    # Додаємо підписи
    for i, (bar, name, size) in enumerate(zip(bars, names, sizes)):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, 
                f'{name}\n({size})', va='center', fontsize=10)
    
    ax.set_xlabel('Розмір шару')
    ax.set_ylabel('Шари моделі')
    ax.set_title('Архітектура DysonianLineCNN', fontsize=16, fontweight='bold')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Архітектура збережена: {save_path}")

def visualize_attention_weights(model, input_data, save_path='attention_weights.png'):
    """
    Візуалізує ваги attention механізму
    
    Args:
        model: DysonianLineCNN модель
        input_data: вхідні дані
        save_path: шлях для збереження
    """
    model.eval()
    with torch.no_grad():
        # Отримуємо attention ваги
        batch_size = input_data.size(0)
        x = input_data.unsqueeze(1)  # (batch_size, 1, input_size)
        
        # Проходимо через перший conv шар
        x = model.conv1(x)
        x = model.bn1(x)
        x = torch.relu(x)
        
        # Reshape для attention
        x_att = x.transpose(1, 2)  # (batch_size, input_size, 64)
        
        # Отримуємо attention ваги
        att_output, att_weights = model.attention(x_att, x_att, x_att)
        
        # Візуалізуємо ваги для першого зразка
        attention_weights = att_weights[0].cpu().numpy()  # (num_heads, seq_len, seq_len)
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i in range(8):  # 8 голів
            im = axes[i].imshow(attention_weights[i], cmap='viridis', aspect='auto')
            axes[i].set_title(f'Голова {i+1}')
            axes[i].set_xlabel('Позиція')
            axes[i].set_ylabel('Позиція')
            plt.colorbar(im, ax=axes[i])
        
        plt.suptitle('Attention ваги для 8 голів', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Attention ваги збережені: {save_path}")

def visualize_feature_maps(model, input_data, save_path='feature_maps.png'):
    """
    Візуалізує feature maps з різних шарів
    
    Args:
        model: DysonianLineCNN модель
        input_data: вхідні дані
        save_path: шлях для збереження
    """
    model.eval()
    with torch.no_grad():
        x = input_data.unsqueeze(1)
        
        # Збираємо feature maps з різних шарів
        feature_maps = []
        layer_names = []
        
        # Conv1
        x = model.conv1(x)
        x = model.bn1(x)
        x = torch.relu(x)
        feature_maps.append(x[0].cpu().numpy())  # Перший зразок
        layer_names.append('Conv1 (64 каналів)')
        
        # Attention
        x_att = x.transpose(1, 2)
        att_output, _ = model.attention(x_att, x_att, x_att)
        x = att_output.transpose(1, 2)
        feature_maps.append(x[0].cpu().numpy())
        layer_names.append('Attention (64 каналів)')
        
        # Conv layers
        for i, (conv, bn) in enumerate(zip(model.conv_layers, model.bn_layers)):
            x = conv(x)
            x = bn(x)
            x = torch.relu(x)
            feature_maps.append(x[0].cpu().numpy())
            layer_names.append(f'Conv{i+2} ({x.size(1)} каналів)')
        
        # Візуалізація
        n_layers = len(feature_maps)
        fig, axes = plt.subplots(n_layers, 1, figsize=(15, 4*n_layers))
        
        for i, (fm, name) in enumerate(zip(feature_maps, layer_names)):
            # Показуємо перші 16 каналів
            n_channels = min(16, fm.shape[0])
            fm_vis = fm[:n_channels]
            
            im = axes[i].imshow(fm_vis, cmap='viridis', aspect='auto')
            axes[i].set_title(f'{name} - Перші {n_channels} каналів')
            axes[i].set_xlabel('Позиція')
            axes[i].set_ylabel('Канал')
            plt.colorbar(im, ax=axes[i])
        
        plt.suptitle('Feature Maps з різних шарів', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Feature maps збережені: {save_path}")

def create_interactive_model_summary(model):
    """
    Створює інтерактивну візуалізацію параметрів моделі
    
    Args:
        model: DysonianLineCNN модель
    """
    # Підраховуємо параметри по шарах
    layer_params = {}
    total_params = 0
    
    for name, param in model.named_parameters():
        layer_name = name.split('.')[0]
        if layer_name not in layer_params:
            layer_params[layer_name] = 0
        layer_params[layer_name] += param.numel()
        total_params += param.numel()
    
    # Створюємо інтерактивну діаграму
    fig = go.Figure(data=[
        go.Bar(
            x=list(layer_params.keys()),
            y=list(layer_params.values()),
            text=[f'{v:,}' for v in layer_params.values()],
            textposition='auto',
            marker_color='lightblue'
        )
    ])
    
    fig.update_layout(
        title='Розподіл параметрів по шарах моделі',
        xaxis_title='Шари',
        yaxis_title='Кількість параметрів',
        showlegend=False
    )
    
    fig.show()
    
    # Створюємо кругову діаграму
    fig_pie = go.Figure(data=[
        go.Pie(
            labels=list(layer_params.keys()),
            values=list(layer_params.values()),
            hole=0.3
        )
    ])
    
    fig_pie.update_layout(
        title='Відсотковий розподіл параметрів'
    )
    
    fig_pie.show()
    
    print(f"✅ Загальна кількість параметрів: {total_params:,}")

def visualize_training_progress(history):
    """
    Візуалізує прогрес навчання
    
    Args:
        history: словник з історією навчання
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Loss', 'R² Score', 'RMSE', 'Learning Rate'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Loss
    fig.add_trace(
        go.Scatter(x=epochs, y=history['train_losses'], name='Train Loss', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history['val_losses'], name='Val Loss', line=dict(color='red')),
        row=1, col=1
    )
    
    # R²
    train_r2 = [metrics['R2'] for metrics in history['train_metrics']]
    val_r2 = [metrics['R2'] for metrics in history['val_metrics']]
    
    fig.add_trace(
        go.Scatter(x=epochs, y=train_r2, name='Train R²', line=dict(color='blue')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=val_r2, name='Val R²', line=dict(color='red')),
        row=1, col=2
    )
    
    # RMSE
    train_rmse = [metrics['RMSE'] for metrics in history['train_metrics']]
    val_rmse = [metrics['RMSE'] for metrics in history['val_metrics']]
    
    fig.add_trace(
        go.Scatter(x=epochs, y=train_rmse, name='Train RMSE', line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=val_rmse, name='Val RMSE', line=dict(color='red')),
        row=2, col=1
    )
    
    fig.update_layout(height=800, title_text="Прогрес навчання моделі")
    fig.show()

def save_model_visualization(model, input_data, output_dir='model_visualizations'):
    """
    Зберігає всі візуалізації моделі
    
    Args:
        model: DysonianLineCNN модель
        input_data: вхідні дані для тестування
        output_dir: директорія для збереження
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("🎨 Створення візуалізацій моделі...")
    
    # Архітектура
    visualize_model_architecture(model, f'{output_dir}/architecture.png')
    
    # Attention ваги
    visualize_attention_weights(model, input_data, f'{output_dir}/attention_weights.png')
    
    # Feature maps
    visualize_feature_maps(model, input_data, f'{output_dir}/feature_maps.png')
    
    # Інтерактивна візуалізація
    create_interactive_model_summary(model)
    
    print(f"✅ Всі візуалізації збережено в: {output_dir}/") 