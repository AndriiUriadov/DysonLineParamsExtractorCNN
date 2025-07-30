import json

# Читаємо notebook
with open('DysonianLineCNN.ipynb', 'r') as f:
    notebook = json.load(f)

# Додаємо нову секцію для візуалізації після навчання
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 8. Візуалізація моделі"]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Імпортуємо функції для візуалізації\n",
            "from model_visualizer import (\n",
            "    visualize_model_architecture, visualize_attention_weights,\n",
            "    visualize_feature_maps, create_interactive_model_summary,\n",
            "    visualize_training_progress, save_model_visualization\n",
            ")\n",
            "\n",
            "# Створюємо тестові дані для візуалізації\n",
            "test_input = torch.randn(1, 4096).to(device)\n",
            "\n",
            "print('🎨 Створення візуалізацій моделі...')\n"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Візуалізуємо архітектуру моделі\n",
            "visualize_model_architecture(model, 'model_architecture.png')\n"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Візуалізуємо attention ваги\n",
            "visualize_attention_weights(model, test_input, 'attention_weights.png')\n"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Візуалізуємо feature maps\n",
            "visualize_feature_maps(model, test_input, 'feature_maps.png')\n"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Створюємо інтерактивну візуалізацію параметрів\n",
            "create_interactive_model_summary(model)\n"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Візуалізуємо прогрес навчання (якщо є історія)\n",
            "if 'history' in locals():\n",
            "    visualize_training_progress(history)\n",
            "else:\n",
            "    print('⚠️  Історія навчання не знайдена. Спочатку навчіть модель.')\n"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Зберігаємо всі візуалізації\n",
            "save_model_visualization(model, test_input, 'model_visualizations')\n",
            "print('✅ Всі візуалізації створено та збережено!')\n"
        ]
    }
]

# Додаємо нові комірки до notebook
notebook['cells'].extend(new_cells)

# Зберігаємо notebook
with open('DysonianLineCNN.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✅ Секцію візуалізації додано до notebook!") 