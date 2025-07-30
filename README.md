# DysonLineParamsExtractorCNN

Проект для роботи з нейронними мережами в Google Colab.

## 📁 Структура проекту

```
DysonLineParamsExtractorCNN/
├── DysonianLineCNN.ipynb          # Основний Jupyter notebook (модульний)
├── colab_gpu_check.py             # Перевірка GPU та RAM
├── data_loader.py                 # Завантаження та підготовка даних
├── model.py                       # Архітектура нейронної мережі
├── trainer.py                     # Навчання моделі
├── utils.py                       # Допоміжні функції
├── DysonianLineCNN_multihead_30K.ipynb  # Старий notebook (монолітний)
└── README.md                      # Документація
```

## 🚀 Запуск у Google Colab

> **Для модульного проекту потрібно клонувати репозиторій у середовищі виконання Colab!**

1. Відкрийте [Google Colab](https://colab.research.google.com/)
2. Відкрийте вкладку "GitHub" і знайдіть цей репозиторій, або відкрийте файл DysonianLineCNN.ipynb напряму з GitHub
3. **У першій комірці notebook виконайте:**

```python
# ⚡️ Клонування репозиторію для запуску в Colab
!git clone https://github.com/AndriiUriadov/DysonLineParamsExtractorCNN.git
%cd DysonLineParamsExtractorCNN
```

4. Далі запускайте всі комірки notebook як зазвичай — всі імпорти працюватимуть, бо всі .py файли будуть у робочій директорії.

---

## 📦 Залежності

- Python >= 3.8
- torch
- numpy
- matplotlib
- seaborn
- scikit-learn
- gdown

(Усі залежності можна встановити через `!pip install ...` у Colab)

---

## 📝 Опис

- `DysonianLineCNN.ipynb` — основний модульний notebook для навчання та аналізу
- `colab_gpu_check.py` — перевірка ресурсів середовища виконання
- `data_loader.py` — завантаження та підготовка даних
- `model.py` — архітектура CNN
- `trainer.py` — навчання моделі
- `utils.py` — допоміжні функції
- `DysonianLineCNN_multihead_30K.ipynb` — старий монолітний notebook (залишено для історії)