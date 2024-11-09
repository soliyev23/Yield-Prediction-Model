# Модель прогнозирования урожайности

Проект по прогнозированию урожайности с использованием ансамблевых методов машинного обучения (Stacking Regressor с Random Forest и Extra Trees).

## Описание проекта

Данный проект представляет собой модель машинного обучения для прогнозирования урожайности на основе различных параметров. Модель использует ансамблевый подход, комбинируя Random Forest и Extra Trees Regressor через Stacking.

## Технологии

- Python 3.x
- scikit-learn
- pandas
- numpy
- optuna
- joblib

## Установка

```bash
# Клонировать репозиторий
git clone https://github.com/your-username/yield-prediction.git

# Перейти в директорию проекта
cd yield-prediction

# Установить зависимости
pip install -r requirements.txt
```

## Структура проекта

```
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── models/
│   └── final_model.joblib
├── notebooks/
│   └── model_development.ipynb
├── README.md
└── requirements.txt
```

## Использование

### Подготовка данных

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Загрузка данных
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Предобработка
train = train.drop(columns=["id", "Row#"])
X = train.drop(columns=['yield'])
y = train['yield']

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Оптимизация гиперпараметров

Проект использует Optuna для автоматической оптимизации гиперпараметров моделей. Результаты оптимизации сохраняются в SQLite базе данных.

```python
study = optuna.create_study(
    storage="sqlite:///optuna.db",
    study_name="tuning",
    load_if_exists=True,
    direction="minimize"
)
study.optimize(objective, n_trials=30)
```

### Обучение модели

```python
# Создание финального пайплайна
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', VarianceThreshold(threshold=0.01)),
    ('model', final_model)
])

# Обучение модели
final_pipeline.fit(X_train, y_train)
```

## Особенности модели

- Использование ансамблевого метода (Stacking) с двумя базовыми моделями
- Автоматическая оптимизация гиперпараметров с помощью Optuna
- Предобработка данных с помощью StandardScaler
- Отбор признаков с помощью VarianceThreshold
- Сохранение истории оптимизации в SQLite базе данных

## Метрики качества

Модель оценивается с помощью Mean Absolute Error (MAE). Текущая версия модели достигает следующих результатов:
- MAE на тестовой выборке: [вставьте ваше значение]

## Requirements.txt

```
pandas
numpy
scikit-learn
optuna
joblib
matplotlib
seaborn
```
