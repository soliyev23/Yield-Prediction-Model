# Yield Prediction Model

A project focused on predicting agricultural yields using ensemble machine learning methods (Stacking Regressor with Random Forest and Extra Trees).

## Project Description

This project features a machine learning model designed to predict crop yields based on various parameters. The model utilizes an ensemble approach, combining Random Forest and Extra Trees Regressors through a Stacking strategy to improve predictive accuracy.

## Tech Stack

- Python 3.x
- scikit-learn
- pandas
- numpy
- optuna
- joblib

## Installation

# Clone the repository
git clone https://github.com/your-username/yield-prediction.git

# Navigate to project directory
cd yield-prediction

# Install dependencies
pip install -r requirements.txt

## Project Structure

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

## Usage

### Data Preparation

import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Preprocessing
train = train.drop(columns=["id", "Row#"])
X = train.drop(columns=['yield'])
y = train['yield']

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Hyperparameter Optimization

The project uses Optuna for automated hyperparameter optimization. Results are stored in a SQLite database.

study = optuna.create_study(
    storage="sqlite:///optuna.db",
    study_name="tuning",
    load_if_exists=True,
    direction="minimize"
)
study.optimize(objective, n_trials=30)

### Model Training

# Create final pipeline
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', VarianceThreshold(threshold=0.01)),
    ('model', final_model)
])

# Fit the model
final_pipeline.fit(X_train, y_train)

## Key Features

- Ensemble Learning: Uses a Stacking Regressor with two base models for better generalization.
- Automated Tuning: Hyperparameter optimization via Optuna.
- Data Preprocessing: Standard scaling using StandardScaler.
- Feature Selection: Automated feature selection using VarianceThreshold.
- Persistence: Optimization history is saved in a local SQLite database.

## Quality Metrics

The model is evaluated using Mean Absolute Error (MAE). 
- MAE on test set: [Insert your value here]

## Requirements List (requirements.txt)

pandas
numpy
scikit-learn
optuna
joblib
matplotlib
seaborn
