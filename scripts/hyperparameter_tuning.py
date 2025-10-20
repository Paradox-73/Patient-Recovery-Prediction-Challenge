
import sys
import os

# Add the project root to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

from src.preprocessing import preprocess_data

def tune_hyperparameters():
    # Load data
    train_df = pd.read_csv(r'E:\IIITB\ML\Project\data\train.csv')

    # Preprocess data
    X = train_df.drop(columns=['Id', 'Recovery Index'])
    y = train_df['Recovery Index']

    X_processed, scaler = preprocess_data(X, is_train=True)

    # Use KFold for robust evaluation during tuning
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    tuning_models = {
        'RandomForestRegressor': {
            'model': RandomForestRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [100, 200, 300, 400],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10, 20]
            }
        },
        'Ridge': {
            'model': Ridge(random_state=42),
            'param_grid': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        },
        'Lasso': {
            'model': Lasso(random_state=42),
            'param_grid': {
                'alpha': [0.001, 0.01, 0.1, 1.0]
            }
        },
        'DecisionTreeRegressor': {
            'model': DecisionTreeRegressor(random_state=42),
            'param_grid': {
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10, 20]
            }
        }
    }

    best_models = {}
    for name, config in tuning_models.items():
        print(f"\nStarting Hyperparameter Tuning for {name}...")
        model = config['model']
        param_grid = config['param_grid']

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                                   cv=kf, n_jobs=-1, verbose=2, scoring='r2')
        grid_search.fit(X_processed, y)

        print(f"Best parameters for {name}: ", grid_search.best_params_)
        print(f"Best R2 score for {name}: ", grid_search.best_score_)

        best_model = grid_search.best_estimator_
        best_models[name] = best_model
        joblib.dump(best_model, fr'E:\IIITB\ML\Project\models/best_{name.replace(" ", "_").lower()}.pkl')
        print(f"Best tuned {name} saved to 'models/best_{name.replace(" ", "_").lower()}.pkl'.")

    # Save the scaler as well
    joblib.dump(scaler, r'E:\IIITB\ML\Project\models/scaler.pkl')
    print("Scaler saved to 'models/scaler.pkl'.")

if __name__ == "__main__":
    tune_hyperparameters()
