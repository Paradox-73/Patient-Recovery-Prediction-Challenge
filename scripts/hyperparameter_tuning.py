
import sys
import os

# Add the project root to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
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
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
            }
        },
        'Lasso': {
            'model': Lasso(random_state=42, selection='random'),
            'param_grid': {
                'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
            }
        },
        'DecisionTreeRegressor': {
            'model': DecisionTreeRegressor(random_state=42),
            'param_grid': {
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10, 20]
            }
        },
        'GradientBoostingRegressor': {
            'model': GradientBoostingRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 4, 5],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        'LinearRegression': {
            'model': LinearRegression(),
            'param_grid': {}
        }
    }

    best_overall_r2 = -np.inf
    best_overall_model_name = ""
    best_overall_model_path = ""

    best_models = {}
    for name, config in tuning_models.items():
        print(f"\nStarting Hyperparameter Tuning for {name}...")
        model = config['model']
        param_grid = config['param_grid']

        if param_grid: # Only perform GridSearchCV if there are parameters to tune
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                                       cv=kf, n_jobs=-1, verbose=2, scoring='r2')
            grid_search.fit(X_processed, y)

            print(f"Best parameters for {name}: ", grid_search.best_params_)
            print(f"Best R2 score for {name}: ", grid_search.best_score_)

            current_best_model = grid_search.best_estimator_
            current_best_r2 = grid_search.best_score_
        else: # For models like Linear Regression with no hyperparameters
            model.fit(X_processed, y)
            y_pred = model.predict(X_processed)
            current_best_r2 = r2_score(y, y_pred) # Evaluate on training data for comparison
            print(f"R2 score for {name}: ", current_best_r2)
            current_best_model = model

        best_models[name] = current_best_model
        model_filename = fr'E:\IIITB\ML\Project\models/best_{name.replace(" ", "_").lower()}.pkl'
        joblib.dump(current_best_model, model_filename)
        print(f"Best tuned {name} saved to '{model_filename}'.")

        if current_best_r2 > best_overall_r2:
            best_overall_r2 = current_best_r2
            best_overall_model_name = name
            best_overall_model_path = model_filename

    # Save the overall best model information
    with open(r'E:\IIITB\ML\Project\models/best_overall_model_info.txt', 'w') as f:
        f.write(f"Best Model: {best_overall_model_name}\n")
        f.write(f"Best R2 Score: {best_overall_r2}\n")
        f.write(f"Model Path: {best_overall_model_path}")
    print(f"Overall best model info saved to 'models/best_overall_model_info.txt'.")

    # Save the scaler as well
    joblib.dump(scaler, r'E:\IIITB\ML\Project\models/scaler.pkl')
    print("Scaler saved to 'models/scaler.pkl'.")

if __name__ == "__main__":
    tune_hyperparameters()
