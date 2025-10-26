
import sys
import os

# Add the project root to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor # Added AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor # Added KNeighborsRegressor
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
        'ElasticNet': { # Added ElasticNet
            'model': ElasticNet(random_state=42),
            'param_grid': {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
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
        'AdaBoostRegressor': { # Added AdaBoostRegressor
            'model': AdaBoostRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0]
            }
        },
        'K-Neighbors Regressor': { # Added KNeighborsRegressor
            'model': KNeighborsRegressor(),
            'param_grid': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
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
            current_best_r2 = round(grid_search.best_score_, 6)
        else: # For models like Linear Regression with no hyperparameters, perform manual KFold cross-validation
            print(f"Evaluating {name} with KFold cross-validation...")
            cv_r2_scores = []
            for train_index, val_index in kf.split(X_processed):
                X_train, X_val = X_processed.iloc[train_index], X_processed.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                cv_r2_scores.append(r2_score(y_val, y_pred))
            current_best_r2 = round(np.mean(cv_r2_scores), 6)
            print(f"R2 score for {name} (KFold cross-validation): ", current_best_r2)
            current_best_model = model

        best_models[name] = current_best_model
        model_filename = fr'E:\IIITB\ML\Project\models/best_{name.replace(" ", "_").lower()}.pkl'
        joblib.dump(current_best_model, model_filename)
        print(f"Best tuned {name} saved to '{model_filename}'.")

        if current_best_r2 > best_overall_r2:
            best_overall_r2 = current_best_r2
            best_overall_model_name = name
            best_overall_model_path = model_filename
        print(f"  After {name}: current_best_r2={current_best_r2}, best_overall_r2={best_overall_r2}, best_overall_model_name={best_overall_model_name}")

    # Save the overall best model information
    print(f"\nFinal Best Model (before saving): {best_overall_model_name}")
    print(f"Final Best R2 Score (before saving): {best_overall_r2}")
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
