
import sys
import os

# Add the project root to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

from src.preprocessing import preprocess_data
from src.models import get_models

def train_evaluate_models():
    # Load data
    train_df = pd.read_csv(r'E:\IIITB\ML\Project\data\train.csv')

    # Separate target variable
    X_full = train_df.drop(columns=['Id', 'Recovery Index'])
    y_full = train_df['Recovery Index']

    # Preprocess data (fit scaler on full training data)
    X_processed_full, scaler = preprocess_data(X_full, is_train=True)

    models = get_models()
    results = {}
    trained_models = {}

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\nTraining and evaluating {name} with KFold cross-validation...")
        mse_scores = []
        rmse_scores = []
        r2_scores = []
        all_y_val = np.array([])
        all_y_pred = np.array([])
        all_indices = np.array([])

        for fold, (train_index, val_index) in enumerate(kf.split(X_processed_full)):
            X_train, X_val = X_processed_full.iloc[train_index], X_processed_full.iloc[val_index]
            y_train, y_val = y_full.iloc[train_index], y_full.iloc[val_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, y_pred)

            mse_scores.append(mse)
            rmse_scores.append(rmse)
            r2_scores.append(r2)

            all_y_val = np.append(all_y_val, y_val)
            all_y_pred = np.append(all_y_pred, y_pred)
            all_indices = np.append(all_indices, val_index)

            print(f"  Fold {fold+1} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

        avg_mse = np.mean(mse_scores)
        avg_rmse = np.mean(rmse_scores)
        avg_r2 = np.mean(r2_scores)

        results[name] = {'Avg MSE': avg_mse, 'Avg RMSE': avg_rmse, 'Avg R2': avg_r2}
        trained_models[name] = model # Store the last trained model (or retrain on full data later)

        print(f"{name} - Avg MSE: {avg_mse:.4f}, Avg RMSE: {avg_rmse:.4f}, Avg R2: {avg_r2:.4f}")

        # Error Analysis: Identify samples with largest errors
        errors = np.abs(all_y_val - all_y_pred)
        error_df = pd.DataFrame({'Original_Index': all_indices, 'Actual': all_y_val, 'Predicted': all_y_pred, 'Error': errors})
        error_df = error_df.sort_values(by='Error', ascending=False).head(5)
        print(f"Top 5 largest errors for {name}:\n{error_df}")

    # Save the best performing models and scaler
    best_model_name = max(results, key=lambda k: results[k]['Avg R2'])
    print(f"\nBest performing model based on Avg R2: {best_model_name}")

    # Retrain the best model on the full dataset for saving
    best_model_instance = models[best_model_name]
    best_model_instance.fit(X_processed_full, y_full)
    joblib.dump(best_model_instance, fr'E:\IIITB\ML\Project\models/{best_model_name.replace(" ", "_").lower()}_final.pkl')
    joblib.dump(scaler, r'E:\IIITB\ML\Project\models/scaler.pkl')

    print("Models and scaler saved to 'models/' directory.")
    return results

if __name__ == "__main__":
    evaluation_results = train_evaluate_models()
    results_df = pd.DataFrame(evaluation_results).T
    print("\nModel Evaluation Results (KFold Cross-Validation):")
    print(results_df.sort_values(by='Avg R2', ascending=False))


if __name__ == "__main__":
    evaluation_results = train_evaluate_models()
    results_df = pd.DataFrame(evaluation_results).T
    print("\nModel Evaluation Results (KFold Cross-Validation):")
    print(results_df.sort_values(by='Avg R2', ascending=False))


if __name__ == "__main__":
    evaluation_results = train_evaluate_models()
    results_df = pd.DataFrame(evaluation_results).T
    print("\nModel Evaluation Results (KFold Cross-Validation):")
    print(results_df.sort_values(by='Avg R2', ascending=False))


if __name__ == "__main__":
    evaluation_results = train_evaluate_models()
    results_df = pd.DataFrame(evaluation_results).T
    print("\nModel Evaluation Results:")
    print(results_df.sort_values(by='Avg R2', ascending=False))
