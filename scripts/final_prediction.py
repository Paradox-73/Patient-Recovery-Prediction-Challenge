import sys
import os

# Add the project root to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib
import numpy as np

from src.preprocessing import preprocess_data

def generate_submission():
    # Load datasets
    train_df = pd.read_csv(r'E:\IIITB\ML\Project\data\train.csv')
    test_df = pd.read_csv(r'E:\IIITB\ML\Project\data\test.csv')

    # Prepare training data
    X_train_full = train_df.drop(columns=['Id', 'Recovery Index'])
    y_train_full = train_df['Recovery Index']

    # Preprocess training data (fit scaler on full training data)
    X_train_processed, scaler = preprocess_data(X_train_full, is_train=True)

    # Save the scaler
    joblib.dump(scaler, r'E:\IIITB\ML\Project\models/final_scaler.pkl')
    print("Final scaler saved.")

    # Prepare test data
    test_ids = test_df['Id']
    X_test = test_df.drop(columns=['Id'])

    # Preprocess test data using the *same* scaler fitted on training data
    X_test_processed, _ = preprocess_data(X_test, is_train=False, scaler=scaler)

    # Load best-tuned models for ensembling
    print("Loading best-tuned models for ensembling...")
    # Assuming these models were saved from hyperparameter_tuning.py
    # You might need to adjust these filenames based on what was actually saved.
    try:
        rf_model = joblib.load(r'E:\IIITB\ML\Project\models/best_randomforestregressor.pkl')
        ridge_model = joblib.load(r'E:\IIITB\ML\Project\models/best_ridge.pkl')
        lasso_model = joblib.load(r'E:\IIITB\ML\Project\models/best_lasso.pkl')
        dt_model = joblib.load(r'E:\IIITB\ML\Project\models/best_decisiontreeregressor.pkl')
        
        models_to_ensemble = [
            ('RandomForest', rf_model),
            ('Ridge', ridge_model),
            ('Lasso', lasso_model),
            ('DecisionTree', dt_model)
        ]
    except FileNotFoundError:
        print("Warning: Not all best-tuned models found. Using available models for ensemble.")
        # Fallback: if any tuned model is missing, we'll just use the best RandomForest
        rf_model = joblib.load(r'E:\IIITB\ML\Project\models/best_randomforestregressor.pkl')
        models_to_ensemble = [('RandomForest', rf_model)]

    # Generate predictions from each model
    all_predictions = []
    for name, model in models_to_ensemble:
        print(f"Generating predictions with {name}...")
        predictions = model.predict(X_test_processed)
        all_predictions.append(predictions)

    # Ensemble: Average the predictions
    print("Ensembling predictions by averaging...")
    ensemble_predictions = np.mean(all_predictions, axis=0)

    # Create submission file
    submission_df = pd.DataFrame({'Id': test_ids, 'Recovery Index': ensemble_predictions})
    submission_df.to_csv(r'E:\IIITB\ML\Project\output/submission.csv', index=False)

    print("Submission file created at 'output/submission.csv'.")

if __name__ == "__main__":
    generate_submission()

