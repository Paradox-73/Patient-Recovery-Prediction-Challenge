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

    # --- THIS IS THE KEY CHANGE ---
    # Load ONLY the best-tuned model. 
    # Your log showed Lasso was best, but Ridge was almost identical.
    # We will load the best_lasso.pkl
    print("Loading the best-tuned model (Lasso)...")
    try:
        # We will use Lasso as it had the (slightly) best R2 in your log 
        best_model = joblib.load(r'E:\IIITB\ML\Project\models/best_lasso.pkl')
    except FileNotFoundError:
        print("Error: 'best_lasso.pkl' not found. Make sure you ran hyperparameter_tuning.py")
        return

    # Generate predictions from the single best model
    print("Generating predictions...")
    final_predictions = best_model.predict(X_test_processed)
    # --- END OF KEY CHANGE ---

    # Create submission file
    submission_df = pd.DataFrame({'Id': test_ids, 'Recovery Index': final_predictions})
    submission_df.to_csv(r'E:\IIITB\ML\Project\output/submission.csv', index=False)

    print("Submission file created at 'output/submission.csv'.")

if __name__ == "__main__":
    generate_submission()

