# Patient Recovery Prediction

## Project Overview
This project aims to build a predictive model to forecast the `Recovery Index` of patients based on various health and lifestyle factors. The goal is to analyze patient data, preprocess it, train multiple machine learning models, evaluate their performance, and ultimately generate predictions for unseen test data.

## Dataset
The project utilizes the following datasets:
- `data/train.csv`: The training dataset containing patient features and their `Recovery Index`.
- `data/test.csv`: The test dataset containing patient features for which the `Recovery Index` needs to be predicted.
- `data/sample_submission.csv`: A sample submission file demonstrating the required format for predictions.

## Project Structure
- `data/`: Contains the raw datasets (`train.csv`, `test.csv`, `sample_submission.csv`).
- `src/`: Houses the core Python modules for data preprocessing and model definitions.
    - `preprocessing.py`: Script for data cleaning, categorical encoding, and feature scaling.
    - `models.py`: Defines and provides instances of various regression models used in the project.
- `scripts/`: Contains executable scripts for different stages of the ML pipeline.
    - `EDA.py`: Script for Exploratory Data Analysis, including visualizations and data summaries.
    - `train_evaluate.py`: Script to train and evaluate multiple regression models, saving the trained models.
    - `hyperparameter_tuning.py`: Script for optimizing model hyperparameters (e.g., using GridSearchCV).
    - `final_prediction.py`: Script to train the best model on the full training data and generate the final submission file.
- `notebooks/`: Intended for Jupyter notebooks or interactive scripts for experimentation (currently contains `EDA.py` as a script).
- `models/`: Directory to store trained machine learning models and scalers.
- `output/`: Stores the final prediction file (`submission.csv`).

## Getting Started

### Prerequisites
- Python 3.x
- pip (Python package installer)

### Installation
1. Clone this repository (if applicable).
2. Navigate to the project root directory.
3. Install the required Python packages:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn joblib
   ```

## Usage
Follow these steps to run the project:

### 1. Exploratory Data Analysis (EDA)
To understand the dataset, its distributions, and relationships:
```bash
python notebooks/EDA.py
```
This script will display various plots and print data summaries.

### 2. Train and Evaluate Models
To train and evaluate all defined models on a training/validation split:
```bash
python scripts/train_evaluate.py
```
This script will print evaluation metrics for each model and save the trained models and the scaler to the `models/` directory.

### 3. Hyperparameter Tuning
To perform hyperparameter tuning on selected models (e.g., RandomForestRegressor):
```bash
python scripts/hyperparameter_tuning.py
```
This script will output the best parameters found and save the best-tuned model to the `models/` directory.

### 4. Final Prediction and Submission
To train the best model on the entire training dataset and generate the `submission.csv` file:
```bash
python scripts/final_prediction.py
```
The `submission.csv` file will be created in the `output/` directory, ready for submission.

## Results
- Trained models and the data scaler are saved in the `models/` directory.
- The final predictions for the test set are saved in `output/submission.csv`.
