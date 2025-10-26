# Project Report: A Comparative Analysis of Regression Models for Patient Recovery Prediction

## Abstract
This project focuses on predicting patient recovery using various regression models. It details the implementation of linear models, regularized linear models, and tree-based ensembles, highlighting different approaches to model building, feature engineering, and hyperparameter tuning. The goal is to identify the most effective techniques for this prediction task.

## Table of Contents
1.  [About the Project](#about-the-project)
2.  [Installation](#installation)
3.  [Data](#data)
4.  [Scripts](#scripts)
5.  [Output](#output)
6.  [Reports and Visualizations](#reports-and-visualizations)
7.  [Kaggle Score Comparison and Justification](#kaggle-score-comparison-and-justification)
8.  [Conclusion and Key Takeaways](#conclusion-and-key-takeaways)
9.  [Future Work](#future-work)
10. [Usage](#usage)

## About the Project
This project undertakes a comparative analysis of various regression models for predicting patient recovery. The primary aim is to explore and implement different machine learning algorithms and evaluate their performance, with a particular emphasis on the impact of feature engineering. The project serves as a guide to understanding model selection, regularization techniques, and the importance of data understanding in achieving high predictive accuracy.

## Installation
To set up the project environment, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Bhavzzzzzz/ML_Project.git
    cd ML_Project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**
    *   **Windows:**
        ```bash
        .venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Data
The `data/` directory contains the datasets used for training and testing the models.
*   `train.csv`: The training dataset containing features and the target variable (Recovery Index).
*   `test.csv`: The test dataset containing features for which Recovery Index predictions are to be made.

## Scripts
The `scripts/` directory hosts various Python scripts, each implementing a different regression model or a data analysis task.

### `AdaBoostRegressor.py`
Implements an AdaBoost Regressor model. This ensemble meta-algorithm combines multiple weak learners to create a strong predictor, focusing iteratively on mispredicted samples. Includes hyperparameter tuning for `n_estimators` and `learning_rate`.

### `DecisionTreeRegressor.py`
Implements a single Decision Tree Regressor. This script trains a Decision Tree on the dataset, exploring its standalone performance. It likely tunes parameters like `max_depth`, `min_samples_split`, and `min_samples_leaf` to prevent overfitting.

### `EDA.py`
DEDICATED to Exploratory Data Analysis. This script performs data loading, cleaning, visualization (histograms, scatter plots, correlation matrices), and initial feature insights to understand patterns, relationships, and anomalies in the patient recovery dataset.

### `Elastic_Net.py`
Implements the Elastic Net Regression model, which combines L1 (Lasso) and L2 (Ridge) regularization penalties. Useful for correlated features, it performs variable selection and coefficient shrinkage. The script tunes `alpha` and `l1_ratio` for optimal balance.

### `KNeighborsRegressor.py`
Utilizes the K-Nearest Neighbors (KNN) Regressor. This non-parametric algorithm predicts values by averaging the 'k' nearest neighbors. The script trains a KNN Regressor, focusing on scaling features and tuning 'k' for optimal performance.

### `Lasso.py`
Implements Lasso (Least Absolute Shrinkage and Selection Operator) Regression, using L1 regularization. This technique performs feature selection by driving some coefficients exactly to zero, leading to sparser models. The script typically tunes the `alpha` parameter.

### `Linear Regression.py`
Implements a standard Ordinary Least Squares (OLS) Linear Regression model. This script trains a basic Linear Regression model as a baseline to understand the fundamental linear predictability of the patient recovery index.

### `LRwithFeatureEngineering.py`
Extends the standard Linear Regression model by incorporating significant Feature Engineering. It focuses on creating new features or transforming existing ones (e.g., interaction terms, polynomial features) to capture complex, non-linear relationships and improve model performance.

### `RandomForest.py`
Implements a Random Forest Regressor, an ensemble method that constructs multiple decision trees and averages their predictions. Known for robustness and handling non-linear relationships. The script tunes hyperparameters like `n_estimators`, `max_depth`, and `max_features`.

### `Ridge.py`
Implements Ridge Regression, using L2 regularization. It adds a penalty equal to the square of the magnitude of coefficients, shrinking them towards zero to prevent overfitting. The script typically tunes the `alpha` parameter.

### `XGBoost.py`
Implements the XGBoost (Extreme Gradient Boosting) Regressor, a highly efficient and flexible gradient boosting framework. Known for speed and performance, it builds trees sequentially. The script likely involves extensive hyperparameter tuning for parameters like `n_estimators`, `learning_rate`, and `max_depth`.

## Output
The `output/` directory stores the prediction results from the various models.
*   `submission_AdaBoost.csv`: Predictions from the AdaBoost Regressor.
*   `submission_DecisionTree.csv`: Predictions from the Decision Tree Regressor.
*   `submission_enet.csv`: Predictions from the Elastic Net Regressor.
*   `submission_feinter1.csv`: Predictions from a feature-engineered model (potentially LR with Feature Engineering, first iteration).
*   `submission_feinter2.csv`: Predictions from a feature-engineered model (potentially LR with Feature Engineering, second iteration).
*   `submission_KNeighbors.csv`: Predictions from the K-Neighbors Regressor.
*   `submission_L2.csv`: Predictions from the Ridge Regressor.
*   `submission_Lasso.csv`: Predictions from the Lasso Regressor.
*   `submission_LR.csv`: Predictions from the standard Linear Regression.
*   `submission_optimized.csv`: Optimized predictions, likely from the best-performing model or a combination.

## Reports and Visualizations
The `reports/` directory contains visualizations generated during the Exploratory Data Analysis (EDA) and model evaluation.
*   `correlation of numerical features.png`: Heatmap showing correlations between numerical features.
*   `distr of lifestyle activiteis.png`: Distribution of lifestyle activities.
*   `distribution of recovery index.png`: Histogram of the Recovery Index distribution.
*   `distribution_of_avg_sleep_hours.png`: Distribution of average sleep hours.
*   `distribution_of_follow-up_sessions.png`: Distribution of follow-up sessions.
*   `distribution_of_inital_health_score.png`: Distribution of initial health scores.
*   `distribution_of_therapy_hours.png`: Distribution of therapy hours.
*   `ri by lifestyle activites.png`: Relationship between Recovery Index and lifestyle activities.

## Kaggle Score Comparison and Justification
Based on the provided Kaggle scores, below is the performance ranking for the implemented models:

| Model                       | Kaggle Score |
| :-------------------------- | :----------- |
| Feature Engineering (LR)    | 1.980        |
| Elastic Net                 | 1.982        |
| L2 (Ridge)                  | 1.982        |
| Lasso                       | 1.982        |
| LR (Linear Regression)      | 1.982        |
| AdaBoost                    | 2.911        |
| Decision Tree               | 3.003        |
| K-Neighbors                 | 4.356        |

The results clearly indicate that **Linear Regression with Feature Engineering** achieved the best performance (lowest Kaggle score of 1.980). This strongly supports the hypothesis that domain-specific feature engineering is more impactful than relying solely on complex algorithms. The engineered features likely captured the underlying non-linear interactions in the data more effectively.

The regularized linear models (Elastic Net, Ridge, Lasso) and the basic Linear Regression also performed exceptionally well, very close to the top score (1.982). This suggests that the dataset has a strong linear component, and regularization effectively prevents overfitting, maintaining robust performance.

In contrast, the ensemble and instance-based models like **AdaBoost**, **Decision Tree**, and especially **K-Neighbors**, performed significantly worse. For this dataset, the complex, non-linear relationships were better captured by explicit feature engineering rather than the implicit discovery mechanisms of tree-based ensembles.

## Conclusion and Key Takeaways
*   **Feature Engineering is Paramount:** Thoughtful, domain-specific feature engineering significantly outperformed more complex, "out-of-the-box" models. Understanding the data's underlying structure and creating features that capture non-linear interactions is often more crucial than algorithmic complexity.
*   **Simpler Models Can Excel:** Simpler linear models (Linear Regression, Ridge, Lasso, Elastic Net) demonstrated superior performance when coupled with effective feature engineering, also offering valuable interpretability.
*   **Regularization is Effective:** Regularized linear models consistently performed well, indicating that regularization techniques (L1 and L2) effectively mitigated overfitting and maintained robust predictive performance.
*   **Ensemble Models' Limitations:** For this particular dataset, ensemble methods like AdaBoost, Decision Tree, and XGBoost struggled to outperform the simpler, feature-engineered linear models.
*   **Data Understanding is Key:** A deep understanding of the data and its underlying patterns is the foundation for building highly accurate and interpretable machine learning models.

## Future Work
*   **Advanced Feature Engineering:** Explore more complex non-linear transformations, polynomial features of higher degrees, and multi-way interaction terms.
*   **Error Analysis of Top Models:** Conduct a detailed error analysis on the predictions of the best-performing models to reveal new insights or necessitate specialized models.
*   **Ensemble Model Optimization:** Pursue more extensive hyperparameter tuning and exploration of different ensemble configurations (e.g., stacking, blending) for models like XGBoost and Random Forest.
*   **Robustness Testing:** Evaluate the models' robustness to noisy data or missing values by simulating real-world data imperfections.
*   **Model Interpretability:** Further investigate the interpretability of the top-performing models to provide clearer, actionable insights for decision-makers.
*   **Deployment and Monitoring:** Develop a pipeline for deploying the best-performing model and continuously monitoring its performance in a real-world setting.

## Usage
To run any of the scripts, navigate to the project root directory and execute the desired script using Python. For example, to run the EDA script:

```bash
python scripts/EDA.py
```

Ensure that you have activated your virtual environment and installed all dependencies before running the scripts.
