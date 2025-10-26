import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

test_ids = test_df['Id']

X = train_df.drop(['Recovery Index', 'Id'], axis=1)
y = train_df['Recovery Index']
X_test_final = test_df.drop('Id', axis=1)

numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

rf_param_grid = {
    'model__n_estimators': [100, 200, 300, 500],
    'model__max_depth': [10, 20, 30, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2', 1.0]
}

rf_random_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=rf_param_grid,
    n_iter=10,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

rf_random_search.fit(X_train, y_train)

best_rf_model = rf_random_search.best_estimator_
print("Best RandomForest Parameters:")
print(rf_random_search.best_params_)

y_pred_rf = best_rf_model.predict(X_val)
rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))
mae_rf = mean_absolute_error(y_val, y_pred_rf)
r2_rf = r2_score(y_val, y_pred_rf)

print(f"Root Mean Squared Error (RMSE): {rmse_rf:.4f}")
print(f"Mean Absolute Error (MAE):     {mae_rf:.4f}")
print(f"R-squared (R²):                {r2_rf:.4f}")

final_predictions_rf = best_rf_model.predict(X_test_final)

final_predictions_rf = np.clip(final_predictions_rf, 10, 100)
final_predictions_rf = np.round(final_predictions_rf).astype(int)

submission_rf = pd.DataFrame({
    'Id': test_ids,
    'Recovery Index': final_predictions_rf
})

submission_rf.to_csv('output/submission_randomforest.csv', index=False)
print("Successfully created 'submission_randomforest.csv'!")
print(submission_rf.head())