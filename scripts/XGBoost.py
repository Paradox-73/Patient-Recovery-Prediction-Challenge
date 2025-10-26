import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
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

xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1))
])

xgb_param_grid = {
    'model__n_estimators': [100, 200, 500, 1000],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': [3, 5, 7, 9],
    'model__subsample': [0.7, 0.8, 1.0],
    'model__colsample_bytree': [0.7, 0.8, 1.0]
}

xgb_random_search = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=xgb_param_grid,
    n_iter=50,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

xgb_random_search.fit(X_train, y_train)

best_xgb_model = xgb_random_search.best_estimator_
print("Best XGBoost Parameters:")
print(xgb_random_search.best_params_)

y_pred_xgb = best_xgb_model.predict(X_val)
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
mae_xgb = mean_absolute_error(y_val, y_pred_xgb)
r2_xgb = r2_score(y_val, y_pred_xgb)

print(f"Root Mean Squared Error (RMSE): {rmse_xgb:.4f}")
print(f"Mean Absolute Error (MAE):     {mae_xgb:.4f}")
print(f"R-squared (R²):                {r2_xgb:.4f}")

final_predictions_xgb = best_xgb_model.predict(X_test_final)

final_predictions_xgb = np.clip(final_predictions_xgb, 10, 100)
final_predictions_xgb = np.round(final_predictions_xgb).astype(int)

submission_xgb = pd.DataFrame({
    'Id': test_ids,
    'Recovery Index': final_predictions_xgb
})

submission_xgb.to_csv('output/submission_xgboost.csv', index=False)
print("Successfully created 'submission_xgboost.csv'!")
print(submission_xgb.head())