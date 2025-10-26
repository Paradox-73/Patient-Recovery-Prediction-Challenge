import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import itertools

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

train_df_processed = train_df.drop('Id', axis=1)
test_df_processed = test_df.drop('Id', axis=1)

train_df_processed['Lifestyle Activities'] = train_df_processed['Lifestyle Activities'].map({'Yes': 1, 'No': 0})
test_df_processed['Lifestyle Activities'] = test_df_processed['Lifestyle Activities'].map({'Yes': 1, 'No': 0})

TARGET_COL = "Recovery Index"
SUBMISSION_FILENAME = "output/submission_feinter1.csv"
DROP_MISSING_FRAC = 0.3
OPS = ("product", "sum", "diff", "ratio")
MAX_PAIRS = None

train_df_new = train_df_processed.copy()
test_df_new  = test_df_processed.copy()

numeric_cols = train_df_new.select_dtypes(include=[np.number]).columns.tolist()
if TARGET_COL in numeric_cols:
    numeric_cols.remove(TARGET_COL)

pairs = list(itertools.combinations(numeric_cols, 2))
if MAX_PAIRS is not None:
    pairs = pairs[:MAX_PAIRS]

for a, b in pairs:
    if "product" in OPS:
        train_df_new[f"{a}__x__{b}"] = train_df_new[a]*train_df_new[b]
        test_df_new[f"{a}__x__{b}"]  = test_df_new[a]*test_df_new[b]
    if "sum" in OPS:
        train_df_new[f"{a}__plus__{b}"] = train_df_new[a]+train_df_new[b]
        test_df_new[f"{a}__plus__{b}"]  = test_df_new[a]+test_df_new[b]
    if "diff" in OPS:
        train_df_new[f"{a}__minus__{b}"] = train_df_new[a]-train_df_new[b]
        test_df_new[f"{a}__minus__{b}"]  = test_df_new[a]-test_df_new[b]
    if "ratio" in OPS:
        train_df_new[f"{a}__div__{b}"] = train_df_new[a]/train_df_new[b].replace(0,np.nan)
        test_df_new[f"{a}__div__{b}"]  = test_df_new[a]/test_df_new[b].replace(0,np.nan)
        train_df_new[f"{b}__div__{a}"] = train_df_new[b]/train_df_new[a].replace(0,np.nan)
        test_df_new[f"{b}__div__{a}"]  = test_df_new[b]/test_df_new[a].replace(0,np.nan)

for col in numeric_cols:
    train_df_new[f"{col}__squared"] = train_df_new[col]**2
    test_df_new[f"{col}__squared"]  = test_df_new[col]**2

if "Therapy Hours" in train_df_new.columns and "Initial Health Score" in train_df_new.columns:
    train_df_new["Therapy_Health_Interaction"] = train_df_new["Therapy Hours"] * train_df_new["Initial Health Score"]
    test_df_new["Therapy_Health_Interaction"]  = test_df_new["Therapy Hours"] * test_df_new["Initial Health Score"]

if "Lifestyle Activities" in train_df_new.columns and "Follow-Up Sessions" in train_df_new.columns:
    train_df_new["Lifestyle_Followup_Interaction"] = train_df_new["Lifestyle Activities"] * train_df_new["Follow-Up Sessions"]
    test_df_new["Lifestyle_Followup_Interaction"]  = test_df_new["Lifestyle Activities"] * test_df_new["Follow-Up Sessions"]

y = train_df_new[TARGET_COL]
X = train_df_new.drop(TARGET_COL, axis=1)
X_test_final = test_df_new.reindex(columns=X.columns)

na_counts = X.isna().sum()
drop_cols = na_counts[na_counts > X.shape[0]*DROP_MISSING_FRAC].index.tolist()
if drop_cols:
    X = X.drop(columns=drop_cols)
    X_test_final = X_test_final.drop(columns=drop_cols, errors='ignore')

imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(X)
X_test_imp = imputer.transform(X_test_final)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)
X_test_scaled = scaler.transform(X_test_imp)

lr = LinearRegression()
lr.fit(X_scaled, y)

preds = lr.predict(X_test_scaled)

preds_final = np.clip(preds, 10, 100)

submission_df = pd.DataFrame({
    "Id": test_df["Id"].values,
    "Recovery Index": preds_final
})

submission_df.to_csv(SUBMISSION_FILENAME, index=False)
print(f"Submission saved to {SUBMISSION_FILENAME}")
submission_df.head()