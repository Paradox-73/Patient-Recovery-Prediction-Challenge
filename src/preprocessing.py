
import pandas as pd

from sklearn.preprocessing import StandardScaler

import numpy as np



def create_features(df):

    # Interaction features

    df['Therapy_Health_Interaction'] = df['Therapy Hours'] * df['Initial Health Score']

    df['Sleep_FollowUp_Interaction'] = df['Average Sleep Hours'] * df['Follow-Up Sessions']

    

    # Log transformations for potentially skewed numerical features (add 1 to handle zeros)

    for col in ['Therapy Hours', 'Initial Health Score', 'Average Sleep Hours', 'Follow-Up Sessions']:

        if col in df.columns:

            df[f'{col}_log'] = np.log1p(df[col])

    

    return df



def preprocess_data(df, is_train=True, scaler=None):

    # Create new features

    df = create_features(df.copy())



    # Handle Categorical Variables: Convert 'Lifestyle Activities' from 'Yes'/'No' to 1/0

    df['Lifestyle Activities'] = df['Lifestyle Activities'].map({'Yes': 1, 'No': 0})



    # Check for Missing Values (and impute if necessary - for this dataset, we assume no missing values based on plan.md)

    if df.isnull().sum().sum() > 0:

        print("Warning: Missing values detected. Imputation strategy not explicitly defined in plan.md.")

        for col in df.select_dtypes(include=['number']).columns:

            if df[col].isnull().any():

                df[col].fillna(df[col].median(), inplace=True)



    # Feature Scaling: Apply StandardScaler to numerical features

    numerical_features = df.select_dtypes(include=['number']).columns.tolist()

    if 'Id' in numerical_features:

        numerical_features.remove('Id')

    if 'Recovery Index' in numerical_features and not is_train:

        numerical_features.remove('Recovery Index') # Don't scale target during prediction



    if is_train:

        scaler = StandardScaler()

        df[numerical_features] = scaler.fit_transform(df[numerical_features])

        return df, scaler

    else:

        if scaler is None:

            raise ValueError("Scaler must be provided for test data preprocessing.")

        df[numerical_features] = scaler.transform(df[numerical_features])

        return df, scaler # Return scaler even if not fitted, for consistency



