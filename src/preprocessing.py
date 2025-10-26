
import pandas as pd

from sklearn.preprocessing import StandardScaler

import numpy as np



def create_features(df):
    # Polynomial features for the most important variables
    df['Health_Score_sq'] = df['Initial Health Score']**2
    df['Therapy_Hours_sq'] = df['Therapy Hours']**2
    df['Health_Score_cub'] = df['Initial Health Score']**3
    df['Therapy_Hours_cub'] = df['Therapy Hours']**3
    
    # Interaction features
    df['Therapy_Health_Interaction'] = df['Therapy Hours'] * df['Initial Health Score']
    df['Sleep_FollowUp_Interaction'] = df['Average Sleep Hours'] * df['Follow-Up Sessions']
    df['Health_Sleep_Interaction'] = df['Initial Health Score'] * df['Average Sleep Hours']
    df['Therapy_FollowUp_Interaction'] = df['Therapy Hours'] * df['Follow-Up Sessions']
    df['Health_FollowUp_Interaction'] = df['Initial Health Score'] * df['Follow-Up Sessions']
    df['Sleep_Therapy_Interaction'] = df['Average Sleep Hours'] * df['Therapy Hours']
    
    # New Feature Engineering:
    # Log transformation for potentially skewed features (e.g., Therapy Hours)
    df['Therapy_Hours_log'] = np.log1p(df['Therapy Hours']) # log1p handles zero values gracefully
    df['Initial_Health_Score_log'] = np.log1p(df['Initial Health Score'])
    df['Average_Sleep_Hours_log'] = np.log1p(df['Average Sleep Hours'])

    # More interaction terms
    df['Health_Score_Therapy_Hours_Ratio'] = df['Initial Health Score'] / (df['Therapy Hours'] + 1e-6) # Add small epsilon to avoid division by zero
    df['FollowUp_Sleep_Ratio'] = df['Follow-Up Sessions'] / (df['Average Sleep Hours'] + 1e-6)

    # Interaction with 'Lifestyle Activities'
    df['Lifestyle_Health_Interaction'] = df['Lifestyle Activities'] * df['Initial Health Score']
    df['Lifestyle_Therapy_Interaction'] = df['Lifestyle Activities'] * df['Therapy Hours']
    df['Lifestyle_Sleep_Interaction'] = df['Lifestyle Activities'] * df['Average Sleep Hours']
    df['Lifestyle_FollowUp_Interaction'] = df['Lifestyle Activities'] * df['Follow-Up Sessions']
    
    return df



def preprocess_data(df, is_train=True, scaler=None):

    # Handle Categorical Variables: Convert 'Lifestyle Activities' from 'Yes'/'No' to 1/0
    df['Lifestyle Activities'] = df['Lifestyle Activities'].map({'Yes': 1, 'No': 0})

    # Create new features
    df = create_features(df.copy())

    



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



