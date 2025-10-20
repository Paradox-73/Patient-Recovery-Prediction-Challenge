
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
train_df = pd.read_csv(r'E:\IIITB\ML\Project\data\train.csv')
test_df = pd.read_csv(r'E:\IIITB\ML\Project\data\test.csv')

print("Train Data Info:")
train_df.info()
print("\nTrain Data Description:")
train_df.describe()

print("\nTest Data Info:")
test_df.info()
print("\nTest Data Description:")
test_df.describe()

# Visualize Distributions for numerical features
numerical_features = ['Therapy Hours', 'Initial Health Score', 'Average Sleep Hours', 'Follow-Up Sessions', 'Recovery Index']
for feature in numerical_features:
    plt.figure(figsize=(8, 5))
    sns.histplot(train_df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

# Count plot for 'Lifestyle Activities'
plt.figure(figsize=(8, 5))
sns.countplot(x='Lifestyle Activities', data=train_df)
plt.title('Distribution of Lifestyle Activities')
plt.show()

# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(train_df[numerical_features].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Box plots for 'Lifestyle Activities' vs 'Recovery Index'
plt.figure(figsize=(8, 5))
sns.boxplot(x='Lifestyle Activities', y='Recovery Index', data=train_df)
plt.title('Recovery Index by Lifestyle Activities')
plt.show()
