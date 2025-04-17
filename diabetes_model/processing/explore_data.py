# Loading the Required Packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Reading Our Dataset
diabetes_dataset = pd.read_csv('diabetes_model/datasets/diabetes_prediction_dataset.csv')
print(f"\nDiabetes Dataset Shape: {diabetes_dataset.shape}")
print(f"\nDiabetes Dataset Columns: {diabetes_dataset.columns.tolist()}")
print(f"\nDiabetes Dataset Head: {diabetes_dataset.head()}")
print(f"\nDiabetes Dataset Info: {diabetes_dataset.info()}")  
print(f"Diabetes Dataset Description:\n {diabetes_dataset.describe()}")
print(f"\nDiabetes Dataset Missing Values: {diabetes_dataset.isnull().sum()}")
print(f"\nDiabetes Dataset Duplicates: {diabetes_dataset.duplicated().sum()}")
#print(f"\nDiabetes Dataset Correlation: {diabetes_dataset.corr()}")
print(f"\nDiabetes Dataset Target Distribution:\n {diabetes_dataset['diabetes'].value_counts()}")
print(f"\nDiabetes Dataset Target Distribution Percentage:\n {diabetes_dataset['diabetes'].value_counts(normalize=True)}")
print(f"\nDiabetes Dataset Target Distribution Percentage:\n {diabetes_dataset['diabetes'].value_counts(normalize=True).plot(kind='bar')}")
plt.title('Diabetes Target Distribution')
plt.xlabel('Diabetes')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.plot
plt.legend(['No Diabetes', 'Diabetes']) 
plt.savefig('diabetes_model/diabetes_target_distribution.png')

plt.xlabel('Diabetes')
plt.ylabel('Count')