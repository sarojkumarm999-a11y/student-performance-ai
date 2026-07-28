import pandas as pd


data = pd.read_csv("student_data.csv")


print("First 5 rows:")
print(data.head())


print("\nDataset Info:")
print(data.info())


print("\nSummary:")
print(data.describe())